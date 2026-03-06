/*
 * GDN Recurrent Kernel Testing Harness
 *
 * Tests the sm90_bf16_gdn_recurrent kernel for both:
 * - Variable length (varlen) mode: 3D tensors (total_tokens, num_heads, head_dim)
 * - Padded (fixed length) mode: 4D tensors (batch, seq_len, num_heads, head_dim)
 *
 * Usage:
 *   ./test_gdn_recurrent [--mode <varlen|padded|all>] [--verbose]
 */

#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <gdn_cuda/utils.h>
#include <torch/torch.h>

#include <apis/gdn_forward.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "test_utils.h"

using test_utils::check_tensor_close;
using test_utils::inspect_tensor;
using test_utils::shape_to_string;

// Test configuration for different shapes
struct GDNTestShape {
    int64_t batch_size;
    int64_t seq_len;  // For padded mode, or max seq_len for varlen
    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t shape_k;  // Key/Query head dimension
    int64_t shape_v;  // Value head dimension
    const char* description;
};

// Test shapes covering various configurations
static const std::vector<GDNTestShape> test_shapes = {
    // Small shapes for quick testing
    {1, 32, 4, 4, 64, 64, "Small (B=1, T=32, H=4, K=64, V=64)"},
    {2, 32, 4, 4, 64, 64, "Small batch (B=2, T=32, H=4, K=64, V=64)"},

    // // Medium shapes
    {1, 16, 8, 8, 128, 128, "Medium (B=1, T=16, H=8, K=128, V=128)"},
    {2, 16, 8, 8, 128, 128, "Medium batch (B=2, T=16, H=8, K=128, V=128)"},

    // GQA configurations (num_k_heads != num_v_heads)
    {1, 8, 8, 32, 64, 64, "GQA 4:1 (B=1, T=32, KH=8, VH=2, K=64, V=64)"},
    {16, 8, 16, 32, 128, 128, "GQA 4:1 larger (B=256, T=16, KH=16, VH=32, K=128, V=128)"},

    // Qwen3-like shapes
    {1, 16, 14, 28, 64, 64, "Qwen3-like (B=1, T=1, KH=14, VH=2, K=64, V=64)"},
};

struct TestConfig {
    bool test_varlen = true;
    bool test_padded = true;
    bool verbose = false;
    bool test_initial_state = false;
    bool test_qk_norm = true;
    bool test_step_state = false;  // Test per-step state storage (kStoreStepState)
    bool test_gate = false;        // Test with gating values
    float atol = 1e-2f;
    float rtol = 1e-2f;
};

void print_usage(const char* program_name) {
    std::cout << "GDN Recurrent Kernel Testing Harness\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --mode <varlen|padded|all>  Test mode (default: all)\n";
    std::cout
        << "  --step-state                 Test per-step state storage (kStoreStepState=true)\n";
    std::cout << "  --gate                       Test with gating values\n";
    std::cout << "  --verbose                    Print detailed output\n";
    std::cout << "  --atol <tol>                Absolute tolerance (default: 0.01)\n";
    std::cout << "  --rtol <tol>                Relative tolerance (default: 0.01)\n";
    std::cout << "  --help                      Show this help message\n";
}

bool parse_args(int argc, char** argv, TestConfig& config) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "varlen") {
                config.test_varlen = true;
                config.test_padded = false;
            } else if (mode == "padded") {
                config.test_varlen = false;
                config.test_padded = true;
            } else if (mode == "all") {
                config.test_varlen = true;
                config.test_padded = true;
            } else {
                std::cerr << "Error: Invalid mode '" << mode << "'\n";
                return false;
            }
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--step-state") {
            config.test_step_state = true;
        } else if (arg == "--initial-state") {
            config.test_initial_state = true;
        } else if (arg == "--gate") {
            config.test_gate = true;
        } else if (arg == "--atol" && i + 1 < argc) {
            config.atol = std::stof(argv[++i]);
        } else if (arg == "--rtol" && i + 1 < argc) {
            config.rtol = std::stof(argv[++i]);
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

// Reference implementation for PADDED mode
// state_t = state_{t-1} - beta * (state_{t-1} @ k - v) * k^T / ||k||^2
// output_t = state_t @ q_t / ||q_t||^2 (if qk_norm) or state_t @ q_t
torch::Tensor gdn_recurrent_reference_padded(
    const torch::Tensor& Q,           // (batch, seq_len, num_k_heads, shape_k)
    const torch::Tensor& K,           // (batch, seq_len, num_k_heads, shape_k)
    const torch::Tensor& V,           // (batch, seq_len, num_v_heads, shape_v)
    const torch::Tensor& init_state,  // (batch, num_v_heads, shape_v, shape_k) - read only
    torch::Tensor& final_state,       // output: (batch, ...) or (batch*seq_len, ...)
    const torch::Tensor& beta,        // (batch, seq_len, num_v_heads)
    bool is_qk_norm, float scale = 1.0f, bool store_step_state = false,
    const std::optional<torch::Tensor>& gate = std::nullopt) {
    auto options = Q.options().dtype(torch::kFloat32);

    int64_t batch_size = Q.size(0);
    int64_t seq_len = Q.size(1);
    int64_t num_k_heads = Q.size(2);
    int64_t num_v_heads = V.size(2);
    int64_t shape_k = Q.size(3);
    int64_t shape_v = V.size(3);

    // Output tensor: one output per token
    torch::Tensor output = torch::zeros({batch_size, seq_len, num_v_heads, shape_v}, options);

    // Convert to float32 for precision
    auto Q_f = Q.to(torch::kFloat32);
    auto K_f = K.to(torch::kFloat32);
    auto V_f = V.to(torch::kFloat32);
    auto init_state_f = init_state.to(torch::kFloat32);
    auto beta_f = beta.to(torch::kFloat32);
    auto gate_f = gate.has_value() ? std::optional<torch::Tensor>(gate.value().to(torch::kFloat32))
                                   : std::nullopt;
    auto final_state_f = final_state.to(torch::kFloat32);

    for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t h = 0; h < num_v_heads; h++) {
            // Get state for this batch/head: (shape_v, shape_k)
            auto S = init_state_f[b][h].clone();  // (shape_v, shape_k)

            // Map K head index for GQA)
            int64_t k_head = (h * num_k_heads) / num_v_heads;

            for (int64_t t = 0; t < seq_len; t++) {
                // Get q, k, v, beta for this timestep
                auto q = Q_f[b][t][k_head];  // (shape_k,)
                auto k = K_f[b][t][k_head];  // (shape_k,)
                auto v = V_f[b][t][h];       // (shape_v,)
                float beta_t = beta_f[b][t][h].item<float>();
                if (gate_f.has_value()) {
                    float g_t = gate_f.value()[b][t][h].item<float>();
                    float gate_mult = std::exp(g_t);
                    S = S * gate_mult;
                }

                // Compute k norm for normalization
                float k_norm_sq = torch::dot(k, k).item<float>();
                float inv_k_norm = is_qk_norm ? (1.0f / std::sqrt(k_norm_sq + 1e-6f)) : 1.0f;

                // S @ k -> (shape_v,)
                auto Sk = torch::mv(S, k);  // (shape_v,)

                // (S @ k) * inv_k_norm - v
                auto diff = Sk * inv_k_norm - v;  // (shape_v,)

                // State update: S = S - beta * diff * k^T * inv_k_norm
                auto outer = torch::outer(diff, k);  // (shape_v, shape_k)
                S = S - beta_t * outer * inv_k_norm;

                // Compute output for this timestep: S @ q (with optional q normalization)
                float q_norm_sq = torch::dot(q, q).item<float>();
                float inv_q_norm = is_qk_norm ? (1.0f / std::sqrt(q_norm_sq + 1e-6f)) : 1.0f;

                auto out = torch::mv(S, q) * (inv_q_norm * scale);  // (shape_v,)
                output[b][t][h] = out;

                // Store step state if requested
                if (store_step_state) {
                    int64_t token_idx = b * seq_len + t;
                    final_state_f[token_idx][h] = S;
                }
            }

            // Store per-batch final state
            if (!store_step_state) {
                final_state_f[b][h] = S;
            }
        }
    }

    // Copy final state back
    final_state.copy_(final_state_f.to(final_state.dtype()));

    return output.to(Q.dtype());
}

// Reference implementation for VARLEN mode
// state_t = state_{t-1} - beta * (state_{t-1} @ k - v) * k^T / ||k||^2
// output_t = state_t @ q_t / ||q_t||^2 (if qk_norm) or state_t @ q_t
torch::Tensor gdn_recurrent_reference_varlen(
    const torch::Tensor& Q,           // (total_tokens, num_k_heads, shape_k)
    const torch::Tensor& K,           // (total_tokens, num_k_heads, shape_k)
    const torch::Tensor& V,           // (total_tokens, num_v_heads, shape_v)
    const torch::Tensor& init_state,  // (batch_size, num_v_heads, shape_v, shape_k) - read only
    torch::Tensor& final_state,       // output: (batch, ...) or (total_tokens, ...)
    const torch::Tensor& beta,        // (total_tokens, num_v_heads)
    bool is_qk_norm, float scale,
    const std::vector<int>& cu_seqlens,  // cumulative sequence lengths
    bool store_step_state = false, const std::optional<torch::Tensor>& gate = std::nullopt) {
    auto options = Q.options().dtype(torch::kFloat32);

    int64_t total_tokens = Q.size(0);
    int64_t num_k_heads = Q.size(1);
    int64_t num_v_heads = V.size(1);
    int64_t shape_k = Q.size(2);
    int64_t shape_v = V.size(2);
    int64_t batch_size = cu_seqlens.size() - 1;

    // Output tensor: one output per token
    torch::Tensor output = torch::zeros({total_tokens, num_v_heads, shape_v}, options);

    // Convert to float32 for precision
    auto Q_f = Q.to(torch::kFloat32);
    auto K_f = K.to(torch::kFloat32);
    auto V_f = V.to(torch::kFloat32);
    auto init_state_f = init_state.to(torch::kFloat32);
    auto beta_f = beta.to(torch::kFloat32);
    auto gate_f = gate.has_value() ? std::optional<torch::Tensor>(gate.value().to(torch::kFloat32))
                                   : std::nullopt;
    auto final_state_f = final_state.to(torch::kFloat32);

    for (int64_t b = 0; b < batch_size; b++) {
        int64_t seq_start = cu_seqlens[b];
        int64_t seq_end = cu_seqlens[b + 1];
        int64_t seq_len = seq_end - seq_start;

        for (int64_t h = 0; h < num_v_heads; h++) {
            // Get state for this batch/head: (shape_v, shape_k)
            auto S = init_state_f[b][h].clone();  // (shape_v, shape_k)

            // Map K head index for GQA
            int64_t k_head = (h * num_k_heads) / num_v_heads;

            for (int64_t t = 0; t < seq_len; t++) {
                int64_t token_idx = seq_start + t;

                // Get q, k, v, beta for this timestep
                auto q = Q_f[token_idx][k_head];  // (shape_k,)
                auto k = K_f[token_idx][k_head];  // (shape_k,)
                auto v = V_f[token_idx][h];       // (shape_v,)
                float beta_t = beta_f[token_idx][h].item<float>();
                if (gate_f.has_value()) {
                    float g_t = gate_f.value()[token_idx][h].item<float>();
                    float gate_mult = std::exp(g_t);
                    S = S * gate_mult;
                }

                // Compute k norm for normalization
                float k_norm_sq = torch::dot(k, k).item<float>();
                float inv_k_norm = is_qk_norm ? (1.0f / std::sqrt(k_norm_sq + 1e-6f)) : 1.0f;

                // S @ k -> (shape_v,)
                auto Sk = torch::mv(S, k);  // (shape_v,)

                // (S @ k) * inv_k_norm - v
                auto diff = Sk * inv_k_norm - v;  // (shape_v,)

                // State update: S = S - beta * diff * k^T * inv_k_norm
                auto outer = torch::outer(diff, k);  // (shape_v, shape_k)
                S = S - beta_t * outer * inv_k_norm;

                // Compute output for this timestep: S @ q (with optional q normalization)
                float q_norm_sq = torch::dot(q, q).item<float>();
                float inv_q_norm = is_qk_norm ? (1.0f / std::sqrt(q_norm_sq + 1e-6f)) : 1.0f;

                auto out = torch::mv(S, q) * (inv_q_norm * scale);  // (shape_v,)
                output[token_idx][h] = out;

                // Store step state if requested
                if (store_step_state) {
                    final_state_f[token_idx][h] = S;
                }
            }

            // Store per-batch final state
            if (!store_step_state) {
                final_state_f[b][h] = S;
            }
        }
    }

    // Copy final state back
    final_state.copy_(final_state_f.to(final_state.dtype()));

    return output.to(Q.dtype());
}

// Wrapper function
torch::Tensor gdn_recurrent_reference(const torch::Tensor& Q, const torch::Tensor& K,
                                      const torch::Tensor& V, const torch::Tensor& init_state,
                                      torch::Tensor& final_state, const torch::Tensor& beta,
                                      bool is_qk_norm, float scale = 1.0f,
                                      const std::vector<int>& cu_seqlens = {},
                                      bool store_step_state = false,
                                      const std::optional<torch::Tensor>& gate = std::nullopt) {
    bool is_varlen = !cu_seqlens.empty();

    if (is_varlen) {
        return gdn_recurrent_reference_varlen(Q, K, V, init_state, final_state, beta, is_qk_norm,
                                              scale, cu_seqlens, store_step_state, gate);
    } else {
        return gdn_recurrent_reference_padded(Q, K, V, init_state, final_state, beta, is_qk_norm,
                                              scale, store_step_state, gate);
    }
}

// Test padded (fixed length) mode
bool test_padded_mode(const GDNTestShape& shape, const TestConfig& config) {
    std::cout << "\n--- Testing Padded Mode: " << shape.description << " ---\n";
    if (config.test_step_state) {
        std::cout << "  [Step State Mode: Storing per-step states]\n";
    }

    torch::Device device = torch::kCUDA;
    torch::TensorOptions bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create input tensors - 4D for padded mode: (batch, seq_len, num_heads, head_dim)
    torch::Tensor Q = torch::normal(
        0.0f, 1.0f, {shape.batch_size, shape.seq_len, shape.num_k_heads, shape.shape_k},
        std::nullopt, bf16_opts);
    torch::Tensor K = torch::normal(
        0.0f, 1.0f, {shape.batch_size, shape.seq_len, shape.num_k_heads, shape.shape_k},
        std::nullopt, bf16_opts);
    torch::Tensor V = torch::normal(
        0.0f, 1.0f, {shape.batch_size, shape.seq_len, shape.num_v_heads, shape.shape_v},
        std::nullopt, bf16_opts);

    // Initial state tensor: (batch, num_v_heads, shape_v, shape_k)
    torch::Tensor init_state =
        config.test_initial_state
            ? torch::normal(0.0f, 1.0f,
                            {shape.batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k},
                            std::nullopt, bf16_opts)
            : torch::zeros({shape.batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k},
                           bf16_opts);

    // Final state tensor shape depends on step_state mode
    int64_t total_tokens = shape.batch_size * shape.seq_len;
    int64_t state_dim0 = config.test_step_state ? total_tokens : shape.batch_size;
    torch::Tensor final_state_ref =
        torch::zeros({state_dim0, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);
    torch::Tensor final_state_kernel =
        torch::zeros({state_dim0, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);

    // Beta tensor: (batch, seq_len, num_v_heads) - per-head scalar
    torch::Tensor beta =
        torch::normal(0.0f, 1.0f, {shape.batch_size, shape.seq_len, shape.num_v_heads},
                      std::nullopt, bf16_opts) *
        0.1f;
    torch::TensorOptions f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    std::optional<torch::Tensor> gate_log_ref_opt = std::nullopt;
    std::optional<at::Tensor> gate_opt = std::nullopt;
    if (config.test_gate) {
        // Both reference and kernel consume log-gate and apply exp(g) internally.
        torch::Tensor gate_log = torch::log(torch::sigmoid(
            torch::rand({shape.batch_size, shape.seq_len, shape.num_v_heads}, f32_opts)));
        torch::Tensor gate_log_bf16 = gate_log.to(torch::kBFloat16);
        gate_log_ref_opt.emplace(gate_log);
        gate_opt.emplace(gate_log_bf16);
    }

    // Output tensor: (batch, seq_len, num_v_heads, shape_v) - one output per token
    torch::Tensor out = torch::zeros(
        {shape.batch_size, shape.seq_len, shape.num_v_heads, shape.shape_v}, bf16_opts);

    if (config.verbose) {
        std::cout << "Q shape: " << shape_to_string(Q.sizes().vec()) << "\n";
        std::cout << "K shape: " << shape_to_string(K.sizes().vec()) << "\n";
        std::cout << "V shape: " << shape_to_string(V.sizes().vec()) << "\n";
        std::cout << "Init state shape: " << shape_to_string(init_state.sizes().vec()) << "\n";
        std::cout << "Final state shape: " << shape_to_string(final_state_kernel.sizes().vec())
                  << "\n";
        std::cout << "Beta shape: " << shape_to_string(beta.sizes().vec()) << "\n";
        if (gate_log_ref_opt.has_value()) {
            std::cout << "Gate (reference log-g) shape: "
                      << shape_to_string(gate_log_ref_opt->sizes().vec()) << "\n";
            std::cout << "Gate (kernel log-g) shape: " << shape_to_string(gate_opt->sizes().vec())
                      << "\n";
        }
        std::cout << "Output shape: " << shape_to_string(out.sizes().vec()) << "\n";
    }

    // Compute scale factor: 1/sqrt(shape_k), matching FLA convention
    float scale = 1.0f / std::sqrt(static_cast<float>(shape.shape_k));

    // Compute reference
    auto ref_output =
        gdn_recurrent_reference(Q, K, V, init_state, final_state_ref, beta, config.test_qk_norm,
                                scale, {}, config.test_step_state, gate_log_ref_opt);

    // Create custom tensors
    at::Tensor q_custom = (Q);
    at::Tensor k_custom = (K);
    at::Tensor v_custom = (V);
    at::Tensor init_state_custom = (init_state);
    at::Tensor final_state_custom = (final_state_kernel);
    at::Tensor out_custom = (out);
    at::Tensor beta_custom = (beta);
    std::optional<at::Tensor> init_state_custom_opt = std::nullopt;
    if (config.test_initial_state) {
        init_state_custom_opt.emplace(init_state_custom);
    }
    // Run kernel
    auto start = std::chrono::high_resolution_clock::now();
    std::optional<at::Tensor> cu_seqlens_opt = std::nullopt;           // empty for padded
    std::optional<at::Tensor> num_accepted_tokens_opt = std::nullopt;  // not used in this test

    printf("beginning kernel execution\n");
    gdn_cuda::bf16_gdn_recurrent(q_custom, k_custom, v_custom, init_state_custom_opt,
                                 final_state_custom, out_custom, gate_opt, beta_custom,
                                 "kvt",  // compiled_dims
                                 stream, cu_seqlens_opt, num_accepted_tokens_opt,
                                 config.test_step_state, config.test_qk_norm, scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Kernel execution time: " << duration.count() << " us\n";

    // Compare output results
    std::cout << "--- Output Comparison ---\n";
    bool output_passed = check_tensor_close(ref_output, out_custom, config.atol, config.rtol);

    // Compare final state results
    std::cout << "--- " << (config.test_step_state ? "Step States" : "Final State")
              << " Comparison ---\n";
    bool state_passed =
        check_tensor_close(final_state_ref, final_state_custom, config.atol, config.rtol);

    if (config.verbose) {
        std::cout << "\n--- Detailed Comparison ---\n";
        std::cout << "Output (first 10 elements):\n";
        std::cout << "  Expected: ";
        inspect_tensor(ref_output, 10);
        std::cout << "  Actual:   ";
        inspect_tensor(out_custom, 10);

        std::cout << (config.test_step_state ? "Step States" : "Final State")
                  << " (first 10 elements):\n";
        std::cout << "  Expected: ";
        inspect_tensor(final_state_ref, 10);
        std::cout << "  Actual:   ";
        inspect_tensor(final_state_custom, 10);
    }

    bool passed = output_passed && state_passed;
    if (!passed) {
        std::cout << "\033[0;31mOutput passed: " << (output_passed ? "YES" : "NO")
                  << ", State passed: " << (state_passed ? "YES" : "NO") << "\033[0m\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));

    return passed;
}

// Varlen-specific test configurations with more batches for proper jagged testing
struct VarlenTestShape {
    int64_t batch_size;   // Number of sequences (should be larger to test scheduling)
    int64_t min_seq_len;  // Minimum sequence length
    int64_t max_seq_len;  // Maximum sequence length
    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t shape_k;
    int64_t shape_v;
    const char* description;
};

static const std::vector<VarlenTestShape> varlen_test_shapes = {
    // Highly jagged sequences
    {8, 16, 256, 4, 4, 64, 64, "Jagged 8 seqs (16-256 tokens)"},
    {16, 32, 512, 4, 4, 64, 64, "Jagged 16 seqs (32-512 tokens)"},
    {32, 16, 128, 4, 4, 64, 64, "Many short jagged (32 seqs, 16-128 tokens)"},

    // GQA with jagged
    {8, 32, 256, 8, 24, 64, 64, "GQA 4:1 jagged (8 seqs)"},
    {12, 64, 384, 16, 32, 128, 128, "GQA 4:1 jagged (12 seqs, 64-384 tokens)"},

    // Stress test with many sequences
    {64, 16, 96, 4, 4, 64, 64, "Stress test (64 seqs, 16-96 tokens)"},

    // // Qwen3-like with jagged
    {8, 32, 512, 14, 28, 64, 64, "Qwen3-like jagged (8 seqs)"},
};

// Test varlen (variable length) mode
bool test_varlen_mode(const VarlenTestShape& shape, const TestConfig& config) {
    std::cout << "\n--- Testing Varlen Mode: " << shape.description << " ---\n";
    if (config.test_step_state) {
        std::cout << "  [Step State Mode: Storing per-step states]\n";
    }

    torch::Device device = torch::kCUDA;
    torch::TensorOptions bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Generate truly random, jagged sequence lengths
    std::vector<int> seq_lens(shape.batch_size);
    alignas(8) std::vector<int> cu_seqlens(shape.batch_size + 1);
    cu_seqlens[0] = 0;

    // Use a seeded RNG for reproducibility
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> len_dist(static_cast<int>(shape.min_seq_len),
                                                static_cast<int>(shape.max_seq_len));

    int64_t total_tokens = 0;
    for (int64_t b = 0; b < shape.batch_size; b++) {
        seq_lens[b] = len_dist(rng);
        total_tokens += seq_lens[b];
        cu_seqlens[b + 1] = static_cast<int>(total_tokens);
    }

    if (config.verbose) {
        std::cout << "Sequence lengths: [";
        for (int64_t b = 0; b < shape.batch_size; b++) {
            std::cout << seq_lens[b];
            if (b < shape.batch_size - 1)
                std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "Total tokens: " << total_tokens << "\n";
    }

    // Create input tensors - 3D for varlen mode: (total_tokens, num_heads, head_dim)
    torch::Tensor Q = torch::normal(0.0f, 1.0f, {total_tokens, shape.num_k_heads, shape.shape_k},
                                    std::nullopt, bf16_opts);
    torch::Tensor K = torch::normal(0.0f, 1.0f, {total_tokens, shape.num_k_heads, shape.shape_k},
                                    std::nullopt, bf16_opts);
    torch::Tensor V = torch::normal(0.0f, 1.0f, {total_tokens, shape.num_v_heads, shape.shape_v},
                                    std::nullopt, bf16_opts);

    // Initial state tensor: (batch, num_v_heads, shape_v, shape_k)
    torch::Tensor init_state =
        config.test_initial_state
            ? torch::normal(0.0f, 1.0f,
                            {shape.batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k},
                            std::nullopt, bf16_opts)
            : torch::zeros({shape.batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k},
                           bf16_opts);

    // Final state tensor shape depends on step_state mode
    int64_t state_dim0 = config.test_step_state ? total_tokens : shape.batch_size;
    torch::Tensor final_state_ref =
        torch::zeros({state_dim0, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);
    torch::Tensor final_state_kernel =
        torch::zeros({state_dim0, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);

    // Beta tensor: (total_tokens, num_v_heads) - per-token, per-head scalar
    torch::Tensor beta =
        torch::normal(0.0f, 1.0f, {total_tokens, shape.num_v_heads}, std::nullopt, bf16_opts) *
        0.1f;
    torch::TensorOptions f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    std::optional<torch::Tensor> gate_log_ref_opt = std::nullopt;
    std::optional<at::Tensor> gate_opt = std::nullopt;
    if (config.test_gate) {
        // Both reference and kernel consume log-gate and apply exp(g) internally.
        torch::Tensor gate_log =
            torch::log(torch::sigmoid(torch::rand({total_tokens, shape.num_v_heads}, f32_opts)));
        torch::Tensor gate_log_bf16 = gate_log.to(torch::kBFloat16);
        gate_log_ref_opt.emplace(gate_log);
        gate_opt.emplace(gate_log_bf16);
    }

    // Output tensor: (total_tokens, num_v_heads, shape_v) - one output per token
    torch::Tensor out = torch::zeros({total_tokens, shape.num_v_heads, shape.shape_v}, bf16_opts);

    if (config.verbose) {
        std::cout << "Q shape: " << shape_to_string(Q.sizes().vec()) << "\n";
        std::cout << "K shape: " << shape_to_string(K.sizes().vec()) << "\n";
        std::cout << "V shape: " << shape_to_string(V.sizes().vec()) << "\n";
        std::cout << "Init state shape: " << shape_to_string(init_state.sizes().vec()) << "\n";
        std::cout << "Final state shape: " << shape_to_string(final_state_kernel.sizes().vec())
                  << "\n";
        std::cout << "Beta shape: " << shape_to_string(beta.sizes().vec()) << "\n";
        if (gate_log_ref_opt.has_value()) {
            std::cout << "Gate (reference log-g) shape: "
                      << shape_to_string(gate_log_ref_opt->sizes().vec()) << "\n";
            std::cout << "Gate (kernel log-g) shape: " << shape_to_string(gate_opt->sizes().vec())
                      << "\n";
        }
        std::cout << "Output shape: " << shape_to_string(out.sizes().vec()) << "\n";
    }

    // Compute scale factor: 1/sqrt(shape_k), matching FLA convention
    float scale = 1.0f / std::sqrt(static_cast<float>(shape.shape_k));

    // Compute reference
    auto ref_output =
        gdn_recurrent_reference(Q, K, V, init_state, final_state_ref, beta, config.test_qk_norm,
                                scale, cu_seqlens, config.test_step_state, gate_log_ref_opt);

    // Create custom tensors
    at::Tensor q_custom = (Q);
    at::Tensor k_custom = (K);
    at::Tensor v_custom = (V);
    at::Tensor init_state_custom = (init_state);
    at::Tensor final_state_custom = (final_state_kernel);
    at::Tensor out_custom = (out);
    at::Tensor beta_custom = (beta);

    std::optional<at::Tensor> init_state_custom_opt = std::nullopt;
    if (config.test_initial_state) {
        init_state_custom_opt.emplace(init_state_custom);
    }

    // Run kernel
    auto start = std::chrono::high_resolution_clock::now();

    // Create cu_seqlens tensor on device
    torch::Tensor cu_seqlens_torch =
        torch::tensor(cu_seqlens, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    at::Tensor cu_seqlens_custom = (cu_seqlens_torch);
    std::optional<at::Tensor> cu_seqlens_opt;

    cu_seqlens_opt.emplace(std::move(cu_seqlens_custom));

    std::optional<at::Tensor> num_accepted_tokens_opt = std::nullopt;  // not used in this test

    gdn_cuda::bf16_gdn_recurrent(q_custom, k_custom, v_custom, init_state_custom_opt,
                                 final_state_custom, out_custom, gate_opt, beta_custom,
                                 "kv",  // compiled_dims
                                 stream, cu_seqlens_opt, num_accepted_tokens_opt,
                                 config.test_step_state, config.test_qk_norm, scale);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Kernel execution time: " << duration.count() << " us\n";

    // Compare output results
    std::cout << "--- Output Comparison ---\n";
    bool output_passed = check_tensor_close(ref_output, out_custom, config.atol, config.rtol);

    // Compare final state results
    std::cout << "--- " << (config.test_step_state ? "Step States" : "Final State")
              << " Comparison ---\n";
    bool state_passed =
        check_tensor_close(final_state_ref, final_state_custom, config.atol, config.rtol);

    if (config.verbose) {
        std::cout << "\n--- Detailed Comparison ---\n";
        std::cout << "Output (first 10 elements):\n";
        std::cout << "  Expected: ";
        inspect_tensor(ref_output, 10);
        std::cout << "  Actual:   ";
        inspect_tensor(out_custom, 10);

        std::cout << (config.test_step_state ? "Step States" : "Final State")
                  << " (first 10 elements):\n";
        std::cout << "  Expected: ";
        inspect_tensor(final_state_ref, 10);
        std::cout << "  Actual:   ";
        inspect_tensor(final_state_custom, 10);
    }

    bool passed = output_passed && state_passed;
    if (!passed) {
        std::cout << "\033[0;31mOutput passed: " << (output_passed ? "YES" : "NO")
                  << ", State passed: " << (state_passed ? "YES" : "NO") << "\033[0m\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));

    return passed;
}

int main(int argc, char** argv) {
    TestConfig config;

    if (!parse_args(argc, argv, config)) {
        return 1;
    }

    // Check CUDA availability
    if (!torch::cuda::is_available()) {
        std::cerr << "Error: CUDA is not available\n";
        return 1;
    }

    // Initialize JIT compiler/runtime paths
    printf("library root path: %s\n", get_env<std::string>("LIBRARY_ROOT_PATH", "").c_str());
    printf("cuda home path: %s\n", get_env<std::string>("CUDA_HOME_PATH", "").c_str());
    gdn_cuda::init(get_env<std::string>("LIBRARY_ROOT_PATH", ""),
                   get_env<std::string>("CUDA_HOME_PATH"));
    std::cout << "============================================\n";
    std::cout << "GDN Recurrent Kernel Testing Harness\n";
    std::cout << "============================================\n";
    std::cout << "Test varlen mode: " << (config.test_varlen ? "yes" : "no") << "\n";
    std::cout << "Test padded mode: " << (config.test_padded ? "yes" : "no") << "\n";
    std::cout << "Test initial state: " << (config.test_initial_state ? "yes" : "no") << "\n";
    std::cout << "Test step state: " << (config.test_step_state ? "yes" : "no") << "\n";
    std::cout << "Test gate: " << (config.test_gate ? "yes" : "no") << "\n";
    std::cout << "Test QK norm: " << (config.test_qk_norm ? "yes" : "no") << "\n";

    int passed = 0, failed = 0;

    // Run padded mode tests
    if (config.test_padded) {
        std::cout << "\n=== Padded Mode Tests ===\n";
        for (const auto& shape : test_shapes) {
            bool result = test_padded_mode(shape, config);
            if (result) {
                std::cout << "\033[0;32m[PASSED] Padded: " << shape.description << "\033[0m\n";
                passed++;
            } else {
                std::cout << "\033[0;31m[FAILED] Padded: " << shape.description << "\033[0m\n";
                failed++;
            }
        }
    }

    // Run varlen mode tests with dedicated jagged test shapes
    if (config.test_varlen) {
        std::cout << "\n=== Varlen Mode Tests (Jagged Sequences) ===\n";
        for (const auto& shape : varlen_test_shapes) {
            bool result = test_varlen_mode(shape, config);
            if (result) {
                std::cout << "\033[0;32m[PASSED] Varlen: " << shape.description << "\033[0m\n";
                passed++;
            } else {
                std::cout << "\033[0;31m[FAILED] Varlen: " << shape.description << "\033[0m\n";
                failed++;
            }
        }
    }

    std::cout << "\n============================================\n";
    std::cout << "Test Summary\n";
    std::cout << "============================================\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";

    if (failed > 0) {
        std::cout << "\033[0;31mSome tests FAILED\033[0m\n";
        return 1;
    } else {
        std::cout << "\033[0;32mAll tests PASSED\033[0m\n";
        return 0;
    }
}
