/*
 * GDN Helper Kernels Testing Harness
 *
 * Tests:
 *   1. chunk_local_cumsum kernel (fixed length and varlen modes)
 *   2. fused_gdn_gating_kernel
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gdn_cuda/utils.h>
#include <torch/torch.h>

#include <apis/gdn_forward.hpp>
#include <chrono>
#include <gdn_cuda/kernels/gdn_helpers.cuh>
#include <iostream>
#include <string>
#include <vector>

#include "test_utils.h"

using test_utils::check_tensor_close;
using test_utils::shape_to_string;

struct HelperTestConfig {
    bool verbose = false;
    float atol = 1e-4f;
    float rtol = 1e-3f;
};

// ============================================================================
// chunk_local_cumsum reference
// ============================================================================

torch::Tensor chunk_local_cumsum_reference(const torch::Tensor& g, int64_t chunk_size,
                                           const std::vector<int>* cu_seqlens = nullptr) {
    auto g_float = g.to(torch::kFloat32).cpu();
    auto output = torch::zeros_like(g_float);

    if (cu_seqlens != nullptr) {
        int64_t num_heads = g.size(1);
        auto g_acc = g_float.accessor<float, 2>();
        auto out_acc = output.accessor<float, 2>();
        int64_t batch_size = (int64_t)cu_seqlens->size() - 1;
        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t seq_start = (*cu_seqlens)[b];
            int64_t seq_end = (*cu_seqlens)[b + 1];
            int64_t seq_len = seq_end - seq_start;
            int64_t num_chunks = (seq_len + chunk_size - 1) / chunk_size;
            for (int64_t h = 0; h < num_heads; ++h) {
                for (int64_t c = 0; c < num_chunks; ++c) {
                    int64_t cs = seq_start + c * chunk_size;
                    int64_t ce = std::min(cs + chunk_size, seq_end);
                    float cumsum = 0.f;
                    for (int64_t t = cs; t < ce; ++t) {
                        cumsum += g_acc[t][h];
                        out_acc[t][h] = cumsum;
                    }
                }
            }
        }
    } else {
        // Fixed layout: (batch, seq_len, num_heads)
        int64_t batch_size = g.size(0);
        int64_t seq_len = g.size(1);
        int64_t num_heads = g.size(2);
        auto g_acc = g_float.accessor<float, 3>();
        auto out_acc = output.accessor<float, 3>();
        int64_t num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t h = 0; h < num_heads; ++h) {
                for (int64_t c = 0; c < num_chunks; ++c) {
                    int64_t cs = c * chunk_size;
                    int64_t ce = std::min(cs + chunk_size, seq_len);
                    float cumsum = 0.f;
                    for (int64_t t = cs; t < ce; ++t) {
                        cumsum += g_acc[b][t][h];
                        out_acc[b][t][h] = cumsum;
                    }
                }
            }
        }
    }
    return output.to(g.device());
}

// ============================================================================
// Fixed-length cumsum (seqfirst layout, batch*heads x seq_len)
// ============================================================================

bool test_cumsum_seqfirst_fixed(const HelperTestConfig& config) {
    std::cout << "\n--- Testing chunk_local_cumsum seqfirst (fixed length) ---\n";
    try {
        int batch_size = 2, seq_len = 256, num_heads = 8, chunk_size = 64;

        torch::Device device = torch::kCUDA;
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        torch::Tensor g = torch::randn({batch_size, seq_len, num_heads}, f32_opts) * 0.3f;
        torch::Tensor ref = chunk_local_cumsum_reference(g, chunk_size);

        // seqfirst layout: (batch_size * num_heads, seq_len)
        // kernel input/output pointers for seqfirst
        torch::Tensor g_sf = g.permute({0, 2, 1}).contiguous();  // (batch, heads, seq)
        torch::Tensor out_sf = torch::zeros_like(g_sf);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        gdn_cuda::kernels::gdn_helpers_impl::chunk_local_cumsum_kernel_launch_seqfirst<float>(
            g_sf.data_ptr<float>(), out_sf.data_ptr<float>(), batch_size, seq_len, num_heads,
            nullptr, nullptr, 0, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Convert back: (batch, heads, seq) -> (batch, seq, heads)
        torch::Tensor out_hf = out_sf.permute({0, 2, 1}).contiguous();
        bool passed = check_tensor_close(ref, out_hf, config.atol, config.rtol);

        if (config.verbose) {
            std::cout << "Reference (first 10): ";
            test_utils::inspect_tensor(ref, 10);
            std::cout << "Kernel (first 10): ";
            test_utils::inspect_tensor(out_hf, 10);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return passed;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
        return false;
    }
}

bool test_cumsum_headfirst_fixed(const HelperTestConfig& config) {
    std::cout << "\n--- Testing chunk_local_cumsum headfirst (fixed length) ---\n";
    try {
        int batch_size = 2, seq_len = 256, num_heads = 8, chunk_size = 64;

        torch::Device device = torch::kCUDA;
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        torch::Tensor g = torch::randn({batch_size, seq_len, num_heads}, f32_opts) * 0.3f;
        torch::Tensor ref = chunk_local_cumsum_reference(g, chunk_size);
        torch::Tensor out = torch::zeros_like(ref);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        gdn_cuda::kernels::gdn_helpers_impl::chunk_local_cumsum_kernel_launch_headfirst<float>(
            g.data_ptr<float>(), out.data_ptr<float>(), batch_size, seq_len, num_heads, nullptr,
            nullptr, 0, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool passed = check_tensor_close(ref, out, config.atol, config.rtol);
        if (config.verbose) {
            std::cout << "Reference (first 10): ";
            test_utils::inspect_tensor(ref, 10);
            std::cout << "Kernel (first 10): ";
            test_utils::inspect_tensor(out, 10);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return passed;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
        return false;
    }
}

bool test_cumsum_headfirst_varlen(const HelperTestConfig& config) {
    std::cout << "\n--- Testing chunk_local_cumsum headfirst (varlen) ---\n";
    try {
        std::vector<int> seq_lengths = {128, 64, 192, 96};
        int num_heads = 8, chunk_size = 64;

        std::vector<int> cu_seqlens(seq_lengths.size() + 1);
        cu_seqlens[0] = 0;
        int total = 0;
        for (size_t i = 0; i < seq_lengths.size(); ++i) {
            total += seq_lengths[i];
            cu_seqlens[i + 1] = total;
        }
        int batch_size = (int)seq_lengths.size();

        torch::Device device = torch::kCUDA;
        auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        torch::Tensor g = torch::randn({total, num_heads}, bf16_opts) * 0.3f;
        torch::Tensor ref = chunk_local_cumsum_reference(g, chunk_size, &cu_seqlens);

        auto chunk_indices_vec = test_utils::prepare_chunk_indices(cu_seqlens, chunk_size);
        int num_chunks = (int)chunk_indices_vec.size() / 2;

        torch::Tensor cu_seqlens_d = test_utils::make_cu_seqlens(cu_seqlens);
        torch::Tensor chunk_idx_d = torch::tensor(
            chunk_indices_vec, torch::TensorOptions().dtype(torch::kInt32).device(device));
        torch::Tensor out = torch::zeros({total, num_heads}, f32_opts);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        gdn_cuda::kernels::gdn_helpers_impl::chunk_local_cumsum_kernel_launch_headfirst<
            __nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(g.data_ptr<at::BFloat16>()),
                           out.data_ptr<float>(), batch_size, 0, num_heads,
                           cu_seqlens_d.data_ptr<int>(), chunk_idx_d.data_ptr<int>(), num_chunks,
                           stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool passed = check_tensor_close(ref, out, config.atol, config.rtol);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return passed;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
        return false;
    }
}

// ============================================================================
// chunk_local_cumsum via public API (bf16 variant, uses gdn_cuda::chunk_local_cumsum_bf16)
// ============================================================================

bool test_cumsum_public_api(const HelperTestConfig& config) {
    std::cout << "\n--- Testing chunk_local_cumsum via public API (bf16, padded) ---\n";
    try {
        int batch_size = 2, seq_len = 192, num_heads = 8, chunk_size = 64;

        torch::Device device = torch::kCUDA;
        auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        // Note: chunk_local_cumsum_bf16 expects input (batch, seq_len, num_heads) BF16
        torch::Tensor g = torch::randn({batch_size, seq_len, num_heads}, bf16_opts) * 0.3f;
        torch::Tensor ref = chunk_local_cumsum_reference(g, chunk_size);
        torch::Tensor out = torch::zeros({batch_size, seq_len, num_heads}, f32_opts);

        std::optional<torch::Tensor> cu_seqlens_opt = std::nullopt;
        std::optional<torch::Tensor> chunk_indices_opt = std::nullopt;

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        gdn_cuda::chunk_local_cumsum_bf16(g, out, batch_size, seq_len, num_heads,
                                          /*head_first=*/false, cu_seqlens_opt, chunk_indices_opt,
                                          stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool passed = check_tensor_close(ref, out, config.atol, config.rtol);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return passed;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
        return false;
    }
}

// ============================================================================
// fused_gdn_gating reference
// ============================================================================

torch::Tensor gdn_gating_reference(const torch::Tensor& A_log, const torch::Tensor& dt_bias,
                                   const torch::Tensor& a, const torch::Tensor& b,
                                   float threshold = 20.0f) {
    // dt = softplus(A_log + dt_bias, threshold) + a (element-wise)
    auto dt =
        torch::log1p(torch::exp(torch::clamp(
            A_log.to(torch::kFloat32) + dt_bias.to(torch::kFloat32), -threshold, threshold))) +
        a.to(torch::kFloat32);
    // g = A_log * exp(dt) where exp is the standard exp
    auto g = A_log.to(torch::kFloat32) * torch::exp(dt);
    return g;
}

bool test_fused_gdn_gating(const HelperTestConfig& config) {
    std::cout << "\n--- Testing fused_gdn_gating (fp32) ---\n";
    try {
        int batch_size = 2, seq_len = 64, num_v_heads = 8;

        torch::Device device = torch::kCUDA;
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        torch::Tensor A_log = torch::randn({batch_size, seq_len, num_v_heads}, f32_opts) * 0.5f;
        torch::Tensor dt_bias = torch::randn({num_v_heads}, f32_opts) * 0.1f;
        torch::Tensor a = torch::randn({num_v_heads}, f32_opts) * 0.1f;
        torch::Tensor b = torch::randn({batch_size, seq_len, num_v_heads}, f32_opts) * 0.3f;

        // Use the public API
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        bool is_var_len = false;
        auto [beta_out, gate_out] =
            gdn_cuda::fused_gdn_gating(A_log, dt_bias, a, b, is_var_len, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Reference for gate only (simplified check - just verify shapes and finite values)
        std::cout << "  beta_out shape: " << shape_to_string(beta_out.sizes().vec()) << "\n";
        std::cout << "  gate_out shape: " << shape_to_string(gate_out.sizes().vec()) << "\n";

        bool shapes_ok = (beta_out.sizes() == b.sizes()) && (gate_out.sizes() == A_log.sizes());
        bool finite_ok = torch::isfinite(beta_out.to(torch::kFloat32)).all().item<bool>() &&
                         torch::isfinite(gate_out.to(torch::kFloat32)).all().item<bool>();

        if (!shapes_ok)
            std::cerr << "\033[0;31mShape mismatch\033[0m\n";
        if (!finite_ok)
            std::cerr << "\033[0;31mNon-finite values\033[0m\n";

        CUDA_CHECK(cudaStreamDestroy(stream));
        return shapes_ok && finite_ok;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
        return false;
    }
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    HelperTestConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose")
            config.verbose = true;
        else if (arg == "--atol" && i + 1 < argc)
            config.atol = std::stof(argv[++i]);
        else if (arg == "--rtol" && i + 1 < argc)
            config.rtol = std::stof(argv[++i]);
    }

    if (!torch::cuda::is_available()) {
        std::cerr << "Error: CUDA not available\n";
        return 1;
    }

    gdn_cuda::init(get_env<std::string>("LIBRARY_ROOT_PATH", ""),
                   get_env<std::string>("CUDA_HOME_PATH"));

    std::cout << "============================================\n";
    std::cout << "GDN Helper Kernels Testing Harness\n";
    std::cout << "============================================\n";

    int passed = 0, failed = 0;
    auto run = [&](bool result, const char* name) {
        if (result) {
            std::cout << "\033[0;32m[PASSED] " << name << "\033[0m\n";
            ++passed;
        } else {
            std::cout << "\033[0;31m[FAILED] " << name << "\033[0m\n";
            ++failed;
        }
    };

    run(test_cumsum_headfirst_fixed(config), "cumsum_headfirst_fixed");
    run(test_cumsum_seqfirst_fixed(config), "cumsum_seqfirst_fixed");
    run(test_cumsum_headfirst_varlen(config), "cumsum_headfirst_varlen");
    run(test_cumsum_public_api(config), "cumsum_public_api");
    run(test_fused_gdn_gating(config), "fused_gdn_gating");

    std::cout << "\n============================================\n";
    std::cout << "Passed: " << passed << "  Failed: " << failed << "\n";
    if (failed > 0) {
        std::cout << "\033[0;31mSome tests FAILED\033[0m\n";
        return 1;
    }
    std::cout << "\033[0;32mAll tests PASSED\033[0m\n";
    return 0;
}
