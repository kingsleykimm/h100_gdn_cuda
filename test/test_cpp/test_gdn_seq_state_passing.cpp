/*
 * GDN Chunked Seq State Passing Testing Harness
 * Tests sm90_bf16_chunked_seq_state_update via the gdn_cuda:: public API.
 */

#include <cuda_runtime.h>
#include <gdn_cuda/utils.h>
#include <torch/torch.h>

#include <apis/gdn_forward.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "test_utils.h"

using test_utils::check_tensor_close;
using test_utils::shape_to_string;

struct StatePassingTestShape {
    int64_t batch_size;
    int64_t seq_len;
    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t shape_k;
    int64_t shape_v;
    int64_t chunk_size;
    const char* description;
};

static const std::vector<StatePassingTestShape> test_shapes = {
    {2, 128, 4, 8, 128, 128, 64, "Padded: B=2, T=128, KH=4, VH=8"},
    {1, 256, 4, 16, 128, 128, 64, "Padded: B=1, T=256, KH=4, VH=16"},
    {32, 256, 4, 16, 128, 128, 64, "Padded: B=32, T=256, KH=4, VH=16"},
    {1, 64, 4, 8, 128, 128, 64, "Padded: B=1, T=64, KH=4, VH=8"},
    {2, 192, 8, 16, 128, 128, 64, "Padded: B=2, T=192, KH=8, VH=16 (3 chunks)"},
    {3, 320, 8, 32, 128, 128, 64, "Padded: B=3, T=320, KH=8, VH=32"},
    {1, 130, 4, 16, 128, 128, 64, "Padded: B=1, T=130, KH=4, VH=16, partial last chunk"},
    {4, 97, 4, 8, 128, 128, 64, "Padded: B=4, T=97, KH=4, VH=8, partial last chunk"},
    {1, 512, 8, 16, 128, 128, 64, "Padded: B=1, T=512, KH=8, VH=16 (8 chunks)"},
    {2, 448, 8, 32, 128, 128, 64, "Padded: B=2, T=448, KH=8, VH=32 (7 chunks)"},
    {8, 160, 4, 16, 128, 128, 64, "Padded: B=8, T=160, KH=4, VH=16"},
    {16, 65, 4, 8, 128, 128, 64, "Padded: B=16, T=65, KH=4, VH=8, near boundary"},
    {5, 255, 8, 16, 128, 128, 64, "Padded: B=5, T=255, KH=8, VH=16"},
};

struct VarlenTestShape {
    int64_t batch_size;
    int64_t min_chunks;
    int64_t max_chunks;
    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t shape_k;
    int64_t shape_v;
    int64_t chunk_size;
    const char* description;
};

static const std::vector<VarlenTestShape> varlen_test_shapes = {
    {8, 1, 4, 4, 8, 128, 128, 64, "Varlen: B=8, chunks [1,4]"},
    {8, 2, 5, 8, 16, 128, 128, 64, "Varlen: B=8, chunks [2,5]"},
    {6, 1, 6, 4, 16, 128, 128, 64, "Varlen: B=6, chunks [1,6], KH=4, VH=16"},
    {10, 3, 6, 8, 16, 128, 128, 64, "Varlen: B=10, chunks [3,6], KH=8, VH=16"},
    {12, 1, 3, 8, 32, 128, 128, 64, "Varlen: B=12, chunks [1,3], KH=8, VH=32"},
    {4, 6, 8, 8, 16, 128, 128, 64, "Varlen: B=4, chunks [6,8], long sequences"},
    {14, 1, 2, 4, 8, 128, 128, 64, "Varlen: B=14, chunks [1,2], many short sequences"},
    {5, 4, 7, 8, 32, 128, 128, 64, "Varlen: B=5, chunks [4,7], KH=8,VH=32"},
    {9, 2, 8, 4, 16, 128, 128, 64, "Varlen: B=9, chunks [2,8], wide spread"},
};

struct VarlenPartialTestShape {
    std::vector<int> seq_lengths;
    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t shape_k;
    int64_t shape_v;
    int64_t chunk_size;
    const char* description;
};

static const std::vector<VarlenPartialTestShape> varlen_partial_test_shapes = {
    {{100, 64, 150, 80}, 4, 8, 128, 128, 64, "Varlen Partial: (100,64,150,80)"},
    {{33, 64, 97, 128, 45},
     4,
     16,
     128,
     128,
     64,
     "Varlen Partial: mixed (33,64,97,128,45), KH=4,VH=16"},
    {{70, 130, 200}, 8, 16, 128, 128, 64, "Varlen Partial: (70,130,200), KH=8,VH=16"},
    {{65, 66, 67, 68, 69, 70}, 4, 8, 128, 128, 64, "Varlen Partial: near-boundary lengths 65-70"},
    {{1, 63, 64, 65, 127, 129}, 4, 8, 128, 128, 64, "Varlen Partial: tiny+boundary stress"},
    {{191, 257, 383}, 8, 16, 128, 128, 64, "Varlen Partial: long odd lengths (191,257,383)"},
    {{2, 17, 31, 46, 62}, 4, 8, 128, 128, 64, "Varlen Partial: all sub-chunk short lengths"},
    {{64, 128, 192, 256}, 8, 32, 128, 128, 64, "Varlen Partial: exact chunk multiples, KH=8,VH=32"},
    {{95, 96, 97, 159, 160, 161},
     4,
     16,
     128,
     128,
     64,
     "Varlen Partial: around 1.5/2.5 chunk boundaries"},
    {{7, 64, 121, 178, 235, 292},
     4,
     8,
     128,
     128,
     64,
     "Varlen Partial: arithmetic progression stress"},
};

struct TestConfig {
    bool test_varlen = true;
    bool test_padded = true;
    bool test_initial_state = false;
    bool test_final_state = true;
    bool use_gate = false;
    bool verbose = false;
    float atol = 0.01f;
    float rtol = 0.01f;
};

bool parse_args(int argc, char** argv, TestConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            config.test_varlen = (m == "varlen" || m == "all");
            config.test_padded = (m == "padded" || m == "all");
        } else if (arg == "--gate") {
            config.use_gate = true;
        } else if (arg == "--initial-state") {
            config.test_initial_state = true;
        } else if (arg == "--no-final-state") {
            config.test_final_state = false;
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--atol" && i + 1 < argc) {
            config.atol = std::stof(argv[++i]);
        } else if (arg == "--rtol" && i + 1 < argc) {
            config.rtol = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Reference
// ============================================================================

torch::Tensor state_passing_chunk_ref(const torch::Tensor& K, torch::Tensor& U,
                                      const torch::Tensor& W, const torch::Tensor& S,
                                      const torch::Tensor* gate_chunk = nullptr,
                                      float gate_last = 0.f) {
    auto W_St = torch::mm(W, S.t());
    auto U_updated = U - W_St;
    U.copy_(U_updated.to(U.dtype()));

    if (gate_chunk != nullptr) {
        auto gate_decay = torch::exp(gate_last - *gate_chunk).unsqueeze(1);
        auto U_gated = U_updated * gate_decay;
        auto S_gated = S * std::exp(gate_last);
        return S_gated + torch::mm(U_gated.t().to(K.dtype()), K);
    } else {
        return S + torch::mm(U_updated.t(), K);
    }
}

torch::Tensor chunk_local_cumsum_reference_padded(const torch::Tensor& gate, int64_t chunk_size) {
    auto g = gate.to(torch::kFloat32).cpu();
    auto out = torch::zeros_like(g);
    auto ga = g.accessor<float, 3>();
    auto oa = out.accessor<float, 3>();
    int64_t B = g.size(0), T = g.size(1), H = g.size(2);
    int64_t nc = (T + chunk_size - 1) / chunk_size;

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t c = 0; c < nc; ++c) {
                int64_t cs = c * chunk_size;
                int64_t ce = std::min(cs + chunk_size, T);
                float s = 0.f;
                for (int64_t t = cs; t < ce; ++t) {
                    s += ga[b][t][h];
                    oa[b][t][h] = s;
                }
            }
        }
    }
    return out.to(gate.device());
}

torch::Tensor chunk_local_cumsum_reference_varlen(const torch::Tensor& gate, int64_t chunk_size,
                                                  const std::vector<int>& cu_seqlens) {
    auto g = gate.to(torch::kFloat32).cpu();
    auto out = torch::zeros_like(g);
    auto ga = g.accessor<float, 2>();
    auto oa = out.accessor<float, 2>();
    int64_t H = g.size(1);
    int64_t B = static_cast<int64_t>(cu_seqlens.size()) - 1;

    for (int64_t b = 0; b < B; ++b) {
        int64_t ss = cu_seqlens[b], se = cu_seqlens[b + 1], sl = se - ss;
        int64_t nc = (sl + chunk_size - 1) / chunk_size;
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t c = 0; c < nc; ++c) {
                int64_t cs = ss + c * chunk_size;
                int64_t ce = std::min(cs + chunk_size, se);
                float s = 0.f;
                for (int64_t t = cs; t < ce; ++t) {
                    s += ga[t][h];
                    oa[t][h] = s;
                }
            }
        }
    }
    return out.to(gate.device());
}

void state_passing_reference_padded(const torch::Tensor& K, torch::Tensor& U,
                                    const torch::Tensor& W, torch::Tensor& initial_state,
                                    torch::Tensor& state, torch::Tensor& final_state,
                                    bool output_final_state, int64_t chunk_size,
                                    const torch::Tensor* gate = nullptr) {
    int64_t B = K.size(0), T = K.size(1), nkh = K.size(2);
    int64_t nvh = U.size(2);
    int64_t nc = (T + chunk_size - 1) / chunk_size;
    auto opts = U.options();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < nvh; ++h) {
            int64_t kh = (h * nkh) / nvh;
            // ret[b][0][h].copy_(state[b][0][h]);
            auto cur_S = initial_state[b].select(0, h);
            for (int64_t c = 0; c < nc; ++c) {
                state[b][c][h].copy_(cur_S.to(opts.dtype()));
                int64_t cs = c * chunk_size, ce = std::min(cs + chunk_size, T);
                auto K_c = K[b].slice(0, cs, ce).select(1, kh);
                auto U_c = U[b].slice(0, cs, ce).select(1, h);
                auto W_c = W[b].slice(0, cs, ce).select(1, h);
                if (gate != nullptr) {
                    auto gs = (*gate)[b].slice(0, cs, ce).select(1, h).contiguous();
                    float gl = gs[-1].item<float>();
                    cur_S = state_passing_chunk_ref(K_c, U_c, W_c, cur_S, &gs, gl);
                } else {
                    cur_S = state_passing_chunk_ref(K_c, U_c, W_c, cur_S);
                }
                if (output_final_state && c == nc - 1)
                    final_state[b][h].copy_(cur_S.to(opts.dtype()));
            }
        }
    }
}

void state_passing_reference_varlen(const torch::Tensor& K, torch::Tensor& U,
                                    const torch::Tensor& W, torch::Tensor& initial_state,
                                    torch::Tensor& state, torch::Tensor& final_state,
                                    bool output_final_state, int64_t chunk_size,
                                    const std::vector<int>& cu_seqlens,
                                    const torch::Tensor* gate = nullptr) {
    int64_t nkh = K.size(1), nvh = U.size(1);
    int64_t batch_size = (int64_t)cu_seqlens.size() - 1;
    auto opts = U.options();
    int64_t co = 0;

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t ss = cu_seqlens[b], se = cu_seqlens[b + 1], sl = se - ss;
        int64_t nc = (sl + chunk_size - 1) / chunk_size;
        for (int64_t h = 0; h < nvh; ++h) {
            int64_t kh = (h * nkh) / nvh;
            auto cur_S = initial_state[b].select(0, h);
            for (int64_t c = 0; c < nc; ++c) {
                state[co + c][h].copy_(cur_S.to(opts.dtype()));
                int64_t cs = ss + c * chunk_size, ce = std::min(cs + chunk_size, se);
                auto K_c = K.slice(0, cs, ce).select(1, kh);
                auto U_c = U.slice(0, cs, ce).select(1, h);
                auto W_c = W.slice(0, cs, ce).select(1, h);
                if (gate != nullptr) {
                    auto gs = gate->slice(0, cs, ce).select(1, h).contiguous();
                    float gl = gs[-1].item<float>();
                    cur_S = state_passing_chunk_ref(K_c, U_c, W_c, cur_S, &gs, gl);
                } else {
                    cur_S = state_passing_chunk_ref(K_c, U_c, W_c, cur_S);
                }
                if (output_final_state && c == nc - 1)
                    final_state[b][h].copy_(cur_S.to(opts.dtype()));
            }
        }
        co += nc;
    }
}

// ============================================================================
// Padded mode test
// ============================================================================

bool test_padded_mode(const StatePassingTestShape& shape, const TestConfig& config) {
    std::cout << "\n--- Testing Padded Mode: " << shape.description << " ---\n";
    try {
        torch::Device device = torch::kCUDA;
        auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        float sk = 1.f / std::sqrt((float)shape.shape_k);
        float sv = 1.f / std::sqrt((float)shape.shape_v);

        int64_t nc = (shape.seq_len + shape.chunk_size - 1) / shape.chunk_size;

        torch::Tensor K = torch::normal(
            0, sk, {shape.batch_size, shape.seq_len, shape.num_k_heads, shape.shape_k},
            std::nullopt, bf16_opts);
        torch::Tensor U = torch::normal(
            0, sv, {shape.batch_size, shape.seq_len, shape.num_v_heads, shape.shape_v},
            std::nullopt, bf16_opts);
        torch::Tensor W = torch::normal(
            0, sk, {shape.batch_size, shape.seq_len, shape.num_v_heads, shape.shape_k},
            std::nullopt, bf16_opts);

        torch::Tensor ref_state;
        if (config.test_initial_state) {
            ref_state = torch::normal(
                0, sk * sv, {shape.batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k},
                std::nullopt, bf16_opts);
        } else {
            ref_state = torch::zeros(
                {shape.batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);
        }

        torch::Tensor final_state_ref = torch::zeros(
            {shape.batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);
        torch::Tensor final_state_kernel = torch::zeros_like(final_state_ref);

        torch::Tensor gate_f32;
        torch::Tensor gate_cumsum_f32;
        std::optional<torch::Tensor> gate_mn_opt = std::nullopt;
        if (config.use_gate) {
            gate_f32 =
                torch::randn({shape.batch_size, shape.seq_len, shape.num_v_heads}, f32_opts) *
                0.05f;
            gate_cumsum_f32 = chunk_local_cumsum_reference_padded(gate_f32, shape.chunk_size);
        }

        torch::Tensor U_ref = U.clone();
        torch::Tensor U_kernel = U.clone();

        torch::Tensor state_ref = torch::zeros(
            {shape.batch_size, nc, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);
        // Reference
        state_passing_reference_padded(K, U_ref, W, ref_state, state_ref, final_state_ref,
                                       config.test_final_state, shape.chunk_size,
                                       config.use_gate ? &gate_cumsum_f32 : nullptr);

        // Kernel state: per-chunk output, shape (B, nc, num_v_heads, shape_v, shape_k)
        torch::Tensor state_kernel = torch::zeros(
            {shape.batch_size, nc, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        if (config.use_gate) {
            auto gm = gdn_cuda::transpose_to_mn_major(gate_cumsum_f32, stream, 128);
            gate_mn_opt = gm;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        std::optional<torch::Tensor> initial_state_opt =
            config.test_initial_state ? std::optional<torch::Tensor>(ref_state) : std::nullopt;
        std::optional<torch::Tensor> final_state_opt =
            config.test_final_state ? std::optional<torch::Tensor>(final_state_kernel)
                                    : std::nullopt;
        std::optional<torch::Tensor> cu_seqlens_opt = std::nullopt;
        std::optional<torch::Tensor> cu_chunks_opt = std::nullopt;
        std::optional<int> total_chunks_opt = std::nullopt;
        gdn_cuda::bf16_chunked_seq_state_update(
            K, U_kernel, W, initial_state_opt, state_kernel, final_state_opt, gate_mn_opt, "t",
            stream, cu_seqlens_opt, cu_chunks_opt, total_chunks_opt, (uint32_t)shape.chunk_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "Checking state:\n";
        bool sp = check_tensor_close(state_ref, state_kernel, config.atol, config.rtol);
        std::cout << "Checking U:\n";
        bool up = check_tensor_close(U_ref, U_kernel, config.atol, config.rtol);
        bool fp = true;
        if (config.test_final_state && final_state_opt.has_value()) {
            std::cout << "Checking final state:\n";
            fp = check_tensor_close(final_state_ref, final_state_opt.value(), config.atol,
                                    config.rtol);
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return sp && up && fp;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
        return false;
    }
}

// ============================================================================
// Varlen mode test
// ============================================================================

bool test_varlen_mode(const VarlenTestShape& shape, const TestConfig& config) {
    std::cout << "\n--- Testing Varlen Mode: " << shape.description << " ---\n";
    try {
        torch::Device device = torch::kCUDA;
        auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        std::vector<int> cu_seqlens(shape.batch_size + 1);
        cu_seqlens[0] = 0;
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist((int)shape.min_chunks, (int)shape.max_chunks);
        int64_t total_tokens = 0, total_chunks = 0;
        for (int64_t b = 0; b < shape.batch_size; ++b) {
            int nc = dist(rng);
            total_tokens += nc * shape.chunk_size;
            total_chunks += nc;
            cu_seqlens[b + 1] = (int)total_tokens;
        }

        float sk = 1.f / std::sqrt((float)shape.shape_k);
        float sv = 1.f / std::sqrt((float)shape.shape_v);

        torch::Tensor K = torch::normal(0, sk, {total_tokens, shape.num_k_heads, shape.shape_k},
                                        std::nullopt, bf16_opts);
        torch::Tensor U = torch::normal(0, sv, {total_tokens, shape.num_v_heads, shape.shape_v},
                                        std::nullopt, bf16_opts);
        torch::Tensor W = torch::normal(0, sk, {total_tokens, shape.num_v_heads, shape.shape_k},
                                        std::nullopt, bf16_opts);

        torch::Tensor ref_state;
        if (config.test_initial_state) {
            ref_state = torch::normal(
                0, sk * sv, {shape.batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k},
                std::nullopt, bf16_opts);
        } else {
            ref_state = torch::zeros(
                {shape.batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);
        }

        torch::Tensor final_state_ref = torch::zeros(
            {shape.batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);
        torch::Tensor final_state_kernel = torch::zeros_like(final_state_ref);

        torch::Tensor gate_f32;
        torch::Tensor gate_cumsum_f32;
        std::optional<torch::Tensor> gate_mn_opt = std::nullopt;
        if (config.use_gate) {
            gate_f32 = torch::randn({total_tokens, shape.num_v_heads}, f32_opts) * 0.05f;
            gate_cumsum_f32 =
                chunk_local_cumsum_reference_varlen(gate_f32, shape.chunk_size, cu_seqlens);
        }

        torch::Tensor U_ref = U.clone();
        torch::Tensor U_kernel = U.clone();

        torch::Tensor state_ref = torch::zeros(
            {total_chunks, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);
        state_passing_reference_varlen(K, U_ref, W, ref_state, state_ref, final_state_ref,
                                       config.test_final_state, shape.chunk_size, cu_seqlens,
                                       config.use_gate ? &gate_cumsum_f32 : nullptr);

        // Kernel state: per-chunk output, shape (total_chunks, num_v_heads, shape_v, shape_k)
        torch::Tensor state_kernel = torch::zeros(
            {total_chunks, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        if (config.use_gate) {
            gate_mn_opt = gdn_cuda::transpose_to_mn_major(gate_cumsum_f32, stream, 128);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        torch::Tensor cu_seqlens_d = test_utils::make_cu_seqlens(cu_seqlens);
        std::optional<torch::Tensor> initial_state_opt =
            config.test_initial_state ? std::optional<torch::Tensor>(ref_state) : std::nullopt;
        std::optional<torch::Tensor> final_state_opt =
            config.test_final_state ? std::optional<torch::Tensor>(final_state_kernel)
                                    : std::nullopt;
        std::optional<torch::Tensor> cu_seqlens_opt = cu_seqlens_d;
        std::optional<torch::Tensor> cu_chunks_opt =
            test_utils::make_cu_chunks(cu_seqlens, (int)shape.chunk_size);
        std::optional<int> total_chunks_opt = (int)total_chunks;

        gdn_cuda::bf16_chunked_seq_state_update(
            K, U_kernel, W, initial_state_opt, state_kernel, final_state_opt, gate_mn_opt, "t",
            stream, cu_seqlens_opt, cu_chunks_opt, total_chunks_opt, (uint32_t)shape.chunk_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "Checking state:\n";
        bool sp = check_tensor_close(state_ref, state_kernel, config.atol, config.rtol);
        std::cout << "Checking U:\n";
        bool up = check_tensor_close(U_ref, U_kernel, config.atol, config.rtol);
        bool fp = true;
        if (config.test_final_state && final_state_opt.has_value()) {
            std::cout << "Checking final state:\n";
            fp = check_tensor_close(final_state_ref, final_state_opt.value(), config.atol,
                                    config.rtol);
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return sp && up && fp;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
        return false;
    }
}

bool test_varlen_partial_mode(const VarlenPartialTestShape& shape, const TestConfig& config) {
    std::cout << "\n--- Testing Varlen Partial Mode: " << shape.description << " ---\n";
    try {
        torch::Device device = torch::kCUDA;
        auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        int64_t batch_size = static_cast<int64_t>(shape.seq_lengths.size());
        std::vector<int> cu_seqlens(batch_size + 1);
        cu_seqlens[0] = 0;
        int64_t total_tokens = 0;
        int64_t total_chunks = 0;
        for (int64_t b = 0; b < batch_size; ++b) {
            total_tokens += shape.seq_lengths[b];
            total_chunks += (shape.seq_lengths[b] + shape.chunk_size - 1) / shape.chunk_size;
            cu_seqlens[b + 1] = static_cast<int>(total_tokens);
        }

        float sk = 1.f / std::sqrt((float)shape.shape_k);
        float sv = 1.f / std::sqrt((float)shape.shape_v);

        torch::Tensor K = torch::normal(0, sk, {total_tokens, shape.num_k_heads, shape.shape_k},
                                        std::nullopt, bf16_opts);
        torch::Tensor U = torch::normal(0, sv, {total_tokens, shape.num_v_heads, shape.shape_v},
                                        std::nullopt, bf16_opts);
        torch::Tensor W = torch::normal(0, sk, {total_tokens, shape.num_v_heads, shape.shape_k},
                                        std::nullopt, bf16_opts);

        torch::Tensor ref_state;
        if (config.test_initial_state) {
            ref_state = torch::normal(0, sk * sv,
                                      {batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k},
                                      std::nullopt, bf16_opts);
        } else {
            ref_state = torch::zeros({batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k},
                                     bf16_opts);
        }

        torch::Tensor final_state_ref =
            torch::zeros({batch_size, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);
        torch::Tensor final_state_kernel = torch::zeros_like(final_state_ref);

        torch::Tensor gate_f32;
        torch::Tensor gate_cumsum_f32;
        std::optional<torch::Tensor> gate_mn_opt = std::nullopt;
        if (config.use_gate) {
            gate_f32 = torch::randn({total_tokens, shape.num_v_heads}, f32_opts) * 0.05f;
            gate_cumsum_f32 =
                chunk_local_cumsum_reference_varlen(gate_f32, shape.chunk_size, cu_seqlens);
        }

        torch::Tensor U_ref = U.clone();
        torch::Tensor U_kernel = U.clone();

        torch::Tensor state_ref = torch::zeros(
            {total_chunks, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);
        state_passing_reference_varlen(K, U_ref, W, ref_state, state_ref, final_state_ref,
                                       config.test_final_state, shape.chunk_size, cu_seqlens,
                                       config.use_gate ? &gate_cumsum_f32 : nullptr);

        // Kernel state: per-chunk output, shape (total_chunks, num_v_heads, shape_v, shape_k)
        torch::Tensor state_kernel = torch::zeros(
            {total_chunks, shape.num_v_heads, shape.shape_v, shape.shape_k}, bf16_opts);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        if (config.use_gate) {
            gate_mn_opt = gdn_cuda::transpose_to_mn_major(gate_cumsum_f32, stream, 128);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        torch::Tensor cu_seqlens_d = test_utils::make_cu_seqlens(cu_seqlens);
        std::optional<torch::Tensor> initial_state_opt =
            config.test_initial_state ? std::optional<torch::Tensor>(ref_state) : std::nullopt;
        std::optional<torch::Tensor> final_state_opt =
            config.test_final_state ? std::optional<torch::Tensor>(final_state_kernel)
                                    : std::nullopt;
        std::optional<torch::Tensor> cu_seqlens_opt = cu_seqlens_d;
        std::optional<torch::Tensor> cu_chunks_opt =
            test_utils::make_cu_chunks(cu_seqlens, (int)shape.chunk_size);
        std::optional<int> total_chunks_opt = (int)total_chunks;

        gdn_cuda::bf16_chunked_seq_state_update(
            K, U_kernel, W, initial_state_opt, state_kernel, final_state_opt, gate_mn_opt, "t",
            stream, cu_seqlens_opt, cu_chunks_opt, total_chunks_opt, (uint32_t)shape.chunk_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "Checking state:\n";
        bool sp = check_tensor_close(state_ref, state_kernel, config.atol, config.rtol);
        std::cout << "Checking U:\n";
        bool up = check_tensor_close(U_ref, U_kernel, config.atol, config.rtol);
        bool fp = true;
        if (config.test_final_state && final_state_opt.has_value()) {
            std::cout << "Checking final state:\n";
            fp = check_tensor_close(final_state_ref, final_state_opt.value(), config.atol,
                                    config.rtol);
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return sp && up && fp;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
        return false;
    }
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    TestConfig config;
    if (!parse_args(argc, argv, config))
        return 1;

    if (!torch::cuda::is_available()) {
        std::cerr << "Error: CUDA not available\n";
        return 1;
    }

    gdn_cuda::init(get_env<std::string>("LIBRARY_ROOT_PATH", ""),
                   get_env<std::string>("CUDA_HOME_PATH"));

    std::cout << "============================================\n";
    std::cout << "GDN Seq State Passing Testing Harness\n";
    std::cout << "============================================\n";

    int passed = 0, failed = 0;

    if (config.test_padded) {
        std::cout << "\n=== Padded Mode Tests ===\n";
        for (const auto& s : test_shapes) {
            bool r = test_padded_mode(s, config);
            if (r) {
                std::cout << "\033[0;32m[PASSED] Padded: " << s.description << "\033[0m\n";
                ++passed;
            } else {
                std::cout << "\033[0;31m[FAILED] Padded: " << s.description << "\033[0m\n";
                ++failed;
            }
        }
    }
    if (config.test_varlen) {
        std::cout << "\n=== Varlen Mode Tests ===\n";
        for (const auto& s : varlen_test_shapes) {
            bool r = test_varlen_mode(s, config);
            if (r) {
                std::cout << "\033[0;32m[PASSED] Varlen: " << s.description << "\033[0m\n";
                ++passed;
            } else {
                std::cout << "\033[0;31m[FAILED] Varlen: " << s.description << "\033[0m\n";
                ++failed;
            }
        }
        std::cout << "\n=== Varlen Partial Mode Tests ===\n";
        for (const auto& s : varlen_partial_test_shapes) {
            bool r = test_varlen_partial_mode(s, config);
            if (r) {
                std::cout << "\033[0;32m[PASSED] Varlen Partial: " << s.description << "\033[0m\n";
                ++passed;
            } else {
                std::cout << "\033[0;31m[FAILED] Varlen Partial: " << s.description << "\033[0m\n";
                ++failed;
            }
        }
    }

    std::cout << "\n============================================\n";
    std::cout << "Passed: " << passed << "  Failed: " << failed << "\n";
    if (failed > 0) {
        std::cout << "\033[0;31mSome tests FAILED\033[0m\n";
        return 1;
    }
    std::cout << "\033[0;32mAll tests PASSED\033[0m\n";
    return 0;
}
