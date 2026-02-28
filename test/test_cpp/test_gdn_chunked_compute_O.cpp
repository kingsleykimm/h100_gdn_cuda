/*
 * GDN Chunked Compute O Testing Harness
 * Tests sm90_bf16_gdn_chunked_compute_O via the gdn_cuda:: public API.
 *
 * The kernel computes per chunk:
 *   O_i = scale * Q_i @ S_i^T + (scale * Q_i @ K_i^T ⊙ M) @ U_i
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

struct ComputeOTestShape {
    int64_t batch_size;
    int64_t seq_len;
    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t shape_k;
    int64_t shape_v;
    int64_t chunk_size;
    const char* description;
};

static const std::vector<ComputeOTestShape> test_shapes = {
    {1, 64, 4, 8, 128, 128, 64, "Single chunk (B=1,T=64,KH=4,VH=8)"},
    {2, 128, 4, 8, 128, 128, 64, "Padded: B=2, T=128, KH=4, VH=8"},
    {1, 256, 4, 16, 128, 128, 64, "Padded: B=1, T=256, KH=4, VH=16"},
    {32, 256, 4, 16, 128, 128, 64, "Padded: B=32, T=256, KH=4, VH=16"},
    {2, 192, 8, 16, 128, 128, 64, "3 chunks batch"},
    {1, 130, 4, 16, 128, 128, 64, "Padded: B=1, T=130, KH=4, VH=16, partial last chunk"},
    {4, 97, 4, 8, 128, 128, 64, "Padded: B=4, T=97, KH=4, VH=8, partial last chunk"},
    {1, 512, 8, 16, 128, 128, 64, "Padded: B=1, T=512, KH=8, VH=16 (8 chunks)"},
    {2, 448, 8, 32, 128, 128, 64, "Padded: B=2, T=448, KH=8, VH=32 (7 chunks)"},
    {8, 160, 4, 16, 128, 128, 64, "Padded: B=8, T=160, KH=4, VH=16"},
    {16, 65, 4, 8, 128, 128, 64, "Padded: B=16, T=65, KH=4, VH=8, near boundary"},
    {5, 255, 8, 16, 128, 128, 64, "Padded: B=5, T=255, KH=8, VH=16"},
    {2, 192, 8, 32, 128, 128, 64, "GQA 4:1 larger"},
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
struct VarlenPartialTestShape {
    std::vector<int> seq_lengths;
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
                int64_t cs = c * chunk_size, ce = std::min(cs + chunk_size, T);
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
                int64_t cs = ss + c * chunk_size, ce = std::min(cs + chunk_size, se);
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

std::tuple<torch::Tensor, torch::Tensor> compute_O_chunk_ref(
    const torch::Tensor& Q, const torch::Tensor& S, const torch::Tensor& K, const torch::Tensor& U,
    float scale, const std::optional<torch::Tensor>& gate_chunk = std::nullopt) {
    auto QS = torch::mm(Q.to(torch::kFloat32), S.to(torch::kFloat32).t());
    auto QKT = torch::mm(Q.to(torch::kFloat32), K.to(torch::kFloat32).t());
    int64_t cs = Q.size(0);
    auto mask = torch::tril(torch::ones({cs, cs}, QKT.options()));
    auto P = QKT * scale * mask;
    if (gate_chunk.has_value()) {
        auto g = gate_chunk->to(torch::kFloat32);
        P = P * torch::exp(g.unsqueeze(1) - g.unsqueeze(0));
    }
    auto PU = torch::mm(P, U.to(torch::kFloat32));
    return {QS, PU};
}

torch::Tensor compute_O_reference_padded(const torch::Tensor& Q, const torch::Tensor& S,
                                         const torch::Tensor& K, const torch::Tensor& U,
                                         int64_t chunk_size,
                                         const std::optional<torch::Tensor>& gate = std::nullopt) {
    int64_t B = Q.size(0), T = Q.size(1), nkh = Q.size(2), nvh = U.size(2);
    int64_t sk = Q.size(3), sv = U.size(3);
    float scale = 1.f / std::sqrt((float)sk);
    int64_t nc = (T + chunk_size - 1) / chunk_size;
    torch::Tensor O = torch::zeros({B, T, nvh, sv}, U.options());

    for (int64_t b = 0; b < B; ++b)
        for (int64_t h = 0; h < nvh; ++h) {
            int64_t kh = (h * nkh) / nvh;
            for (int64_t c = 0; c < nc; ++c) {
                int64_t cs = c * chunk_size, ce = std::min(cs + chunk_size, T);
                auto Q_c = Q[b].slice(0, cs, ce).select(1, kh);
                auto K_c = K[b].slice(0, cs, ce).select(1, kh);
                auto U_c = U[b].slice(0, cs, ce).select(1, h);
                auto S_c = S[b][c][h];
                std::optional<torch::Tensor> g_c = std::nullopt;
                if (gate.has_value())
                    g_c = gate.value()[b].slice(0, cs, ce).select(1, h).to(torch::kFloat32);
                auto [QS, PU] = compute_O_chunk_ref(Q_c, S_c, K_c, U_c, scale, g_c);
                torch::Tensor O_c;
                if (gate.has_value()) {
                    auto g = g_c->to(torch::kFloat32);
                    O_c = QS * scale * torch::exp(g.unsqueeze(-1)) + PU;
                } else {
                    O_c = QS * scale + PU;
                }
                O[b].slice(0, cs, ce).select(1, h).copy_(O_c.to(U.options().dtype()));
            }
        }
    return O;
}

torch::Tensor compute_O_reference_varlen(const torch::Tensor& Q, const torch::Tensor& S,
                                         const torch::Tensor& K, const torch::Tensor& U,
                                         int64_t chunk_size, const std::vector<int>& cu_seqlens,
                                         const std::optional<torch::Tensor>& gate = std::nullopt) {
    int64_t nkh = Q.size(1), nvh = U.size(1);
    int64_t sk = Q.size(2), sv = U.size(2);
    int64_t total = Q.size(0);
    int64_t batch_size = (int64_t)cu_seqlens.size() - 1;
    float scale = 1.f / std::sqrt((float)sk);
    torch::Tensor O = torch::zeros({total, nvh, sv}, U.options());
    int64_t co = 0;

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t ss = cu_seqlens[b], se = cu_seqlens[b + 1], sl = se - ss;
        int64_t nc = (sl + chunk_size - 1) / chunk_size;
        for (int64_t h = 0; h < nvh; ++h) {
            int64_t kh = (h * nkh) / nvh;
            for (int64_t c = 0; c < nc; ++c) {
                int64_t cs = ss + c * chunk_size, ce = std::min(cs + chunk_size, se);
                auto Q_c = Q.slice(0, cs, ce).select(1, kh);
                auto K_c = K.slice(0, cs, ce).select(1, kh);
                auto U_c = U.slice(0, cs, ce).select(1, h);
                auto S_c = S[co + c][h];
                std::optional<torch::Tensor> g_c = std::nullopt;
                if (gate.has_value())
                    g_c = gate.value().slice(0, cs, ce).select(1, h).to(torch::kFloat32);
                auto [QS, PU] = compute_O_chunk_ref(Q_c, S_c, K_c, U_c, scale, g_c);
                torch::Tensor O_c;
                if (gate.has_value()) {
                    auto g = g_c->to(torch::kFloat32);
                    O_c = QS * scale * torch::exp(g.unsqueeze(-1)) + PU;
                } else {
                    O_c = QS * scale + PU;
                }
                O.slice(0, cs, ce).select(1, h).copy_(O_c.to(U.options().dtype()));
            }
        }
        co += nc;
    }
    return O;
}

// ============================================================================
// Padded mode test
// ============================================================================

bool test_padded_mode(const ComputeOTestShape& shape, const TestConfig& config) {
    std::cout << "\n--- Testing Padded Mode: " << shape.description << " ---\n";
    try {
        torch::Device device = torch::kCUDA;
        auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        float sk = 0.1f, sv = 0.1f;
        int64_t nc = (shape.seq_len + shape.chunk_size - 1) / shape.chunk_size;

        torch::Tensor Q = torch::normal(
            0, sk, {shape.batch_size, shape.seq_len, shape.num_k_heads, shape.shape_k},
            std::nullopt, bf16_opts);
        torch::Tensor K = torch::normal(
            0, sk, {shape.batch_size, shape.seq_len, shape.num_k_heads, shape.shape_k},
            std::nullopt, bf16_opts);
        torch::Tensor U = torch::normal(
            0, sv, {shape.batch_size, shape.seq_len, shape.num_v_heads, shape.shape_v},
            std::nullopt, bf16_opts);
        torch::Tensor S = torch::normal(
            0, sk * sv, {shape.batch_size, nc, shape.num_v_heads, shape.shape_v, shape.shape_k},
            std::nullopt, bf16_opts);

        torch::Tensor gate_f32;
        torch::Tensor gate_cumsum_f32;
        std::optional<torch::Tensor> gate_ref_opt = std::nullopt;
        if (config.use_gate) {
            gate_f32 = torch::normal(0, 0.3f, {shape.batch_size, shape.seq_len, shape.num_v_heads},
                                     std::nullopt, f32_opts);
            gate_cumsum_f32 = chunk_local_cumsum_reference_padded(gate_f32, shape.chunk_size);
            gate_ref_opt = gate_cumsum_f32;
        }

        torch::Tensor O_ref =
            compute_O_reference_padded(Q, S, K, U, shape.chunk_size, gate_ref_opt);
        torch::Tensor O_kernel = torch::zeros_like(U);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::optional<torch::Tensor> gate_mn_opt = std::nullopt;
        if (config.use_gate) {
            gate_mn_opt = gdn_cuda::transpose_to_mn_major(gate_cumsum_f32, stream, 128);
        }

        std::optional<torch::Tensor> cu_seqlens_opt = std::nullopt;
        std::optional<torch::Tensor> chunk_indices_opt = std::nullopt;
        std::optional<torch::Tensor> cu_chunks_opt = std::nullopt;
        std::optional<float> scale_opt = std::nullopt;
        gdn_cuda::bf16_gdn_chunked_compute_O(Q, S, K, U, O_kernel, gate_mn_opt, scale_opt, "t",
                                             stream, cu_seqlens_opt, chunk_indices_opt,
                                             cu_chunks_opt, (uint32_t)shape.chunk_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "Checking O:\n";
        bool passed = check_tensor_close(O_ref, O_kernel, config.atol, config.rtol);

        CUDA_CHECK(cudaStreamDestroy(stream));
        return passed;
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

        torch::Tensor Q = torch::normal(0, sk, {total_tokens, shape.num_k_heads, shape.shape_k},
                                        std::nullopt, bf16_opts);
        torch::Tensor K = torch::normal(0, sk, {total_tokens, shape.num_k_heads, shape.shape_k},
                                        std::nullopt, bf16_opts);
        torch::Tensor U = torch::normal(0, sv, {total_tokens, shape.num_v_heads, shape.shape_v},
                                        std::nullopt, bf16_opts);
        torch::Tensor S = torch::normal(
            0, sk * sv, {total_chunks, shape.num_v_heads, shape.shape_v, shape.shape_k},
            std::nullopt, bf16_opts);

        torch::Tensor gate_f32;
        torch::Tensor gate_cumsum_f32;
        std::optional<torch::Tensor> gate_ref_opt = std::nullopt;
        if (config.use_gate) {
            gate_f32 =
                torch::normal(0, 0.3f, {total_tokens, shape.num_v_heads}, std::nullopt, f32_opts);
            gate_cumsum_f32 =
                chunk_local_cumsum_reference_varlen(gate_f32, shape.chunk_size, cu_seqlens);
            gate_ref_opt = gate_cumsum_f32;
        }

        torch::Tensor O_ref =
            compute_O_reference_varlen(Q, S, K, U, shape.chunk_size, cu_seqlens, gate_ref_opt);
        torch::Tensor O_kernel = torch::zeros_like(U);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::optional<torch::Tensor> gate_mn_opt = std::nullopt;
        if (config.use_gate) {
            gate_mn_opt = gdn_cuda::transpose_to_mn_major(gate_cumsum_f32, stream, 128);
        }

        torch::Tensor cu_seqlens_d = test_utils::make_cu_seqlens(cu_seqlens);
        torch::Tensor chunk_indices_d =
            test_utils::make_chunk_indices(cu_seqlens, (int)shape.chunk_size);
        torch::Tensor cu_chunks_d = test_utils::make_cu_chunks(cu_seqlens, (int)shape.chunk_size);
        std::optional<torch::Tensor> cu_seqlens_opt = cu_seqlens_d;
        std::optional<torch::Tensor> chunk_indices_opt = chunk_indices_d;
        std::optional<torch::Tensor> cu_chunks_opt = cu_chunks_d;

        std::optional<float> scale_opt = std::nullopt;
        gdn_cuda::bf16_gdn_chunked_compute_O(Q, S, K, U, O_kernel, gate_mn_opt, scale_opt, "t",
                                             stream, cu_seqlens_opt, chunk_indices_opt,
                                             cu_chunks_opt, (uint32_t)shape.chunk_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "Checking O:\n";
        bool passed = check_tensor_close(O_ref, O_kernel, config.atol, config.rtol);

        CUDA_CHECK(cudaStreamDestroy(stream));
        return passed;
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

        torch::Tensor Q = torch::normal(0, sk, {total_tokens, shape.num_k_heads, shape.shape_k},
                                        std::nullopt, bf16_opts);
        torch::Tensor K = torch::normal(0, sk, {total_tokens, shape.num_k_heads, shape.shape_k},
                                        std::nullopt, bf16_opts);
        torch::Tensor U = torch::normal(0, sv, {total_tokens, shape.num_v_heads, shape.shape_v},
                                        std::nullopt, bf16_opts);
        torch::Tensor S = torch::normal(
            0, sk * sv, {total_chunks, shape.num_v_heads, shape.shape_v, shape.shape_k},
            std::nullopt, bf16_opts);

        torch::Tensor gate_f32;
        torch::Tensor gate_cumsum_f32;
        std::optional<torch::Tensor> gate_ref_opt = std::nullopt;
        if (config.use_gate) {
            gate_f32 =
                torch::normal(0, 0.3f, {total_tokens, shape.num_v_heads}, std::nullopt, f32_opts);
            gate_cumsum_f32 =
                chunk_local_cumsum_reference_varlen(gate_f32, shape.chunk_size, cu_seqlens);
            gate_ref_opt = gate_cumsum_f32;
        }

        torch::Tensor O_ref =
            compute_O_reference_varlen(Q, S, K, U, shape.chunk_size, cu_seqlens, gate_ref_opt);
        torch::Tensor O_kernel = torch::zeros_like(U);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::optional<torch::Tensor> gate_mn_opt = std::nullopt;
        if (config.use_gate) {
            gate_mn_opt = gdn_cuda::transpose_to_mn_major(gate_cumsum_f32, stream, 128);
        }
        std::optional<float> scale_opt = std::nullopt;
        torch::Tensor cu_seqlens_d = test_utils::make_cu_seqlens(cu_seqlens);
        torch::Tensor chunk_indices_d =
            test_utils::make_chunk_indices(cu_seqlens, (int)shape.chunk_size);
        torch::Tensor cu_chunks_d = test_utils::make_cu_chunks(cu_seqlens, (int)shape.chunk_size);
        std::optional<torch::Tensor> cu_seqlens_opt = cu_seqlens_d;
        std::optional<torch::Tensor> chunk_indices_opt = chunk_indices_d;
        std::optional<torch::Tensor> cu_chunks_opt = cu_chunks_d;

        gdn_cuda::bf16_gdn_chunked_compute_O(Q, S, K, U, O_kernel, gate_mn_opt, scale_opt, "t",
                                             stream, cu_seqlens_opt, chunk_indices_opt,
                                             cu_chunks_opt, (uint32_t)shape.chunk_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "Checking O:\n";
        bool passed = check_tensor_close(O_ref, O_kernel, config.atol, config.rtol);

        CUDA_CHECK(cudaStreamDestroy(stream));
        return passed;
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
    std::cout << "GDN Chunked Compute O Testing Harness\n";
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
