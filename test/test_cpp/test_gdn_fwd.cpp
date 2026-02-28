/*
 * GDN Chunked Forward End-to-End Testing Harness
 *
 * Validates gdn_cuda::chunked_forward against a composed reference.
 *
 * Reference pipeline:
 *   1. chunk_local_cumsum on gate
 *   2. compute_u_w_reference
 *   3. state_passing_reference
 *   4. compute_O_reference
 */

#include <cuda_runtime.h>
#include <gdn_cuda/utils.h>
#include <torch/torch.h>

#include <apis/gdn_forward.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "test_utils.h"

using test_utils::calc_diff;
using test_utils::check_tensor_close;
using test_utils::shape_to_string;

namespace {

inline int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}
inline constexpr double kCosineDiffTol = 0.01;

struct GdnFwdTestShape {
    int64_t batch_size;
    int64_t seq_len;
    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t shape_k;
    int64_t shape_v;
    int64_t chunk_size;
    float gate_normalization_factor = 1.0f;
    const char* description;
};

struct GdnFwdVarlenShape {
    int64_t batch_size;
    int64_t min_chunks;
    int64_t max_chunks;
    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t shape_k;
    int64_t shape_v;
    int64_t chunk_size;
    float gate_normalization_factor = 1.0f;
    const char* description;
};

struct GdnFwdVarlenPartialShape {
    std::vector<int> seq_lengths;
    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t shape_k;
    int64_t shape_v;
    int64_t chunk_size;
    float gate_normalization_factor = 1.0f;
    const char* description;
};

struct TestConfig {
    bool test_varlen = true;
    bool test_padded = true;
    bool verbose = false;
    bool test_initial_state = false;
    float atol = 0.01f;
    float rtol = 0.01f;
};

static const std::vector<GdnFwdTestShape> padded_shapes = {
    {2, 128, 4, 8, 128, 128, 64, 1.0f, "Padded: B=2, T=128, KH=4, VH=8"},
    {1, 256, 4, 16, 128, 128, 64, 10, "Padded: B=1, T=256, KH=4, VH=16"},
    {32, 256, 4, 16, 128, 128, 64, 0.5f, "Padded: B=32, T=256, KH=4, VH=16"},
    {1, 64, 4, 8, 128, 128, 64, 0.5f, "Padded: B=1, T=64, KH=4, VH=8, gate norm=0.5"},
    {2, 192, 8, 16, 128, 128, 64, 10, "Padded: B=2, T=192, KH=8, VH=16 (3 chunks)"},
    {3, 320, 8, 32, 128, 128, 64, 1.0f, "Padded: B=3, T=320, KH=8, VH=32, gate norm=10"},
    {1, 130, 4, 16, 128, 128, 64, 1.0f, "Padded: B=1, T=130, KH=4, VH=16, partial last chunk"},
    {4, 97, 4, 8, 128, 128, 64, 1.0f, "Padded: B=4, T=97, KH=4, VH=8, partial last chunk"},
    {1, 512, 8, 16, 128, 128, 64, 2.0, "Padded: B=1, T=512, KH=8, VH=16 (8 chunks)"},
    {2, 448, 8, 32, 128, 128, 64, 0.5, "Padded: B=2, T=448, KH=8, VH=32 (7 chunks)"},
    {8, 160, 4, 16, 128, 128, 64, 0.5, "Padded: B=8, T=160, KH=4, VH=16, gate norm=0.75"},
    {16, 65, 4, 8, 128, 128, 64, 3.0, "Padded: B=16, T=65, KH=4, VH=8, near boundary"},
    {5, 255, 8, 16, 128, 128, 64, 0.8f, "Padded: B=5, T=255, KH=8, VH=16, strong gate norm"},
};

static const std::vector<GdnFwdVarlenShape> varlen_shapes = {
    {8, 1, 4, 4, 8, 128, 128, 64, 1.0f, "Varlen: B=8, chunks [1,4]"},
    {8, 2, 5, 8, 16, 128, 128, 64, 10, "Varlen: B=8, chunks [2,5]"},
    {6, 1, 6, 4, 16, 128, 128, 64, 3.0f, "Varlen: B=6, chunks [1,6], KH=4, VH=16"},
    {10, 3, 6, 8, 16, 128, 128, 64, 1.0f, "Varlen: B=10, chunks [3,6], KH=8, VH=16"},
    {12, 1, 3, 8, 32, 128, 128, 64, 10.0f, "Varlen: B=12, chunks [1,3], KH=8, VH=32"},
    {4, 6, 8, 8, 16, 128, 128, 64, 1.0f, "Varlen: B=4, chunks [6,8], long sequences"},
    {14, 1, 2, 4, 8, 128, 128, 64, 4.0f, "Varlen: B=14, chunks [1,2], many short sequences"},
    {5, 4, 7, 8, 32, 128, 128, 64, 0.5f, "Varlen: B=5, chunks [4,7], KH=8,VH=32"},
    {9, 2, 8, 4, 16, 128, 128, 64, 12.0f, "Varlen: B=9, chunks [2,8], wide spread"},
};

static const std::vector<GdnFwdVarlenPartialShape> varlen_partial_shapes = {
    {{100, 64, 150, 80},
     4,
     8,
     128,
     128,
     64,
     10.0f,
     "Varlen Partial: (100,64,150,80), gate norm=10"},
    {{33, 64, 97, 128, 45},
     4,
     16,
     128,
     128,
     64,
     3.0f,
     "Varlen Partial: mixed (33,64,97,128,45), KH=4,VH=16"},
    {{70, 130, 200}, 8, 16, 128, 128, 64, 1.0f, "Varlen Partial: (70,130,200), KH=8,VH=16"},
    {{65, 66, 67, 68, 69, 70},
     4,
     8,
     128,
     128,
     64,
     5.0f,
     "Varlen Partial: near-boundary lengths 65-70"},
    {{1, 63, 64, 65, 127, 129}, 4, 8, 128, 128, 64, 10.0f, "Varlen Partial: tiny+boundary stress"},
    {{191, 257, 383}, 8, 16, 128, 128, 64, 2.0f, "Varlen Partial: long odd lengths (191,257,383)"},
    {{2, 17, 31, 46, 62}, 4, 8, 128, 128, 64, 0.5f, "Varlen Partial: all sub-chunk short lengths"},
    {{64, 128, 192, 256},
     8,
     32,
     128,
     128,
     64,
     1.0f,
     "Varlen Partial: exact chunk multiples, KH=8,VH=32"},
    {{95, 96, 97, 159, 160, 161},
     4,
     16,
     128,
     128,
     64,
     7.0f,
     "Varlen Partial: around 1.5/2.5 chunk boundaries"},
    {{7, 64, 121, 178, 235, 292},
     4,
     8,
     128,
     128,
     64,
     15.0f,
     "Varlen Partial: arithmetic progression stress"},
};

bool parse_args(int argc, char** argv, TestConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            config.test_varlen = (m == "varlen" || m == "all");
            config.test_padded = (m == "padded" || m == "all");
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--initial-state") {
            config.test_initial_state = true;
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

torch::Tensor normalize_gate_for_exp(const torch::Tensor& gate, float gate_normalization_factor) {
    if (std::abs(gate_normalization_factor - 1.0f) < 1e-6f) {
        return gate;
    }
    return gate / gate_normalization_factor;
}

bool check_tensor_close_or_cosine(const torch::Tensor& ref, const torch::Tensor& out, float atol,
                                  float rtol, const char* label) {
    if (check_tensor_close(ref, out, atol, rtol)) {
        return true;
    }

    torch::Tensor ref_tmp = ref;
    torch::Tensor out_tmp = out;
    const double cosine_diff = calc_diff(ref_tmp, out_tmp);
    std::cout << "  " << label << " cosine diff (1 - cosine similarity): " << cosine_diff << "\n";
    if (cosine_diff < kCosineDiffTol) {
        std::cout << "\033[0;33m  Accepting via cosine diff threshold (< " << kCosineDiffTol
                  << ")\033[0m\n";
        return true;
    }
    return false;
}

// ============================================================================
// Reference helpers
// ============================================================================

torch::Tensor chunk_local_cumsum_reference(const torch::Tensor& g, int64_t chunk_size,
                                           const std::vector<int>* cu_seqlens = nullptr) {
    auto g_f = g.to(torch::kFloat32).cpu();
    auto out = torch::zeros_like(g_f);
    auto ga = g_f.accessor<float, 2>();
    auto oa = out.accessor<float, 2>();
    int64_t nh = g.size(1);

    if (cu_seqlens != nullptr) {
        for (int64_t b = 0; b + 1 < static_cast<int64_t>(cu_seqlens->size()); ++b) {
            int64_t ss = (*cu_seqlens)[b], se = (*cu_seqlens)[b + 1], sl = se - ss;
            int64_t nc = ceil_div(sl, chunk_size);
            for (int64_t h = 0; h < nh; ++h) {
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
    } else {
        int64_t total = g.size(0);
        int64_t nc = ceil_div(total, chunk_size);
        for (int64_t h = 0; h < nh; ++h) {
            for (int64_t c = 0; c < nc; ++c) {
                int64_t cs = c * chunk_size, ce = std::min(cs + chunk_size, total);
                float s = 0.f;
                for (int64_t t = cs; t < ce; ++t) {
                    s += ga[t][h];
                    oa[t][h] = s;
                }
            }
        }
    }

    return out.to(g.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_uw_chunk_ref(
    const torch::Tensor& K, const torch::Tensor& V, const torch::Tensor& beta,
    const std::optional<torch::Tensor>& gate = std::nullopt) {
    int64_t cs = K.size(0);
    // auto K_f = K.to(torch::kFloat32), V_f = V.to(torch::kFloat32), b_f =
    // beta.to(torch::kFloat32);
    auto A = torch::tril(torch::mm(K, K.t()), -1);
    if (gate.has_value()) {
        auto g = gate.value().to(K.dtype());
        A = A * torch::exp(g.unsqueeze(1) - g.unsqueeze(0));
    }
    A = A * beta.unsqueeze(1);
    auto opts = torch::TensorOptions().dtype(K.dtype()).device(K.device());
    auto T = torch::inverse((A + torch::eye(cs, opts).to(torch::kFloat32)));
    T = T.to(K.dtype());
    T = T * beta.unsqueeze(0);
    auto U = torch::mm(T, V);
    auto W = torch::mm(T, K);
    if (gate.has_value()) {
        auto g = gate.value().to(K.dtype());
        W = W * torch::exp(g).unsqueeze(1);
    }
    return {U.to(K.dtype()), W.to(K.dtype()), T.to(K.dtype())};
}

std::tuple<torch::Tensor, torch::Tensor> compute_uw_reference(
    const torch::Tensor& K, const torch::Tensor& V, const torch::Tensor& beta, int64_t chunk_size,
    const std::vector<int>& cu_seqlens = {},
    const std::optional<torch::Tensor>& gate = std::nullopt) {
    bool is_varlen = !cu_seqlens.empty();
    torch::Tensor U = torch::zeros_like(V);
    torch::Tensor W = is_varlen
                          ? torch::zeros({V.size(0), V.size(1), K.size(2)}, V.options())
                          : torch::zeros({V.size(0), V.size(1), V.size(2), K.size(3)}, V.options());

    auto do_chunk = [&](int64_t cs, int64_t ce, int64_t kh, int64_t h, bool is_vl) {
        auto K_c =
            is_vl ? K.slice(0, cs, ce).select(1, kh) : K[cs].slice(0, ce, ce + 1);  // placeholder
        // (handled inline below)
    };
    (void)do_chunk;

    if (is_varlen) {
        int64_t B = (int64_t)cu_seqlens.size() - 1, nkh = K.size(1), nvh = V.size(1);
        for (int64_t b = 0; b < B; ++b) {
            int64_t ss = cu_seqlens[b], se = cu_seqlens[b + 1], sl = se - ss;
            int64_t nc = ceil_div(sl, chunk_size);
            for (int64_t h = 0; h < nvh; ++h) {
                int64_t kh = (h * nkh) / nvh;
                for (int64_t c = 0; c < nc; ++c) {
                    int64_t cs = ss + c * chunk_size, ce = std::min(cs + chunk_size, se);
                    auto K_c = K.slice(0, cs, ce).select(1, kh);
                    auto V_c = V.slice(0, cs, ce).select(1, h);
                    auto b_c = beta.slice(0, cs, ce).select(1, h);
                    std::optional<torch::Tensor> g_c = std::nullopt;
                    if (gate.has_value())
                        g_c = gate->slice(0, cs, ce).select(1, h);
                    auto [Uc, Wc, _] = compute_uw_chunk_ref(K_c, V_c, b_c, g_c);
                    U.slice(0, cs, ce).select(1, h).copy_(Uc);
                    W.slice(0, cs, ce).select(1, h).copy_(Wc);
                }
                // partial last chunk handled by ceil_div above
            }
        }
    } else {
        int64_t B = K.size(0), T = K.size(1), nkh = K.size(2), nvh = V.size(2);
        int64_t nc = ceil_div(T, chunk_size);
        for (int64_t b = 0; b < B; ++b)
            for (int64_t h = 0; h < nvh; ++h) {
                int64_t kh = (h * nkh) / nvh;
                for (int64_t c = 0; c < nc; ++c) {
                    int64_t cs = c * chunk_size, ce = std::min(cs + chunk_size, T);
                    auto K_c = K[b].slice(0, cs, ce).select(1, kh);
                    auto V_c = V[b].slice(0, cs, ce).select(1, h);
                    auto b_c = beta[b].slice(0, cs, ce).select(1, h);
                    std::optional<torch::Tensor> g_c = std::nullopt;
                    if (gate.has_value())
                        g_c = (*gate)[b].slice(0, cs, ce).select(1, h);
                    auto [Uc, Wc, _] = compute_uw_chunk_ref(K_c, V_c, b_c, g_c);
                    U[b].slice(0, cs, ce).select(1, h).copy_(Uc);
                    W[b].slice(0, cs, ce).select(1, h).copy_(Wc);
                }
            }
    }
    return {U, W};
}

torch::Tensor state_passing_chunk_ref2(const torch::Tensor& K, torch::Tensor& U,
                                       const torch::Tensor& W, const torch::Tensor& S,
                                       const torch::Tensor* gc = nullptr, float gl = 0.f) {
    auto W_St = torch::mm(W, S.t());
    auto Uu = U - W_St;
    U.copy_(Uu.to(U.dtype()));
    if (gc != nullptr) {
        auto d = torch::exp(gl - *gc).to(Uu.dtype()).unsqueeze(1);
        return S * std::exp(gl) + torch::mm((Uu * d).t(), K);
    }
    return S + torch::mm(Uu.t(), K);
}

void state_passing_ref_padded(
    const torch::Tensor& K, torch::Tensor& U, const torch::Tensor& W,
    const torch::Tensor& initial_state,  // {B, nvh, sv, sk}
    torch::Tensor& state,                // {B, nc, nvh, sv, sk} - state before each chunk
    torch::Tensor& final_state, int64_t cs, const torch::Tensor* gate = nullptr) {
    int64_t B = K.size(0), T = K.size(1), nkh = K.size(2), nvh = U.size(2);
    int64_t nc = ceil_div(T, cs);
    auto opts = U.options();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t h = 0; h < nvh; ++h) {
            int64_t kh = (h * nkh) / nvh;
            auto curS = initial_state[b].select(0, h);
            for (int64_t c = 0; c < nc; ++c) {
                int64_t cst = c * cs, cen = std::min(cst + cs, T);
                state[b][c][h].copy_(curS.to(opts.dtype()));
                auto K_c = K[b].slice(0, cst, cen).select(1, kh);
                auto U_c = U[b].slice(0, cst, cen).select(1, h);
                auto W_c = W[b].slice(0, cst, cen).select(1, h);
                if (gate != nullptr) {
                    int64_t fs = b * T + cst, fe = b * T + cen;
                    auto gs = gate->slice(0, fs, fe).select(1, h).contiguous();
                    float gl = gs[-1].item<float>();
                    curS = state_passing_chunk_ref2(K_c, U_c, W_c, curS, &gs, gl);
                } else {
                    curS = state_passing_chunk_ref2(K_c, U_c, W_c, curS);
                }
                if (c == nc - 1)
                    final_state[b][h].copy_(curS.to(opts.dtype()));
            }
        }
}

void state_passing_ref_varlen(
    const torch::Tensor& K, torch::Tensor& U, const torch::Tensor& W,
    const torch::Tensor& initial_state,  // {batch_size, nvh, sv, sk}
    torch::Tensor& state,                // {total_chunks, nvh, sv, sk} - state before each chunk
    torch::Tensor& final_state, int64_t cs, const std::vector<int>& cus,
    const torch::Tensor* gate = nullptr) {
    int64_t nkh = K.size(1), nvh = U.size(1);
    int64_t B = (int64_t)cus.size() - 1;
    auto opts = U.options();
    int64_t co = 0;
    for (int64_t b = 0; b < B; ++b) {
        int64_t ss = cus[b], se = cus[b + 1], sl = se - ss;
        int64_t nc = ceil_div(sl, cs);
        for (int64_t h = 0; h < nvh; ++h) {
            int64_t kh = (h * nkh) / nvh;
            auto curS = initial_state[b].select(0, h);
            for (int64_t c = 0; c < nc; ++c) {
                int64_t cst = ss + c * cs, cen = std::min(cst + cs, se);
                state[co + c][h].copy_(curS.to(opts.dtype()));
                auto K_c = K.slice(0, cst, cen).select(1, kh);
                auto U_c = U.slice(0, cst, cen).select(1, h);
                auto W_c = W.slice(0, cst, cen).select(1, h);
                if (gate != nullptr) {
                    auto gs = gate->slice(0, cst, cen).select(1, h).contiguous();
                    float gl = gs[-1].item<float>();
                    curS = state_passing_chunk_ref2(K_c, U_c, W_c, curS, &gs, gl);
                } else {
                    curS = state_passing_chunk_ref2(K_c, U_c, W_c, curS);
                }
                if (c == nc - 1)
                    final_state[b][h].copy_(curS.to(opts.dtype()));
            }
        }
        co += nc;
    }
}

std::tuple<torch::Tensor, torch::Tensor> compute_O_chunk_ref2(
    const torch::Tensor& Q, const torch::Tensor& S, const torch::Tensor& K, const torch::Tensor& U,
    float scale, const std::optional<torch::Tensor>& gc = std::nullopt) {
    auto QS = torch::mm(Q, S.t());
    auto QKT = torch::mm(Q, K.t());
    int64_t cs = Q.size(0);
    auto P = QKT * scale * torch::tril(torch::ones({cs, cs}, QKT.options()));
    if (gc.has_value()) {
        auto g = gc.value().to(Q.dtype());
        P = P * torch::exp(g.unsqueeze(1) - g.unsqueeze(0));
    }
    return {QS, torch::mm(P, U)};
}

torch::Tensor compute_O_ref_padded(const torch::Tensor& Q, const torch::Tensor& S,
                                   const torch::Tensor& K, const torch::Tensor& U, int64_t cs,
                                   const std::optional<torch::Tensor>& gate = std::nullopt) {
    int64_t B = Q.size(0), T = Q.size(1), nkh = Q.size(2), nvh = U.size(2);
    float scale = 1.f / std::sqrt((float)Q.size(3));
    int64_t nc = ceil_div(T, cs);
    torch::Tensor O = torch::zeros({B, T, nvh, U.size(3)}, U.options());
    for (int64_t b = 0; b < B; ++b)
        for (int64_t h = 0; h < nvh; ++h) {
            int64_t kh = (h * nkh) / nvh;
            for (int64_t c = 0; c < nc; ++c) {
                int64_t cst = c * cs, cen = std::min(cst + cs, T);
                auto Q_c = Q[b].slice(0, cst, cen).select(1, kh);
                auto K_c = K[b].slice(0, cst, cen).select(1, kh);
                auto U_c = U[b].slice(0, cst, cen).select(1, h);
                auto S_c = S[b][c][h];
                std::optional<torch::Tensor> gc2 = std::nullopt;
                if (gate.has_value())
                    gc2 = gate.value()[b].slice(0, cst, cen).select(1, h);
                auto [QS, PU] = compute_O_chunk_ref2(Q_c, S_c, K_c, U_c, scale, gc2);
                torch::Tensor Oc;
                if (gate.has_value()) {
                    auto g = gc2.value();
                    Oc = QS * scale * torch::exp(g.unsqueeze(-1)) + PU;
                } else
                    Oc = QS * scale + PU;
                O[b].slice(0, cst, cen).select(1, h).copy_(Oc.to(U.options().dtype()));
            }
        }
    return O;
}

torch::Tensor compute_O_ref_varlen(const torch::Tensor& Q, const torch::Tensor& S,
                                   const torch::Tensor& K, const torch::Tensor& U, int64_t cs,
                                   const std::vector<int>& cus,
                                   const std::optional<torch::Tensor>& gate = std::nullopt) {
    int64_t nkh = Q.size(1), nvh = U.size(1);
    int64_t B = (int64_t)cus.size() - 1;
    float scale = 1.f / std::sqrt((float)Q.size(2));
    torch::Tensor O = torch::zeros({Q.size(0), nvh, U.size(2)}, U.options());
    int64_t co = 0;
    for (int64_t b = 0; b < B; ++b) {
        int64_t ss = cus[b], se = cus[b + 1], sl = se - ss;
        int64_t nc = ceil_div(sl, cs);
        for (int64_t h = 0; h < nvh; ++h) {
            int64_t kh = (h * nkh) / nvh;
            for (int64_t c = 0; c < nc; ++c) {
                int64_t cst = ss + c * cs, cen = std::min(cst + cs, se);
                auto Q_c = Q.slice(0, cst, cen).select(1, kh);
                auto K_c = K.slice(0, cst, cen).select(1, kh);
                auto U_c = U.slice(0, cst, cen).select(1, h);
                auto S_c = S[co + c][h];
                std::optional<torch::Tensor> gc2 = std::nullopt;
                if (gate.has_value())
                    gc2 = gate.value().slice(0, cst, cen).select(1, h);
                auto [QS, PU] = compute_O_chunk_ref2(Q_c, S_c, K_c, U_c, scale, gc2);
                torch::Tensor Oc;
                if (gate.has_value()) {
                    auto g = gc2.value();
                    Oc = QS * scale * torch::exp(g.unsqueeze(-1)) + PU;
                } else
                    Oc = QS * scale + PU;
                O.slice(0, cst, cen).select(1, h).copy_(Oc.to(U.options().dtype()));
            }
        }
        co += nc;
    }
    return O;
}

// ============================================================================
// Padded mode test
// ============================================================================

bool test_padded_mode(const GdnFwdTestShape& shape, const TestConfig& config) {
    std::cout << "\n--- Testing Padded Mode: " << shape.description << " ---\n";
    try {
        torch::Device device = torch::kCUDA;
        auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

        int64_t B = shape.batch_size, T = shape.seq_len;
        int64_t nkh = shape.num_k_heads, nvh = shape.num_v_heads;
        int64_t sk = shape.shape_k, sv = shape.shape_v, cs = shape.chunk_size;
        float fsk = 1.f / std::sqrt((float)sk), fsv = 1.f / std::sqrt((float)sv);

        torch::Tensor Q = torch::normal(0, fsk, {B, T, nkh, sk}, std::nullopt, bf16_opts);
        torch::Tensor K = torch::normal(0, fsk, {B, T, nkh, sk}, std::nullopt, bf16_opts);
        torch::Tensor V = torch::normal(0, fsv, {B, T, nvh, sv}, std::nullopt, bf16_opts);
        torch::Tensor beta = torch::sigmoid(torch::randn({B, T, nvh}, bf16_opts) * 0.3f);

        // gate_raw: (B*T, nvh) bf16
        torch::Tensor gate_raw = torch::normal(0, 0.3f, {B * T, nvh}, std::nullopt, bf16_opts);
        torch::Tensor gate_raw_normalized =
            normalize_gate_for_exp(gate_raw.to(torch::kFloat32), shape.gate_normalization_factor);

        // Reference path: preserve per-batch boundaries for padded mode.
        std::vector<int> cu_seqlens_padded(B + 1);
        for (int64_t b = 0; b <= B; ++b)
            cu_seqlens_padded[b] = static_cast<int>(b * T);
        torch::Tensor gate_cumsum =
            chunk_local_cumsum_reference(gate_raw_normalized, cs, &cu_seqlens_padded);
        auto gcr = gate_cumsum.view({B, T, nvh});
        auto [U_ref, W_ref] = compute_uw_reference(K, V, beta, cs, {}, gcr);

        int64_t nc = ceil_div(T, cs);
        torch::Tensor initial_state =
            config.test_initial_state
                ? torch::normal(0, fsk * fsv, {B, nvh, sv, sk}, std::nullopt, bf16_opts)
                : torch::zeros({B, nvh, sv, sk}, bf16_opts);

        torch::Tensor state_per_chunk = torch::zeros({B, nc, nvh, sv, sk}, bf16_opts);
        torch::Tensor final_state_ref = torch::zeros({B, nvh, sv, sk}, bf16_opts);
        torch::Tensor U_state = U_ref.clone();
        state_passing_ref_padded(K, U_state, W_ref, initial_state, state_per_chunk, final_state_ref,
                                 cs, &gate_cumsum);

        torch::Tensor O_ref = compute_O_ref_padded(Q, state_per_chunk, K, U_state, cs, gcr);

        torch::Tensor state_kernel = torch::zeros_like(state_per_chunk);
        // Kernel path
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        torch::Tensor gate_raw_f32 = gate_raw_normalized.view({B, T, nvh});
        std::optional<torch::Tensor> initial_state_opt =
            config.test_initial_state ? std::optional<torch::Tensor>{initial_state} : std::nullopt;
        std::optional<torch::Tensor> cu_seqlens_opt = std::nullopt;
        std::optional<torch::Tensor> chunk_indices_opt = std::nullopt;
        std::optional<torch::Tensor> cu_chunks_opt = std::nullopt;
        std::optional<int> total_chunks_opt = std::nullopt;
        std::optional<float> scale_opt = std::nullopt;

        auto [O_kernel, final_state_kernel] = gdn_cuda::chunked_forward(
            Q, K, V, beta, gate_raw_f32, scale_opt, initial_state_opt, cu_seqlens_opt,
            chunk_indices_opt, cu_chunks_opt, total_chunks_opt, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "Checking O:\n";
        bool op = check_tensor_close_or_cosine(O_ref, O_kernel, config.atol, config.rtol, "O");
        std::cout << "Checking final state:\n";
        bool sp = check_tensor_close_or_cosine(final_state_ref, final_state_kernel, config.atol,
                                               config.rtol, "final_state");

        std::cout << "Checking state tensor: \n";
        bool st = check_tensor_close_or_cosine(state_per_chunk, state_kernel, config.atol,
                                               config.rtol, "state_tensor");

        CUDA_CHECK(cudaStreamDestroy(stream));
        return op && sp;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
        return false;
    }
}

// ============================================================================
// Varlen mode test
// ============================================================================

bool test_varlen_mode(const GdnFwdVarlenShape& shape, const TestConfig& config) {
    std::cout << "\n--- Testing Varlen Mode: " << shape.description << " ---\n";
    try {
        torch::Device device = torch::kCUDA;
        auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

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

        int64_t nkh = shape.num_k_heads, nvh = shape.num_v_heads;
        int64_t sk = shape.shape_k, sv = shape.shape_v, cs = shape.chunk_size;
        float fsk = 1.f / std::sqrt((float)sk), fsv = 1.f / std::sqrt((float)sv);

        torch::Tensor Q = torch::normal(0, fsk, {total_tokens, nkh, sk}, std::nullopt, bf16_opts);
        torch::Tensor K = torch::normal(0, fsk, {total_tokens, nkh, sk}, std::nullopt, bf16_opts);
        torch::Tensor V = torch::normal(0, fsv, {total_tokens, nvh, sv}, std::nullopt, bf16_opts);
        torch::Tensor beta = torch::sigmoid(torch::randn({total_tokens, nvh}, bf16_opts) * 0.3f);
        torch::Tensor gate_raw =
            torch::normal(0, 0.1f, {total_tokens, nvh}, std::nullopt, bf16_opts);
        torch::Tensor gate_raw_normalized =
            normalize_gate_for_exp(gate_raw.to(torch::kFloat32), shape.gate_normalization_factor);

        // Reference path
        torch::Tensor gate_cumsum =
            chunk_local_cumsum_reference(gate_raw_normalized, cs, &cu_seqlens);
        auto [U_ref, W_ref] = compute_uw_reference(K, V, beta, cs, cu_seqlens, gate_cumsum);

        torch::Tensor initial_state =
            config.test_initial_state ? torch::normal(0, fsk * fsv, {shape.batch_size, nvh, sv, sk},
                                                      std::nullopt, bf16_opts)
                                      : torch::zeros({shape.batch_size, nvh, sv, sk}, bf16_opts);
        torch::Tensor state_per_chunk = torch::zeros({total_chunks, nvh, sv, sk}, bf16_opts);
        torch::Tensor final_state_ref = torch::zeros({shape.batch_size, nvh, sv, sk}, bf16_opts);
        torch::Tensor U_state = U_ref.clone();
        state_passing_ref_varlen(K, U_state, W_ref, initial_state, state_per_chunk, final_state_ref,
                                 cs, cu_seqlens, &gate_cumsum);

        torch::Tensor O_ref =
            compute_O_ref_varlen(Q, state_per_chunk, K, U_state, cs, cu_seqlens, gate_cumsum);

        torch::Tensor state_kernel = torch::zeros_like(state_per_chunk);
        // Kernel path
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        torch::Tensor cu_seqlens_d = test_utils::make_cu_seqlens(cu_seqlens);
        torch::Tensor chunk_indices_d = test_utils::make_chunk_indices(cu_seqlens, (int)cs);
        torch::Tensor cu_chunks_d = test_utils::make_cu_chunks(cu_seqlens, (int)cs);
        std::optional<torch::Tensor> cu_seqlens_opt = cu_seqlens_d;
        std::optional<torch::Tensor> chunk_indices_opt = chunk_indices_d;
        std::optional<torch::Tensor> cu_chunks_opt = cu_chunks_d;
        std::optional<int> total_chunks_opt = (int)total_chunks;

        torch::Tensor gate_raw_f32 = gate_raw_normalized;
        std::optional<torch::Tensor> initial_state_opt =
            config.test_initial_state ? std::optional<torch::Tensor>{initial_state} : std::nullopt;

        std::optional<float> scale_opt = std::nullopt;
        auto [O_kernel, final_state_kernel] = gdn_cuda::chunked_forward(
            Q, K, V, beta, gate_raw_f32, scale_opt, initial_state_opt, cu_seqlens_opt,
            chunk_indices_opt, cu_chunks_opt, total_chunks_opt, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "Checking O:\n";
        bool op = check_tensor_close_or_cosine(O_ref, O_kernel, config.atol, config.rtol, "O");
        std::cout << "Checking final state:\n";
        bool sp = check_tensor_close_or_cosine(final_state_ref, final_state_kernel, config.atol,
                                               config.rtol, "final_state");

        CUDA_CHECK(cudaStreamDestroy(stream));
        return op && sp;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
        return false;
    }
}

bool test_varlen_partial_mode(const GdnFwdVarlenPartialShape& shape, const TestConfig& config) {
    std::cout << "\n--- Testing Varlen Partial Mode: " << shape.description << " ---\n";
    try {
        torch::Device device = torch::kCUDA;
        auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

        int64_t batch_size = static_cast<int64_t>(shape.seq_lengths.size());
        std::vector<int> cu_seqlens(batch_size + 1);
        cu_seqlens[0] = 0;
        int64_t total_tokens = 0;
        int64_t total_chunks = 0;
        for (int64_t b = 0; b < batch_size; ++b) {
            total_tokens += shape.seq_lengths[b];
            total_chunks += ceil_div(shape.seq_lengths[b], shape.chunk_size);
            cu_seqlens[b + 1] = static_cast<int>(total_tokens);
        }

        int64_t nkh = shape.num_k_heads, nvh = shape.num_v_heads;
        int64_t sk = shape.shape_k, sv = shape.shape_v, cs = shape.chunk_size;
        float fsk = 1.f / std::sqrt((float)sk), fsv = 1.f / std::sqrt((float)sv);

        torch::Tensor Q = torch::normal(0, fsk, {total_tokens, nkh, sk}, std::nullopt, bf16_opts);
        torch::Tensor K = torch::normal(0, fsk, {total_tokens, nkh, sk}, std::nullopt, bf16_opts);
        torch::Tensor V = torch::normal(0, fsv, {total_tokens, nvh, sv}, std::nullopt, bf16_opts);
        torch::Tensor beta = torch::sigmoid(torch::randn({total_tokens, nvh}, bf16_opts) * 0.3f);
        torch::Tensor gate_raw =
            torch::normal(0, 0.1f, {total_tokens, nvh}, std::nullopt, bf16_opts);
        torch::Tensor gate_raw_normalized =
            normalize_gate_for_exp(gate_raw.to(torch::kFloat32), shape.gate_normalization_factor);

        // Reference path
        torch::Tensor gate_cumsum =
            chunk_local_cumsum_reference(gate_raw_normalized, cs, &cu_seqlens);
        auto [U_ref, W_ref] = compute_uw_reference(K, V, beta, cs, cu_seqlens, gate_cumsum);

        torch::Tensor initial_state =
            config.test_initial_state
                ? torch::normal(0, fsk * fsv, {batch_size, nvh, sv, sk}, std::nullopt, bf16_opts)
                : torch::zeros({batch_size, nvh, sv, sk}, bf16_opts);
        torch::Tensor state_per_chunk = torch::zeros({total_chunks, nvh, sv, sk}, bf16_opts);
        torch::Tensor final_state_ref = torch::zeros({batch_size, nvh, sv, sk}, bf16_opts);
        torch::Tensor U_state = U_ref.clone();
        state_passing_ref_varlen(K, U_state, W_ref, initial_state, state_per_chunk, final_state_ref,
                                 cs, cu_seqlens, &gate_cumsum);

        torch::Tensor O_ref =
            compute_O_ref_varlen(Q, state_per_chunk, K, U_state, cs, cu_seqlens, gate_cumsum);

        // Kernel path
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        torch::Tensor cu_seqlens_d = test_utils::make_cu_seqlens(cu_seqlens);
        torch::Tensor chunk_indices_d = test_utils::make_chunk_indices(cu_seqlens, (int)cs);
        torch::Tensor cu_chunks_d = test_utils::make_cu_chunks(cu_seqlens, (int)cs);
        std::optional<torch::Tensor> cu_seqlens_opt = cu_seqlens_d;
        std::optional<torch::Tensor> chunk_indices_opt = chunk_indices_d;
        std::optional<torch::Tensor> cu_chunks_opt = cu_chunks_d;
        std::optional<int> total_chunks_opt = (int)total_chunks;

        torch::Tensor state_kernel = torch::zeros_like(state_per_chunk);
        std::optional<torch::Tensor> initial_state_opt =
            config.test_initial_state ? std::optional<torch::Tensor>{initial_state} : std::nullopt;
        std::optional<float> scale_opt = std::nullopt;

        auto [O_kernel, final_state_kernel] = gdn_cuda::chunked_forward(
            Q, K, V, beta, gate_raw_normalized, scale_opt, initial_state_opt, cu_seqlens_opt,
            chunk_indices_opt, cu_chunks_opt, total_chunks_opt, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "Checking O:\n";
        bool op = check_tensor_close_or_cosine(O_ref, O_kernel, config.atol, config.rtol, "O");
        std::cout << "Checking final state:\n";
        bool sp = check_tensor_close_or_cosine(final_state_ref, final_state_kernel, config.atol,
                                               config.rtol, "final_state");

        CUDA_CHECK(cudaStreamDestroy(stream));
        return op && sp;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
        return false;
    }
}

}  // anonymous namespace

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
    std::cout << "GDN Chunked Forward End-to-End Testing Harness\n";
    std::cout << "============================================\n";

    int passed = 0, failed = 0;

    if (config.test_padded) {
        std::cout << "\n=== Padded Mode Tests ===\n";
        for (const auto& s : padded_shapes) {
            bool r = test_padded_mode(s, config);
            if (r) {
                std::cout << "\033[0;32m[PASSED] " << s.description << "\033[0m\n";
                ++passed;
            } else {
                std::cout << "\033[0;31m[FAILED] " << s.description << "\033[0m\n";
                ++failed;
            }
        }
    }
    if (config.test_varlen) {
        std::cout << "\n=== Varlen Mode Tests ===\n";
        for (const auto& s : varlen_shapes) {
            bool r = test_varlen_mode(s, config);
            if (r) {
                std::cout << "\033[0;32m[PASSED] " << s.description << "\033[0m\n";
                ++passed;
            } else {
                std::cout << "\033[0;31m[FAILED] " << s.description << "\033[0m\n";
                ++failed;
            }
        }
        std::cout << "\n=== Varlen Partial Mode Tests ===\n";
        for (const auto& s : varlen_partial_shapes) {
            bool r = test_varlen_partial_mode(s, config);
            if (r) {
                std::cout << "\033[0;32m[PASSED] " << s.description << "\033[0m\n";
                ++passed;
            } else {
                std::cout << "\033[0;31m[FAILED] " << s.description << "\033[0m\n";
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
