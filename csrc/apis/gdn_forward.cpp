#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <gdn_cuda/types.h>

#include <apis/gdn.hpp>
#include <apis/gdn_forward.hpp>
#include <apis/layout.hpp>
#include <cmath>
#include <jit/compiler.hpp>
#include <jit/utils/common.hpp>
#include <kernels/internal_api.hpp>

namespace gdn_cuda {

void init(const std::string& library_root, const std::string& cuda_home) {
    Compiler::init_static_vars(library_root, cuda_home);
}

std::pair<at::Tensor, at::Tensor> chunked_forward(
    at::Tensor& query, at::Tensor& key, at::Tensor& value, at::Tensor& beta, at::Tensor& gate,
    std::optional<float> scale, std::optional<at::Tensor>& initial_state,
    std::optional<at::Tensor>& cu_seqlens, std::optional<at::Tensor>& chunk_indices,
    std::optional<at::Tensor>& cu_chunks, std::optional<int> total_chunks,
    ChunkedForwardWorkspace* workspace, cudaStream_t stream) {
    // --- 1. Compute head_dim and aligned_dim (pure math, no allocations) ---
    const int head_dim = value.size(-1);
    HOST_ASSERT(head_dim > 0, "Head dimension must be greater than 0");
    int aligned_dim;
    if (head_dim % 64 != 0) {
        aligned_dim = ti_align(head_dim, 64);
        if (aligned_dim == 0) {
            aligned_dim = 64;
        }
    } else {
        aligned_dim = head_dim;
    }
    const bool needs_padding = (head_dim % 64 != 0);

    // --- 2. Compute dimensions ---
    const bool is_varlen = cu_seqlens.has_value();
    const int batch_size = is_varlen ? 1 : query.size(0);
    const int seq_len = is_varlen ? query.size(0) : query.size(1);
    const int num_heads = value.size(-2);
    const bool is_initial_state = initial_state.has_value();

    int total_chunks_int = 0;
    int n_seq = 0;
    int num_chunks = 0;
    if (is_varlen) {
        total_chunks_int = total_chunks.has_value()
                               ? total_chunks.value()
                               : (chunk_indices ? static_cast<int>(chunk_indices->size(0) / 2) : 0);
        n_seq = static_cast<int>(cu_seqlens->size(0) - 1);
    } else {
        num_chunks = (seq_len + 63) / 64;
        total_chunks_int = num_chunks;
    }

    // --- 3. Check workspace validity ---
    HOST_ASSERT(gate.scalar_type() == at::kFloat, "Gate must be Float32");
    const bool ws_valid =
        workspace && workspace->is_valid_for(batch_size, seq_len, num_heads, head_dim, aligned_dim,
                                             is_varlen, total_chunks_int, n_seq);

    // --- 4. Handle padding ---
    if (needs_padding) {
        if (ws_valid) {
            // Copy into pre-allocated padded buffers (no cudaMalloc)
            workspace->padded_query.narrow(-1, 0, head_dim).copy_(query);
            workspace->padded_key.narrow(-1, 0, head_dim).copy_(key);
            workspace->padded_value.narrow(-1, 0, head_dim).copy_(value);
            query = workspace->padded_query;
            key = workspace->padded_key;
            value = workspace->padded_value;
            if (initial_state.has_value()) {
                workspace->padded_initial_state.narrow(-1, 0, head_dim)
                    .narrow(-2, 0, head_dim)
                    .copy_(initial_state.value());
                initial_state.emplace(workspace->padded_initial_state);
            }
        } else {
            // Fresh allocation via torch::pad
            const int padding = aligned_dim - head_dim;
            std::vector<int64_t> padding_qkv(query.dim() * 2, 0);
            padding_qkv[1] = padding;

            query = torch::pad(query, padding_qkv, "constant", 0);
            key = torch::pad(key, padding_qkv, "constant", 0);
            value = torch::pad(value, padding_qkv, "constant", 0);
            if (initial_state.has_value()) {
                std::vector<int64_t> padding_initial_state(initial_state->dim() * 2, 0);
                padding_initial_state[1] = padding;
                padding_initial_state[3] = padding;
                initial_state.emplace(
                    torch::pad(initial_state.value(), padding_initial_state, "constant", 0));
            }
        }
    }

    // --- 5. Allocate/reuse intermediate tensors ---
    at::Tensor beta_mn, gate_cumsum, gate_mn, u, w, state, out;
    std::optional<at::Tensor> final_state_storage;

    if (ws_valid) {
        // --- Reuse workspace tensors (zero allocations) ---
        api::transpose_to_mn_major_into(beta, workspace->beta_mn, stream);
        beta_mn = workspace->beta_mn;
        gate_cumsum = workspace->gate_cumsum;
        u = workspace->u;
        w = workspace->w;
        state = workspace->state;
        out = workspace->out;
        final_state_storage = workspace->final_state;

        if (!is_initial_state) {
            workspace->default_initial_state.zero_();
            initial_state.emplace(workspace->default_initial_state);
        }
    } else {
        // --- Fresh allocations ---
        beta_mn = api::transpose_to_mn_major(beta, stream);
        gate_cumsum = at::empty_like(gate, gate.options().dtype(at::kFloat));
        u = at::empty_like(value);
        w = at::empty_like(value);
        out = at::empty_like(value);

        if (is_varlen) {
            state = at::empty(
                {static_cast<int64_t>(total_chunks_int), value.size(-2), aligned_dim, aligned_dim},
                value.options());
            if (!is_initial_state) {
                initial_state.emplace(at::zeros(
                    {(long)n_seq, value.size(-2), aligned_dim, aligned_dim}, value.options()));
            }
            final_state_storage.emplace(at::empty(
                {(long)n_seq, value.size(-2), aligned_dim, aligned_dim}, value.options()));
        } else {
            state = at::empty({static_cast<int64_t>(batch_size), static_cast<int64_t>(num_chunks),
                               value.size(-2), aligned_dim, aligned_dim},
                              value.options());
            if (!is_initial_state) {
                initial_state.emplace(at::zeros(
                    {(long)batch_size, value.size(-2), aligned_dim, aligned_dim}, value.options()));
            }
            final_state_storage.emplace(at::empty(
                {query.size(0), value.size(-2), aligned_dim, aligned_dim}, value.options()));
        }
    }

    // --- 6. Cumsum kernel (writes into gate_cumsum) ---
    kernels::chunk_local_cumsum<float>(gate, gate_cumsum, batch_size, seq_len, num_heads, true,
                                       cu_seqlens, chunk_indices, stream);

    // --- Gate MN-major transpose ---
    if (ws_valid) {
        api::transpose_to_mn_major_into(gate_cumsum, workspace->gate_mn, stream);
        gate_mn = workspace->gate_mn;
    } else {
        gate_mn = api::transpose_to_mn_major(gate_cumsum, stream);
    }
    std::optional<at::Tensor> gate_mn_opt = gate_mn;

    // --- Save to workspace on first allocation ---
    if (workspace && !ws_valid) {
        workspace->beta_mn = beta_mn;
        workspace->gate_cumsum = gate_cumsum;
        workspace->gate_mn = gate_mn;
        workspace->u = u;
        workspace->w = w;
        workspace->state = state;
        workspace->out = out;
        workspace->final_state = final_state_storage.value();

        // Save padded input buffers
        if (needs_padding) {
            workspace->padded_query = query;
            workspace->padded_key = key;
            workspace->padded_value = value;
            if (is_initial_state) {
                workspace->padded_initial_state = initial_state.value();
            } else {
                // Pre-allocate padded_initial_state for future calls that provide one
                if (is_varlen) {
                    workspace->padded_initial_state = at::zeros(
                        {(long)n_seq, value.size(-2), aligned_dim, aligned_dim}, value.options());
                } else {
                    workspace->padded_initial_state =
                        at::zeros({(long)batch_size, value.size(-2), aligned_dim, aligned_dim},
                                  value.options());
                }
            }
        }

        // Save default_initial_state with correct 4D shape
        if (!is_initial_state) {
            workspace->default_initial_state = initial_state.value();
        } else {
            // Pre-allocate default_initial_state for future calls that may not provide one.
            if (is_varlen) {
                workspace->default_initial_state = at::zeros(
                    {(long)n_seq, value.size(-2), aligned_dim, aligned_dim}, value.options());
            } else {
                workspace->default_initial_state = at::zeros(
                    {(long)batch_size, value.size(-2), aligned_dim, aligned_dim}, value.options());
            }
        }
        workspace->allocated_ = true;
        workspace->batch_ = batch_size;
        workspace->seq_ = seq_len;
        workspace->heads_ = num_heads;
        workspace->head_dim_ = head_dim;
        workspace->aligned_dim_ = aligned_dim;
        workspace->is_varlen_ = is_varlen;
        workspace->total_chunks_ = total_chunks_int;
        workspace->n_seq_ = n_seq;
    }

    // --- 7. Compute kernels ---
    api::bf16_gdn_compute_u_w(key, value, u, w, beta_mn, gate_mn_opt, "kvt", stream, cu_seqlens,
                              chunk_indices);

    api::bf16_chunked_seq_state_update(key, u, w, initial_state, state, final_state_storage,
                                       gate_mn_opt, "kvt", stream, cu_seqlens, cu_chunks,
                                       total_chunks);

    api::bf16_gdn_chunked_compute_O(query, state, key, u, out, gate_mn_opt, scale, "kvt", stream,
                                    cu_seqlens, chunk_indices, cu_chunks, total_chunks);

    // --- Slice to original head_dim ---
    out = out.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, head_dim)});
    auto final_state = final_state_storage.value();
    final_state = final_state.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, head_dim),
                                     torch::indexing::Slice(0, head_dim)});

    return std::make_pair(out, final_state);
}

std::pair<at::Tensor, at::Tensor> recurrent_forward(
    at::Tensor& query, at::Tensor& key, at::Tensor& value, std::optional<at::Tensor>& initial_state,
    at::Tensor& beta, at::Tensor& gate, std::optional<at::Tensor>& cu_seqlens,
    std::optional<at::Tensor>& num_accepted_tokens, InferenceMode inference_mode,
    cudaStream_t stream, bool is_qk_norm, std::optional<float> scale) {
    const bool store_step_state = (inference_mode == InferenceMode::SpecVerify);
    const bool is_varlen = cu_seqlens.has_value();

    const int64_t batch_size = is_varlen ? cu_seqlens->size(0) - 1 : query.size(0);
    const int64_t seq_len = is_varlen ? query.size(0) : query.size(1);
    const int64_t num_v_heads = value.size(-2);

    int64_t head_dim = query.size(-1);

    int64_t aligned_head_dim;
    if (head_dim % 64 != 0) {
        aligned_head_dim = ti_align(head_dim, 64);
        if (aligned_head_dim == 0) {
            aligned_head_dim = 64;
        }
        const int64_t padding = aligned_head_dim - head_dim;
        std::vector<int64_t> padding_qkv(query.dim() * 2, 0);
        // right pad on last dimension
        padding_qkv[1] = padding;

        // we zero pad the tensors along the head dim
        query = torch::pad(query, padding_qkv, "constant", 0);
        key = torch::pad(key, padding_qkv, "constant", 0);
        value = torch::pad(value, padding_qkv, "constant", 0);

        if (initial_state.has_value()) {
            std::vector<int64_t> padding_initial_state(initial_state->dim() * 2, 0);
            padding_initial_state[1] = padding;  // right side of dim = 0
            padding_initial_state[3] = padding;  // right side of dim = 1
            initial_state.emplace(
                torch::pad(initial_state.value(), padding_initial_state, "constant", 0));
        }
    } else {
        aligned_head_dim = head_dim;
    }

    at::Tensor final_state_storage;
    if (store_step_state) {
        if (is_varlen) {
            final_state_storage =
                at::empty({value.size(0), value.size(1), aligned_head_dim, aligned_head_dim},
                          value.options());
        } else {
            final_state_storage = at::empty(
                {query.size(0), query.size(1), value.size(2), aligned_head_dim, aligned_head_dim},
                value.options());
        }
    } else {
        final_state_storage =
            at::empty({static_cast<int64_t>(batch_size), static_cast<int64_t>(num_v_heads),
                       aligned_head_dim, aligned_head_dim},
                      value.options());
    }

    at::Tensor out = at::empty_like(value);
    std::optional<at::Tensor> gate_opt = gate;
    float actual_scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(head_dim)));
    api::bf16_gdn_recurrent(query, key, value, initial_state, final_state_storage, out, gate_opt,
                            beta, "kvt", stream, cu_seqlens, num_accepted_tokens, store_step_state,
                            is_qk_norm, actual_scale);

    out = out.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, head_dim)});
    final_state_storage =
        final_state_storage.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, head_dim),
                                   torch::indexing::Slice(0, head_dim)});

    return std::make_pair(std::move(out), std::move(final_state_storage));
}

std::pair<at::Tensor, at::Tensor> fused_gdn_gating(at::Tensor& A_log, at::Tensor& dt_bias,
                                                   at::Tensor& a, at::Tensor& b, bool is_var_len,
                                                   cudaStream_t stream) {
    at::Tensor beta_storage = at::empty_like(b);
    at::Tensor g_storage = at::empty_like(A_log);

    const int seqlen = is_var_len ? 1 : A_log.size(1);
    const int batch_size = A_log.size(0);
    const int num_v_heads = A_log.size(-1);

    if (A_log.scalar_type() == at::kFloat) {
        kernels::gdn_gating<float, float>(A_log, dt_bias, a, b, beta_storage, g_storage,
                                          num_v_heads, batch_size, seqlen, stream);
    } else if (A_log.scalar_type() == at::kBFloat16) {
        kernels::gdn_gating<__nv_bfloat16, __nv_bfloat16>(
            A_log, dt_bias, a, b, beta_storage, g_storage, num_v_heads, batch_size, seqlen, stream);
    } else {
        HOST_ASSERT(false, "Unsupported dtype for GDN Gating");
    }

    return std::make_pair(std::move(beta_storage), std::move(g_storage));
}

void bf16_gdn_compute_u_w(at::Tensor& k, at::Tensor& v, at::Tensor& u, at::Tensor& w,
                          at::Tensor& beta, std::optional<at::Tensor>& gate,
                          const std::string& compiled_dims, cudaStream_t stream,
                          std::optional<at::Tensor>& cu_seqlens,
                          std::optional<at::Tensor>& chunk_indices) {
    api::bf16_gdn_compute_u_w(k, v, u, w, beta, gate, compiled_dims, stream, cu_seqlens,
                              chunk_indices);
}

void bf16_gdn_recurrent(at::Tensor& q, at::Tensor& k, at::Tensor& v,
                        std::optional<at::Tensor>& initial_state, at::Tensor& final_state,
                        at::Tensor& out, std::optional<at::Tensor>& gate, at::Tensor& beta,
                        const std::string& compiled_dims, cudaStream_t stream,
                        std::optional<at::Tensor>& cu_seqlens,
                        std::optional<at::Tensor>& num_accepted_tokens, bool store_step_state,
                        bool is_qk_norm, float scale) {
    api::bf16_gdn_recurrent(q, k, v, initial_state, final_state, out, gate, beta, compiled_dims,
                            stream, cu_seqlens, num_accepted_tokens, store_step_state, is_qk_norm,
                            scale);
}

void bf16_chunked_seq_state_update(
    at::Tensor& k, at::Tensor& u, at::Tensor& w, std::optional<at::Tensor>& initial_state,
    at::Tensor& state, std::optional<at::Tensor>& final_state, std::optional<at::Tensor>& gate,
    const std::string& compiled_dims, cudaStream_t stream, std::optional<at::Tensor>& cu_seqlens,
    std::optional<at::Tensor>& cu_chunks, std::optional<int> total_chunks, uint32_t chunk_size) {
    api::bf16_chunked_seq_state_update(k, u, w, initial_state, state, final_state, gate,
                                       compiled_dims, stream, cu_seqlens, cu_chunks, total_chunks,
                                       chunk_size);
}

void bf16_gdn_chunked_compute_O(at::Tensor& q, at::Tensor& state, at::Tensor& k, at::Tensor& u,
                                at::Tensor& o, std::optional<at::Tensor>& gate,
                                std::optional<float> scale, const std::string& compiled_dims,
                                cudaStream_t stream, std::optional<at::Tensor>& cu_seqlens,
                                std::optional<at::Tensor>& chunk_indices,
                                std::optional<at::Tensor>& cu_chunks,
                                std::optional<int> total_chunks, uint32_t chunk_size) {
    api::bf16_gdn_chunked_compute_O(q, state, k, u, o, gate, scale, compiled_dims, stream,
                                    cu_seqlens, chunk_indices, cu_chunks, total_chunks, chunk_size);
}

void chunk_local_cumsum_bf16(at::Tensor& input, at::Tensor& output, int batch_size, int seq_len,
                             int num_heads, bool head_first, std::optional<at::Tensor>& cu_seqlens,
                             std::optional<at::Tensor>& chunk_indices, cudaStream_t stream) {
    kernels::chunk_local_cumsum<__nv_bfloat16>(input, output, batch_size, seq_len, num_heads,
                                               head_first, cu_seqlens, chunk_indices, stream);
}

at::Tensor transpose_to_mn_major(at::Tensor& input, cudaStream_t stream, uint32_t alignment) {
    return api::transpose_to_mn_major(input, stream, alignment);
}

}  // namespace gdn_cuda
