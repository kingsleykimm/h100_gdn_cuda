#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <gdn_cuda/types.h>

#include <apis/gdn.hpp>
#include <apis/layout.hpp>
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
    std::optional<at::Tensor>& cu_chunks, std::optional<int> total_chunks, cudaStream_t stream) {
    // Keep preprocessing on the same stream used by downstream kernels.
    at::Tensor beta_mn = api::transpose_to_mn_major(beta, stream);

    // we want to align the head_dims to the closest multiple of 16 bytes for TMA loads

    const int head_dim = value.size(-1);
    HOST_ASSERT(head_dim > 0, "Head dimension must be greater than 0");
    int aligned_dim;
    if (head_dim % 64 != 0) {  // just align to 64 eleemnts for simplicity

        aligned_dim = ti_align(head_dim, 64);
        if (aligned_dim == 0) {
            aligned_dim = 64;
        }
        const int padding = aligned_dim - head_dim;
        std::vector<int64_t> padding_qkv(query.dim() * 2, 0);
        // right pad
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
        aligned_dim = head_dim;
    }

    const bool is_varlen = cu_seqlens.has_value();
    const int batch_size = is_varlen ? 1 : query.size(0);
    const int seq_len = is_varlen ? query.size(0) : query.size(1);
    const int num_heads = value.size(-2);

    at::Tensor state;
    if (is_varlen) {
        int total_chunks_int = total_chunks.has_value()
                                   ? total_chunks.value()
                                   : (chunk_indices ? chunk_indices->size(0) / 2 : 0);
        state = at::empty(
            {static_cast<int64_t>(total_chunks_int), value.size(-2), aligned_dim, aligned_dim},
            value.options());
    } else {
        int num_chunks = (seq_len + 63) / 64;
        state = at::empty({static_cast<int64_t>(batch_size), static_cast<int64_t>(num_chunks),
                           value.size(-2), aligned_dim, aligned_dim},
                          value.options());
    }

    // Compute chunk-local cumsum on original [*, seq, heads] layout.
    // Then build MN-major gate tensors with per-kernel alignment requirements.
    HOST_ASSERT(gate.scalar_type() == at::kFloat, "Gate must be Float32");
    at::Tensor gate_cumsum = at::empty_like(gate, gate.options().dtype(at::kFloat));
    kernels::chunk_local_cumsum<float>(gate, gate_cumsum, batch_size, seq_len, num_heads, true,
                                       cu_seqlens, chunk_indices, stream);

    at::Tensor gate_mn = api::transpose_to_mn_major(gate_cumsum, stream);
    std::optional<at::Tensor> gate_mn_opt = gate_mn;
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    at::Tensor u = at::empty_like(value);
    at::Tensor w = at::empty_like(value);

    const bool is_initial_state = initial_state.has_value();
    std::optional<at::Tensor> final_state_storage;

    if (is_varlen) {
        if (!is_initial_state) {
            const int total_chunks = chunk_indices->size(0) / 2;
            initial_state.emplace(at::zeros(
                {(long)total_chunks, value.size(-2), aligned_dim, aligned_dim}, value.options()));
        }
        final_state_storage.emplace(at::empty(
            {cu_seqlens->size(0) - 1, value.size(-2), aligned_dim, aligned_dim}, value.options()));
    } else {
        const int num_chunks = (seq_len + 63) / 64;
        if (!is_initial_state) {
            initial_state.emplace(at::zeros(
                {(long)batch_size, (long)num_chunks, value.size(-2), aligned_dim, aligned_dim},
                value.options()));
        }
        final_state_storage.emplace(
            at::empty({query.size(0), value.size(-2), aligned_dim, aligned_dim}, value.options()));
    }

    api::bf16_gdn_compute_u_w(key, value, u, w, beta_mn, gate_mn_opt, "t", stream, cu_seqlens,
                              chunk_indices);

    api::bf16_chunked_seq_state_update(key, u, w, initial_state, state, final_state_storage,
                                       gate_mn_opt, "t", stream, cu_seqlens, cu_chunks,
                                       total_chunks);

    at::Tensor out = at::empty_like(value);
    api::bf16_gdn_chunked_compute_O(query, state, key, u, out, gate_mn_opt, scale, "t", stream,
                                    cu_seqlens, chunk_indices, cu_chunks);

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
    cudaStream_t stream, bool is_qk_norm) {
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
    api::bf16_gdn_recurrent(query, key, value, initial_state, final_state_storage, out, gate_opt,
                            beta, "kvt", stream, cu_seqlens, num_accepted_tokens, store_step_state,
                            is_qk_norm);

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
                        bool is_qk_norm) {
    api::bf16_gdn_recurrent(q, k, v, initial_state, final_state, out, gate, beta, compiled_dims,
                            stream, cu_seqlens, num_accepted_tokens, store_step_state, is_qk_norm);
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
                                std::optional<at::Tensor>& cu_chunks, uint32_t chunk_size) {
    api::bf16_gdn_chunked_compute_O(q, state, k, u, o, gate, scale, compiled_dims, stream,
                                    cu_seqlens, chunk_indices, cu_chunks, chunk_size);
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
