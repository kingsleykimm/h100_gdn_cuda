#pragma once
// Inline JIT kernel wrappers.
// Dispatches directly to jit_kernels/impls/ based on device compute capability.

#include <gdn_cuda/device.hpp>
#include <jit_kernels/impls/sm90_bf16_chunked_seq_state.hpp>
#include <jit_kernels/impls/sm90_bf16_compute_u_w.hpp>
#include <jit_kernels/impls/sm90_bf16_gdn_chunked_compute_O.hpp>
#include <jit_kernels/impls/sm90_bf16_gdn_recurrent.hpp>

namespace gdn_cuda {
namespace api {

inline void bf16_gdn_compute_u_w(at::Tensor& k, at::Tensor& v, at::Tensor& u, at::Tensor& w,
                                 at::Tensor& beta, std::optional<at::Tensor>& gate,
                                 const std::string& compiled_dims, cudaStream_t stream,
                                 std::optional<at::Tensor>& cu_seqlens,
                                 std::optional<at::Tensor>& chunk_indices) {
    int major = device_prop->get_major_minor().first;
    if (major == 9) {
        sm90_bf16_compute_u_w(k, v, u, w, beta, gate, compiled_dims, stream, cu_seqlens,
                              chunk_indices);
    }
}

inline void bf16_gdn_recurrent(at::Tensor& q, at::Tensor& k, at::Tensor& v,
                               std::optional<at::Tensor>& initial_state, at::Tensor& final_state,
                               at::Tensor& out, std::optional<at::Tensor>& gate, at::Tensor& beta,
                               const std::string& compiled_dims, cudaStream_t stream,
                               std::optional<at::Tensor>& cu_seqlens,
                               std::optional<at::Tensor>& num_accepted_tokens,
                               bool store_step_state, bool is_qk_norm, float scale = 1.0f) {
    int major = device_prop->get_major_minor().first;
    if (major == 9) {
        sm90_bf16_gdn_recurrent(q, k, v, initial_state, final_state, out, gate, beta, compiled_dims,
                                stream, cu_seqlens, num_accepted_tokens, store_step_state,
                                is_qk_norm, scale);
    }
}

inline void bf16_chunked_seq_state_update(
    at::Tensor& k, at::Tensor& u, at::Tensor& w, std::optional<at::Tensor>& initial_state,
    at::Tensor& state, std::optional<at::Tensor>& final_state, std::optional<at::Tensor>& gate,
    const std::string& compiled_dims, cudaStream_t stream, std::optional<at::Tensor>& cu_seqlens,
    std::optional<at::Tensor>& cu_chunks, std::optional<int> total_chunks,
    const uint32_t chunk_size = 64) {
    int major = device_prop->get_major_minor().first;
    // HOST_ASSERT(output_final_state, "Final state must be provided if output_final_state is
    // true");
    if (major == 9) {
        sm90_bf16_chunked_seq_state_update(k, u, w, initial_state, state, final_state, gate,
                                           compiled_dims, stream, cu_seqlens, cu_chunks,
                                           total_chunks, chunk_size);
    }
}

inline void bf16_gdn_chunked_compute_O(
    at::Tensor& q, at::Tensor& state, at::Tensor& k, at::Tensor& u, at::Tensor& o,
    std::optional<at::Tensor>& gate, std::optional<float> scale, const std::string& compiled_dims,
    cudaStream_t stream, std::optional<at::Tensor>& cu_seqlens,
    std::optional<at::Tensor>& chunk_indices, std::optional<at::Tensor>& cu_chunks,
    std::optional<int> total_chunks, const uint32_t chunk_size = 64) {
    int major = device_prop->get_major_minor().first;
    if (major == 9) {
        sm90_bf16_gdn_chunked_compute_O(q, state, k, u, o, gate, scale, compiled_dims, stream,
                                        cu_seqlens, chunk_indices, cu_chunks, total_chunks,
                                        chunk_size);
    }
}

}  // namespace api
}  // namespace gdn_cuda
