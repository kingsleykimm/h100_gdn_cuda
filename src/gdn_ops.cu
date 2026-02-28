// Non-JIT GDN kernel template instantiations.
// Compiled by nvcc (required for gdn_helpers.cuh CUDA templates).

#include <gdn_cuda/kernels/gdn_helpers.cuh>
#include <kernels/internal_api.hpp>

namespace gdn_cuda {
namespace kernels {

// ============================================================================
// GDN Gating operations (non-JIT, compiled directly)
// ============================================================================
template <typename input_t, typename output_t>
void gdn_gating(at::Tensor& A_log, at::Tensor& dt_bias, at::Tensor& a, at::Tensor& b,
                at::Tensor& beta, at::Tensor& g, int num_v_heads, int batch_size, int seqlen,
                cudaStream_t stream, float threshold) {
    gdn_cuda::kernels::gdn_helpers_impl::fused_gdn_gating_kernel_launch<input_t, output_t>(
        reinterpret_cast<input_t*>(A_log.data_ptr()),
        reinterpret_cast<input_t*>(dt_bias.data_ptr()), reinterpret_cast<input_t*>(a.data_ptr()),
        reinterpret_cast<input_t*>(b.data_ptr()), reinterpret_cast<output_t*>(beta.data_ptr()),
        reinterpret_cast<output_t*>(g.data_ptr()), num_v_heads, batch_size, seqlen, stream,
        threshold);
}

template void gdn_gating<__nv_bfloat16, __nv_bfloat16>(at::Tensor&, at::Tensor&, at::Tensor&,
                                                       at::Tensor&, at::Tensor&, at::Tensor&, int,
                                                       int, int, cudaStream_t, float);
template void gdn_gating<__nv_bfloat16, float>(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                                               at::Tensor&, at::Tensor&, int, int, int,
                                               cudaStream_t, float);
template void gdn_gating<float, float>(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                                       at::Tensor&, at::Tensor&, int, int, int, cudaStream_t,
                                       float);

// ============================================================================
// Chunk local cumsum (non-JIT, compiled directly)
// ============================================================================
template <typename input_t>
void chunk_local_cumsum(at::Tensor& input, at::Tensor& output, int batch_size, int seq_len,
                        int num_heads, bool head_first, std::optional<at::Tensor>& cu_seqlens,
                        std::optional<at::Tensor>& chunk_indices, cudaStream_t stream) {
    if (head_first) {
        gdn_cuda::kernels::gdn_helpers_impl::chunk_local_cumsum_kernel_launch_headfirst<input_t>(
            reinterpret_cast<input_t*>(input.data_ptr()), output.data_ptr<float>(), batch_size,
            seq_len, num_heads,
            cu_seqlens.has_value() ? cu_seqlens.value().data_ptr<int>() : nullptr,
            chunk_indices.has_value() ? chunk_indices.value().data_ptr<int>() : nullptr,
            chunk_indices.has_value() ? (int)(chunk_indices->size(0) / 2) : 0, stream);
    } else {
        gdn_cuda::kernels::gdn_helpers_impl::chunk_local_cumsum_kernel_launch_seqfirst<input_t>(
            reinterpret_cast<input_t*>(input.data_ptr()), output.data_ptr<float>(), batch_size,
            seq_len, num_heads,
            cu_seqlens.has_value() ? cu_seqlens.value().data_ptr<int>() : nullptr,
            chunk_indices.has_value() ? chunk_indices.value().data_ptr<int>() : nullptr,
            chunk_indices.has_value() ? (int)(chunk_indices->size(0) / 2) : 0, stream);
    }
}

template void chunk_local_cumsum<__nv_bfloat16>(at::Tensor&, at::Tensor&, int, int, int, bool,
                                                std::optional<at::Tensor>&,
                                                std::optional<at::Tensor>&, cudaStream_t);
template void chunk_local_cumsum<float>(at::Tensor&, at::Tensor&, int, int, int, bool,
                                        std::optional<at::Tensor>&, std::optional<at::Tensor>&,
                                        cudaStream_t);

}  // namespace kernels
}  // namespace gdn_cuda
