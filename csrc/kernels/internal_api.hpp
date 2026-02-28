#pragma once
// Non-JIT kernel declarations (gdn_gating, chunk_local_cumsum).
// Implementations are template instantiations in src/gdn_ops.cu.

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <optional>
#include <string>

namespace gdn_cuda {
namespace kernels {

template <typename input_t, typename output_t>
void gdn_gating(at::Tensor& A_log, at::Tensor& dt_bias, at::Tensor& a, at::Tensor& b,
                at::Tensor& beta, at::Tensor& g, int num_v_heads, int batch_size, int seqlen,
                cudaStream_t stream, float threshold = 20.0f);

template <typename input_t>
void chunk_local_cumsum(at::Tensor& input, at::Tensor& output, int batch_size, int seq_len,
                        int num_heads, bool head_first, std::optional<at::Tensor>& cu_seqlens,
                        std::optional<at::Tensor>& chunk_indices, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace gdn_cuda
