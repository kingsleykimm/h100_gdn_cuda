#pragma once
// Forward declarations for high-level GDN forward passes and individual kernel wrappers.
// Implementations live in gdn_forward.cpp.

#include <cuda_runtime.h>
#include <gdn_cuda/types.h>
#include <torch/torch.h>

#include <optional>
#include <string>
#include <utility>

namespace gdn_cuda {

// Pre-allocated workspace for chunked_forward to eliminate per-call cudaMalloc overhead.
// Create once, pass on every call. Tensors are lazily allocated on the first call and
// reused on subsequent calls when the input shapes match.  When shapes change the
// workspace automatically reallocates.
struct ChunkedForwardWorkspace {
    // Padded input buffers (only used when head_dim % 64 != 0)
    at::Tensor padded_query;
    at::Tensor padded_key;
    at::Tensor padded_value;
    at::Tensor padded_initial_state;  // for user-provided initial_state padding

    // Intermediate tensors
    at::Tensor beta_mn;
    at::Tensor gate_cumsum;
    at::Tensor gate_mn;
    at::Tensor u;
    at::Tensor w;
    at::Tensor state;
    at::Tensor out;
    at::Tensor final_state;
    at::Tensor default_initial_state;  // zeros for when no initial_state provided

    // Shape tracking
    bool allocated_ = false;
    int64_t batch_ = 0, seq_ = 0, heads_ = 0;
    int64_t head_dim_ = 0, aligned_dim_ = 0;
    bool is_varlen_ = false;
    int64_t total_chunks_ = 0, n_seq_ = 0;

    bool is_valid_for(int64_t batch, int64_t seq, int64_t heads, int64_t head_dim,
                      int64_t aligned_dim, bool is_varlen, int64_t total_chunks,
                      int64_t n_seq) const {
        return allocated_ && batch_ == batch && seq_ == seq && heads_ == heads &&
               head_dim_ == head_dim && aligned_dim_ == aligned_dim && is_varlen_ == is_varlen &&
               total_chunks_ == total_chunks && n_seq_ == n_seq;
    }
};

void init(const std::string& library_root, const std::string& cuda_home);

// High-level forward passes

std::pair<at::Tensor, at::Tensor> chunked_forward(
    at::Tensor& query, at::Tensor& key, at::Tensor& value, at::Tensor& beta, at::Tensor& gate,
    std::optional<float> scale, std::optional<at::Tensor>& initial_state,
    std::optional<at::Tensor>& cu_seqlens, std::optional<at::Tensor>& chunk_indices,
    std::optional<at::Tensor>& cu_chunks, std::optional<int> total_chunks,
    ChunkedForwardWorkspace* workspace, cudaStream_t stream);

std::pair<at::Tensor, at::Tensor> recurrent_forward(
    at::Tensor& query, at::Tensor& key, at::Tensor& value, std::optional<at::Tensor>& initial_state,
    at::Tensor& beta, at::Tensor& gate, std::optional<at::Tensor>& cu_seqlens,
    std::optional<at::Tensor>& num_accepted_tokens, InferenceMode inference_mode,
    cudaStream_t stream, bool is_qk_norm = false, std::optional<float> scale = std::nullopt);

std::pair<at::Tensor, at::Tensor> fused_gdn_gating(at::Tensor& A_log, at::Tensor& dt_bias,
                                                   at::Tensor& a, at::Tensor& b, bool is_var_len,
                                                   cudaStream_t stream);

// Individual kernel wrappers (for testing / benchmarking)

void bf16_gdn_compute_u_w(at::Tensor& k, at::Tensor& v, at::Tensor& u, at::Tensor& w,
                          at::Tensor& beta, std::optional<at::Tensor>& gate,
                          const std::string& compiled_dims, cudaStream_t stream,
                          std::optional<at::Tensor>& cu_seqlens,
                          std::optional<at::Tensor>& chunk_indices);

void bf16_gdn_recurrent(at::Tensor& q, at::Tensor& k, at::Tensor& v,
                        std::optional<at::Tensor>& initial_state, at::Tensor& final_state,
                        at::Tensor& out, std::optional<at::Tensor>& gate, at::Tensor& beta,
                        const std::string& compiled_dims, cudaStream_t stream,
                        std::optional<at::Tensor>& cu_seqlens,
                        std::optional<at::Tensor>& num_accepted_tokens,
                        bool store_step_state = false, bool is_qk_norm = false, float scale = 1.0f);

void bf16_chunked_seq_state_update(at::Tensor& k, at::Tensor& u, at::Tensor& w,
                                   std::optional<at::Tensor>& initial_state, at::Tensor& state,
                                   std::optional<at::Tensor>& final_state,
                                   std::optional<at::Tensor>& gate,
                                   const std::string& compiled_dims, cudaStream_t stream,
                                   std::optional<at::Tensor>& cu_seqlens,
                                   std::optional<at::Tensor>& cu_chunks,
                                   std::optional<int> total_chunks, uint32_t chunk_size = 64);

void bf16_gdn_chunked_compute_O(at::Tensor& q, at::Tensor& state, at::Tensor& k, at::Tensor& u,
                                at::Tensor& o, std::optional<at::Tensor>& gate,
                                std::optional<float> scale, const std::string& compiled_dims,
                                cudaStream_t stream, std::optional<at::Tensor>& cu_seqlens,
                                std::optional<at::Tensor>& chunk_indices,
                                std::optional<at::Tensor>& cu_chunks,
                                std::optional<int> total_chunks, uint32_t chunk_size = 64);

void chunk_local_cumsum_bf16(at::Tensor& input, at::Tensor& output, int batch_size, int seq_len,
                             int num_heads, bool head_first, std::optional<at::Tensor>& cu_seqlens,
                             std::optional<at::Tensor>& chunk_indices,
                             cudaStream_t stream = nullptr);

at::Tensor transpose_to_mn_major(at::Tensor& input, cudaStream_t stream, uint32_t alignment = 16);

}  // namespace gdn_cuda
