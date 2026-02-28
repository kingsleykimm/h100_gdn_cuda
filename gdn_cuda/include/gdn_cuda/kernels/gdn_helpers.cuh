#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <gdn_cuda/device.hpp>
#include <gdn_cuda/error.hpp>
#include <gdn_cuda/kernels/common/common.hpp>

namespace gdn_cuda {
namespace kernels {
namespace gdn_helpers_impl {

template <typename input_t_, typename output_t_>
struct GDNGatingParams {
    input_t_ *A_log_ptr;
    input_t_ *dt_bias_ptr;
    input_t_ *b_ptr;
    input_t_ *a_ptr;
    output_t_ *beta_ptr;
    output_t_ *g_ptr;

    float threshold;
    float beta = 1.0;

    int block_head_size;
    int num_v_heads;
    int batch_size;
    int seqlen;

    int num_seq_per_block;

    using input_t = input_t_;
    using output_t = output_t_;

    static constexpr int vec_load_size = 8;
};

// could jit this kernel for num_v_heads, maybe later
template <class kParams>
__global__ void __launch_bounds__(256, 1) gdn_gating_kernel(kParams params) {
    int seq_block_idx = blockIdx.x;
    int v_head_block_idx = blockIdx.y;

    using input_t = typename kParams::input_t;
    using output_t = typename kParams::output_t;

    // Each thread handles one sequence from the block of 32 sequences
    int actual_seq_idx = seq_block_idx * params.num_seq_per_block + threadIdx.x;

    // Each block handles 8 consecutive v_heads (block_head_size elements per head)
    int v_head_start = v_head_block_idx * 8;

    // Bounds check
    if (actual_seq_idx >= params.seqlen || v_head_start >= params.num_v_heads) {
        return;
    }

    // Calculate offsets
    // A_log: shape [num_v_heads, block_head_size]
    input_t *A_log_arr = params.A_log_ptr + v_head_start;
    input_t *dt_bias_arr = params.dt_bias_ptr + v_head_start;

    // a, b, g, beta: shape [seqlen, num_v_heads, block_head_size]
    int base_offset = actual_seq_idx * params.num_v_heads + v_head_start;

    input_t *a_arr = params.a_ptr + base_offset;
    input_t *b_arr = params.b_ptr + base_offset;
    auto g_ptr = params.g_ptr + base_offset;
    auto beta_ptr = params.beta_ptr + base_offset;

    // we assume here that dt_bias = 1 for all

#pragma unroll
    for (int i = 0; i < 8; i++) {
        input_t A_log_val = __ldg(A_log_arr + i);
        input_t a_val = __ldg(a_arr + i);
        input_t b_val = __ldg(b_arr + i);

        float x = to_float(a_val) + to_float(dt_bias_arr[i]);
        float softplus_x = (params.beta * x <= params.threshold)
                               ? (1 / params.beta) * __logf(1 + __expf(params.beta * x))
                               : x;
        float g = -1.f * __expf(to_float(A_log_val)) * softplus_x;
        float beta_full = 1 / (1 + __expf(-to_float(b_val)));

        g_ptr[i] = g;
        beta_ptr[i] = beta_full;
    }
}

template <typename input_t, typename output_t = float>
inline void fused_gdn_gating_kernel_launch(input_t *A_log_ptr, input_t *dt_bias_ptr, input_t *a_ptr,
                                           input_t *b_ptr, output_t *beta_ptr, output_t *g_ptr,
                                           int num_v_heads, int batch_size, int seqlen,
                                           cudaStream_t stream, float threshold = 20.f) {
    // Each block handles 32 sequences (one warp) and 8 v_heads
    int num_seq_per_block = 32;
    int total_seqlen = batch_size * seqlen;
    dim3 blockDim = dim3(32 * 8);
    dim3 gridDim = dim3((total_seqlen + 31) / 32, (num_v_heads + 7) / 8);
    GDNGatingParams<input_t, output_t> params;
    params.A_log_ptr = A_log_ptr;
    params.dt_bias_ptr = dt_bias_ptr;
    params.a_ptr = a_ptr;
    params.b_ptr = b_ptr;
    params.beta_ptr = beta_ptr;
    params.g_ptr = g_ptr;
    params.threshold = threshold;
    params.num_v_heads = num_v_heads;
    params.block_head_size = 8;
    params.batch_size = 1;
    params.seqlen = total_seqlen;
    params.num_seq_per_block = num_seq_per_block;
    params.beta = 1.0;
    // right now num_v_heads and block_head_size are hardcoded to 8
    HOST_ASSERT(num_v_heads >= 8, "num_v_heads must be greater than or equal to vec_load_size");
    HOST_ASSERT(num_v_heads % 8 == 0, "num_v_heads must be a multiple of 8");

    gdn_gating_kernel<<<gridDim, blockDim, 0, stream>>>(params);
}

template <typename input_t_, int kChunkSize_ = 64>
struct ChunkLocalCumsumParams {
    using input_t = input_t_;
    static constexpr int kChunkSize = kChunkSize_;
    static constexpr int kThreads = kChunkSize_;

    input_t *g_ptr;
    float *output_ptr;
    int *cu_seqlens_ptr;     // device pointer, nullptr for fixed length
    int *chunk_indices_ptr;  // device pointer, nullptr for fixed length

    int seq_len;
    int num_heads;
};

// referenced from FLA's chunk local cumsum implementation (cumsum.py)
// grid: (num_chunks, batch_size * num_heads), block: (kChunkSize)
// gate layout is non-head-first: (total_seq, num_heads) with stride num_heads between timesteps
template <class kParams>
__global__ void __launch_bounds__(kParams::kThreads)
    chunk_local_cumsum_kernel_headfirst(kParams params) {
    using input_t = typename kParams::input_t;
    constexpr int kChunkSize = kParams::kChunkSize;

    int chunk_idx = blockIdx.x;
    int batch_head_idx = blockIdx.y;

    int head_idx = batch_head_idx % params.num_heads;
    int batch_idx = batch_head_idx / params.num_heads;
    bool is_varlen = params.cu_seqlens_ptr != nullptr;

    int bos, eos, local_chunk_idx;

    if (is_varlen) {
        int batch_index = __ldg(params.chunk_indices_ptr + chunk_idx * 2);
        local_chunk_idx = __ldg(params.chunk_indices_ptr + chunk_idx * 2 + 1);
        bos = __ldg(params.cu_seqlens_ptr + batch_index);
        eos = __ldg(params.cu_seqlens_ptr + batch_index + 1);
    } else {
        bos = batch_idx * params.seq_len;
        eos = bos + params.seq_len;
        local_chunk_idx = chunk_idx;
    }

    int chunk_start = bos + local_chunk_idx * kChunkSize;
    int tid = threadIdx.x;

    // Load gate value with bounds check, converting to float for precision
    // gate layout: (total_seq, num_heads), stride between timesteps = num_heads
    float val = 0.0f;
    if (chunk_start + tid < eos) {
        val = to_float(__ldg(params.g_ptr + (chunk_start + tid) * params.num_heads + head_idx));
    }

    // Inclusive prefix sum via CUB BlockScan
    using BlockScan = cub::BlockScan<float, kChunkSize>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    float cumsum;
    BlockScan(temp_storage).InclusiveSum(val, cumsum);

    // Store result as float — the state update kernel consumes gate as float
    if (chunk_start + tid < eos) {
        params.output_ptr[(chunk_start + tid) * params.num_heads + head_idx] = cumsum;
    }
}

template <class kParams>
__global__ void __launch_bounds__(kParams::kThreads)
    chunk_local_cumsum_kernel_seqfirst(kParams params) {
    using input_t = typename kParams::input_t;
    constexpr int kChunkSize = kParams::kChunkSize;

    int chunk_idx = blockIdx.y;
    int batch_head_idx = blockIdx.x;

    int head_idx = batch_head_idx % params.num_heads;
    int batch_idx = batch_head_idx / params.num_heads;
    bool is_varlen = params.cu_seqlens_ptr != nullptr;

    int bos, eos, local_chunk_idx;

    if (is_varlen) {
        int batch_index = __ldg(params.chunk_indices_ptr + chunk_idx * 2);
        local_chunk_idx = __ldg(params.chunk_indices_ptr + chunk_idx * 2 + 1);
        bos = __ldg(params.cu_seqlens_ptr + batch_index);
        eos = __ldg(params.cu_seqlens_ptr + batch_index + 1);
    } else {
        bos = batch_idx * params.seq_len;
        eos = bos + params.seq_len;
        local_chunk_idx = chunk_idx;
    }

    // chunk_start_global: absolute token position (for bounds checking against eos)
    // chunk_start_local: offset within this (batch, head) sequence (for seqfirst addressing)
    int chunk_start_global = bos + local_chunk_idx * kChunkSize;
    int chunk_start_local = local_chunk_idx * kChunkSize;
    int tid = threadIdx.x;

    // Load gate value with bounds check, converting to float for precision
    // seqfirst layout: (batch * num_heads, seq_len) for fixed-length,
    //                  (num_heads, total_tokens) for varlen

    float val = 0.0f;
    if (chunk_start_global + tid < eos) {
        if (is_varlen) {
            // varlen seqfirst: (num_heads, total_tokens), use global token position
            val = to_float(
                __ldg(params.g_ptr + head_idx * params.seq_len + chunk_start_global + tid));
        } else {
            // fixed seqfirst: (batch * num_heads, seq_len), use local offset within this batch's
            // sequence
            val = to_float(
                __ldg(params.g_ptr + (batch_head_idx)*params.seq_len + chunk_start_local + tid));
        }
    }

    // Inclusive prefix sum via CUB BlockScan
    using BlockScan = cub::BlockScan<float, kChunkSize>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    float cumsum;
    BlockScan(temp_storage).InclusiveSum(val, cumsum);

    // Store result as float — the state update kernel consumes gate as float
    if (chunk_start_global + tid < eos) {
        if (is_varlen) {
            params.output_ptr[head_idx * params.seq_len + chunk_start_global + tid] = cumsum;
        } else {
            params.output_ptr[(batch_head_idx)*params.seq_len + chunk_start_local + tid] = cumsum;
        }
    }
}

// Host launch wrapper for chunk-local cumsum
// g:              device pointer to input gate values (input_t, non-head-first layout)
// output_ptr:     device pointer to output cumsum values (float)
// d_cu_seqlens:   device pointer to cumulative sequence lengths (nullptr for fixed length)
// d_chunk_indices: device pointer to flattened (batch_idx, chunk_idx) pairs (nullptr for fixed
// length) num_chunks:     total number of chunks across all sequences (only used for varlen)
template <typename input_t>
inline void chunk_local_cumsum_kernel_launch_headfirst(input_t *g, float *output_ptr,
                                                       int batch_size, int seq_len, int num_heads,
                                                       int *d_cu_seqlens = nullptr,
                                                       int *d_chunk_indices = nullptr,
                                                       int num_chunks = 0,
                                                       cudaStream_t stream = nullptr) {
    constexpr int kChunkSize = 64;

    dim3 grid;
    if (d_cu_seqlens != nullptr) {
        HOST_ASSERT(num_chunks > 0, "num_chunks must be provided for varlen");
        grid = dim3(num_chunks, num_heads);
    } else {
        grid = dim3((seq_len + kChunkSize - 1) / kChunkSize, num_heads * batch_size);
    }
    dim3 block(kChunkSize);

    ChunkLocalCumsumParams<input_t, kChunkSize> params;
    params.g_ptr = g;
    params.output_ptr = output_ptr;
    params.cu_seqlens_ptr = d_cu_seqlens;
    params.chunk_indices_ptr = d_chunk_indices;
    params.seq_len = seq_len;
    params.num_heads = num_heads;

    chunk_local_cumsum_kernel_headfirst<<<grid, block, 0, stream>>>(params);
}

template <typename input_t>
inline void chunk_local_cumsum_kernel_launch_seqfirst(input_t *g, float *output_ptr, int batch_size,
                                                      int seq_len, int num_heads,
                                                      int *d_cu_seqlens = nullptr,
                                                      int *d_chunk_indices = nullptr,
                                                      int num_chunks = 0,
                                                      cudaStream_t stream = nullptr) {
    constexpr int kChunkSize = 64;

    dim3 grid;
    if (d_cu_seqlens != nullptr) {
        HOST_ASSERT(num_chunks > 0, "num_chunks must be provided for varlen");
        grid = dim3(num_heads,
                    num_chunks);  // since the gate tensor is shape (total_seq_len, num_heads), but
                                  // with seq-major layout
    } else {
        grid = dim3(batch_size * num_heads, ti_ceil_div(seq_len, kChunkSize));
    }
    dim3 block(kChunkSize);

    ChunkLocalCumsumParams<input_t, kChunkSize> params;
    params.g_ptr = g;
    params.output_ptr = output_ptr;
    params.cu_seqlens_ptr = d_cu_seqlens;
    params.chunk_indices_ptr = d_chunk_indices;
    params.seq_len = seq_len;
    params.num_heads = num_heads;

    chunk_local_cumsum_kernel_seqfirst<<<grid, block, 0, stream>>>(params);
}
}  // namespace gdn_helpers_impl
}  // namespace kernels
}  // namespace gdn_cuda
