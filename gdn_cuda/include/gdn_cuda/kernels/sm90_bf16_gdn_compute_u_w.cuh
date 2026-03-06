#pragma once
// #include "cute/atom/copy_traits_sm90_tma.hpp"
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/config.hpp>
#include <cute/swizzle.hpp>
#include <gdn_cuda/kernels/common/common.hpp>
#include <gdn_cuda/kernels/common/scheduler.cuh>
#include <gdn_cuda/kernels/common/sm90_utils.cuh>

#include "cute/arch/mma_sm90_desc.hpp"
#include "cutlass/arch/memory.h"

namespace gdn_cuda {
namespace kernels {
namespace sm90_gdn_compute_u_w_impl {

// helper functions

template <int kChunkSize = 64>
__device__ __forceinline__ void forward_sub_warp(__nv_bfloat16* sL, __nv_bfloat16* beta,
                                                 int& seq_end_in_chunk) {
    // Only first warp participates
    // Only first warp participates

    const int lane = threadIdx.x;
    const int col0 = lane * 2;
    const int col1 = lane * 2 + 1;

    // Capture the beta values for the columns this thread is responsible for
    float beta_col0 = __bfloat162float(beta[col0]);
    float beta_col1 = __bfloat162float(beta[col1]);

    // 128 registers per thread for X
    float X0[kChunkSize];  // Column col0
    float X1[kChunkSize];  // Column col1

    // Forward substitution - read from row-major KKT matrix
    // sL layout: sL[row * kChunkSize + col]
    for (int row = 0; row < kChunkSize; row++) {
        float row_beta = __bfloat162float(beta[row]);
        float sum0 = 0.0f;
        float sum1 = 0.0f;

        for (int k = 0; k < row; k++) {
            // Simple row-major indexing
            float L_row_k = __bfloat162float(sL[row * kChunkSize + k]);
            sum0 += L_row_k * X0[k];
            sum1 += L_row_k * X1[k];
        }
        float b_value_0 = (row == col0) ? 1.0f : 0.0f;
        float b_value_1 = (row == col1) ? 1.0f : 0.0f;

        // Recurrence for inverse: X = (I + diag(beta) * tril(K K^T))^{-1}
        // Forward substitution: X[row] = b[row] - beta[row] * sum_{k<row} L[row,k] * X[k]
        X0[row] = b_value_0 - row_beta * sum0;
        X1[row] = b_value_1 - row_beta * sum1;
    }

    __syncwarp();

    // Write back in row-major layout for ldmatrix to read
    for (int row = 0; row < kChunkSize; row++) {
        const bool row_valid = (seq_end_in_chunk < 0) || (row <= seq_end_in_chunk);
        __nv_bfloat16 val0 = (row_valid && row >= col0) ? (__float2bfloat16(X0[row] * beta_col0))
                                                        : __float2bfloat16(0.0f);
        __nv_bfloat16 val1 = (row_valid && row >= col1) ? (__float2bfloat16(X1[row] * beta_col1))
                                                        : __float2bfloat16(0.0f);
        // Simple row-major indexing
        sL[row * kChunkSize + col0] = val0;
        sL[row * kChunkSize + col1] = val1;
    }
}

/*
plan - we follow along the chunkwise algorithm from
https://sustcsonglin.github.io/blog/2024/deltanet-2/


To leverage tensor cores, chunk wise should be a multiple of 16

Accordingly, each of QKV, O will be of size C x d, where d is the hidden size (128) for Qwen3 GDN.
We should load through TMA here

We use the WY representation of Householder matrics, which is used to decompose the more complex
state recurrence equation of GDN compared to GDA.

GDA has a straightforward chunked algorithm, GDN is slightly more complex since we need to use the
UT transform to calculate the U and W matrices used in the blocked matmuls.

in order to perform sequential chunk-level passing, we need to calculate W and U first

for the recurrent rule, define a 2D grid over the batch_size x number of heads - each SM will try to
take one. We can also use heuristics to determine appropriate load sizes, since storing a large
hidden matrix will be harder when the hidden dim is larger

input tensors are of shape (batch_size, num_heads, seq_len, hidden_dim) for prefill
(batch_size, num_heads, hidden_dim) for decoding. When the prefill size is less than 16, should
still use recurrent

Query tensor has shape (B, NH, T, D) -> (B * NH, D), and we iterate over the D dimension
 */

// decoding phase

/*
prefill phase - to make this kernel jit compatible, we should try to always pad or use varlen
varlen is probably the best performance option

the chunk size C is important here - that is the number of rows we calculate per iteration
unfortunately we probably have to store the S in shared memory, of size d^2

beta is now of shape (chunk_size)

order of operands / dependence, look for possible overlaps here

load in K matrices to compute A = tril(diag(beta) * K * K^T) - staged GEMM

Compute T_t = (I-A_t)^{-1} using forward substitution

WGMMA on W_t = T_tK_t, U_t = T_tV_t -> diag(beta_t)


split here
  - why? everything above this line is independent of sequence causality - this is where we can
compute each of the UW representation matrices chunkwise parallel
  - after that, we are dependent on states

WGMMA (parallel)
  - QK^T
  - WS^T

Then run these two WGMMAs in parallel:
- QS^T
- second one

And then finally add

The nice thing is that all of these have the same swizzle mode
we only need the V matrix after forward substition, so we can set up double barriers for that

for large key_value head dims, exploring splitting the head dimension is worthwhile
*/

// gridDims (num_seq_chunks, batch_size * num_heads)

// here, the head dims must be compile time constants

template <uint32_t SHAPE_K, uint32_t SHAPE_V, uint32_t kBatchSize, uint32_t kChunkSize,
          uint32_t kNumVHeads, uint32_t kNumKHeads, uint32_t BLOCK_K, uint32_t kNumBlocks,
          uint32_t kNumTMAMulticast, uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kSwizzleKMode, uint32_t kSwizzleVMode, uint32_t kSwizzleAMode, uint32_t kSeqLen,
          bool kIsVarLen, uint32_t kUseGating>
__global__ void __launch_bounds__(kNumMathThreads + kNumTMAThreads, 1)
    sm90_bf16_compute_u_w(CUTE_GRID_CONSTANT const cute::TmaDescriptor k_tensor_map,
                          CUTE_GRID_CONSTANT const cute::TmaDescriptor v_tensor_map,
                          CUTE_GRID_CONSTANT const cute::TmaDescriptor u_tensor_map,
                          CUTE_GRID_CONSTANT const cute::TmaDescriptor w_tensor_map, int batch_size,
                          int shape_T, int chunk_indices_length, int* chunk_indices,
                          int* cu_seqlens, __nv_bfloat16* U_ptr, __nv_bfloat16* W_ptr,
                          __nv_bfloat16* beta_ptr, float* gate_ptr, uint32_t beta_stride,
                          uint32_t gate_stride) {
    CUTE_STATIC_ASSERT(SHAPE_K > 0 && SHAPE_V > 0, "Shape K and V must be positive");
    CUTE_STATIC_ASSERT(kNumBlocks > 0, "Blocks cannot be 0");
    CUTE_STATIC_ASSERT(SHAPE_K == SHAPE_V, "Different K dim and V dim not supported yet");
    CUTE_STATIC_ASSERT(BLOCK_K % 16 == 0, "Needs to be divisible by WGMMA::K atom size");
    // we want kIsVarLen is true or kSeqLen > 0, exclusive or
    // CUTE_STATIC_ASSERT(kIsVarLen ^ (kSeqLen > 0), "Exactly one of kIsVarLen or fixed kSeqLen must
    // be active");
    DEVICE_ASSERT(!kIsVarLen || cu_seqlens != nullptr);
    CUTE_STATIC_ASSERT(kChunkSize == 64, "Invalid chunk size");
    CUTE_STATIC_ASSERT(kNumMathThreads >= 128, "We need at least one WGMMA warpgroup");
    CUTE_STATIC_ASSERT(kNumTMAThreads >= 32, "We need at least one warp for TMA");
    CUTE_STATIC_ASSERT(
        SHAPE_K <= 256 && SHAPE_V <= 256,
        "N dimension of W and U is too large for WGMMA atom, use Split-N atom instead");
    CUTE_STATIC_ASSERT(kNumKHeads <= kNumVHeads,
                       "kNumKHeads must be less than or equal to kNumVHeads");

    // set up compiled dims if needed
    batch_size = kBatchSize != 0 ? kBatchSize : batch_size;
    shape_T = kSeqLen != 0 ? kSeqLen : shape_T;

    // DEVICE_ASSERT(shape_T % kChunkSize == 0); // we need this to ensure no sequence mixing

    // N = kChunkSize
    using KKTMMA = typename BF16MMASelector<kChunkSize, Major::K, Major::K>::type;

    // N = SHAPE_V
    using UMMA =
        typename BF16MMASelectorRS<SHAPE_V,
                                   Major::MN>::type;  // U = sA * sV, N = SHAPE_V, sV is normally
                                                      // (kChunkSize, SHAPE_V), but we transpose to
    // (SHAPE_V, kChunkSize), kChunkSize = K, SHAPE_V = N
    using WMMA = typename BF16MMASelectorRS<BLOCK_K, Major::MN>::type;  // W = sA * sK, N = SHAPE_K

    auto lane_predicate = cute::elect_one_sync();
    int lane_idx = threadIdx.x & 0x1f;
    int warpIdx = threadIdx.x / 32;
    if (lane_predicate) {
        cute::prefetch_tma_descriptor(&k_tensor_map);
        cute::prefetch_tma_descriptor(&v_tensor_map);
        cute::prefetch_tma_descriptor(&u_tensor_map);
        cute::prefetch_tma_descriptor(&w_tensor_map);
    }
    __syncwarp();

    using TMABarrier = cutlass::arch::ClusterTransactionBarrier;
    using MathBarrier = cutlass::arch::ClusterBarrier;
    // shared memory setup - separate buffers for K/V (input) and U/W (output)
    // This allows overlapping TMA stores of U/W with loading next iteration's K/V
    extern __shared__ __align__(1024) char shared[];
    constexpr uint32_t SMEM_K_SIZE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkSize * SHAPE_K, 1024);
    constexpr uint32_t SMEM_V_SIZE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkSize * SHAPE_V, 1024);
    constexpr uint32_t SMEM_U_SIZE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkSize * SHAPE_V, 1024);  // U output buffer
    constexpr uint32_t SMEM_W_SIZE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkSize * SHAPE_K, 1024);  // W output buffer
    constexpr uint32_t SMEM_A_SIZE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkSize * kChunkSize, 128);
    constexpr uint32_t SMEM_BETA_SIZE = constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkSize, 128);

    constexpr uint32_t SMEM_GATE_SIZE = kUseGating ? sizeof(float) * kChunkSize : 0;
    constexpr uint32_t SMEM_BARRIER_OFFSET = SMEM_K_SIZE + SMEM_V_SIZE + SMEM_U_SIZE + SMEM_W_SIZE +
                                             SMEM_A_SIZE + SMEM_BETA_SIZE + SMEM_GATE_SIZE;

    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(&shared);
    __nv_bfloat16* sV = reinterpret_cast<__nv_bfloat16*>(shared + SMEM_K_SIZE);
    __nv_bfloat16* sU = reinterpret_cast<__nv_bfloat16*>(shared + SMEM_K_SIZE + SMEM_V_SIZE);
    __nv_bfloat16* sW =
        reinterpret_cast<__nv_bfloat16*>(shared + SMEM_K_SIZE + SMEM_V_SIZE + SMEM_U_SIZE);
    __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(shared + SMEM_K_SIZE + SMEM_V_SIZE +
                                                         SMEM_U_SIZE + SMEM_W_SIZE);
    __nv_bfloat16* sBeta = reinterpret_cast<__nv_bfloat16*>(
        shared + SMEM_K_SIZE + SMEM_V_SIZE + SMEM_U_SIZE + SMEM_W_SIZE + SMEM_A_SIZE);
    float* s_gate;
    if constexpr (kUseGating) {
        s_gate = reinterpret_cast<float*>(shared + SMEM_K_SIZE + SMEM_V_SIZE + SMEM_U_SIZE +
                                          SMEM_W_SIZE + SMEM_A_SIZE + SMEM_BETA_SIZE);
    }
    TMABarrier* tma_barrier = reinterpret_cast<TMABarrier*>(shared + SMEM_BARRIER_OFFSET);
    MathBarrier* math_barrier =
        reinterpret_cast<MathBarrier*>(shared + SMEM_BARRIER_OFFSET + sizeof(TMABarrier));
    // in the future, this should probably be a TMA-load, but then sequence length must be the
    // unit-strided dimension

    if (threadIdx.x == 0) {
        tma_barrier->init(1);
        math_barrier->init(kNumMathThreads /
                           32);  // TMA warp signals math threads when loads complete
    }
    cutlass::arch::fence_barrier_init();
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Create chunk scheduler - handles both varlen and fixed padded modes
    // For varlen: uses chunk_indices to map global chunk index -> (batch_idx, chunk_idx)
    // For fixed: derives batch_idx and chunk_idx from linear index
    auto scheduler = [&]() {
        if constexpr (kIsVarLen) {
            return ChunkGDNScheduler<kChunkSize, kNumVHeads, kNumKHeads, kNumBlocks, SHAPE_V, true>(
                batch_size, SHAPE_V, chunk_indices_length, cu_seqlens, chunk_indices);
        } else {
            return ChunkGDNScheduler<kChunkSize, kNumVHeads, kNumKHeads, kNumBlocks, SHAPE_V,
                                     false>(batch_size, SHAPE_V, shape_T);
        }
    }();

    constexpr uint32_t kNumMathRegisters = 232;
    constexpr uint32_t kNumTMARegisters = 40;

    // Total bytes for all K and V tiles (no staging)
    constexpr uint32_t num_k_blocks = cute::ceil_div(SHAPE_K, BLOCK_K);

    int phase = 0;
    auto advance_pipeline = [&]() {
        phase ^= 1;
    };
    // Scheduler outputs
    int chunk_idx, batch_idx, head_idx, seq_end_in_chunk, v_block_idx;

    constexpr uint32_t TMA_U_BLOCK_N = kSwizzleVMode / sizeof(__nv_bfloat16);
    constexpr uint32_t TMA_W_BLOCK_N = kSwizzleKMode / sizeof(__nv_bfloat16);
    constexpr uint32_t num_u_tma_blocks = SHAPE_V / TMA_U_BLOCK_N;
    constexpr uint32_t num_w_tma_blocks = SHAPE_K / TMA_W_BLOCK_N;
    CUTE_STATIC_ASSERT(num_u_tma_blocks <= 32, "Too many TMA stores for U");
    CUTE_STATIC_ASSERT(num_w_tma_blocks <= 32, "Too many TMA stores for W");
    if (warpIdx >= kNumMathThreads / 32) {
        // TMA warp - issues loads and stores with pipelining to overlap stores with next
        // iteration's loads
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        while (scheduler.get_next_block(chunk_idx, batch_idx, head_idx, seq_end_in_chunk,
                                        v_block_idx)) {
            // if (threadIdx.x == kNumMathThreads) {
            //   printf("getting next iteration of while block \n");
            // }
            // Calculate global row offset for TMA loads

            int global_row_offset;
            if constexpr (kIsVarLen) {
                global_row_offset = scheduler.seq_start;
            } else {
                global_row_offset = batch_idx * kSeqLen + chunk_idx * kChunkSize;
            }
            int k_head_idx = head_idx / (kNumVHeads / kNumKHeads);

            // Issue TMA loads for current iteration into sK/sV
            math_barrier->wait(phase ^ 1);
            if (warpIdx == kNumMathThreads / 32 && lane_predicate) {
                // Load all K blocks
                auto& barrier = tma_barrier[0];
                for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
                    if constexpr (kIsVarLen) {
                        tma_copy<BLOCK_K, kChunkSize, kSwizzleKMode, __nv_bfloat16, 3>(
                            &k_tensor_map, &barrier, sK + k_block_idx * BLOCK_K * kChunkSize,
                            k_block_idx * BLOCK_K, k_head_idx, 1, global_row_offset);
                    } else {
                        tma_copy<BLOCK_K, kChunkSize, kSwizzleKMode, __nv_bfloat16, 4>(
                            &k_tensor_map, &barrier, sK + k_block_idx * BLOCK_K * kChunkSize,
                            k_block_idx * BLOCK_K, k_head_idx, 1, chunk_idx * kChunkSize,
                            batch_idx);
                    }
                }
                // Load all V blocks

                if constexpr (kIsVarLen) {
                    tma_copy<SHAPE_V, kChunkSize, kSwizzleVMode, __nv_bfloat16, 3>(
                        &v_tensor_map, &barrier, sV, 0, head_idx, 1, global_row_offset);
                } else {
                    tma_copy<SHAPE_V, kChunkSize, kSwizzleVMode, __nv_bfloat16, 4>(
                        &v_tensor_map, &barrier, sV, 0, head_idx, 1, chunk_idx * kChunkSize,
                        batch_idx);
                }

                barrier.arrive_and_expect_tx(SMEM_K_SIZE + SMEM_V_SIZE);
            }
            // ensures tma stores are ordered with a memory fence
            advance_pipeline();
        }

    } else {
        // Math threads - wait for TMA then do WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        auto smem_k_desc = create_wgmma_desc<Major::K, kChunkSize, BLOCK_K, kSwizzleKMode>(sK);
        const uint32_t smem_k_desc_base = __shfl_sync(uint32_t(-1), smem_k_desc.reg32_[0], 0);
        int math_barrier_phase = 0;
        while (scheduler.get_next_block(chunk_idx, batch_idx, head_idx, seq_end_in_chunk,
                                        v_block_idx)) {
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
            auto math_barrier_arrive = [&]() {
                if constexpr (kNumTMAMulticast == 1) {
                    (lane_idx == 0) ? math_barrier[0].arrive() : void();
                } else {
                    int target_cta = scheduler.get_effective_multicast() > 1
                                         ? lane_idx
                                         : cute::block_rank_in_cluster();
                    (lane_idx < kNumTMAMulticast) ? math_barrier[0].arrive(target_cta) : void();
                }
            };

            int global_row_offset;
            if constexpr (kIsVarLen) {
                global_row_offset = scheduler.seq_start;
            } else {
                global_row_offset = batch_idx * kSeqLen + chunk_idx * kChunkSize;
            }

            bool can_load_gate_and_beta;
            uint32_t beta_offset;
            uint32_t gate_offset;
            if constexpr (kIsVarLen) {
                can_load_gate_and_beta = (seq_end_in_chunk < 0 || threadIdx.x <= seq_end_in_chunk);
                beta_offset = head_idx * beta_stride + global_row_offset;
                gate_offset = head_idx * gate_stride + global_row_offset;
            } else {
                can_load_gate_and_beta = (chunk_idx * kChunkSize + threadIdx.x) < shape_T;
                beta_offset = batch_idx * (kNumVHeads * beta_stride) + head_idx * beta_stride +
                              chunk_idx * kChunkSize;
                gate_offset = batch_idx * (kNumVHeads * gate_stride) + head_idx * gate_stride +
                              chunk_idx * kChunkSize;
            }

            if (threadIdx.x < kChunkSize) {
                if (can_load_gate_and_beta) {
                    if constexpr (kUseGating) {
                        s_gate[threadIdx.x] = __ldg(gate_ptr + gate_offset + threadIdx.x);
                    }
                    sBeta[threadIdx.x] = __ldg(beta_ptr + beta_offset + threadIdx.x);
                } else {
                    if constexpr (kUseGating) {
                        s_gate[threadIdx.x] = 0.0f;
                    }
                    sBeta[threadIdx.x] = 0.0f;
                }
            }
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
            // seq_end_in_chunk indicates where sequence boundary is within chunk (-1 if none)
            // This can be used for masking in the math operations if needed

            // Wait for TMA loads to complete (both tma_barrier for data and math_barrier for TMA
            // thread signal)
            tma_barrier->wait(phase);
            // KKT multiplication - iterate over all K blocks
            float accum[KKTMMA::kNumAccum] = {0};

#pragma unroll
            for (int i = 0; i < KKTMMA::kNumAccum; i++) {
                cute::warpgroup_fence_operand(accum[i]);
            }
            cute::warpgroup_arrive();

            // #pragma unroll 8
            for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
                const uint32_t k_desc_lo =
                    smem_k_desc_base +
                    ((k_block_idx * BLOCK_K * kChunkSize * sizeof(__nv_bfloat16)) >> 4);
#pragma unroll
                for (int k = 0; k < BLOCK_K / KKTMMA::K; k++) {
                    smem_k_desc.reg32_[0] =
                        k_desc_lo + ((k * KKTMMA::K *
                                      get_gmma_desc_stride_k<Major::K, kChunkSize, BLOCK_K,
                                                             kSwizzleKMode, __nv_bfloat16>() *
                                      sizeof(__nv_bfloat16)) >>
                                     4);
                    KKTMMA::wgmma(smem_k_desc.desc_, smem_k_desc.desc_, accum, 1);
                }
            }

            cute::warpgroup_commit_batch();
#pragma unroll
            for (int i = 0; i < KKTMMA::kNumAccum; i++) {
                cute::warpgroup_fence_operand(accum[i]);
            }
            cute::warpgroup_wait<0>();

            // Store KKT accumulators to sA in row-major layout using scalar stores
            // WGMMA accumulator layout for m64n64k16:
            // - 4 warps, each handles 16 rows
            // - Within warp: row = (lane_idx % 8) + ((lane_idx / 16) * 8)
            // - Column determined by (lane_idx / 8) % 2 and accumulator index
            constexpr uint32_t WGMMA_M_PER_WARP = KKTMMA::M / 4;  // 16
            const int a_row_index = WGMMA_M_PER_WARP * warpIdx + lane_idx;
            if constexpr (kUseGating) {
#pragma unroll
                for (int i = 0; i < KKTMMA::kNumAccum; i++) {
                    auto [row, col] = get_accum_row_col(threadIdx.x, i);

                    if (row != col) {
                        accum[i] *=
                            expf(s_gate[row] -
                                 s_gate[col]);  // gets the cumulative gate product, by subtracting
                                                // the row gate from the column gate
                    }
                }
            }

#pragma unroll
            for (int i = 0; i < KKTMMA::kNumAccum / 4; i++) {
                // zero-swizzle
                uint8_t* smem_ptr =
                    reinterpret_cast<uint8_t*>(sA + a_row_index * kChunkSize + i * 8);
                // Each iteration stores 4 consecutive fp32 accumulators as 4 bf16 values
                // Scalar stores in row-major layout: sA[row * kChunkSize + col]
                custom_SM90_U32x2_STSM_N<__nv_bfloat162>::copy(
                    __float22bfloat162_rn({accum[i * 4 + 0], accum[i * 4 + 1]}),
                    __float22bfloat162_rn({accum[i * 4 + 2], accum[i * 4 + 3]}), smem_ptr);
            }

            cutlass::arch::NamedBarrier::sync(kNumMathThreads,
                                              1);  // ensure STSM writes are visible
            // forward substitution
            if (warpIdx == 0)
                forward_sub_warp(sA, sBeta, seq_end_in_chunk);
            cutlass::arch::NamedBarrier::sync(kNumMathThreads,
                                              1);  // ensure forward sub writes are visible
            // Now compute U = sA * sV and W = sA * sK sequentially
            // sA is kChunkSize x kChunkSize, sV is kChunkSize x SHAPE_V, sK is kChunkSize x SHAPE_K
            // WGMMA K atom is 16, so we need kChunkSize/16 iterations
            auto smem_v_desc = create_wgmma_desc<Major::MN, SHAPE_V, kChunkSize, kSwizzleVMode>(sV);

            const uint32_t smem_v_desc_base = __shfl_sync(uint32_t(-1), smem_v_desc.reg32_[0], 0);
            // ========== U = sA * sV ==========
            float u_accum[UMMA::kNumAccum] = {0};

#pragma unroll
            for (int i = 0; i < UMMA::kNumAccum; i++) {
                cute::warpgroup_fence_operand(u_accum[i]);
            }
            cute::warpgroup_arrive();
#pragma unroll
            for (int k = 0; k < kChunkSize / UMMA::K; k++) {
                const uint2 load1 =
                    custom_SM90_U32x2_STLM_N::load(sA + a_row_index * kChunkSize + k * UMMA::K);
                const uint2 load2 =
                    custom_SM90_U32x2_STLM_N::load(sA + a_row_index * kChunkSize + 8 + k * UMMA::K);
                // B descriptor: sV at row k*16, stride is SHAPE_V
                smem_v_desc.reg32_[0] =
                    smem_v_desc_base + ((k * UMMA::K *
                                         get_gmma_desc_stride_k<Major::MN, SHAPE_V, kChunkSize,
                                                                kSwizzleVMode, __nv_bfloat16>() *
                                         sizeof(__nv_bfloat16)) >>
                                        4u);
                // rs wgmma
                UMMA::wgmma(load1.x, load1.y, load2.x, load2.y, smem_v_desc.desc_, u_accum, 1);
            }
            cute::warpgroup_commit_batch();
#pragma unroll
            for (int i = 0; i < UMMA::kNumAccum; i++) {
                cute::warpgroup_fence_operand(u_accum[i]);
            }
            // ========== W = sA * sK ==========
            // because sK was partitioned into BLOCK_K columns for the load, we need to iterate
            // across each
            float w_accum[WMMA::kNumAccum * num_k_blocks] = {0};
#pragma unroll
            for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
                auto smem_k_desc_mn =
                    create_wgmma_desc<Major::MN, BLOCK_K, kChunkSize, kSwizzleKMode>(
                        sK + k_block_idx * kChunkSize * BLOCK_K);
                const uint32_t smem_k_desc_mn_base =
                    __shfl_sync(uint32_t(-1), smem_k_desc_mn.reg32_[0], 0);
                float* shifted_accum = w_accum + k_block_idx * WMMA::kNumAccum;

#pragma unroll
                for (int i = 0; i < WMMA::kNumAccum; i++) {
                    cute::warpgroup_fence_operand(shifted_accum[i]);
                }
                cute::warpgroup_arrive();

#pragma unroll
                for (int k = 0; k < kChunkSize / WMMA::K; k++) {
                    // A descriptor: sA at column k*16
                    const uint2 load1 =
                        custom_SM90_U32x2_STLM_N::load(sA + a_row_index * kChunkSize + k * WMMA::K);
                    const uint2 load2 = custom_SM90_U32x2_STLM_N::load(
                        sA + a_row_index * kChunkSize + 8 + k * WMMA::K);
                    // B descriptor
                    smem_k_desc_mn.reg32_[0] =
                        smem_k_desc_mn_base +
                        ((k * WMMA::K *
                          get_gmma_desc_stride_k<Major::MN, BLOCK_K, kChunkSize, kSwizzleKMode,
                                                 __nv_bfloat16>() *
                          sizeof(__nv_bfloat16)) >>
                         4);
                    WMMA::wgmma(load1.x, load1.y, load2.x, load2.y, smem_k_desc_mn.desc_,
                                shifted_accum, 1);
                }
                cute::warpgroup_commit_batch();

#pragma unroll
                for (int i = 0; i < WMMA::kNumAccum; i++) {
                    cute::warpgroup_fence_operand(shifted_accum[i]);
                }
            }
            cute::warpgroup_wait<0>();

            if constexpr (kUseGating) {
#pragma unroll
                for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
                    float* shifted_w_accum = w_accum + k_block_idx * WMMA::kNumAccum;
#pragma unroll
                    for (int i = 0; i < WMMA::kNumAccum; i++) {
                        auto [row, col] = get_accum_row_col(threadIdx.x, i);
                        shifted_w_accum[i] *= expf(s_gate[row]);
                    }
                }
            }

            // wait for all previous tma stores to finish before writing into U or W
            if (warpIdx == 0 && lane_predicate) {
                cute::tma_store_wait<0>();
            }
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
            // Store U result to sU (separate output buffer for overlap with next iteration's loads)
            constexpr uint32_t U_WGMMA_M_PER_WARP = UMMA::M / 4;

            store_accum_to_swizzled_smem<SHAPE_V, kChunkSize, kSwizzleVMode, U_WGMMA_M_PER_WARP>(
                u_accum, sU, warpIdx, lane_idx);
            // Store W result to sW (separate output buffer for overlap with next iteration's loads)
            constexpr uint32_t W_WGMMA_M_PER_WARP = WMMA::M / 4;
#pragma unroll
            for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
                float* shifted_w_accum = w_accum + k_block_idx * WMMA::kNumAccum;
                store_accum_to_swizzled_smem<BLOCK_K, kChunkSize, kSwizzleKMode,
                                             W_WGMMA_M_PER_WARP>(
                    shifted_w_accum, sW + k_block_idx * kChunkSize * BLOCK_K, warpIdx, lane_idx);
            }
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
            if constexpr (kIsVarLen) {
                if (seq_end_in_chunk > -1) {
#pragma unroll
                    for (int val_idx = 0; val_idx < UMMA::kNumAccum; val_idx++) {
                        const auto [row_idx, col_idx] = get_accum_row_col(threadIdx.x, val_idx);
                        bool pred = row_idx <= seq_end_in_chunk;
                        __nv_bfloat16* U_offset =
                            U_ptr +
                            ((global_row_offset + row_idx) * kNumVHeads + head_idx) * SHAPE_V +
                            col_idx;
                        cutlass::arch::global_store<__nv_bfloat16, sizeof(__nv_bfloat16)>(
                            __float2bfloat16_rn(u_accum[val_idx]), U_offset, pred);
                    }
                } else {
                    if (warpIdx == 0 && lane_idx < num_u_tma_blocks) {
                        auto smem_offset = sU + lane_idx * kChunkSize * TMA_U_BLOCK_N;
                        cute::SM90_TMA_STORE_3D::copy(&u_tensor_map, smem_offset,
                                                      lane_idx * TMA_U_BLOCK_N, head_idx,
                                                      global_row_offset);
                        cute::tma_store_arrive();
                    }
                }
            } else {
                if (warpIdx == 0 && lane_idx < num_u_tma_blocks) {
                    auto smem_offset = sU + lane_idx * kChunkSize * TMA_U_BLOCK_N;
                    cute::SM90_TMA_STORE_4D::copy(&u_tensor_map, smem_offset,
                                                  lane_idx * TMA_U_BLOCK_N, head_idx,
                                                  chunk_idx * kChunkSize, batch_idx);
                    cute::tma_store_arrive();
                }
            }
            if constexpr (kIsVarLen) {
                if (seq_end_in_chunk > -1) {
#pragma unroll
                    for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
                        auto n_chunk_offset = k_block_idx * BLOCK_K;
                        float* shifted_w_accum = w_accum + k_block_idx * WMMA::kNumAccum;
#pragma unroll
                        for (int val_idx = 0; val_idx < WMMA::kNumAccum; val_idx++) {
                            const auto [row_idx, col_idx] = get_accum_row_col(threadIdx.x, val_idx);
                            bool pred = row_idx <= seq_end_in_chunk;
                            __nv_bfloat16* W_offset =
                                W_ptr +
                                ((global_row_offset + row_idx) * kNumVHeads + head_idx) * SHAPE_V +
                                n_chunk_offset + col_idx;
                            cutlass::arch::global_store<__nv_bfloat16, sizeof(__nv_bfloat16)>(
                                __float2bfloat16_rn(shifted_w_accum[val_idx]), W_offset, pred);
                        }
                    }
                } else {
                    if (warpIdx == 0 && lane_idx < num_w_tma_blocks) {
                        auto smem_offset = sW + lane_idx * kChunkSize * TMA_W_BLOCK_N;
                        cute::SM90_TMA_STORE_3D::copy(&w_tensor_map, smem_offset,
                                                      lane_idx * TMA_W_BLOCK_N, head_idx,
                                                      global_row_offset);
                        cute::tma_store_arrive();
                    }
                }
            } else {
                if (warpIdx == 0 && lane_idx < num_w_tma_blocks) {
                    auto smem_offset = sW + lane_idx * kChunkSize * TMA_W_BLOCK_N;
                    cute::SM90_TMA_STORE_4D::copy(&w_tensor_map, smem_offset,
                                                  lane_idx * TMA_W_BLOCK_N, head_idx,
                                                  chunk_idx * kChunkSize, batch_idx);
                    cute::tma_store_arrive();
                }
            }

            __syncwarp();
            math_barrier_arrive();
            advance_pipeline();
        }
    }
}

// now we do the sequential stae passing

// format for the sequential state passing - i'm going to use cutlass/cute here because this one is
// not as specialized assume all row major here for strides
// t

}  // namespace sm90_gdn_compute_u_w_impl
}  // namespace kernels
}  // namespace gdn_cuda
