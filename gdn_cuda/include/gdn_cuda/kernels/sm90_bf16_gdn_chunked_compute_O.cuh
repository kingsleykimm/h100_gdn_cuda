#pragma once
// #include "cute/atom/copy_traits_sm90_tma.hpp"
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/config.hpp>
#include <cute/swizzle.hpp>
#include <gdn_cuda/kernels/common/common.hpp>
#include <gdn_cuda/kernels/common/scheduler.cuh>
#include <gdn_cuda/kernels/common/sm90_utils.cuh>

#include "cute/arch/mma_sm90_desc.hpp"
#include "cutlass/arch/memory.h"

namespace gdn_cuda {
namespace kernels {
namespace sm90_gdn_compute_o_impl {

template <uint32_t kBatchSize, uint32_t kChunkShape, uint32_t kChunkBlock, uint32_t kNumKHeads,
          uint32_t kNumVHeads, uint32_t SHAPE_K, uint32_t BLOCK_K, uint32_t SHAPE_V,
          uint32_t BLOCK_V, uint32_t kSwizzleKMode, uint32_t kSwizzleUMode, uint32_t kSwizzleQMode,
          uint32_t kSwizzleSMode, uint32_t kNumMathThreads, uint32_t kNumTMAThreads,
          uint32_t kNumBlocks, uint32_t kNumStages, uint32_t kNumTMAMulticast, bool kIsVarLen,
          uint32_t kSeqLen, bool kUseGating>
__global__ void __launch_bounds__(kNumMathThreads + kNumTMAThreads, 1)
    sm90_bf16_gdn_chunked_compute_O(CUTE_GRID_CONSTANT const cute::TmaDescriptor state_tensor_map,
                                    CUTE_GRID_CONSTANT const cute::TmaDescriptor k_tensor_map,
                                    CUTE_GRID_CONSTANT const cute::TmaDescriptor q_tensor_map,
                                    CUTE_GRID_CONSTANT const cute::TmaDescriptor u_tensor_map,
                                    CUTE_GRID_CONSTANT const cute::TmaDescriptor o_tensor_map,
                                    __nv_bfloat16* O_ptr, float* gate_ptr, uint32_t gate_stride,
                                    int batch_size, int shape_T, int* cu_seqlens,
                                    int chunk_indices_length, int* chunk_indices, int* cu_chunks,
                                    float scale_factor) {
    CUTE_STATIC_ASSERT(SHAPE_V <= 256 && SHAPE_K <= 256, "Head dim too large");
    CUTE_STATIC_ASSERT(SHAPE_K == SHAPE_V, "Different K dim and V dim not supported yet");
    CUTE_STATIC_ASSERT(BLOCK_K % 16 == 0, "Needs to be divisible by WGMMA::K atom size");
    // we want kIsVarLen is true or kSeqLen > 0, exclusive or
    // CUTE_STATIC_ASSERT(kIsVarLen ^ (kSeqLen > 0), "Exactly one of kIsVarLen or fixed kSeqLen must
    // be active");
    DEVICE_ASSERT(!kIsVarLen || (cu_seqlens != nullptr && chunk_indices != nullptr));
    CUTE_STATIC_ASSERT(kChunkShape == 64, "Invalid chunk shape");
    CUTE_STATIC_ASSERT(kChunkBlock >= 16 && kChunkBlock % 16 == 0,
                       "K chunk block must be divisible by 16");
    CUTE_STATIC_ASSERT(kNumTMAThreads >= 128, "We need at least one warp for TMA");
    CUTE_STATIC_ASSERT(
        SHAPE_K <= 256 && SHAPE_V <= 256,
        "N dimension of W and U is too large for WGMMA atom, use Split-N atom instead");
    CUTE_STATIC_ASSERT(kNumKHeads <= kNumVHeads,
                       "kNumKHeads must be less than or equal to kNumVHeads");
    // a big issue is that for RS atoms, the A operand must be K-major. Maybe it's easier than to
    // load in U with different strides

    batch_size = kBatchSize != 0 ? kBatchSize : batch_size;
    shape_T = kSeqLen != 0 ? kSeqLen : shape_T;

    using QK_MMA = typename BF16MMASelector<kChunkBlock, Major::K, Major::K>::type;

    // run these two in parallel
    using QS_MMA = typename BF16MMASelector<BLOCK_V, Major::K, Major::K>::type;
    using PU_MMA = typename BF16MMASelectorRS<BLOCK_V, Major::MN>::type;

    auto lane_predicate = cute::elect_one_sync();
    int lane_idx = threadIdx.x & 0x1f;
    int warpIdx = threadIdx.x / 32;
    if (lane_predicate) {
        cute::prefetch_tma_descriptor(&state_tensor_map);
        cute::prefetch_tma_descriptor(&k_tensor_map);
        cute::prefetch_tma_descriptor(&u_tensor_map);
        cute::prefetch_tma_descriptor(&q_tensor_map);
        cute::prefetch_tma_descriptor(&o_tensor_map);
    }
    __syncwarp();

    using TMABarrier = cutlass::arch::ClusterTransactionBarrier;
    using MathBarrier = cutlass::arch::ClusterBarrier;
    extern __shared__ __align__(1024) char shared[];
    constexpr uint32_t SMEM_STATE_SIZE_PER_STAGE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * BLOCK_V * BLOCK_K, 1024);
    constexpr uint32_t SMEM_K_SIZE_PER_STAGE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkBlock * BLOCK_K, 1024);
    constexpr uint32_t SMEM_U_SIZE_PER_STAGE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkBlock * BLOCK_V, 1024);  // U output buffer
    constexpr uint32_t SMEM_Q_SIZE_PER_STAGE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkShape * BLOCK_K,
                           1024);  // W output buffer
    constexpr uint32_t SMEM_O_SIZE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * BLOCK_V * kChunkShape, 1024);
    constexpr uint32_t SMEM_GATE_SIZE = kUseGating ? kChunkShape * sizeof(float) : 0;
    constexpr uint32_t SMEM_BARRIER_OFFSET = (SMEM_STATE_SIZE_PER_STAGE + SMEM_U_SIZE_PER_STAGE +
                                              SMEM_Q_SIZE_PER_STAGE + SMEM_K_SIZE_PER_STAGE) *
                                                 kNumStages +
                                             SMEM_O_SIZE + SMEM_GATE_SIZE;

    auto sState = PatternVisitor([&](const uint32_t& stage_idx) {
        return reinterpret_cast<__nv_bfloat16*>(shared + stage_idx * SMEM_STATE_SIZE_PER_STAGE);
    });

    auto sK = PatternVisitor([&](const uint32_t& stage_idx) {
        return reinterpret_cast<__nv_bfloat16*>(shared + kNumStages * SMEM_STATE_SIZE_PER_STAGE +
                                                stage_idx * SMEM_K_SIZE_PER_STAGE);
    });

    auto sU = PatternVisitor([&](const uint32_t& stage_idx) {
        return reinterpret_cast<__nv_bfloat16*>(
            shared + kNumStages * (SMEM_STATE_SIZE_PER_STAGE + SMEM_K_SIZE_PER_STAGE) +
            stage_idx * SMEM_U_SIZE_PER_STAGE);
    });

    auto sQ = PatternVisitor([&](const uint32_t& stage_idx) {
        return reinterpret_cast<__nv_bfloat16*>(
            shared +
            kNumStages *
                (SMEM_STATE_SIZE_PER_STAGE + SMEM_K_SIZE_PER_STAGE + SMEM_U_SIZE_PER_STAGE) +
            stage_idx * SMEM_Q_SIZE_PER_STAGE);
    });

    auto sO = reinterpret_cast<__nv_bfloat16*>(shared +
                                               (SMEM_STATE_SIZE_PER_STAGE + SMEM_K_SIZE_PER_STAGE +
                                                SMEM_U_SIZE_PER_STAGE + SMEM_Q_SIZE_PER_STAGE) *
                                                   kNumStages);

    float* s_gate;
    if constexpr (kUseGating) {
        s_gate = reinterpret_cast<float*>(shared +
                                          kNumStages *
                                              (SMEM_STATE_SIZE_PER_STAGE + SMEM_K_SIZE_PER_STAGE +
                                               SMEM_U_SIZE_PER_STAGE + SMEM_Q_SIZE_PER_STAGE) +
                                          SMEM_O_SIZE);
    }

    auto tma_barriers = PatternVisitor([&](const uint32_t& stage_idx) {
        return reinterpret_cast<TMABarrier*>(shared + SMEM_BARRIER_OFFSET +
                                             stage_idx * sizeof(TMABarrier));
    });

    auto math_barriers = PatternVisitor([&](const uint32_t& stage_idx) {
        return reinterpret_cast<MathBarrier*>(shared + SMEM_BARRIER_OFFSET +
                                              kNumStages * sizeof(TMABarrier) +
                                              stage_idx * sizeof(MathBarrier));
    });
    constexpr uint32_t num_k1_blocks = SHAPE_K / BLOCK_K;
    constexpr uint32_t num_k2_blocks = kChunkShape / kChunkBlock;
    constexpr uint32_t WGMMA_M_PER_WARP = QS_MMA::M / 4;
    constexpr uint32_t NUM_TMA_O_BLOCKS = BLOCK_V / (kSwizzleUMode / sizeof(__nv_bfloat16));
    constexpr uint32_t O_BLOCK_SIZE = kSwizzleUMode / sizeof(__nv_bfloat16);

    if (warpIdx == 0 && lane_predicate) {
#pragma unroll
        for (int stage_idx = 0; stage_idx < kNumStages; ++stage_idx) {
            tma_barriers[stage_idx]->init(1);
            math_barriers[stage_idx]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }
        cutlass::arch::fence_barrier_init();
    }
    kNumTMAMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    int stage_idx = 0, phase = 0;
    auto advance_pipeline = [&]() {
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
    // we can now do the double looped similar to FA. Q_i will have shapes (M, bK1), K_i will have
    // shape (bK1, bK2), and U_i will have shape (bK2, bN)) So each iteration, QK^T produces a (M,
    // bK2) matrix, which we mask out and then multiply to U. so there needs to be two loops, one
    // across the kChunkShape / kChunkBlock dimension, for the PU mma, and then one for the QK^T
    // mma, across SHAPE_K / BLOCK_K
    int chunk_idx, batch_idx, v_head_idx, seq_end_in_chunk, v_block_idx;

    // here v_block_idx is equivalent to n_block_idx
    auto scheduler = [&]() {
        if constexpr (kIsVarLen) {
            return ChunkGDNScheduler<kChunkShape, kNumVHeads, kNumKHeads, kNumBlocks, BLOCK_V,
                                     true>(batch_size, SHAPE_V, chunk_indices_length, cu_seqlens,
                                           chunk_indices);
        } else {
            return ChunkGDNScheduler<kChunkShape, kNumVHeads, kNumKHeads, kNumBlocks, BLOCK_V,
                                     false>(batch_size, SHAPE_V, shape_T);
        }
    }();

    if (threadIdx.x >= kNumMathThreads) {
        if (warpIdx == kNumMathThreads / 32 && lane_predicate) {
            while (scheduler.get_next_block(chunk_idx, batch_idx, v_head_idx, seq_end_in_chunk,
                                            v_block_idx)) {
                // the mma stages will be for the inner KU multiplication, so we actually need two
                // barriers
                int global_row_offset, global_state_offset;
                int seq_len, seq_start, seq_end;
                if constexpr (kIsVarLen) {
                    global_state_offset = __ldg(cu_chunks + batch_idx) + chunk_idx;
                    global_row_offset = scheduler.seq_start;
                } else {
                    global_state_offset =
                        batch_idx * constexpr_ti_ceil_div(kSeqLen, kChunkShape) + chunk_idx;
                    global_row_offset = batch_idx * kSeqLen + chunk_idx * kChunkShape;
                }
                int k_head_idx = v_head_idx / (kNumVHeads / kNumKHeads);

                for (int k2_block_idx = 0; k2_block_idx < num_k2_blocks; ++k2_block_idx) {
                    for (int k1_block_idx = 0; k1_block_idx < num_k1_blocks; ++k1_block_idx) {
                        math_barriers[stage_idx]->wait(phase ^ 1);
                        auto barrier = tma_barriers[stage_idx];
                        // we group the Q, S matrix loads together since they're only done on the
                        // first iteration however, we need to use a different stage idx for these
                        // shared memory loads, to avoid
                        if (k2_block_idx == 0) {
                            tma_copy<BLOCK_K, BLOCK_V, kSwizzleSMode, __nv_bfloat16, 4>(
                                &state_tensor_map, barrier, sState[stage_idx],
                                k1_block_idx * BLOCK_K, v_block_idx * BLOCK_V, 1, v_head_idx,
                                global_state_offset);
                        }
                        if constexpr (kIsVarLen) {
                            tma_copy<BLOCK_K, kChunkShape, kSwizzleQMode, __nv_bfloat16, 3>(
                                &q_tensor_map, barrier, sQ[stage_idx], k1_block_idx * BLOCK_K,
                                k_head_idx, 1, global_row_offset);
                        } else {
                            tma_copy<BLOCK_K, kChunkShape, kSwizzleQMode, __nv_bfloat16, 4>(
                                &q_tensor_map, barrier, sQ[stage_idx], k1_block_idx * BLOCK_K,
                                k_head_idx, 1, chunk_idx * kChunkShape, batch_idx);
                        }
                        if constexpr (kIsVarLen) {
                            // copy K
                            tma_copy<BLOCK_K, kChunkBlock, kSwizzleKMode, __nv_bfloat16, 3>(
                                &k_tensor_map, barrier, sK[stage_idx], k1_block_idx * BLOCK_K,
                                k_head_idx, 1, global_row_offset + k2_block_idx * kChunkBlock);
                            tma_copy<BLOCK_V, kChunkBlock, kSwizzleUMode, __nv_bfloat16, 3>(
                                &u_tensor_map, barrier, sU[stage_idx], v_block_idx * BLOCK_V,
                                v_head_idx, 1, global_row_offset + k2_block_idx * kChunkBlock);
                        } else {
                            tma_copy<BLOCK_K, kChunkBlock, kSwizzleKMode, __nv_bfloat16, 4>(
                                &k_tensor_map, barrier, sK[stage_idx], k1_block_idx * BLOCK_K,
                                k_head_idx, 1, chunk_idx * kChunkShape + k2_block_idx * kChunkBlock,
                                batch_idx);
                            tma_copy<BLOCK_V, kChunkBlock, kSwizzleUMode, __nv_bfloat16, 4>(
                                &u_tensor_map, barrier, sU[stage_idx], v_block_idx * BLOCK_V,
                                v_head_idx, 1, chunk_idx * kChunkShape + k2_block_idx * kChunkBlock,
                                batch_idx);
                        }
                        if (k2_block_idx == 0) {
                            tma_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_STATE_SIZE_PER_STAGE + SMEM_K_SIZE_PER_STAGE +
                                SMEM_Q_SIZE_PER_STAGE + SMEM_U_SIZE_PER_STAGE);
                        } else {
                            tma_barriers[stage_idx]->arrive_and_expect_tx(SMEM_U_SIZE_PER_STAGE +
                                                                          SMEM_K_SIZE_PER_STAGE +
                                                                          SMEM_Q_SIZE_PER_STAGE);
                        }

                        advance_pipeline();
                    }
                }
            }
        }
    }

    else {
        auto math_barrier_arrive = [&](int stage_idx) {
            if constexpr (kNumTMAMulticast == 1) {
                (lane_idx == 0) ? math_barriers[stage_idx]->arrive() : void();
            } else {
                int target_cta = scheduler.get_effective_multicast() > 1
                                     ? lane_idx
                                     : cute::block_rank_in_cluster();
                (lane_idx < kNumTMAMulticast) ? math_barriers[stage_idx]->arrive(target_cta)
                                              : void();
            }
        };
        while (scheduler.get_next_block(chunk_idx, batch_idx, v_head_idx, seq_end_in_chunk,
                                        v_block_idx)) {
            int global_row_offset, global_state_offset;
            int seq_len, seq_start, seq_end;
            if constexpr (kIsVarLen) {
                global_state_offset = __ldg(cu_chunks + batch_idx) + chunk_idx;
                global_row_offset = scheduler.seq_start;
            } else {
                global_state_offset =
                    batch_idx * constexpr_ti_ceil_div(kSeqLen, kChunkShape) + chunk_idx;
                global_row_offset = batch_idx * kSeqLen + chunk_idx * kChunkShape;
            }
            int k_head_idx = v_head_idx / (kNumVHeads / kNumKHeads);

            float QS_accum[QS_MMA::kNumAccum] = {0};

            __nv_bfloat16 QKT_accum_bf16[QK_MMA::kNumAccum];
            float PU_accum[PU_MMA::kNumAccum] = {0};
            float QKT_accum[QK_MMA::kNumAccum] = {0};
            if constexpr (kUseGating) {
                bool can_load_gate;
                uint32_t gate_offset;
                if constexpr (kIsVarLen) {
                    can_load_gate = (seq_end_in_chunk < 0 || threadIdx.x <= seq_end_in_chunk);
                    gate_offset = v_head_idx * gate_stride + global_row_offset;
                } else {
                    can_load_gate = (chunk_idx * kChunkShape + threadIdx.x) < shape_T;
                    gate_offset = batch_idx * (kNumVHeads * gate_stride) +
                                  v_head_idx * gate_stride + chunk_idx * kChunkShape;
                }

                if (threadIdx.x < kChunkShape) {
                    if (can_load_gate) {
                        s_gate[threadIdx.x] = __ldg(gate_ptr + gate_offset + threadIdx.x);
                    } else {
                        s_gate[threadIdx.x] = 0.0f;
                    }
                }
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
            }

#pragma unroll 4
            for (int k2_block_idx = 0; k2_block_idx < num_k2_blocks; ++k2_block_idx) {
                // #pragma unroll
                //         for (int val_idx = 0; val_idx < QK_MMA::kNumAccum; val_idx++) {
                //           QKT_accum[val_idx] = 0;
                //         }
#pragma unroll 4
                for (int k1_block_idx = 0; k1_block_idx < num_k1_blocks;
                     ++k1_block_idx) {  // QK and QS reduction dimensions
                    tma_barriers[stage_idx]->wait(phase);

                    auto smem_q_desc =
                        create_wgmma_desc<Major::K, kChunkShape, BLOCK_K, kSwizzleQMode>(
                            sQ[stage_idx]);
                    auto smem_k_desc =
                        create_wgmma_desc<Major::K, kChunkBlock, BLOCK_K, kSwizzleKMode>(
                            sK[stage_idx]);
                    auto smem_state_desc =
                        create_wgmma_desc<Major::K, BLOCK_V, BLOCK_K, kSwizzleSMode>(
                            sState[stage_idx]);

                    const auto smem_q_desc_lo = __shfl_sync(uint32_t(-1), smem_q_desc.reg32_[0], 0);
                    const auto smem_k_desc_lo = __shfl_sync(uint32_t(-1), smem_k_desc.reg32_[0], 0);
                    const auto smem_state_desc_lo =
                        __shfl_sync(uint32_t(-1), smem_state_desc.reg32_[0], 0);

#pragma unroll
                    for (int i = 0; i < QK_MMA::kNumAccum; i++) {
                        cute::warpgroup_fence_operand(QKT_accum[i]);
                    }
                    cute::warpgroup_arrive();

#pragma unroll
                    for (int k = 0; k < BLOCK_K / QK_MMA::K; k++) {
                        smem_q_desc.reg32_[0] =
                            smem_q_desc_lo +
                            ((k * QK_MMA::K *
                              get_gmma_desc_stride_k<Major::K, kChunkShape, BLOCK_K, kSwizzleQMode,
                                                     __nv_bfloat16>() *
                              sizeof(__nv_bfloat16)) >>
                             4);
                        smem_k_desc.reg32_[0] =
                            smem_k_desc_lo +
                            ((k * QK_MMA::K *
                              get_gmma_desc_stride_k<Major::K, kChunkBlock, BLOCK_K, kSwizzleKMode,
                                                     __nv_bfloat16>() *
                              sizeof(__nv_bfloat16)) >>
                             4);

                        QK_MMA::wgmma(smem_q_desc, smem_k_desc, QKT_accum, k);
                    }
                    cute::warpgroup_commit_batch();

#pragma unroll
                    for (int i = 0; i < QK_MMA::kNumAccum; i++) {
                        cute::warpgroup_fence_operand(QKT_accum[i]);
                    }

                    if (k2_block_idx == 0) {
#pragma unroll
                        for (int i = 0; i < QS_MMA::kNumAccum; i++) {
                            cute::warpgroup_fence_operand(QS_accum[i]);
                        }
                        cute::warpgroup_arrive();
#pragma unroll
                        for (int k = 0; k < BLOCK_K / QS_MMA::K; k++) {
                            smem_q_desc.reg32_[0] =
                                smem_q_desc_lo +
                                ((k * QS_MMA::K *
                                  get_gmma_desc_stride_k<Major::K, kChunkShape, BLOCK_K,
                                                         kSwizzleQMode, __nv_bfloat16>() *
                                  sizeof(__nv_bfloat16)) >>
                                 4);
                            smem_state_desc.reg32_[0] =
                                smem_state_desc_lo +
                                ((k * QS_MMA::K *
                                  get_gmma_desc_stride_k<Major::K, BLOCK_V, BLOCK_K, kSwizzleSMode,
                                                         __nv_bfloat16>() *
                                  sizeof(__nv_bfloat16)) >>
                                 4);
                            QS_MMA::wgmma(smem_q_desc, smem_state_desc, QS_accum, 1);
                        }
                        cute::warpgroup_commit_batch();
#pragma unroll
                        for (int i = 0; i < QS_MMA::kNumAccum; i++) {
                            cute::warpgroup_fence_operand(QS_accum[i]);
                        }
                    }

                    // TODO: this is actually only done per k1 == 0

                    // on the first k2_block_idx, we also issue the QS mma, so we
                    // if (k2_block_idx == 0) {
                    //   cute::warpgroup_wait<BLOCK_K / QS_MMA::K>();
                    // } else {
                    //   cute::warpgroup_wait<0>();
                    // }

                    cute::warpgroup_wait<0>();
                    // Fence QS_accum after wait to ensure compiler sees the updated registers

// now we know that the QK^T accum is always done, so we perform the diagonal masking here
// partial tile of shape (kChunkShape, kChunkBlock)
#pragma unroll
                    for (int i = 0; i < QK_MMA::kNumAccum; i++) {
                        auto [row_idx, col_idx] = get_accum_row_col(threadIdx.x, i);
                        int global_col = col_idx + k2_block_idx * kChunkBlock;
                        bool valid = row_idx >= global_col;
                        if (seq_end_in_chunk >= 0) {
                            valid &=
                                (row_idx <= seq_end_in_chunk) && (global_col <= seq_end_in_chunk);
                        }
                        float gate_val = 1.0f;
                        if constexpr (kUseGating) {
                            gate_val = __expf(s_gate[row_idx] - s_gate[global_col]);
                        }
                        QKT_accum_bf16[i] =
                            valid ? __float2bfloat16_rn(QKT_accum[i] * scale_factor * gate_val)
                                  : __nv_bfloat16(0.0f);
                    }

                    auto smem_u_desc =
                        create_wgmma_desc<Major::MN, BLOCK_V, kChunkBlock, kSwizzleUMode>(
                            sU[stage_idx]);
                    const auto smem_u_desc_lo = __shfl_sync(uint32_t(-1), smem_u_desc.reg32_[0], 0);
#pragma unroll
                    for (int i = 0; i < PU_MMA::kNumAccum; i++) {
                        cute::warpgroup_fence_operand(PU_accum[i]);
                    }
                    cute::warpgroup_arrive();
#pragma unroll
                    for (int k = 0; k < kChunkBlock / PU_MMA::K; k++) {
                        __nv_bfloat16* shifted_QKT_accum = QKT_accum_bf16 + k * 8;
                        uint32_t* packed_accum = reinterpret_cast<uint32_t*>(shifted_QKT_accum);
                        smem_u_desc.reg32_[0] =
                            smem_u_desc_lo +
                            ((k * PU_MMA::K *
                              get_gmma_desc_stride_k<Major::MN, BLOCK_V, kChunkBlock, kSwizzleUMode,
                                                     __nv_bfloat16>() *
                              sizeof(__nv_bfloat16)) >>
                             4);
                        PU_MMA::wgmma(packed_accum[0], packed_accum[1], packed_accum[2],
                                      packed_accum[3], smem_u_desc, PU_accum, 1);
                    }
                    cute::warpgroup_commit_batch();

#pragma unroll
                    for (int i = 0; i < PU_MMA::kNumAccum; i++) {
                        cute::warpgroup_fence_operand(PU_accum[i]);
                    }

                    cute::warpgroup_wait<0>();  // wait for the PU and possibly QS mma to finish
                    math_barrier_arrive(stage_idx);
                    advance_pipeline();
                }
            }

            //   printf("PU_accum: %f \n", PU_accum[0]);
            //   printf("PU_accum 1 : %f \n", PU_accum[1]);
            // }

            float gate_val = 1.0f;
#pragma unroll
            for (int i = 0; i < PU_MMA::kNumAccum; i++) {
                if constexpr (kUseGating) {
                    auto [row, col] = get_accum_row_col(threadIdx.x, i);
                    gate_val = __expf(s_gate[row]);
                }
                QS_accum[i] = gate_val * QS_accum[i] * scale_factor + PU_accum[i];
            }

            if (seq_end_in_chunk < 0 ||
                !kIsVarLen) {  // only for full chunks do we need to do this storing
                if (warpIdx == 0) {
                    cute::tma_store_wait<0>();
                }
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
                store_accum_to_swizzled_smem<BLOCK_V, kChunkShape, kSwizzleUMode, WGMMA_M_PER_WARP>(
                    QS_accum, sO, warpIdx, lane_idx);
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
            }

            if constexpr (kIsVarLen) {
                if (seq_end_in_chunk > -1) {
#pragma unroll
                    for (int val_idx = 0; val_idx < QS_MMA::kNumAccum; val_idx++) {
                        const auto [row_idx, col_idx] = get_accum_row_col(threadIdx.x, val_idx);
                        bool pred = row_idx <= seq_end_in_chunk;
                        __nv_bfloat16* O_offset =
                            O_ptr +
                            ((global_row_offset + row_idx) * kNumVHeads + v_head_idx) * SHAPE_V +
                            v_block_idx * BLOCK_V + col_idx;
                        cutlass::arch::global_store<__nv_bfloat16, sizeof(__nv_bfloat16)>(
                            __float2bfloat16_rn(QS_accum[val_idx]), O_offset, pred);
                    }
                } else {
                    if (warpIdx == 0 && lane_idx < NUM_TMA_O_BLOCKS) {
                        auto smem_offset = sO + lane_idx * kChunkShape * O_BLOCK_SIZE;
                        cute::SM90_TMA_STORE_3D::copy(
                            &o_tensor_map, smem_offset,
                            v_block_idx * BLOCK_V + lane_idx * O_BLOCK_SIZE, v_head_idx,
                            global_row_offset);
                        cute::tma_store_arrive();
                    }
                }
            } else {
                if (warpIdx == 0 && lane_idx < NUM_TMA_O_BLOCKS) {
                    auto smem_offset = sO + lane_idx * kChunkShape * O_BLOCK_SIZE;
                    cute::SM90_TMA_STORE_4D::copy(&o_tensor_map, smem_offset,
                                                  v_block_idx * BLOCK_V + lane_idx * O_BLOCK_SIZE,
                                                  v_head_idx, chunk_idx * kChunkShape, batch_idx);
                    cute::tma_store_arrive();
                }
            }
        }
    }
}

}  // namespace sm90_gdn_compute_o_impl
}  // namespace kernels
}  // namespace gdn_cuda
