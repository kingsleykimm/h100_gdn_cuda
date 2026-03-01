#pragma once
// #include "cute/atom/copy_traits_sm90_tma.hpp"
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory.h>
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

template <uint32_t kBatchSize, uint32_t kChunkSize, uint32_t kNumKHeads, uint32_t kNumVHeads,
          uint32_t SHAPE_K, uint32_t BLOCK_K, uint32_t SHAPE_V, uint32_t BLOCK_V,
          uint32_t kSwizzleKMode, uint32_t kSwizzleUMode, uint32_t kSwizzleWMode,
          uint32_t kSwizzleSMode, uint32_t kNumMathThreads, uint32_t kNumTMAThreads,
          uint32_t kNumBlocks, uint32_t kNumTMAMulticast, bool kIsInitialState, bool kIsVarLen,
          uint32_t kSeqLen, bool kOutputFinalState, bool kUseGate>
__global__ void __launch_bounds__(kNumMathThreads + kNumTMAThreads, 1)
    sm90_bf16_chunked_seq_state_update(
        CUTE_GRID_CONSTANT const cute::TmaDescriptor state_tensor_map,
        CUTE_GRID_CONSTANT const cute::TmaDescriptor k_tensor_map,
        CUTE_GRID_CONSTANT const cute::TmaDescriptor w_tensor_map,
        CUTE_GRID_CONSTANT const cute::TmaDescriptor u_tensor_map,
        CUTE_GRID_CONSTANT const cute::TmaDescriptor initial_state_tensor_map,
        CUTE_GRID_CONSTANT const cute::TmaDescriptor final_state_tensor_map, __nv_bfloat16* U_ptr,
        float* gate_ptr, uint32_t gate_stride, int batch_size, int shape_T, int* cu_seqlens,
        int* cu_chunks) {
    CUTE_STATIC_ASSERT(SHAPE_V <= 256 && SHAPE_K <= 256, "Head dim too large");
    CUTE_STATIC_ASSERT(SHAPE_K == SHAPE_V, "Different K dim and V dim not supported yet");
    CUTE_STATIC_ASSERT(BLOCK_K % 16 == 0, "Needs to be divisible by WGMMA::K atom size");
    // we want kIsVarLen is true or kSeqLen > 0, exclusive or
    // CUTE_STATIC_ASSERT(kIsVarLen ^ (kSeqLen > 0), "Exactly one of kIsVarLen or
    // fixed kSeqLen must be active");
    DEVICE_ASSERT(!kIsVarLen || cu_seqlens != nullptr);
    CUTE_STATIC_ASSERT(kChunkSize == 64, "Invalid chunk size");
    CUTE_STATIC_ASSERT(kNumTMAThreads >= 128, "We need at least one warp for TMA");
    CUTE_STATIC_ASSERT(SHAPE_K <= 256 && SHAPE_V <= 256,
                       "N dimension of W and U is too large for WGMMA atom, use "
                       "Split-N atom instead");
    CUTE_STATIC_ASSERT(kNumKHeads <= kNumVHeads,
                       "kNumKHeads must be less than or equal to kNumVHeads");
    // a big issue is that for RS atoms, the A operand must be K-major. Maybe it's
    // easier than to load in U with different strides

    batch_size = kBatchSize != 0 ? kBatchSize : batch_size;
    shape_T = kSeqLen != 0 ? kSeqLen : shape_T;
    // for state, it is of shape (num_chunks, num_v_heads, shape_v, shape_k) if
    // varlen is on, other wise batcH_size is the// fifth dimension

    // we're actually going to compute a tile of (shape_k, shape_v), which will be
    // transposed. however, since we know the accumulator layout, we can easily
    // find the untransposed indices, accumulate these with the register copies of
    // state that we'll store continuously, and also perform tma stores of both
    // the state matrix (per step), and the (U - WS^T) matrix in registers. This
    // is important because it will save a WGMMA for the parallel output
    // computation

    using WSMMA =
        typename BF16MMASelector<BLOCK_V, Major::K, Major::K, cute::GMMA::ScaleIn::Neg>::type;
    using STATE_MMA =
        typename BF16MMASelector<SHAPE_K, Major::MN,
                                 Major::MN>::type;  // second, outer matrix multiplication
    // first matmul is WS^T, second one is the (U - WS^T)^TK.
    // the full equation is S_{i+1} = S_i + (U^T-SW^T_i)K_i = S_i + (U^T / K *
    // K-(S1W1 + S2W2 + .. SKWK))K_i if we load in a row tile chunk of S and U,
    // then the SW^T is a SS matmul we partition the U^T into registers, align
    // with results and subtract U^T - SW^T in registers perform RS matmul for
    // outer matmul, then add to S_i, we get the new state which we store and move
    // to next chunk we can pipeline W/U/K tma loads with the S_{i+1} store and
    // the gmma

    // make a 2D accumulator loop - the first index tiles across the contract
    // dimension of first matmul, and the second one tiles across the columns of
    // the A operand in the second matmul. So if we have B * C * D, we're tililng
    // C into a 2d tile of (i, j) coordinates and summing
    auto lane_predicate = cute::elect_one_sync();
    int lane_idx = threadIdx.x & 0x1f;
    int warpIdx = threadIdx.x >> 5;
    if (lane_predicate) {
        cute::prefetch_tma_descriptor(&state_tensor_map);
        cute::prefetch_tma_descriptor(&k_tensor_map);
        cute::prefetch_tma_descriptor(&u_tensor_map);
        cute::prefetch_tma_descriptor(&w_tensor_map);
        cute::prefetch_tma_descriptor(&initial_state_tensor_map);
        cute::prefetch_tma_descriptor(&final_state_tensor_map);
    }
    __syncwarp();

    using TMABarrier = cutlass::arch::ClusterTransactionBarrier;
    using MathBarrier = cutlass::arch::ClusterBarrier;
    // shared memory setup - separate buffers for K/V (input) and U/W (output)
    // This allows overlapping TMA stores of U/W with loading next iteration's K/V

    // although the WGMMA is 'staged', I'm just going to set the number of stages
    // to num_k/v_blocks, because I want the shapes to be written linearly
    extern __shared__ __align__(1024) char shared[];
    constexpr uint32_t SMEM_STATE_SIZE_PER_STAGE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * BLOCK_V * SHAPE_K, 1024);
    constexpr uint32_t SMEM_K_SIZE_PER_STAGE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkSize * SHAPE_K, 1024);
    constexpr uint32_t SMEM_U_SIZE_PER_STAGE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkSize * BLOCK_V, 1024);  // U output buffer
    constexpr uint32_t SMEM_W_SIZE_PER_STAGE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * kChunkSize * SHAPE_K, 1024);  // W output buffer
    constexpr uint32_t SMEM_GATE_SIZE = kUseGate ? sizeof(float) * kChunkSize : 0;
    constexpr uint32_t SMEM_BARRIER_OFFSET =
        (SMEM_STATE_SIZE_PER_STAGE + SMEM_U_SIZE_PER_STAGE + SMEM_W_SIZE_PER_STAGE +
         SMEM_K_SIZE_PER_STAGE + SMEM_GATE_SIZE);

    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(&shared);
    __nv_bfloat16* sState = reinterpret_cast<__nv_bfloat16*>(shared + SMEM_K_SIZE_PER_STAGE);
    __nv_bfloat16* sU = reinterpret_cast<__nv_bfloat16*>(
        shared + (SMEM_K_SIZE_PER_STAGE + SMEM_STATE_SIZE_PER_STAGE));
    __nv_bfloat16* sW = reinterpret_cast<__nv_bfloat16*>(
        shared + (SMEM_STATE_SIZE_PER_STAGE + SMEM_U_SIZE_PER_STAGE + SMEM_K_SIZE_PER_STAGE));
    float* s_gate;
    if constexpr (kUseGate) {
        s_gate =
            reinterpret_cast<float*>(shared + SMEM_STATE_SIZE_PER_STAGE + SMEM_U_SIZE_PER_STAGE +
                                     SMEM_W_SIZE_PER_STAGE + SMEM_K_SIZE_PER_STAGE);
    }
    TMABarrier* tma_barrier_ws = reinterpret_cast<TMABarrier*>(shared + SMEM_BARRIER_OFFSET);
    TMABarrier* tma_barrier_k =
        reinterpret_cast<TMABarrier*>(shared + SMEM_BARRIER_OFFSET + sizeof(TMABarrier));
    MathBarrier* math_barrier =
        reinterpret_cast<MathBarrier*>(shared + SMEM_BARRIER_OFFSET + sizeof(TMABarrier) * 2);
    // Gate values shared buffer — kChunkSize floats (256 bytes) fits within
    // barrier region's 1024-byte alignment padding
    constexpr uint32_t TMA_U_BLOCK_N = kSwizzleUMode / sizeof(__nv_bfloat16);
    constexpr uint32_t TMA_S_BLOCK_N = kSwizzleSMode / sizeof(__nv_bfloat16);
    constexpr uint32_t NUM_U_TMA_BLOCKS = BLOCK_V / TMA_U_BLOCK_N;
    constexpr uint32_t NUM_S_TMA_BLOCKS = SHAPE_K / TMA_S_BLOCK_N;

    CUTE_STATIC_ASSERT(NUM_U_TMA_BLOCKS <= 32, "Too many TMA stores for U");
    CUTE_STATIC_ASSERT(NUM_S_TMA_BLOCKS <= 32, "Too many TMA stores for S");

    if (lane_predicate) {
        tma_barrier_ws[0].init(1);
        tma_barrier_k[0].init(1);
        math_barrier[0].init(kNumTMAMulticast * kNumMathThreads /
                             32);  // TMA warp signals math threads when loads complete
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    auto scheduler =
        RecurrentGDNScheduler<BLOCK_V, kNumVHeads, kNumKHeads, kNumBlocks, 1>(SHAPE_V, batch_size);

    constexpr uint32_t kNumMathRegisters = 232;
    constexpr uint32_t kNumTMARegisters = 48;
    int phase = 0;
    auto advance_pipeline = [&]() {
        phase ^= 1;
    };
    // Scheduler outputs
    int v_block_idx, v_head_idx, k_head_idx, batch_index;
    int global_row_offset;

    constexpr int num_v_blocks = constexpr_ti_ceil_div(SHAPE_V, BLOCK_V);

    // tma warpgroup
    if (threadIdx.x >= kNumMathThreads) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        while (scheduler.get_next_block(v_block_idx, v_head_idx, k_head_idx, batch_index)) {
            int seq_len, seq_start, seq_end;
            int global_state_offset;
            uint32_t num_chunks;

            if constexpr (kIsVarLen) {
                seq_start = __ldg(cu_seqlens + batch_index);
                seq_end = __ldg(cu_seqlens + batch_index + 1);
                seq_len = seq_end - seq_start;
                global_state_offset = __ldg(cu_chunks + batch_index);
            } else {
                seq_start = batch_index * shape_T;
                seq_end = (batch_index + 1) * shape_T;
                seq_len = shape_T;
                global_state_offset = batch_index * constexpr_ti_ceil_div(shape_T, kChunkSize);
            }

            num_chunks = ti_ceil_div(seq_len, kChunkSize);
            int global_row_offset;

            for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                global_row_offset = seq_start + chunk_idx * kChunkSize;
                math_barrier[0].wait(phase ^ 1);
                if (warpIdx == kNumMathThreads / 32 && lane_predicate) {
                    auto& barrier = tma_barrier_ws[0];
                    auto& barrier_k = tma_barrier_k[0];
                    // load the wS^T mma first, with reduction dimension SHAPE_V
                    if (chunk_idx == 0) {
                        if constexpr (kIsInitialState) {
                            tma_copy<SHAPE_K, BLOCK_V, kSwizzleSMode, __nv_bfloat16, 4>(
                                &initial_state_tensor_map, &barrier, sState, 0,
                                v_block_idx * BLOCK_V, 1, v_head_idx, batch_index);
                        }
                    }
                    if constexpr (kIsVarLen) {
                        tma_copy<SHAPE_K, kChunkSize, kSwizzleWMode, __nv_bfloat16, 3>(
                            &w_tensor_map, &barrier, sW, 0, v_head_idx, 1, global_row_offset);
                        tma_copy<BLOCK_V, kChunkSize, kSwizzleUMode, __nv_bfloat16, 3>(
                            &u_tensor_map, &barrier, sU, v_block_idx * BLOCK_V, v_head_idx, 1,
                            global_row_offset);
                    } else {
                        tma_copy<SHAPE_K, kChunkSize, kSwizzleWMode, __nv_bfloat16, 4>(
                            &w_tensor_map, &barrier, sW, 0, v_head_idx, 1, chunk_idx * kChunkSize,
                            batch_index);
                        tma_copy<BLOCK_V, kChunkSize, kSwizzleUMode, __nv_bfloat16, 4>(
                            &u_tensor_map, &barrier, sU, v_block_idx * BLOCK_V, v_head_idx, 1,
                            chunk_idx * kChunkSize, batch_index);
                    }

                    (chunk_idx == 0)
                        ? tma_barrier_ws[0].arrive_and_expect_tx(
                              kIsInitialState ? SMEM_STATE_SIZE_PER_STAGE + SMEM_U_SIZE_PER_STAGE +
                                                    SMEM_W_SIZE_PER_STAGE
                                              : SMEM_U_SIZE_PER_STAGE + SMEM_W_SIZE_PER_STAGE)
                        : tma_barrier_ws[0].arrive_and_expect_tx(SMEM_U_SIZE_PER_STAGE +
                                                                 SMEM_W_SIZE_PER_STAGE);
                    if constexpr (kIsVarLen) {
                        tma_copy<SHAPE_K, kChunkSize, kSwizzleKMode, __nv_bfloat16, 3>(
                            &k_tensor_map, &barrier_k, sK, 0, k_head_idx, 1, global_row_offset);
                    } else {
                        tma_copy<SHAPE_K, kChunkSize, kSwizzleKMode, __nv_bfloat16, 4>(
                            &k_tensor_map, &barrier_k, sK, 0, k_head_idx, 1, chunk_idx * kChunkSize,
                            batch_index);
                    }

                    tma_barrier_k[0].arrive_and_expect_tx(SMEM_K_SIZE_PER_STAGE);
                }

                advance_pipeline();
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        constexpr uint32_t WGMMA_M_PER_WARP = STATE_MMA::M / 4;
        while (scheduler.get_next_block(v_block_idx, v_head_idx, k_head_idx, batch_index)) {
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
            int seq_len, seq_start, seq_end, global_state_offset, global_row_offset;
            if constexpr (kIsVarLen) {
                seq_start = __ldg(cu_seqlens + batch_index);
                seq_end = __ldg(cu_seqlens + batch_index + 1);
                seq_len = seq_end - seq_start;
                global_state_offset = __ldg(cu_chunks + batch_index);
            } else {
                seq_start = batch_index * shape_T;
                seq_end = (batch_index + 1) * shape_T;
                seq_len = shape_T;
                global_state_offset = batch_index * constexpr_ti_ceil_div(shape_T, kChunkSize);
            }

            uint32_t num_chunks = ti_ceil_div(seq_len, kChunkSize);

            float state_accum[STATE_MMA::kNumAccum];
            float ws_accum[WSMMA::kNumAccum];
            float gate_last = 0.f;

            for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                global_row_offset = seq_start + chunk_idx * kChunkSize;
                int last_idx = (kIsVarLen) ? min(global_row_offset + kChunkSize, seq_end) - 1
                                           : min((chunk_idx + 1) * kChunkSize, shape_T) - 1;
                if constexpr (kUseGate) {
                    bool can_load_gate_and_beta;
                    uint32_t offset;
                    can_load_gate_and_beta = (threadIdx.x < seq_end - global_row_offset);
                    offset = kIsVarLen ? v_head_idx * gate_stride + global_row_offset
                                       : gate_stride * (batch_index * kNumVHeads + v_head_idx) +
                                             chunk_idx * kChunkSize;
                    if (threadIdx.x < kChunkSize) {
                        if (can_load_gate_and_beta) {
                            s_gate[threadIdx.x] = __ldg(gate_ptr + offset + threadIdx.x);
                        } else {
                            s_gate[threadIdx.x] = 0.0f;
                        }
                    }

                    gate_last = (kIsVarLen)
                                    ? __ldg(gate_ptr + v_head_idx * gate_stride + last_idx)
                                    : __ldg(gate_ptr + batch_index * (kNumVHeads * gate_stride) +
                                            v_head_idx * gate_stride + last_idx);

                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
                }

                tma_barrier_ws[0].wait(phase);

                // we only do the smem -> rmem copy at the very beginning - after this
                // the registers will hold the copies to accumulate to, and we copy from
                // rmem -> smem at the end of each loop in order to do a tma store
                if (chunk_idx == 0) {
                    if constexpr (!kIsInitialState) {
                        // this is the only time we load from shared memory into register
                        // memory
#pragma unroll
                        for (int i = 0; i < STATE_MMA::kNumAccum; i++) {
                            state_accum[i] = 0;
                        }
                        store_accum_to_swizzled_smem<SHAPE_K, BLOCK_V, kSwizzleSMode,
                                                     WGMMA_M_PER_WARP>(state_accum, sState, warpIdx,
                                                                       lane_idx);
                    } else {
                        load_swizzled_smem_to_accum<SHAPE_K, BLOCK_V, kSwizzleSMode,
                                                    WGMMA_M_PER_WARP>(sState, state_accum, warpIdx,
                                                                      lane_idx);
                    }
                }

                // initiate tma store here for the current chunk's STARTING state
                if (warpIdx == 0 && lane_idx < NUM_S_TMA_BLOCKS) {
                    auto smem_offset = sState + lane_idx * TMA_S_BLOCK_N * BLOCK_V;
                    cute::SM90_TMA_STORE_4D::copy(
                        &state_tensor_map, smem_offset, lane_idx * TMA_S_BLOCK_N,
                        v_block_idx * BLOCK_V, v_head_idx,
                        global_state_offset +
                            (chunk_idx));  // state tensor has shape (num_chunks', num_v_heads,
                                           // shape_v, shape_k), but num_chunks is actually
                                           // divided down by the chunk size
                    cute::tma_store_arrive();
                }

                load_swizzled_smem_to_accum<BLOCK_V, kChunkSize, kSwizzleUMode, WGMMA_M_PER_WARP>(
                    sU, ws_accum, warpIdx, lane_idx);
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);

                // if (blockIdx.x == 0 && threadIdx.x == 3) {
                //   printf("ws_accum far : %f \n", ws_accum[WSMMA::kNumAccum - 3]);
                // }
                auto smem_w_desc =
                    create_wgmma_desc<Major::K, kChunkSize, BLOCK_K, kSwizzleWMode>(sW);
                auto smem_state_desc =
                    create_wgmma_desc<Major::K, BLOCK_V, BLOCK_K, kSwizzleSMode>(sState);

                const auto smem_w_desc_base = __shfl_sync(uint32_t(-1), smem_w_desc.reg32_[0], 0);
                const auto smem_state_desc_base =
                    __shfl_sync(uint32_t(-1), smem_state_desc.reg32_[0], 0);
                // Ensure all warps in the warpgroup have completed loading before
                // proceeding

#pragma unroll 8
                for (int k_block_idx = 0; k_block_idx < SHAPE_K / BLOCK_K; k_block_idx++) {
                    const uint32_t w_desc_lo =
                        smem_w_desc_base +
                        ((k_block_idx * BLOCK_K * kChunkSize * sizeof(__nv_bfloat16)) >> 4);
                    const uint32_t s_desc_lo =
                        smem_state_desc_base +
                        ((k_block_idx * BLOCK_K * BLOCK_V * sizeof(__nv_bfloat16)) >> 4);

#pragma unroll
                    for (int i = 0; i < WSMMA::kNumAccum; i++) {
                        cute::warpgroup_fence_operand(ws_accum[i]);
                    }
                    cute::warpgroup_arrive();
#pragma unroll
                    for (int k = 0; k < BLOCK_K / WSMMA::K; k++) {
                        smem_w_desc.reg32_[0] =
                            w_desc_lo + (((k * WSMMA::K) *
                                          get_gmma_desc_stride_k<Major::K, kChunkSize, BLOCK_K,
                                                                 kSwizzleWMode, __nv_bfloat16>() *
                                          sizeof(__nv_bfloat16)) >>
                                         4u);
                        // state is transposed
                        smem_state_desc.reg32_[0] =
                            s_desc_lo + (((k * WSMMA::K) *
                                          get_gmma_desc_stride_k<Major::K, BLOCK_V, BLOCK_K,
                                                                 kSwizzleSMode, __nv_bfloat16>() *
                                          sizeof(__nv_bfloat16)) >>
                                         4u);
                        WSMMA::wgmma(smem_w_desc, smem_state_desc, ws_accum, 1);
                    }
                    cute::warpgroup_commit_batch();

#pragma unroll
                    for (int i = 0; i < WSMMA::kNumAccum; i++) {
                        cute::warpgroup_fence_operand(ws_accum[i]);
                    }
                }
                cute::warpgroup_wait<0>();

                // we want to wait for the previous tma stores of U, S to finish so that
                // we can overwrite them with new tma stores
                if (warpIdx == 0) {
                    cute::tma_store_wait<0>();
                }
                bool full_chunk = global_row_offset + kChunkSize - 1 < seq_end;
                store_accum_to_swizzled_smem<BLOCK_V, kChunkSize, kSwizzleUMode, WGMMA_M_PER_WARP>(
                    ws_accum, sU, warpIdx, lane_idx);
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
                // }

                if constexpr (kIsVarLen) {
                    if (!full_chunk) {
                        DEVICE_ASSERT(shape_T > 0);
#pragma unroll
                        for (int val_idx = 0; val_idx < WSMMA::kNumAccum; val_idx++) {
                            auto [row_idx, col_idx] = get_accum_row_col(threadIdx.x, val_idx);
                            bool pred = global_row_offset + row_idx < seq_end;
                            __nv_bfloat16* U_offset =
                                U_ptr + (global_row_offset + row_idx) * kNumVHeads * SHAPE_V +
                                v_head_idx * SHAPE_V + v_block_idx * BLOCK_V + col_idx;
                            // include this to zero out the U tensor on ragged tiles for
                            // downstream computation
                            if constexpr (kOutputFinalState) {
                                if (!pred) {
                                    ws_accum[val_idx] = 0.f;
                                }
                            }
                            cutlass::arch::global_store<__nv_bfloat16, sizeof(__nv_bfloat16)>(
                                __float2bfloat16_rn(ws_accum[val_idx]), U_offset, pred);
                        }
                    } else {
                        if (warpIdx == 0 && lane_idx < NUM_U_TMA_BLOCKS) {
                            auto smem_offset = sU + lane_idx * TMA_U_BLOCK_N * kChunkSize;
                            cute::SM90_TMA_STORE_3D::copy(
                                &u_tensor_map, smem_offset,
                                v_block_idx * BLOCK_V + lane_idx * TMA_U_BLOCK_N, v_head_idx,
                                global_row_offset);
                            cute::tma_store_arrive();
                        }
                    }
                } else {
                    if (warpIdx == 0 && lane_idx < NUM_U_TMA_BLOCKS) {
                        auto smem_offset = sU + lane_idx * TMA_U_BLOCK_N * kChunkSize;
                        cute::SM90_TMA_STORE_4D::copy(
                            &u_tensor_map, smem_offset,
                            v_block_idx * BLOCK_V + lane_idx * TMA_U_BLOCK_N, v_head_idx,
                            chunk_idx * kChunkSize, batch_index);
                        cute::tma_store_arrive();
                    }
                }

                tma_barrier_k[0].wait(phase);

                // now that K is loaded, on ragged tiles we need to zero out the K as
                // well, but only if we care about outputting the final state
                if constexpr ((kIsVarLen && kOutputFinalState) || kUseGate) {
                    // zero out the K tensor on ragged tiles, and store back
                    // so this ensures that the rows are zeroed out for ragged tiles for
                    // the A operand
                    if (!full_chunk) {
                        // if we're using gating, it's already stored in the gating side
                        // #pragma unroll
                        //                         for (int val_idx = 0; val_idx < WSMMA::kNumAccum;
                        //                         val_idx++) {
                        //                             auto [row_idx, col_idx] =
                        //                             get_accum_row_col(threadIdx.x, val_idx); bool
                        //                             pred = global_row_offset + row_idx < seq_end;
                        //                             if (!pred) {
                        //                                 ws_accum[val_idx] = 0.0f;
                        //                             }
                        //                         }
                        store_accum_to_swizzled_smem<BLOCK_V, kChunkSize, kSwizzleUMode,
                                                     WGMMA_M_PER_WARP>(ws_accum, sU, warpIdx,
                                                                       lane_idx);
                    }
                    // we now need to ensure for K that they are zeroed out
                    constexpr int kNumMathWarps = kNumMathThreads / 32;
                    float sK_accum[SHAPE_K / 2];
                    load_swizzled_smem_to_accum<SHAPE_K, kChunkSize, kSwizzleKMode,
                                                WGMMA_M_PER_WARP>(sK, sK_accum, warpIdx, lane_idx);
#pragma unroll
                    for (int val_idx = 0; val_idx < SHAPE_K / 8; val_idx++) {
                        const auto [row_idx, col_idx] = get_accum_row_col(threadIdx.x, val_idx);
                        bool pred = global_row_offset + row_idx < seq_end;
                        if (!pred) {
                            sK_accum[val_idx] = 0.f;
                        }

                        if constexpr (kUseGate) {
                            sK_accum[val_idx] *= __expf(gate_last - s_gate[row_idx]);
                        }
                    }
                    gate_last = __expf(gate_last);
                    if constexpr (kUseGate) {
#pragma unroll
                        for (int i = 0; i < STATE_MMA::kNumAccum; i++) {
                            state_accum[i] *= gate_last;  // gating decay
                        }
                    }
                    store_accum_to_swizzled_smem<SHAPE_K, kChunkSize, kSwizzleKMode,
                                                 WGMMA_M_PER_WARP>(sK_accum, sK, warpIdx, lane_idx);
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
                }
                auto smem_u_desc =
                    create_wgmma_desc<Major::MN, BLOCK_V, kChunkSize, kSwizzleUMode>(sU);
                auto smem_k_desc =
                    create_wgmma_desc<Major::MN, SHAPE_K, kChunkSize, kSwizzleKMode>(sK);
                const auto smem_u_desc_base = __shfl_sync(uint32_t(-1), smem_u_desc.reg32_[0], 0);
                const auto smem_k_desc_base = __shfl_sync(uint32_t(-1), smem_k_desc.reg32_[0], 0);

#pragma unroll
                for (int i = 0; i < STATE_MMA::kNumAccum; i++) {
                    cute::warpgroup_fence_operand(state_accum[i]);
                }
                cute::warpgroup_arrive();

#pragma unroll
                for (int k = 0; k < kChunkSize / STATE_MMA::K; k++) {
                    smem_u_desc.reg32_[0] =
                        smem_u_desc_base +
                        ((k * STATE_MMA::K *
                          get_gmma_desc_stride_k<Major::MN, BLOCK_V, kChunkSize, kSwizzleUMode,
                                                 __nv_bfloat16>() *
                          sizeof(__nv_bfloat16)) >>
                         4);
                    smem_k_desc.reg32_[0] =
                        smem_k_desc_base +
                        ((k * STATE_MMA::K *
                          get_gmma_desc_stride_k<Major::MN, SHAPE_K, kChunkSize, kSwizzleKMode,
                                                 __nv_bfloat16>() *
                          sizeof(__nv_bfloat16)) >>
                         4);
                    STATE_MMA::wgmma(smem_u_desc, smem_k_desc, state_accum, 1);
                }
                cute::warpgroup_commit_batch();

#pragma unroll
                for (int i = 0; i < STATE_MMA::kNumAccum; i++) {
                    cute::warpgroup_fence_operand(state_accum[i]);
                }
                cute::warpgroup_wait<0>();

                // once this is done, we store the new state_{i+1} tensor back into
                // smem, for the state load and then begin next iteration
                store_accum_to_swizzled_smem<SHAPE_K, BLOCK_V, kSwizzleSMode, WGMMA_M_PER_WARP>(
                    state_accum, sState, warpIdx, lane_idx);
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);  // we're storing S_{i}

                if constexpr (kOutputFinalState) {
                    if (chunk_idx == num_chunks - 1) {
                        if (warpIdx == 0 && lane_idx < NUM_S_TMA_BLOCKS) {
                            auto smem_offset = sState + lane_idx * TMA_S_BLOCK_N * BLOCK_V;
                            cute::SM90_TMA_STORE_4D::copy(
                                &final_state_tensor_map, smem_offset, lane_idx * TMA_S_BLOCK_N,
                                v_block_idx * BLOCK_V, v_head_idx,
                                batch_index);  // write state to next chunk,
                                               // since we're calcing S_{i+1}
                            cute::tma_store_arrive();
                        }
                    }
                    cute::tma_store_wait<0>();
                }
                // at the end of the current (batch_idx, head_idx) iteration, we want
                // to wait for all state stores to finish before we overwrite, just in
                // case a new tma load for the next while loop iteration overwrites
                // our current stores
                advance_pipeline();
                math_barrier_arrive();
            }
            cute::tma_store_wait<0>();
        }
    }
}
