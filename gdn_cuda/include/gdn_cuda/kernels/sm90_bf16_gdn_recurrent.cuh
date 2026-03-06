#pragma once
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

#include "cute/arch/cluster_sm90.hpp"

namespace gdn_cuda {
namespace kernels {
namespace sm90_gdn_recurrent_impl {

template <uint32_t SHAPE_K, uint32_t SHAPE_V, uint32_t kNumVHeads, uint32_t kNumKHeads,
          uint32_t BLOCK_V, uint32_t kNumBlocks, uint32_t kNumTMAMulticast, uint32_t kNumTMAThreads,
          uint32_t kNumMathThreads, uint32_t kSeqLen, bool kIsVarLen, bool kIsInitialState,
          bool kStoreStepState, bool kIsQKNorm, bool kUseGate>
__global__ void fused_recurrent_gated_delta_rule_bf16(
    const CUTE_GRID_CONSTANT cute::TmaDescriptor q_tensor_map,
    const CUTE_GRID_CONSTANT cute::TmaDescriptor k_tensor_map,
    const CUTE_GRID_CONSTANT cute::TmaDescriptor v_tensor_map,
    const CUTE_GRID_CONSTANT cute::TmaDescriptor state_tensor_map,
    const CUTE_GRID_CONSTANT cute::TmaDescriptor final_state_tensor_map, __nv_bfloat16* out,
    const __nv_bfloat16* beta_ptr, __nv_bfloat16* g_ptr, int* num_accepted_tokens_ptr,
    int batch_size, int shape_k, int shape_v, int* cu_seqlens, float scale) {
    CUTE_STATIC_ASSERT(BLOCK_V >= 64, "BLOCK_V >= 64 In order for TMA-aligned loads");
    CUTE_STATIC_ASSERT(kNumBlocks % kNumTMAMulticast == 0,
                       "kNumBlocks must be divisible by kNumTMAMulticast");
    CUTE_STATIC_ASSERT(kNumVHeads >= kNumKHeads);
    // we want kIsVarLen is true or kSeqLen > 0, exclusive or
    CUTE_STATIC_ASSERT(kIsVarLen ^ (kSeqLen > 0),
                       "Exactly one of kIsVarLen or fixed kSeqLen must be active");
    DEVICE_ASSERT(!kIsVarLen || cu_seqlens != nullptr);
    DEVICE_ASSERT(uintptr_t(beta_ptr) % 8 == 0);
    DEVICE_ASSERT(!kUseGate || g_ptr != nullptr);

    auto lane_predicate = cute::elect_one_sync();
    int warpIdx = threadIdx.x / 32;
    constexpr int kNumMathWarps = kNumMathThreads / 32;
    int lane_idx = threadIdx.x & 0x1f;
    CUTE_STATIC_ASSERT(BLOCK_V % kNumMathWarps == 0, "Invalid number of math warps");
    if (lane_predicate) {
        cute::prefetch_tma_descriptor(&q_tensor_map);
        cute::prefetch_tma_descriptor(&k_tensor_map);
        cute::prefetch_tma_descriptor(&v_tensor_map);
        if constexpr (kIsInitialState) {
            cute::prefetch_tma_descriptor(&state_tensor_map);
        }
        cute::prefetch_tma_descriptor(&final_state_tensor_map);
    }
    __syncwarp();
    // set up shared memory

    // we need to store a QKV, O, M, U and W
    // no swizzling

    extern __shared__ __align__(1024) char shared[];
    constexpr uint32_t SMEM_Q_SIZE = constexpr_ti_align(sizeof(__nv_bfloat16) * SHAPE_K, 128);
    constexpr uint32_t SMEM_K_SIZE = constexpr_ti_align(sizeof(__nv_bfloat16) * SHAPE_K, 128);
    constexpr uint32_t SMEM_V_SIZE = constexpr_ti_align(sizeof(__nv_bfloat16) * BLOCK_V, 128);
    constexpr uint32_t SMEM_HIDDEN_SIZE =
        constexpr_ti_align(sizeof(__nv_bfloat16) * SHAPE_K * BLOCK_V, 128);

    __nv_bfloat16* state = reinterpret_cast<__nv_bfloat16*>(&shared);
    __nv_bfloat16* query = reinterpret_cast<__nv_bfloat16*>(shared + SMEM_HIDDEN_SIZE);

    __nv_bfloat16* key = reinterpret_cast<__nv_bfloat16*>(shared + SMEM_HIDDEN_SIZE + SMEM_Q_SIZE);
    __nv_bfloat16* value =
        reinterpret_cast<__nv_bfloat16*>(shared + SMEM_HIDDEN_SIZE + SMEM_Q_SIZE + SMEM_K_SIZE);
    // barriers
    using MathBarrier = cutlass::arch::ClusterBarrier;
    using TMABarrier = cutlass::arch::ClusterTransactionBarrier;
    TMABarrier* tma_barrier = reinterpret_cast<TMABarrier*>(
        shared + SMEM_HIDDEN_SIZE + SMEM_K_SIZE + SMEM_Q_SIZE + SMEM_V_SIZE);
    MathBarrier* math_barrier = reinterpret_cast<MathBarrier*>(
        shared + SMEM_HIDDEN_SIZE + SMEM_K_SIZE + SMEM_Q_SIZE + SMEM_V_SIZE + sizeof(TMABarrier));
    // overlap tma loads with other calculations
    constexpr uint32_t kFirstIterTMABytes =
        kIsInitialState ? SMEM_HIDDEN_SIZE + SMEM_Q_SIZE + SMEM_K_SIZE + SMEM_V_SIZE
                        : SMEM_Q_SIZE + SMEM_K_SIZE + SMEM_V_SIZE;
    constexpr uint32_t kTMATransbytes = SMEM_Q_SIZE + SMEM_K_SIZE + SMEM_V_SIZE;
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 200;  // overallocate anyways

    if (warpIdx == kNumMathThreads / 32 && lane_predicate) {
        tma_barrier->init(1);
        math_barrier->init(kNumTMAMulticast * kNumMathThreads / 32);
    }
    cutlass::arch::fence_barrier_init();

    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    auto scheduler =
        RecurrentGDNScheduler<BLOCK_V, kNumVHeads, kNumKHeads, kNumBlocks, kNumTMAMulticast>(
            shape_v, batch_size);
    int v_block_idx, v_head_idx, k_head_idx, batch_index;
    int phase = 0, stage_idx = 0;

    auto advance_pipeline = [&]() {
        phase ^= 1;
    };

    if (threadIdx.x >= kNumMathThreads) {
        // cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        while (scheduler.get_next_block(v_block_idx, v_head_idx, k_head_idx, batch_index)) {
            size_t T;
            size_t bos, eos;
            if constexpr (kIsVarLen) {
                bos = __ldg(cu_seqlens + batch_index);
                eos = __ldg(cu_seqlens + batch_index + 1);
                T = eos - bos;
            } else {
                T = kSeqLen;
                bos = batch_index * kSeqLen;
                eos = (batch_index + 1) * kSeqLen;
            }

            size_t global_state_offset = kStoreStepState ? bos : batch_index;

            if (num_accepted_tokens_ptr != nullptr) {
                T = __ldg(num_accepted_tokens_ptr + batch_index);
            }
            // inner_idx should be the offset
            if (warpIdx == kNumMathThreads / 32 && lane_predicate) {
#pragma unroll 1
                for (int seq_idx = 0; seq_idx < T; seq_idx++, advance_pipeline()) {
                    uint32_t num_tma_multicast = scheduler.get_effective_multicast();
                    // uint32_t num_tma_multicast = 1;
                    // math_barrier->wait(tma_state.phase());
                    math_barrier[0].wait(phase ^ 1);

                    auto& barrier = tma_barrier[0];

                    // note make sure to include multicast here

                    tma_copy<SHAPE_K, 1, 0, __nv_bfloat16, 2>(
                        &q_tensor_map, &barrier, query, 0,
                        (bos + seq_idx) * kNumKHeads + k_head_idx, num_tma_multicast);
                    tma_copy<SHAPE_K, 1, 0, __nv_bfloat16, 2>(
                        &k_tensor_map, &barrier, key, 0, (bos + seq_idx) * kNumKHeads + k_head_idx,
                        num_tma_multicast);
                    tma_copy<BLOCK_V, 1, 0, __nv_bfloat16, 2>(
                        &v_tensor_map, &barrier, value, v_block_idx * BLOCK_V,
                        (bos + seq_idx) * kNumVHeads + v_head_idx, 1);

                    if (seq_idx == 0) {
                        if constexpr (kIsInitialState) {
                            tma_copy<SHAPE_K, BLOCK_V, 0, __nv_bfloat16, 4>(
                                &state_tensor_map, &barrier, state, 0, v_block_idx * BLOCK_V, 1,
                                v_head_idx, batch_index);
                            barrier.arrive_and_expect_tx(SMEM_HIDDEN_SIZE + SMEM_Q_SIZE +
                                                         SMEM_K_SIZE + SMEM_V_SIZE);
                        } else {
                            barrier.arrive_and_expect_tx(SMEM_Q_SIZE + SMEM_K_SIZE + SMEM_V_SIZE);
                        }
                    } else {
                        barrier.arrive_and_expect_tx(SMEM_Q_SIZE + SMEM_K_SIZE + SMEM_V_SIZE);
                    }
                }
                // advance_pipeline();
                // math_barrier->wait(phase ^ 1);
            }
        }
        // if constexpr (kNumTMAMulticast > 1) {
        //   advance_pipeline();
        //   math_barrier->wait(phase ^ 1);
        // }
    } else {  // warpgroup units, so you should be using one warpgroup to store as well
        // cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        auto math_barrier_arrive = [&]() {
            if constexpr (kNumTMAMulticast == 1) {
                (lane_idx == 0) ? math_barrier[0].arrive() : void();
            } else {
                math_barrier[0].arrive(lane_idx, lane_idx < kNumTMAMulticast);
            }
        };
        while (scheduler.get_next_block(v_block_idx, v_head_idx, k_head_idx, batch_index)) {
            // if (cute::thread0()) {
            //   printf("v_block_idx : %d, v_head_idx : %d, k_head_idx : %d, batch_index : %d \n",
            //   v_block_idx, v_head_idx,
            //          k_head_idx, batch_index);
            // }
            constexpr uint32_t numRowsPerWarp = BLOCK_V / kNumMathWarps;
            using vec_load =
                VEC_LOAD<SHAPE_K * cutlass::sizeof_bits<__nv_bfloat16>::value, __nv_bfloat16>;
            using load_type = typename vec_load::ptr_type;
            constexpr int kNumVec =
                cute::min(SHAPE_K / 32, (int)(sizeof(load_type) / sizeof(__nv_bfloat16)));
            // 1 Warp Per Row - SHAPE_K / (kNumVec * 32) equals the number of active threads, we
            // want this to be >= 1 so kNumVec <= SHAPE_K / 32, and each thread will have SHAPE_K /
            // 32 * numRowsPerThread registers allocated for state. the max kNumVec is 8, so if
            // SHAPE_K > 32 * 8, we just use it. SO this is a min operation. if SHAPE_K = 128, then
            // the kNumVec is always 4

            // rewrite the load_type pointer to match the kNumVec determined above
            using vec_load_ptr =
                typename VEC_LOAD_PTR<kNumVec,
                                      cutlass::sizeof_bits<__nv_bfloat16>::value>::ptr_type;

            constexpr uint32_t threadLoadSize = kNumVec * 32;

            // current stage is loaded into

            size_t T;
            size_t bos, eos;
            if constexpr (kIsVarLen) {
                bos = __ldg(cu_seqlens + batch_index);
                eos = __ldg(cu_seqlens + batch_index + 1);
                T = eos - bos;
            } else {
                T = kSeqLen;
                bos = batch_index * kSeqLen;
                eos = (batch_index + 1) * kSeqLen;
            }

            if (num_accepted_tokens_ptr != nullptr) {
                T = __ldg(num_accepted_tokens_ptr + batch_index);
            }
            size_t global_state_offset = kStoreStepState ? bos : batch_index;
            float state_regs[SHAPE_K / 32 * numRowsPerWarp];
            float key_regs[SHAPE_K / 32];
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
            // wait for the previous tma store to finish before we can start writing into the new
            // state shared memory
            for (int seq_idx = 0; seq_idx < T; seq_idx++, advance_pipeline()) {
                __nv_bfloat16 beta = __ldg(beta_ptr + (bos + seq_idx) * kNumVHeads + v_head_idx);
                __nv_bfloat16 g;
                if constexpr (kUseGate) {
                    g = __ldg(g_ptr + (bos + seq_idx) * kNumVHeads + v_head_idx);
                }
                tma_barrier[0].wait(phase);
                // initial state load from gmem -> smem -> rmem
                if (seq_idx == 0) {
                    if constexpr (kIsInitialState) {
#pragma unroll
                        for (int row_offset = 0; row_offset < numRowsPerWarp; row_offset++) {
                            int row = warpIdx + row_offset * kNumMathWarps;
                            float* shifted_state = state_regs + SHAPE_K / 32 * row_offset;

#pragma unroll
                            for (int col = lane_idx * kNumVec; col < SHAPE_K;
                                 col += threadLoadSize) {
                                // No swizzle - use simple linear indexing
                                uint32_t offset = row * SHAPE_K + col;
                                vec_load_ptr state_load =
                                    ld_shared(reinterpret_cast<vec_load_ptr*>(state + offset));
                                __nv_bfloat16* state_bf16 =
                                    reinterpret_cast<__nv_bfloat16*>(&state_load);
#pragma unroll
                                for (int j = 0; j < kNumVec; j++) {
                                    shifted_state[j] = __bfloat162float(state_bf16[j]);
                                }
                                shifted_state += kNumVec;
                            }
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < SHAPE_K / 32 * numRowsPerWarp; i++) {
                            state_regs[i] = 0;
                        }
                    }
                }
                // row offset - we are going to guarantee in the heuristic that number of warps ==
                // BLOCK_V
#pragma unroll
                for (int row_offset = 0; row_offset < numRowsPerWarp; row_offset++) {
                    int row = warpIdx + row_offset * kNumMathWarps;
                    float* shifted_state = state_regs + SHAPE_K / 32 * row_offset;
                    float* shifted_key = key_regs;
                    float accum = 0;
                    float k2_sum = 0;
#pragma unroll
                    for (int col = lane_idx * kNumVec; col < SHAPE_K; col += threadLoadSize) {
                        vec_load_ptr k_load = ld_shared(reinterpret_cast<vec_load_ptr*>(key + col));
                        __nv_bfloat16* k_bf16 = reinterpret_cast<__nv_bfloat16*>(&k_load);

#pragma unroll
                        for (int j = 0; j < kNumVec; j++) {
                            shifted_key[j] = __bfloat162float(k_bf16[j]);
                            accum += shifted_state[j] * shifted_key[j];
                            k2_sum += shifted_key[j] * shifted_key[j];
                        }
                        shifted_state += kNumVec;
                        shifted_key += kNumVec;
                    }
// warp reduction
#pragma unroll
                    for (int offset = 1; offset < 32; offset <<= 1) {
                        accum += __shfl_xor_sync(uint32_t(-1), accum, offset);
                        k2_sum += __shfl_xor_sync(uint32_t(-1), k2_sum, offset);
                    }

                    float inv_k2_norm = kIsQKNorm ? rsqrt(k2_sum + 1e-6f) : 1.0f;

                    float v = __bfloat162float(value[row]);
                    // unified register trick
                    float row_val = __shfl_sync(uint32_t(-1), (accum * inv_k2_norm - v),
                                                0);  // (Sk * norm  - v)
                    shifted_state = state_regs + SHAPE_K / 32 * row_offset;
                    shifted_key = key_regs;
// now we have to access K through vectorized loads again - this time using to perform
// a fused outerproduct + state addition
#pragma unroll
                    for (int col = lane_idx * kNumVec; col < SHAPE_K; col += threadLoadSize) {
#pragma unroll
                        for (int j = 0; j < kNumVec; j++) {
                            if constexpr (kUseGate) {
                                shifted_state[j] *= __expf(__bfloat162float(g));
                            }
                            shifted_state[j] -=
                                __bfloat162float(beta) * row_val * shifted_key[j] * inv_k2_norm;
                        }
                        shifted_state += kNumVec;
                        shifted_key += kNumVec;
                    }

                    __syncwarp();

                    // make sure all state_regs are updated
                    shifted_state = state_regs + SHAPE_K / 32 * row_offset;
                    float q2_sum = 0;
                    float Sq_accum = 0;
// once state_bf16 is finished, we need to perform the last S*q vector multiplication
#pragma unroll
                    for (int col = lane_idx * kNumVec; col < SHAPE_K; col += threadLoadSize) {
                        vec_load_ptr q_load =
                            ld_shared(reinterpret_cast<vec_load_ptr*>(query + col));
                        __nv_bfloat16* q_bf16 = reinterpret_cast<__nv_bfloat16*>(&q_load);
#pragma unroll
                        for (int j = 0; j < kNumVec; j++) {
                            float q_val = __bfloat162float(q_bf16[j]);
                            Sq_accum += shifted_state[j] * q_val;
                            q2_sum += q_val * q_val;
                        }
                        shifted_state += kNumVec;
                    }

#pragma unroll
                    for (int offset = 1; offset < 32; offset <<= 1) {
                        Sq_accum += __shfl_xor_sync(uint32_t(-1), Sq_accum, offset);
                        q2_sum += __shfl_xor_sync(uint32_t(-1), q2_sum, offset);
                    }

                    float inv_q2_norm = kIsQKNorm ? rsqrt(q2_sum + 1e-6f) : 1.0f;
                    Sq_accum *= inv_q2_norm * scale;
                    __nv_bfloat16 Sq_accum_bf16 = __float2bfloat16(Sq_accum);

                    if (lane_predicate) {
                        // if (blockIdx.x == 0 && warpIdx == 0)
                        //   printf("storing to : %d\n", (bos + seq_idx));
                        size_t offset =
                            (bos + seq_idx) * static_cast<size_t>(kNumVHeads) * SHAPE_V +
                            v_head_idx * SHAPE_V + v_block_idx * BLOCK_V + row;
                        out[offset] = Sq_accum_bf16;
                        // __stcs(out + (bos + seq_idx) * kNumVHeads * SHAPE_V + v_head_idx *
                        // SHAPE_V + v_block_idx * BLOCK_V + row,
                        //        Sq_accum_bf16);
                    }
                    __syncwarp();
                }

                // we are done with all the QKV tensors for current token
                if (seq_idx < T - 1) {
                    math_barrier_arrive();
                }
                if constexpr (kStoreStepState) {
                    if (warpIdx == 0 && lane_predicate) {
                        cute::tma_store_wait<0>();
                    }
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
#pragma unroll
                    for (int row_offset = 0; row_offset < numRowsPerWarp; row_offset++) {
                        int row = warpIdx + row_offset * kNumMathWarps;
                        float* shifted_state = state_regs + SHAPE_K / 32 * row_offset;
#pragma unroll
                        for (int col = lane_idx * kNumVec; col < SHAPE_K; col += threadLoadSize) {
                            uint32_t offset = row * SHAPE_K + col;
                            __nv_bfloat16 state_bf16[kNumVec];
#pragma unroll
                            for (int j = 0; j < kNumVec; j++) {
                                state_bf16[j] = __float2bfloat16(shifted_state[j]);
                            }
                            st_shared(reinterpret_cast<const vec_load_ptr*>(state + offset),
                                      *reinterpret_cast<vec_load_ptr*>(&state_bf16[0]));
                        }
                    }
                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
                    if (warpIdx == 0 && lane_predicate) {
                        cute::SM90_TMA_STORE_4D::copy(&final_state_tensor_map, state, 0,
                                                      v_block_idx * BLOCK_V, v_head_idx,
                                                      bos + seq_idx);
                        cute::tma_store_arrive();
                    }
                }
            }

            if constexpr (!kStoreStepState) {
#pragma unroll
                for (int row_offset = 0; row_offset < numRowsPerWarp; row_offset++) {
                    int row = warpIdx + row_offset * kNumMathWarps;
                    float* shifted_state = state_regs + SHAPE_K / 32 * row_offset;
#pragma unroll
                    for (int col = lane_idx * kNumVec; col < SHAPE_K; col += threadLoadSize) {
                        uint32_t offset = row * SHAPE_K + col;
                        __nv_bfloat16 state_bf16[kNumVec];
#pragma unroll
                        for (int j = 0; j < kNumVec; j++) {
                            state_bf16[j] = __float2bfloat16(shifted_state[j]);
                        }
                        st_shared(reinterpret_cast<const vec_load_ptr*>(state + offset),
                                  *reinterpret_cast<vec_load_ptr*>(&state_bf16[0]));
                    }
                }
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
                if (warpIdx == 0 && lane_predicate) {
                    cute::SM90_TMA_STORE_4D::copy(&final_state_tensor_map, state, 0,
                                                  v_block_idx * BLOCK_V, v_head_idx, batch_index);
                    cute::tma_store_arrive();
                }
                // Signal TMA warp AFTER TMA store is issued to prevent race condition
                // where TMA warp's tma_store_wait<0>() executes before store is issued
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
            }

            if (warpIdx == 0 && lane_predicate) {
                cute::tma_store_wait<0>();
            }
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
            math_barrier_arrive();
        }
    }
}

}  // namespace sm90_gdn_recurrent_impl
}  // namespace kernels
}  // namespace gdn_cuda
