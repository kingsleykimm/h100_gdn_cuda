#pragma once
#include <jit_kernels/heuristics/sm90_arch.hpp>
#include <string>
#include <vector>

#include "gdn_cuda/device.hpp"

inline std::tuple<int, int, int> get_transpose_config(int mn, int sf_k,
                                                      size_t element_size = sizeof(float)) {
    const std::vector<int> block_mn_candidates = {128, 64, 32, 16, 8};
    const std::vector<int> num_threads_candidates = {512, 256, 128, 64, 32};
    int best_sm_occupancy = 0;
    int best_block_mn = 0;
    int best_threads = 0;
    int best_smem_size = 0;
    for (const auto block_mn : block_mn_candidates) {
        int padded_sf_k = sf_k + (sf_k + 1) % 2;
        int smem_size = block_mn * padded_sf_k * static_cast<int>(element_size);
        bool valid = smem_size < device_prop->get_smem_size();
        if (valid) {
            int num_blocks_per_sm = device_prop->get_prop()->sharedMemPerMultiprocessor / smem_size;
            if (num_blocks_per_sm == 0)
                continue;
            for (const auto thread : num_threads_candidates) {
                if (thread > device_prop->get_prop()->maxThreadsPerBlock)
                    continue;
                int actual_blocks_per_sm =
                    std::min(num_blocks_per_sm,
                             device_prop->get_prop()->maxThreadsPerMultiProcessor / thread);
                float occupancy = (float)actual_blocks_per_sm * thread /
                                  device_prop->get_prop()->maxThreadsPerMultiProcessor;
                if (occupancy > best_sm_occupancy) {
                    best_sm_occupancy = occupancy;
                    best_block_mn = block_mn;
                    best_threads = thread;
                    best_smem_size = smem_size;
                } else if (occupancy == best_sm_occupancy) {
                    if (best_threads < thread) {
                        best_block_mn = block_mn;
                        best_threads = thread;
                        best_smem_size = smem_size;
                    }
                }
            }
        }
    }
    HOST_ASSERT(best_block_mn > 0 && best_threads > 0, "Error in heuristic search");
    return std::make_tuple(best_block_mn, best_threads, best_smem_size);
}

inline GDNConfig get_recurrent_config(const uint32_t& shape_k, const uint32_t& shape_v,
                                      const uint32_t& num_v_heads, const uint32_t& num_k_heads,
                                      const uint32_t& batch_size,
                                      const uint32_t& num_sms = SM90Arch::kMaxSMs) {
    constexpr size_t bf16_size = 2;
    constexpr int smem_capacity = SM90Arch::kMaxSharedMemoryPerBlock;

    HOST_ASSERT(shape_v >= 64 && shape_v % 64 == 0,
                "Shape_v doesn't fit minimum TMA load requirements");
    HOST_ASSERT(shape_k >= 64 && shape_k % 64 == 0,
                "Shape_k doesn't fit minimum TMA load requirements");
    const uint32_t block_v = std::min(shape_v, (uint32_t)64);
    const uint32_t num_blocks = ti_ceil_div(shape_v, block_v);

    const uint32_t smem_hidden_size = ti_align(bf16_size * shape_k * block_v, 128);
    const uint32_t smem_q_size = ti_align(bf16_size * shape_k, 128);
    const uint32_t smem_k_size = ti_align(bf16_size * shape_k, 128);
    const uint32_t smem_v_size = ti_align(bf16_size * block_v, 128);
    const uint32_t smem_barrier_size = ti_align(SM90Arch::get_barrier_size() * 2, 128);

    const uint32_t total_smem_size =
        smem_hidden_size + smem_q_size + smem_k_size + smem_v_size + smem_barrier_size;

    if (total_smem_size > static_cast<uint32_t>(smem_capacity)) {
        std::string msg = "GDN recurrent kernel shared memory exceeds capacity: " +
                          std::to_string(total_smem_size) + " > " + std::to_string(smem_capacity);
        HOST_ERROR(msg.c_str());
    }

    const uint32_t num_math_warps = std::min(block_v, (uint32_t)8);
    const uint32_t num_math_threads = num_math_warps * 32;
    const uint32_t num_tma_threads = 128;
    constexpr uint32_t kRecurrentMaxMulticast = 2;
    uint32_t num_multicast =
        std::min(kRecurrentMaxMulticast, std::max(1u, num_v_heads / num_k_heads));
    if ((num_v_heads / num_k_heads) % num_multicast != 0) {
        num_multicast = 1;
    }

    const uint32_t smem_constrained_blocks = smem_capacity / total_smem_size;
    const uint32_t thread_constrained_num_blocks =
        SM90Arch::kMaxThreadsPerSM / (num_math_threads + num_tma_threads);
    uint32_t total_blocks_count = num_blocks * num_v_heads * batch_size;

    const uint32_t per_sm_blocks =
        (std::min(smem_constrained_blocks, thread_constrained_num_blocks) / num_multicast) *
        num_multicast;

    const uint32_t total_waves = ti_ceil_div(total_blocks_count, per_sm_blocks * num_sms);
    uint32_t min_sms = num_sms;
    if (SM90Arch::should_minimize_sms()) {
        if (total_waves == 1) {
            min_sms = ti_ceil_div(total_blocks_count, per_sm_blocks);
        } else {
            uint32_t possible =
                ti_align(ti_ceil_div(total_blocks_count, total_waves), per_sm_blocks);
            if (possible > min_sms * per_sm_blocks) {
                HOST_ASSERT(false, "error : possible sms is larger than hardware limits");
            }
            min_sms = possible / per_sm_blocks;
        }
    }

    GDNConfig config;
    config.gdn_type = GDNType::Recurrent;
    config.block_v = block_v;
    config.block_k = shape_k;
    config.swizzle_state_mode = 0;
    config.num_tma_threads = num_tma_threads;
    config.num_math_threads = num_math_threads;
    config.num_blocks = min_sms * per_sm_blocks;
    config.swizzle_k_mode = 0;
    config.swizzle_v_mode = 0;
    config.swizzle_a_mode = 0;
    config.smem_size = total_smem_size;
    config.num_sms = min_sms;
    config.num_tma_multicast = num_multicast;
    return config;
}

template <bool kIsVarLen>
inline GDNConfig get_uw_config(const uint32_t& shape_k, const uint32_t& shape_v,
                               const uint32_t& batch_size, const uint32_t& seq_len,
                               const uint32_t& num_chunks, const uint32_t& num_v_heads,
                               const bool use_gate, const uint32_t& chunk_size = 64,
                               const uint32_t& num_sms = SM90Arch::kMaxSMs) {
    constexpr size_t bf16_size = 2;
    constexpr int smem_capacity = SM90Arch::kMaxSharedMemoryPerBlock;
    const uint32_t block_k = std::min(shape_k, (uint32_t)64);

    const int swizzle_k_mode = get_swizzle_mode(block_k, bf16_size);
    const int swizzle_v_mode = get_swizzle_mode(shape_v, bf16_size);
    const int swizzle_a_mode = get_swizzle_mode(chunk_size, bf16_size);

    const uint32_t smem_k_size = ti_align(bf16_size * chunk_size * shape_k, 1024);
    const uint32_t smem_v_size = ti_align(bf16_size * chunk_size * shape_v, 1024);
    const uint32_t smem_u_size = ti_align(bf16_size * chunk_size * shape_v, 1024);
    const uint32_t smem_w_size = ti_align(bf16_size * chunk_size * shape_k, 1024);
    const uint32_t smem_a_size = ti_align(bf16_size * chunk_size * chunk_size, 128);
    const uint32_t smem_beta_size = ti_align(bf16_size * chunk_size, 128);
    const uint32_t smem_barrier_size = ti_align(SM90Arch::get_barrier_size(), 8);

    uint32_t total_smem_size = smem_k_size + smem_v_size + smem_a_size + smem_u_size + smem_w_size +
                               smem_beta_size + smem_barrier_size;
    if (use_gate) {
        total_smem_size += sizeof(float) * chunk_size;
    }

    if (total_smem_size > static_cast<uint32_t>(smem_capacity)) {
        HOST_ERROR("GDN UW kernel shared memory usage exceeds capacity");
    }

    constexpr uint32_t num_math_threads = 128;
    constexpr uint32_t num_tma_threads = 128;

    const uint32_t smem_constrained_blocks = smem_capacity / total_smem_size;
    const uint32_t thread_constrained_num_blocks =
        SM90Arch::kMaxThreadsPerSM / (num_math_threads + num_tma_threads);
    uint32_t total_blocks;
    if (kIsVarLen) {
        total_blocks = num_chunks * num_v_heads;
    } else {
        total_blocks = ti_ceil_div(seq_len, chunk_size) * batch_size * num_v_heads;
    }

    const uint32_t per_sm_blocks = std::min(smem_constrained_blocks, thread_constrained_num_blocks);
    const uint32_t total_waves = ti_ceil_div(total_blocks, per_sm_blocks * num_sms);

    uint32_t min_sms = num_sms;
    if (SM90Arch::should_minimize_sms()) {
        if (total_waves == 1) {
            min_sms = ti_ceil_div(total_blocks, per_sm_blocks);
        } else {
            uint32_t possible = ti_align(ti_ceil_div(total_blocks, total_waves), per_sm_blocks);
            if (possible > min_sms * per_sm_blocks) {
                HOST_ASSERT(false, "error : possible sms is larger than hardware limits");
            }
            min_sms = possible / per_sm_blocks;
        }
    }

    GDNConfig config;
    config.gdn_type = GDNType::Chunked;
    config.block_v = shape_v;
    config.block_k = block_k;
    config.swizzle_state_mode = 0;
    config.num_tma_threads = num_tma_threads;
    config.num_math_threads = num_math_threads;
    config.num_blocks = min_sms * per_sm_blocks;
    config.swizzle_k_mode = static_cast<uint32_t>(swizzle_k_mode);
    config.swizzle_v_mode = static_cast<uint32_t>(swizzle_v_mode);
    config.swizzle_a_mode = static_cast<uint32_t>(swizzle_a_mode);
    config.smem_size = total_smem_size;
    config.num_sms = min_sms;
    config.num_tma_multicast = 1;
    return config;
}

template <bool kIsVarLen>
inline GDNConfig get_compute_O_config(const uint32_t& shape_k, const uint32_t& shape_v,
                                      const uint32_t& num_v_heads, const uint32_t& num_k_heads,
                                      const uint32_t& batch_size, const uint32_t& seq_len,
                                      const uint32_t& num_chunks, const uint32_t& use_gate,
                                      const uint32_t& chunk_size = 64,
                                      const uint32_t& num_sms = SM90Arch::kMaxSMs) {
    constexpr size_t bf16_size = 2;
    constexpr int smem_capacity = SM90Arch::kMaxSharedMemoryPerBlock;

    HOST_ASSERT(shape_v >= 64 && shape_v % 64 == 0,
                "Shape_v doesn't fit minimum TMA load requirements");
    HOST_ASSERT(shape_k >= 64 && shape_k % 64 == 0,
                "Shape_k doesn't fit minimum TMA load requirements");
    HOST_ASSERT(chunk_size == 64, "Only chunk_size=64 is supported currently");

    const uint32_t block_v = std::min(shape_v, (uint32_t)64);
    const uint32_t block_k = std::min(shape_k, (uint32_t)64);
    const uint32_t chunk_block = std::min(chunk_size, (uint32_t)32);

    HOST_ASSERT(chunk_size % chunk_block == 0, "chunk_block must divide chunk_size");
    HOST_ASSERT(chunk_block >= 16 && chunk_block % 16 == 0,
                "chunk_block must be >= 16 and divisible by 16");

    const int swizzle_k_mode = get_swizzle_mode(block_k, bf16_size);
    const int swizzle_q_mode = get_swizzle_mode(block_k, bf16_size);
    const int swizzle_u_mode = get_swizzle_mode(block_v, bf16_size);
    const int swizzle_s_mode = get_swizzle_mode(block_v, bf16_size);

    const uint32_t smem_state_per_stage = ti_align(bf16_size * block_v * block_k, 1024);
    const uint32_t smem_k_per_stage = ti_align(bf16_size * chunk_block * block_k, 1024);
    const uint32_t smem_u_per_stage = ti_align(bf16_size * chunk_block * block_v, 1024);
    const uint32_t smem_q_per_stage = ti_align(bf16_size * chunk_size * block_k, 1024);

    const uint32_t smem_gate_size = use_gate ? sizeof(float) * chunk_size : 0;

    const uint32_t smem_per_stage =
        smem_state_per_stage + smem_k_per_stage + smem_u_per_stage + smem_q_per_stage;

    const uint32_t smem_o_size = ti_align(bf16_size * block_v * chunk_size, 1024);
    const uint32_t smem_barrier_per_stage = SM90Arch::get_barrier_size();

    constexpr uint32_t num_math_threads = 128;
    constexpr uint32_t num_tma_threads = 128;

    uint32_t best_num_stages = 1;
    uint32_t best_smem_size = 0;

    const uint32_t num_k1_blocks = shape_k / block_k;
    const uint32_t num_k2_blocks = chunk_size / chunk_block;
    const uint32_t ideal_stages = std::min((uint32_t)8, num_k1_blocks * num_k2_blocks);

    for (uint32_t num_stages = ideal_stages; num_stages >= 1; num_stages--) {
        uint32_t total_smem = smem_per_stage * num_stages + smem_o_size + smem_gate_size +
                              smem_barrier_per_stage * num_stages;
        if (total_smem <= static_cast<uint32_t>(smem_capacity)) {
            best_num_stages = num_stages;
            best_smem_size = total_smem;
            break;
        }
    }

    HOST_ASSERT(best_num_stages > 0,
                "Cannot fit even 1 stage in shared memory for compute_O kernel");

    const uint32_t num_v_blocks = ti_ceil_div(shape_v, block_v);
    uint32_t total_blocks;

    if constexpr (kIsVarLen) {
        total_blocks = num_chunks * num_v_heads * num_v_blocks;
    } else {
        const uint32_t chunks_per_seq = ti_ceil_div(seq_len, chunk_size);
        total_blocks = batch_size * chunks_per_seq * num_v_heads * num_v_blocks;
    }

    const uint32_t smem_constrained_blocks = smem_capacity / best_smem_size;
    const uint32_t thread_constrained_blocks =
        SM90Arch::kMaxThreadsPerSM / (num_math_threads + num_tma_threads);
    const uint32_t per_sm_blocks = std::min(smem_constrained_blocks, thread_constrained_blocks);

    const uint32_t total_waves = ti_ceil_div(total_blocks, per_sm_blocks * num_sms);

    uint32_t min_sms = num_sms;
    if (SM90Arch::should_minimize_sms()) {
        if (total_waves == 1) {
            min_sms = ti_ceil_div(total_blocks, per_sm_blocks);
        } else {
            uint32_t possible = ti_align(ti_ceil_div(total_blocks, total_waves), per_sm_blocks);
            if (possible > min_sms * per_sm_blocks) {
                HOST_ASSERT(false, "error: computed SMs exceeds hardware limits");
            }
            min_sms = possible / per_sm_blocks;
        }
    }

    GDNConfig config;
    config.gdn_type = GDNType::Chunked;
    config.block_v = block_v;
    config.block_k = block_k;
    config.chunk_block = chunk_block;
    config.num_stages = best_num_stages;
    config.swizzle_state_mode = static_cast<uint32_t>(swizzle_s_mode);
    config.swizzle_k_mode = static_cast<uint32_t>(swizzle_k_mode);
    config.swizzle_q_mode = static_cast<uint32_t>(swizzle_q_mode);
    config.swizzle_u_mode = static_cast<uint32_t>(swizzle_u_mode);
    config.swizzle_v_mode = static_cast<uint32_t>(swizzle_u_mode);
    config.swizzle_a_mode = 0;
    config.num_tma_threads = num_tma_threads;
    config.num_math_threads = num_math_threads;
    config.num_blocks = min_sms * per_sm_blocks;
    config.smem_size = best_smem_size;
    config.num_sms = min_sms;
    config.num_tma_multicast = 1;

    return config;
}

template <bool kIsVarLen>
inline GDNConfig get_seq_state_config(const uint32_t& shape_k, const uint32_t& shape_v,
                                      const uint32_t& num_v_heads, const uint32_t& num_k_heads,
                                      const uint32_t& batch_size, const uint32_t& seq_len,
                                      const uint32_t& chunk_size = 64, const bool& use_gate = false,
                                      const uint32_t& num_sms = SM90Arch::kMaxSMs) {
    constexpr size_t bf16_size = 2;
    constexpr int smem_capacity = SM90Arch::kMaxSharedMemoryPerBlock;
    HOST_ASSERT(shape_v >= 64 && shape_v % 64 == 0,
                "Shape_v doesn't fit minimum TMA load requirements");
    HOST_ASSERT(shape_k >= 64 && shape_k % 64 == 0,
                "Shape_k doesn't fit minimum TMA load requirements");
    const uint32_t block_v = std::min(shape_v, (uint32_t)64);

    const int swizzle_k_mode = get_swizzle_mode(shape_k, bf16_size);
    const int swizzle_v_mode = get_swizzle_mode(block_v, bf16_size);
    const int swizzle_s_mode = get_swizzle_mode(shape_k, bf16_size);

    const uint32_t smem_state_size = ti_align(bf16_size * block_v * shape_k, 1024);
    const uint32_t smem_k_size = ti_align(bf16_size * shape_k * chunk_size, 1024);
    const uint32_t smem_u_size = ti_align(bf16_size * chunk_size * block_v, 1024);
    const uint32_t smem_w_size = ti_align(bf16_size * chunk_size * shape_k, 1024);
    const uint32_t smem_barrier_size = ti_align(SM90Arch::get_barrier_size() * 3 / 2, 8);
    const uint32_t smem_gate_size = use_gate ? sizeof(float) * chunk_size : 0;

    const uint32_t total_smem_size = smem_state_size + smem_k_size + smem_u_size + smem_w_size +
                                     smem_barrier_size + smem_gate_size;

    if (total_smem_size > static_cast<uint32_t>(smem_capacity)) {
        HOST_ERROR("GDN seq state kernel shared memory usage exceeds capacity");
    }

    constexpr uint32_t num_math_threads = 128;
    constexpr uint32_t num_tma_threads = 128;

    const uint32_t smem_constrained_blocks = smem_capacity / total_smem_size;
    const uint32_t thread_constrained_num_blocks =
        SM90Arch::kMaxThreadsPerSM / (num_math_threads + num_tma_threads);
    uint32_t total_blocks;
    total_blocks = batch_size * num_v_heads * ti_ceil_div(shape_v, block_v);

    const uint32_t per_sm_blocks = std::min(smem_constrained_blocks, thread_constrained_num_blocks);
    const uint32_t total_waves = ti_ceil_div(total_blocks, per_sm_blocks * num_sms);

    uint32_t min_sms = num_sms;
    if (SM90Arch::should_minimize_sms()) {
        if (total_waves == 1) {
            min_sms = ti_ceil_div(total_blocks, per_sm_blocks);
        } else {
            uint32_t possible = ti_align(ti_ceil_div(total_blocks, total_waves), per_sm_blocks);
            if (possible > min_sms * per_sm_blocks) {
                HOST_ASSERT(false, "error : possible sms is larger than hardware limits");
            }
            min_sms = possible / per_sm_blocks;
        }
    }

    GDNConfig config;
    config.gdn_type = GDNType::Chunked;
    config.block_v = block_v;
    config.block_k = swizzle_k_mode / sizeof(__nv_bfloat16);
    config.swizzle_state_mode = swizzle_s_mode;
    config.num_tma_threads = num_tma_threads;
    config.num_math_threads = num_math_threads;
    config.num_blocks = min_sms * per_sm_blocks;
    config.swizzle_k_mode = static_cast<uint32_t>(swizzle_k_mode);
    config.swizzle_v_mode = static_cast<uint32_t>(swizzle_v_mode);
    config.swizzle_state_mode = static_cast<uint32_t>(swizzle_s_mode);
    config.smem_size = total_smem_size;
    config.num_sms = min_sms;
    config.num_tma_multicast = 1;
    return config;
}
