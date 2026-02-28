#pragma once
#include <cstdint>
#include <gdn_cuda/kernels/common/types.hpp>
#include <jit/utils/common.hpp>
#include <vector>

struct SM90Arch {
    static constexpr uint32_t kMaxSharedMemoryPerBlock = 232448;
    static constexpr uint32_t kMaxSharedMemoryPerSM = 232448;
    static constexpr uint32_t kMaxSMs = 132;
    static constexpr uint32_t kWarpSize = 32;
    static constexpr uint32_t kMaxThreadsPerBlock = 1024;
    static constexpr uint32_t kMaxRegistersPerThread = 255;
    static constexpr uint32_t kMaxThreadsPerSM = 2048;
    static constexpr uint32_t kMaxMulticast = 8;

    static bool should_minimize_sms() { return true; }

    static int get_barrier_size() { return 8 * 2; }
};
