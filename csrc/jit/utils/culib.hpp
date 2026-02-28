#pragma once
#include <cuda.h>
#include <driver_types.h>

#include <gdn_cuda/device.hpp>
#include <gdn_cuda/error.hpp>
#include <jit/utils/common.hpp>

using KernelHandle = CUfunction;

using LaunchConfigHandle = CUlaunchConfig;
using LibraryHandle = CUmodule;

template <typename... Args>
CUresult launch_kernel(KernelHandle& kernel, const LaunchConfigHandle& launch_config,
                       Args&&... args) {
    void* kernelParams[] = {const_cast<void*>(static_cast<const void*>(&args))...};

    if (get_env<int>("JIT_DEBUG", 0) == 1) {
        printf("  Launching kernel: grid=(%u,%u,%u), block=(%u,%u,%u), smem=%u, numAttrs=%u\n",
               launch_config.gridDimX, launch_config.gridDimY, launch_config.gridDimZ,
               launch_config.blockDimX, launch_config.blockDimY, launch_config.blockDimZ,
               launch_config.sharedMemBytes, launch_config.numAttrs);
        fflush(stdout);
    }

    CUresult result = cuLaunchKernelEx(&launch_config, kernel, kernelParams, nullptr);

    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        printf("  cuLaunchKernelEx FAILED: %s (code %d)\n", error_str, result);
    }
    return result;
}

inline LaunchConfigHandle create_launch_config(KernelHandle& kernel, const int& smem_size,
                                               const dim3& blockDim, const dim3& gridDim,
                                               const uint32_t& num_multicast, cudaStream_t stream) {
    if (smem_size > 0) {
        CUDA_CHECK(
            cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size));
    }

    LaunchConfigHandle launch_handle;
    std::memset(&launch_handle, 0, sizeof(launch_handle));
    launch_handle.blockDimX = blockDim.x;
    launch_handle.blockDimY = blockDim.y;
    launch_handle.blockDimZ = blockDim.z;
    launch_handle.gridDimX = gridDim.x;
    launch_handle.gridDimY = gridDim.y;
    launch_handle.gridDimZ = gridDim.z;
    launch_handle.hStream = (CUstream)stream;
    launch_handle.sharedMemBytes = smem_size;
    launch_handle.attrs = nullptr;
    launch_handle.numAttrs = 0;
    static CUlaunchAttribute attr;
    if (num_multicast > 1) {
        if (gridDim.x % num_multicast != 0) {
            printf("ERROR: gridDim.x=%u is not divisible by cluster size=%u\n", gridDim.x,
                   num_multicast);
        }
        if (num_multicast > device_prop->get_max_clusters().first) {
            CUDA_CHECK(
                cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
        }
        attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attr.value.clusterDim.x = num_multicast;
        attr.value.clusterDim.y = 1;
        attr.value.clusterDim.z = 1;
        launch_handle.attrs = &attr;
        launch_handle.numAttrs = 1;

        if (get_env<int>("JIT_DEBUG", 0) == 1) {
            printf("  Cluster launch: grid=%u, cluster=%u\n", gridDim.x, num_multicast);
        }
    }
    return launch_handle;
}
