#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <iostream>

static void HOST_WARNING_(const char* msg, const char* file, int line) {
    printf("WARNING: %s in %s:%d\n", msg, file, line);
}

static void HOST_ERROR_(const char* msg, const char* file, int line) {
    printf("ERROR: %s in %s:%d\n", msg, file, line);
    exit(EXIT_FAILURE);
}

#define HOST_WARNING(cond, msg) \
    if (!(cond))                \
    HOST_WARNING_(msg, __FILE__, __LINE__)
#define HOST_ERROR(msg) HOST_ERROR_(msg, __FILE__, __LINE__)
#define HOST_ASSERT(cond, msg) \
    if (!(cond))               \
    HOST_ERROR(msg)

static void cuda_check_error_(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA ERROR: %s in %s:%d\n", cudaGetErrorName(error), file, line);
        exit(EXIT_FAILURE);
    }
}

static void cuda_check_error_(CUresult result, const char* file, int line) {
    if (result != CUDA_SUCCESS) {
        const char* error_string;
        cuGetErrorString(result, &error_string);
        printf("CUDA ERROR: %s in %s:%d\n", error_string, file, line);
        exit(EXIT_FAILURE);
    }
}

static void nvrtc_check_error_(nvrtcResult result, const char* file, int line) {
    if (result != NVRTC_SUCCESS) {
        const char* error_string = nvrtcGetErrorString(result);
        printf("NVRTC ERROR: %s in %s:%d\n", error_string, file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(error) cuda_check_error_(error, __FILE__, __LINE__)

#ifdef CUDA_DEBUG_SYNC
#define CUDA_SYNC_DEBUG()                                                                          \
    do {                                                                                           \
        cudaError_t err = cudaDeviceSynchronize();                                                 \
        if (err != cudaSuccess) {                                                                  \
            printf("CUDA SYNC ERROR at %s:%d\n%s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
#else
#define CUDA_SYNC_DEBUG() ((void)0)
#endif

#define NVRTC_CHECK(error) nvrtc_check_error_(error, __FILE__, __LINE__);
