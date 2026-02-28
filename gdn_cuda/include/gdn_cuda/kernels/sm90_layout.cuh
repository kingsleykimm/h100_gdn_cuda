#pragma once
#include <gdn_cuda/kernels/common/common.hpp>
#include <gdn_cuda/kernels/common/sm90_utils.cuh>

namespace gdn_cuda {
namespace kernels {
namespace sm90_layout_impl {

// transpose K major SF tensor to MN major SF tensor
template <size_t kNumThreads, size_t BLOCK_MN, size_t SF_K,
          size_t PADDED_SF_K = (SF_K + ((SF_K + 1) % 2))>
__global__ void transpose_fp32_sf(float *sf, float *out, size_t mn, size_t aligned_mn) {
    extern __shared__ float
        smem_buffer[];  // shape (BLOCK_MN< PADDED_SF_K) (to avoid shared memory bank conflicts)
    const int rem_rows = min(aligned_mn - blockIdx.x * BLOCK_MN, BLOCK_MN);
    using vec_load = VEC_LOAD<SF_K * sizeof(float) * 8, float>;
    using load_type = typename vec_load::ptr_type;
    constexpr int kNumVec = sizeof(load_type) / sizeof(float);
    // this is over batch indices / third dimension
    sf += blockIdx.y * mn * SF_K;
    out += blockIdx.y * aligned_mn * SF_K;

    auto sf_vec = reinterpret_cast<load_type *>(sf) + blockIdx.x * BLOCK_MN * SF_K / kNumVec;
    for (int i = threadIdx.x; i < rem_rows * SF_K / kNumVec; i += kNumThreads) {
        auto load = __ldg(sf_vec + i);  // float1/2/4
        float *values = reinterpret_cast<float *>(&load);
        int row = i * kNumVec / SF_K;
        int col = (i % (SF_K / kNumVec)) * kNumVec;
#pragma unroll
        for (int j = 0; j < kNumVec; j++) {
            // padding to prevent smem bank conflicts
            st_shared(smem_buffer + row * PADDED_SF_K + col + j, values[j]);
        }
    }
    __syncthreads();
    // once loaded into shared memory, calculate the per-thread amount, we transpose by making MN
    // major

    for (int i = threadIdx.x; i < rem_rows * SF_K; i += kNumThreads) {
        int row = i / SF_K;
        int col = i % SF_K;
        int global_mn = blockIdx.x * BLOCK_MN + row;
        out[col * aligned_mn + global_mn] = ld_shared(smem_buffer + row * PADDED_SF_K + col);
    }
}

// more generic version of the transpose_fp32_sf kernel
template <typename input_t, size_t kNumThreads, size_t BLOCK_MN, size_t SF_K,
          size_t PADDED_SF_K = (SF_K + ((SF_K + 1) % 2))>
__global__ void transpose_generic(input_t *sf, input_t *out, size_t mn, size_t aligned_mn) {
    extern __shared__ input_t
        smem_buffer[];  // shape (BLOCK_MN< PADDED_SF_K) (to avoid shared memory bank conflicts)
    const int rem_rows = min(aligned_mn - blockIdx.x * BLOCK_MN, BLOCK_MN);
    using vec_load = VEC_LOAD<SF_K * sizeof(input_t) * 8, input_t>;
    using load_type = typename vec_load::ptr_type;
    constexpr int kNumInputVec = vec_load::SIZE;
    // this is over batch indices / third dimension
    sf += blockIdx.y * mn * SF_K;
    out += blockIdx.y * aligned_mn * SF_K;

    auto sf_vec = reinterpret_cast<load_type *>(sf) + blockIdx.x * BLOCK_MN * SF_K / kNumInputVec;
    for (int i = threadIdx.x; i < rem_rows * SF_K / kNumInputVec; i += kNumThreads) {
        input_t load[kNumInputVec];

        load_type uint_load = ld_global_uint_dispatch(sf_vec + i);

        VEC_LOAD_CVT<load_type, input_t>::convert(uint_load, &load[0]);

        int row = i * kNumInputVec / SF_K;
        int col = (i % (SF_K / kNumInputVec)) * kNumInputVec;
#pragma unroll
        for (int j = 0; j < kNumInputVec; j++) {
            // padding to prevent smem bank conflicts
            st_shared(smem_buffer + row * PADDED_SF_K + col + j, load[j]);
        }
    }
    __syncthreads();
    // once loaded into shared memory, calculate the per-thread amount, we transpose by making MN
    // major

    for (int i = threadIdx.x; i < rem_rows * SF_K; i += kNumThreads) {
        int row = i / SF_K;
        int col = i % SF_K;
        int global_mn = blockIdx.x * BLOCK_MN + row;
        out[col * aligned_mn + global_mn] = ld_shared(smem_buffer + row * PADDED_SF_K + col);
    }
}

}  // namespace sm90_layout_impl
}  // namespace kernels
}  // namespace gdn_cuda
