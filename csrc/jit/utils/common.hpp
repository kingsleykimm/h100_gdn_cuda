#pragma once
#include <cuda.h>
#include <gdn_cuda/utils.h>
#include <torch/torch.h>

#include <gdn_cuda/error.hpp>
#include <gdn_cuda/kernels/common/common.hpp>
#include <gdn_cuda/kernels/common/types.hpp>
#include <jit/utils/files.hpp>

struct GDNConfig {
    GDNType gdn_type;
    uint32_t block_v;
    uint32_t block_k;
    uint32_t swizzle_state_mode;
    uint32_t num_tma_threads;
    uint32_t num_math_threads;
    uint32_t num_blocks;
    uint32_t swizzle_k_mode;
    uint32_t swizzle_v_mode;
    uint32_t swizzle_a_mode;
    uint32_t smem_size;
    uint32_t num_sms;
    uint32_t num_tma_multicast;
    uint32_t chunk_block;
    uint32_t num_stages;
    uint32_t swizzle_q_mode;
    uint32_t swizzle_u_mode;
};

struct LaunchConfig {
    dim3 blockDim;
    dim3 gridDim;
    cudaStream_t stream;
    int smem_size;
    uint32_t num_multicast;
};

inline int get_type_size(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:
            return sizeof(float);
        case at::kHalf:
            return sizeof(half);
        case at::kBFloat16:
            return sizeof(__nv_bfloat16);
        case at::kInt:
            return sizeof(int);
        case at::kFloat8_e4m3fn:
            return 1;
        default:
            HOST_ERROR("Unsupported dtype => Type size");
            return 0;
    }
}

inline bool is_multicast_legal(const uint32_t& shape_dim, const uint32_t& block_dim,
                               const uint32_t& num_multicast, const uint32_t& num_sms,
                               bool require_divisible) {
    bool divisible = !require_divisible || ti_ceil_div(shape_dim, block_dim) % num_multicast == 0;
    return divisible && num_sms % num_multicast == 0;
}

inline int get_swizzle_mode(const uint32_t& block_size, const uint32_t& elem_size) {
    const uint32_t& num_elements_bytes = block_size * elem_size;
    for (const auto mode : {128, 64, 32, 16}) {
        if (num_elements_bytes % mode == 0)
            return mode;
    }
    std::string msg = "There does not exist a compatible swizzle mode for the current block size " +
                      std::to_string(block_size) + " and elem_size " + std::to_string(elem_size);
    HOST_ERROR(msg.c_str());
    return 0;
}

inline int get_compiled_dim(const std::string& compiled_dims, char dim, int dim_value) {
    for (const auto c : compiled_dims) {
        if (c == dim)
            return dim_value;
    }
    return 0;
}

static CUtensorMapDataType convert_to_cudtype(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case at::kHalf:
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
        case at::kBFloat16:
            return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        case at::kInt:
            return CU_TENSOR_MAP_DATA_TYPE_INT32;
        default:
            HOST_ERROR("Unsupported dtype => CuTensorMap conversion");
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    }
}

static CUtensorMapSwizzle getCuSwizzle(const int& swizzle_size) {
    switch (swizzle_size) {
        case 0:
        case 16:
            return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 32:
            return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64:
            return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128:
            return CU_TENSOR_MAP_SWIZZLE_128B;
        default:
            HOST_ERROR("getCuSwizzle: unsupported swizzle mode!");
            return CU_TENSOR_MAP_SWIZZLE_NONE;
    }
}

static CUtensorMapL2promotion getCuL2PromotionSize(const int& l2_promotion_size) {
    switch (l2_promotion_size) {
        case 0:
            return CU_TENSOR_MAP_L2_PROMOTION_NONE;
        case 64:
            return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
        case 128:
            return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
        default:
            return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    }
}

inline std::pair<int, int> get_inner_outer_dims(Major major, int mn, int k) {
    int inner = (major == Major::MN) ? mn : k;
    int outer = (major == Major::MN) ? k : mn;
    return std::make_pair(inner, outer);
}

inline bool is_mn_major(const at::Tensor& tensor) {
    if (tensor.dim() < 2) {
        return false;
    }
    return tensor.stride(-2) == 1;
}

static CUtensorMap make_tma_2d_desc(
    at::Tensor& t, size_t gmem_inner_dim, size_t gmem_outer_dim, size_t gmem_outer_stride,
    size_t smem_inner_dim, size_t smem_outer_dim, const int& swizzle_size,
    const int& l2_promotion_size,
    CUtensorMapFloatOOBfill_enum fill_mode = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
    HOST_ASSERT(t.data_ptr() != nullptr, "Tensor data is null");
    CUtensorMap map;
    size_t type_size = t.element_size();
    if (swizzle_size > 0) {
        smem_inner_dim = swizzle_size / type_size;
    }
    const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim),
                                     static_cast<cuuint64_t>(gmem_outer_dim)};
    const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * type_size)};
    const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim),
                                     static_cast<cuuint32_t>(smem_outer_dim)};
    CUtensorMapSwizzle swizzle_type = getCuSwizzle(swizzle_size);
    if (get_env<int>("JIT_DEBUG")) {
        printf(
            "Making TMA desc: global memory: %zu %zu, shared memory: %zu %zu, outer stride: %zu, "
            "swizzle: %d elem size: %zu\n",
            gmem_inner_dim, gmem_outer_dim, smem_inner_dim, smem_outer_dim,
            gmem_outer_stride * type_size, swizzle_size, type_size);
    }
    const cuuint32_t elem_strides[2] = {1, 1};
    CUDA_CHECK(cuTensorMapEncodeTiled(&map, convert_to_cudtype(t.scalar_type()), 2, t.data_ptr(),
                                      gmem_dims, gmem_strides, smem_dims, elem_strides,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_type,
                                      getCuL2PromotionSize(l2_promotion_size), fill_mode));
    return map;
}

static CUtensorMap make_tma_3d_desc(at::Tensor& t, size_t gmem_dim_0, size_t gmem_dim_1,
                                    size_t gmem_dim_2, size_t gmem_stride_1, size_t gmem_stride_2,
                                    size_t smem_dim_0, size_t smem_dim_1, size_t smem_dim_2,
                                    const int& swizzle_size, const int& l2_promotion_size) {
    HOST_ASSERT(t.data_ptr() != nullptr, "Tensor data is null");
    size_t type_size = t.element_size();
    if (swizzle_size > 0) {
        smem_dim_0 = swizzle_size / type_size;
    }
    CUtensorMap map;
    const cuuint64_t gmem_dims[3] = {static_cast<cuuint64_t>(gmem_dim_0),
                                     static_cast<cuuint64_t>(gmem_dim_1),
                                     static_cast<cuuint64_t>(gmem_dim_2)};
    const cuuint64_t gmem_strides[2] = {static_cast<cuuint64_t>(gmem_stride_1 * type_size),
                                        static_cast<cuuint64_t>(gmem_stride_2 * type_size)};
    const cuuint32_t smem_dims[3] = {static_cast<cuuint32_t>(smem_dim_0),
                                     static_cast<cuuint32_t>(smem_dim_1),
                                     static_cast<cuuint32_t>(smem_dim_2)};
    const cuuint32_t elem_strides[3] = {1, 1, 1};
    CUtensorMapSwizzle swizzle_type = getCuSwizzle(swizzle_size);
    if (get_env<int>("JIT_DEBUG")) {
        printf(
            "Making TMA desc: global memory: %zu %zu %zu, shared memory: %zu %zu %zu, outer "
            "stride: %zu %zu, swizzle: %d elem size: %zu \n",
            gmem_dim_0, gmem_dim_1, gmem_dim_2, smem_dim_0, smem_dim_1, smem_dim_2, gmem_stride_1,
            gmem_stride_2, swizzle_size, type_size);
    }
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &map, convert_to_cudtype(t.scalar_type()), 3, t.data_ptr(), gmem_dims, gmem_strides,
        smem_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_type,
        getCuL2PromotionSize(l2_promotion_size), CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

static CUtensorMap make_tma_4d_desc(at::Tensor& t, size_t gmem_dim_0, size_t gmem_dim_1,
                                    size_t gmem_dim_2, size_t gmem_dim_3, size_t gmem_stride_1,
                                    size_t gmem_stride_2, size_t gmem_stride_3, size_t smem_dim_0,
                                    size_t smem_dim_1, size_t smem_dim_2, size_t smem_dim_3,
                                    const int& swizzle_size, const int& l2_promotion_size) {
    HOST_ASSERT(t.data_ptr() != nullptr, "Tensor data is null");
    size_t type_size = t.element_size();
    if (swizzle_size > 0) {
        smem_dim_0 = swizzle_size / type_size;
    }
    CUtensorMap map;
    const cuuint64_t gmem_dims[4] = {
        static_cast<cuuint64_t>(gmem_dim_0), static_cast<cuuint64_t>(gmem_dim_1),
        static_cast<cuuint64_t>(gmem_dim_2), static_cast<cuuint64_t>(gmem_dim_3)};
    const cuuint64_t gmem_strides[3] = {static_cast<cuuint64_t>(gmem_stride_1 * type_size),
                                        static_cast<cuuint64_t>(gmem_stride_2 * type_size),
                                        static_cast<cuuint64_t>(gmem_stride_3 * type_size)};
    const cuuint32_t smem_dims[4] = {
        static_cast<cuuint32_t>(smem_dim_0), static_cast<cuuint32_t>(smem_dim_1),
        static_cast<cuuint32_t>(smem_dim_2), static_cast<cuuint32_t>(smem_dim_3)};
    const cuuint32_t elem_strides[4] = {1, 1, 1, 1};
    CUtensorMapSwizzle swizzle_type = getCuSwizzle(swizzle_size);
    if (get_env<int>("JIT_DEBUG")) {
        printf(
            "Making TMA desc: global memory: %zu %zu %zu %zu, shared memory: %zu %zu %zu %zu, "
            "outer stride: %zu %zu %zu, swizzle: %d elem size: %zu \n",
            gmem_dim_0, gmem_dim_1, gmem_dim_2, gmem_dim_3, smem_dim_0, smem_dim_1, smem_dim_2,
            smem_dim_3, gmem_stride_1, gmem_stride_2, gmem_stride_3, swizzle_size, type_size);
    }
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &map, convert_to_cudtype(t.scalar_type()), 4, t.data_ptr(), gmem_dims, gmem_strides,
        smem_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_type,
        getCuL2PromotionSize(l2_promotion_size), CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

inline std::vector<int> prepare_chunk_indices(const std::vector<int>& cu_seqlens,
                                              int chunk_size = 64) {
    alignas(8) std::vector<int> chunks(cu_seqlens.size() - 1);
    for (int i = 1; i < cu_seqlens.size(); i++) {
        int length = cu_seqlens[i] - cu_seqlens[i - 1];
        chunks[i - 1] = ti_ceil_div(length, chunk_size);
    }
    std::vector<int> chunk_indices;
    for (int i = 0; i < chunks.size(); i++) {
        int num_chunks = chunks[i];
        for (int j = 0; j < num_chunks; j++) {
            chunk_indices.push_back(i);
            chunk_indices.push_back(j);
        }
    }
    return chunk_indices;
}

inline std::vector<int> prepare_cu_chunks(const std::vector<int>& cu_seqlens, int chunk_size = 64,
                                          bool output_final_state = false) {
    alignas(8) std::vector<int> chunks(cu_seqlens.size() - 1);
    for (int i = 1; i < cu_seqlens.size(); i++) {
        int length = cu_seqlens[i] - cu_seqlens[i - 1];
        chunks[i - 1] = ti_ceil_div(length, chunk_size);
    }
    std::vector<int> cu_chunks;
    cu_chunks.push_back(0);
    int chunk_sum = 0;
    for (int i = 1; i < chunks.size() + 1; i++) {
        chunk_sum += chunks[i - 1];
        cu_chunks.push_back(chunk_sum);
    }
    return cu_chunks;
}
