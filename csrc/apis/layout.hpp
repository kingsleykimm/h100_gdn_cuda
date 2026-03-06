#pragma once
// Inline transpose-to-MN-major wrapper.
// Dispatches directly to jit_kernels/impls/sm90_layout.hpp.

#include <gdn_cuda/device.hpp>
#include <jit/utils/common.hpp>
#include <jit_kernels/impls/sm90_layout.hpp>

namespace gdn_cuda {
namespace api {

// default alignment to 16 bytes in case future  code wants to use TMA, nice alignment anyways
inline at::Tensor transpose_to_mn_major(at::Tensor& input, cudaStream_t stream,
                                        uint32_t alignment = 16) {
    HOST_ASSERT(input.dim() >= 2, "Input must have at least two dimensions");
    const int64_t mn = input.size(-2);
    const int64_t k = input.size(-1);
    const size_t elem_size = input.element_size();
    HOST_ASSERT(alignment % elem_size == 0, "Alignment must be divisible by tensor element size");
    const int64_t aligned_mn = ti_align(mn, alignment / elem_size);
    size_t num_groups = 1;
    for (int64_t i = 0; i < input.dim() - 2; i++) {
        num_groups *= static_cast<size_t>(input.size(i));
    }

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < input.dim() - 2; i++) {
        out_shape.push_back(input.size(i));
    }
    out_shape.push_back(aligned_mn);
    out_shape.push_back(k);

    std::vector<int64_t> out_strides(out_shape.size(), 1);
    out_strides[out_shape.size() - 2] = 1;
    out_strides[out_shape.size() - 1] = aligned_mn;
    int64_t group_stride = aligned_mn * k;
    for (int64_t i = static_cast<int64_t>(out_shape.size()) - 3; i >= 0; --i) {
        out_strides[i] = group_stride;
        group_stride *= out_shape[i];
    }

    auto opts = input.options();
    at::Tensor out_storage = at::empty_strided(out_shape, out_strides, opts);

    if (input.scalar_type() == at::kFloat) {
        sm90_transpose_fp32(input.data_ptr<float>(), out_storage.data_ptr<float>(), mn, k,
                            num_groups, alignment, stream);
    } else if (input.scalar_type() == at::kBFloat16) {
        sm90_transpose_bf16(reinterpret_cast<__nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                            reinterpret_cast<__nv_bfloat16*>(out_storage.data_ptr<at::BFloat16>()),
                            mn, k, num_groups, alignment, stream);
    } else {
        HOST_ERROR("transpose_to_mn_major: unsupported dtype (only FP32 and BF16)");
    }

    return out_storage;
}

// Transpose into a pre-allocated output tensor (no allocation).
// The output tensor must already have the correct shape/strides from a prior
// call to transpose_to_mn_major with the same input shape and alignment.
inline void transpose_to_mn_major_into(at::Tensor& input, at::Tensor& output, cudaStream_t stream,
                                       uint32_t alignment = 16) {
    HOST_ASSERT(input.dim() >= 2, "Input must have at least two dimensions");
    const int64_t mn = input.size(-2);
    const int64_t k = input.size(-1);
    size_t num_groups = 1;
    for (int64_t i = 0; i < input.dim() - 2; i++) {
        num_groups *= static_cast<size_t>(input.size(i));
    }

    if (input.scalar_type() == at::kFloat) {
        sm90_transpose_fp32(input.data_ptr<float>(), output.data_ptr<float>(), mn, k, num_groups,
                            alignment, stream);
    } else if (input.scalar_type() == at::kBFloat16) {
        sm90_transpose_bf16(reinterpret_cast<__nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), mn,
                            k, num_groups, alignment, stream);
    } else {
        HOST_ERROR("transpose_to_mn_major_into: unsupported dtype (only FP32 and BF16)");
    }
}

}  // namespace api
}  // namespace gdn_cuda
