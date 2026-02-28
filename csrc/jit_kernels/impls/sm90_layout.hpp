#pragma once
#include <gdn_cuda/device.hpp>
#include <gdn_cuda/format.hpp>
#include <jit/compiler.hpp>
#include <jit/runtime.hpp>
#include <jit/utils/culib.hpp>
#include <jit_kernels/heuristics/common.hpp>

class SM90_Transpose_SF_Runtime : LaunchRuntime<SM90_Transpose_SF_Runtime> {
   public:
    struct Args {
        uint32_t num_threads;
        uint32_t block_mn;
        uint32_t sf_k;

        float* sf;
        float* out;
        size_t mn, aligned_mn;

        cudaStream_t stream;
        LaunchConfig launch_config;
    };

    static std::string generate_impl(const Args& args) {
        const std::string code = fmt::format(R"(
            #include <gdn_cuda/kernels/sm90_layout.cuh>

            using namespace gdn_cuda::kernels::sm90_layout_impl;

            static void __instantiate_kernel() {{

            auto kernel_ptr = reinterpret_cast<void *>(&transpose_fp32_sf<
            {}, {}, {}
            >);
            }}
            )",
                                             args.num_threads, args.block_mn, args.sf_k);
        return code;
    }

    static void launch_impl(KernelHandle& kernel, const LaunchConfigHandle& launch_config,
                            const Args& args) {
        CUDA_CHECK(
            launch_kernel(kernel, launch_config, args.sf, args.out, args.mn, args.aligned_mn));
    }
};

class SM90_Transpose_BF16_Runtime : LaunchRuntime<SM90_Transpose_BF16_Runtime> {
   public:
    struct Args {
        uint32_t num_threads;
        uint32_t block_mn;
        uint32_t sf_k;

        __nv_bfloat16* sf;
        __nv_bfloat16* out;
        size_t mn, aligned_mn;

        cudaStream_t stream;
        LaunchConfig launch_config;
    };

    static std::string generate_impl(const Args& args) {
        const std::string code = fmt::format(R"(
            #include <gdn_cuda/kernels/sm90_layout.cuh>

            using namespace gdn_cuda::kernels::sm90_layout_impl;

            static void __instantiate_kernel() {{

            auto kernel_ptr = reinterpret_cast<void *>(&transpose_generic<
            __nv_bfloat16,
            {}, {}, {}
            >);
            }}
            )",
                                             args.num_threads, args.block_mn, args.sf_k);
        return code;
    }

    static void launch_impl(KernelHandle& kernel, const LaunchConfigHandle& launch_config,
                            const Args& args) {
        CUDA_CHECK(
            launch_kernel(kernel, launch_config, args.sf, args.out, args.mn, args.aligned_mn));
    }
};

inline void sm90_transpose_bf16(__nv_bfloat16* input, __nv_bfloat16* output, size_t mn, size_t k,
                                size_t num_groups, uint32_t alignment, cudaStream_t stream) {
    const auto [block_mn, num_threads, smem_size] =
        get_transpose_config(mn, k, sizeof(__nv_bfloat16));
    const size_t aligned_mn = ti_align(mn, alignment / 2);
    LaunchConfig launch_config = {dim3(num_threads), dim3(ti_ceil_div(mn, block_mn), num_groups),
                                  stream, smem_size, 1};

    SM90_Transpose_BF16_Runtime::Args args = {static_cast<uint32_t>(num_threads),
                                              (uint32_t)block_mn,
                                              static_cast<uint32_t>(k),
                                              input,
                                              output,
                                              mn,
                                              aligned_mn,
                                              stream,
                                              launch_config};

    const std::string& code = LaunchRuntime<SM90_Transpose_BF16_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_transpose_bf16", code);
    LaunchRuntime<SM90_Transpose_BF16_Runtime>::launch(runtime, args);
}

inline void sm90_transpose_fp32(float* input, float* output, size_t mn, size_t k, size_t num_groups,
                                uint32_t alignment, cudaStream_t stream) {
    const auto [block_mn, num_threads, smem_size] = get_transpose_config(mn, k);
    const size_t aligned_mn = ti_align(mn, alignment / 4);
    LaunchConfig launch_config = {dim3(num_threads), dim3(ti_ceil_div(mn, block_mn), num_groups),
                                  stream, smem_size, 1};

    SM90_Transpose_SF_Runtime::Args args = {static_cast<uint32_t>(num_threads),
                                            (uint32_t)block_mn,
                                            static_cast<uint32_t>(k),
                                            input,
                                            output,
                                            mn,
                                            aligned_mn,
                                            stream,
                                            launch_config};

    const std::string& code = LaunchRuntime<SM90_Transpose_SF_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_transpose_fp32", code);
    LaunchRuntime<SM90_Transpose_SF_Runtime>::launch(runtime, args);
}
