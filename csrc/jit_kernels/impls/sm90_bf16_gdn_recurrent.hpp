
/*

some quick notes - remember that threads must be at least warpgroup-sized for tma loads and stores

remember that the minimum tma payload size is 128 bytes - this should be accounted for when loading
any of the QKV 1d vectors which are small and may not fit that requirement - this also plays in
tandem with the shared memory size
*/

#pragma once
#include <gdn_cuda/device.hpp>
#include <gdn_cuda/format.hpp>
#include <jit/compiler.hpp>
#include <jit/runtime.hpp>
#include <jit/utils/common.hpp>
#include <jit/utils/culib.hpp>
#include <jit_kernels/heuristics/common.hpp>

#include "jit_kernels/heuristics/sm90_arch.hpp"

// Heuristic for recurrent GDN kernel configuration

class SM90_BF16_GDN_Recurrent_Runtime : public LaunchRuntime<SM90_BF16_GDN_Recurrent_Runtime> {
   public:
    struct Args {
        // Template parameters
        uint32_t shape_k;
        uint32_t shape_v;
        uint32_t num_v_heads;
        uint32_t num_k_heads;
        uint32_t block_v;
        uint32_t num_blocks;
        GDNType gdn_type;
        uint32_t num_tma_threads;
        uint32_t num_math_threads;
        uint32_t seq_len;
        bool is_var_len;
        bool is_initial_state;
        bool store_step_state;
        bool is_qk_norm;
        bool use_gate;

        // Runtime arguments (passed to kernel)
        CUtensorMap q_tensor_map;
        CUtensorMap k_tensor_map;
        CUtensorMap v_tensor_map;
        CUtensorMap state_tensor_map;
        CUtensorMap final_state_tensor_map;
        __nv_bfloat16* out;
        __nv_bfloat16* beta;
        __nv_bfloat16* gate;
        int* num_accepted_tokens;
        int batch_size;
        int* cu_seqlens;
        float scale;

        // JIT compile settings
        std::string compiled_dims;
        LaunchConfig launch_config;
    };

    static std::string generate_impl(const Args& args) {
        const std::string code = fmt::format(
            R"(
#include <gdn_cuda/kernels/sm90_bf16_gdn_recurrent.cuh>

using namespace gdn_cuda::kernels::sm90_gdn_recurrent_impl;

using GdnRecurrentKernelPtr = decltype(&fused_recurrent_gated_delta_rule_bf16<
    {}, {}, {}, {},
    {}, {},
    {}, {}, {},
    {}, {}, {}, {}, {}, {}
>);

extern "C" __attribute__((used)) GdnRecurrentKernelPtr __gdn_recurrent_kernel_ref =
    &fused_recurrent_gated_delta_rule_bf16<
        {}, {}, {}, {},
        {}, {},
        {}, {}, {},
        {}, {}, {}, {}, {}, {}
    >;
            )",
            get_compiled_dim(args.compiled_dims, 'k', args.shape_k),
            get_compiled_dim(args.compiled_dims, 'v', args.shape_v), args.num_v_heads,
            args.num_k_heads, args.block_v, args.num_blocks, args.launch_config.num_multicast,
            args.num_tma_threads, args.num_math_threads,
            get_compiled_dim(args.compiled_dims, 't', args.seq_len),
            args.is_var_len ? "true" : "false", args.is_initial_state ? "true" : "false",
            args.store_step_state ? "true" : "false", args.is_qk_norm ? "true" : "false",
            args.use_gate ? "true" : "false",
            get_compiled_dim(args.compiled_dims, 'k', args.shape_k),
            get_compiled_dim(args.compiled_dims, 'v', args.shape_v), args.num_v_heads,
            args.num_k_heads, args.block_v, args.num_blocks, args.launch_config.num_multicast,
            args.num_tma_threads, args.num_math_threads,
            get_compiled_dim(args.compiled_dims, 't', args.seq_len),
            args.is_var_len ? "true" : "false", args.is_initial_state ? "true" : "false",
            args.store_step_state ? "true" : "false", args.is_qk_norm ? "true" : "false",
            args.use_gate ? "true" : "false");
        return code;
    }

    static void launch_impl(KernelHandle& kernel, const LaunchConfigHandle& launch_config,
                            const Args& args) {
        CUDA_CHECK(launch_kernel(
            kernel, launch_config, args.q_tensor_map, args.k_tensor_map, args.v_tensor_map,
            args.state_tensor_map, args.final_state_tensor_map, args.out, args.beta, args.gate,
            args.num_accepted_tokens, args.batch_size, static_cast<int>(args.shape_k),
            static_cast<int>(args.shape_v), args.cu_seqlens, args.scale));
    }
};

// API expectations:
// - Purpose: recurrent decode/spec-verify step with optional gating.
// - Padded mode expects rank-4 q/k/v and rank-4 state; varlen expects rank-3 q/k/v with cu_seqlens.
// - DTypes: q/k/v/state/final_state/out/beta are BF16; gate (optional) follows recurrent kernel
// requirements.
// - `out` and `final_state` are written on the provided stream.
inline void sm90_bf16_gdn_recurrent(
    at::Tensor& q,  // [batch, seq_len, num_k_heads, shape_v]
    at::Tensor& k, at::Tensor& v, std::optional<at::Tensor>& initial_state, at::Tensor& final_state,
    at::Tensor& out,                  // [batch, num_v_heads, shape_v] output
    std::optional<at::Tensor>& gate,  // [batch, seq_len, num_v_heads] gating values
    at::Tensor& beta, const std::string& compiled_dims, cudaStream_t stream,
    std::optional<at::Tensor>& cu_seqlens, std::optional<at::Tensor>& num_accepted_tokens,
    bool store_step_state = false, bool is_qk_norm = false, float scale = 1.0f) {
    bool is_var_len = cu_seqlens.has_value();
    bool is_initial_state = initial_state.has_value();
    if (is_var_len) {
        // Variable length: tensors are (total_seq_len, num_heads, head_dim)
        HOST_ASSERT(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                    "Variable length expects 3D tensors (total_seq, heads, dim)");
        if (is_initial_state) {
            HOST_ASSERT(initial_state->dim() == 4,
                        "State should be 4D (batch, heads, v_dim, k_dim)");
        }

        at::Tensor& cu_seqlens_tensor = cu_seqlens.value();
        HOST_ASSERT(cu_seqlens_tensor.dim() == 1, "cu_seqlens must be a 1D tensor");

        const uint32_t batch_size = static_cast<uint32_t>(cu_seqlens_tensor.size(0) - 1);
        const uint32_t total_seq_len = static_cast<uint32_t>(q.size(0));
        const uint32_t num_k_heads = static_cast<uint32_t>(q.size(1));
        const uint32_t num_v_heads = static_cast<uint32_t>(v.size(1));
        const uint32_t shape_k = static_cast<uint32_t>(q.size(2));
        const uint32_t shape_v = static_cast<uint32_t>(v.size(2));

        uint32_t state_size = batch_size;
        if (store_step_state) {
            state_size = total_seq_len;
        }
        // Get kernel configuration
        GDNConfig gdn_config =
            get_recurrent_config(shape_k, shape_v, num_v_heads, num_k_heads, batch_size);
        // Create TMA descriptors
        // Q: (total_seq_len, num_k_heads, shape_k) - load 1D vectors of shape_k
        CUtensorMap q_tensor_map = make_tma_2d_desc(q, shape_k, num_k_heads * total_seq_len,
                                                    q.stride(1), shape_k, 1, 0, 0);
        // K: (total_seq_len, num_k_heads, shape_k)
        CUtensorMap k_tensor_map =
            make_tma_2d_desc(k, shape_k, num_k_heads * total_seq_len, k.stride(1), shape_k, 1, 0,
                             ti_align(shape_k * 2, 64));
        // V: (total_seq_len, num_v_heads, shape_v) - load 1D vectors of block_v
        CUtensorMap v_tensor_map =
            make_tma_2d_desc(v, shape_v, num_v_heads * total_seq_len, v.stride(1),
                             gdn_config.block_v, 1, 0, ti_align(gdn_config.block_v * 2, 64));
        // State: (batch, num_v_heads, shape_v, shape_k) - load 2D blocks, no swizzle or
        // (total_tokens, num-v_heads, shape_v, shape_k)
        CUtensorMap state_tensor_map =
            is_initial_state
                ? make_tma_4d_desc(initial_state.value(), shape_k, shape_v, num_v_heads, batch_size,
                                   initial_state->stride(-2), initial_state->stride(-3),
                                   initial_state->stride(-4), shape_k, gdn_config.block_v, 1, 1, 0,
                                   ti_align(shape_k * 2, 64))
                : CUtensorMap();
        // Step State: (batch, num_v_heads, shape_v, shape_k) - load 2D blocks, no swizzle
        CUtensorMap final_state_tensor_map =
            make_tma_4d_desc(final_state, shape_k, shape_v, num_v_heads, state_size,
                             final_state.stride(-2), final_state.stride(-3), final_state.stride(-4),
                             shape_k, gdn_config.block_v, 1, 1, 0, ti_align(shape_k * 2, 64));
        // Launch configuration
        LaunchConfig launch_config;
        launch_config.blockDim =
            dim3(gdn_config.num_math_threads + gdn_config.num_tma_threads, 1, 1);
        launch_config.gridDim = dim3(gdn_config.num_blocks, 1, 1);
        launch_config.stream = stream;
        launch_config.smem_size = static_cast<int>(gdn_config.smem_size);
        launch_config.num_multicast = 1;

        // cu_seqlens is already on device as a tensor
        int* d_cu_seqlens = cu_seqlens_tensor.data_ptr<int>();

        // Build Args
        SM90_BF16_GDN_Recurrent_Runtime::Args args;
        args.shape_k = shape_k;
        args.shape_v = shape_v;
        args.num_v_heads = num_v_heads;
        args.num_k_heads = num_k_heads;
        args.block_v = gdn_config.block_v;
        args.num_blocks = gdn_config.num_blocks;
        args.gdn_type = GDNType::Recurrent;
        args.num_tma_threads = gdn_config.num_tma_threads;
        args.num_math_threads = gdn_config.num_math_threads;
        args.seq_len = 0;  // varlen doesn't use fixed seq_len
        args.is_var_len = true;
        args.is_initial_state = is_initial_state;
        args.is_qk_norm = is_qk_norm;
        args.store_step_state = store_step_state;
        args.use_gate = gate.has_value();
        args.q_tensor_map = q_tensor_map;
        args.k_tensor_map = k_tensor_map;
        args.v_tensor_map = v_tensor_map;
        args.state_tensor_map = state_tensor_map;
        args.final_state_tensor_map = final_state_tensor_map;
        args.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
        args.beta = reinterpret_cast<__nv_bfloat16*>(beta.data_ptr());
        args.gate = gate.has_value() ? reinterpret_cast<__nv_bfloat16*>(gate->data_ptr()) : nullptr;
        args.num_accepted_tokens =
            num_accepted_tokens.has_value() ? num_accepted_tokens->data_ptr<int>() : nullptr;
        args.batch_size = static_cast<int>(batch_size);
        args.cu_seqlens = d_cu_seqlens;
        args.scale = scale;
        args.compiled_dims = compiled_dims;
        args.launch_config = launch_config;

        // Generate, build, and launch
        const std::string code = LaunchRuntime<SM90_BF16_GDN_Recurrent_Runtime>::generate(args);
        std::shared_ptr<KernelRuntime> runtime =
            compiler->build("sm90_bf16_gdn_recurrent_varlen", code);
        LaunchRuntime<SM90_BF16_GDN_Recurrent_Runtime>::launch(runtime, args);
    } else {
        // Fixed length: tensors are (batch, seq_len, num_heads, head_dim)
        HOST_ASSERT(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                    "Fixed length expects 4D tensors (batch, seq, heads, dim)");
        if (is_initial_state) {
            HOST_ASSERT(initial_state->dim() == 4,
                        "State should be 4D (batch, heads, k_dim, v_dim)");
        }

        const uint32_t batch_size = static_cast<uint32_t>(q.size(0));
        const uint32_t seq_len = static_cast<uint32_t>(q.size(1));
        const uint32_t num_k_heads = static_cast<uint32_t>(q.size(2));
        const uint32_t num_v_heads = static_cast<uint32_t>(v.size(2));
        const uint32_t shape_k = static_cast<uint32_t>(q.size(3));
        const uint32_t shape_v = static_cast<uint32_t>(v.size(3));

        const uint32_t total_seq_len = batch_size * seq_len;
        uint32_t state_size = batch_size;
        if (store_step_state) {
            state_size = total_seq_len;
        }
        // Get kernel configuration

        GDNConfig gdn_config =
            get_recurrent_config(shape_k, shape_v, num_v_heads, num_k_heads, batch_size);

        // Create TMA descriptors - reshape 4D to 3D view for TMA
        // Q: viewed as (batch*seq, num_k_heads, shape_k)
        CUtensorMap q_tensor_map = make_tma_2d_desc(q, shape_k, num_k_heads * total_seq_len,
                                                    q.stride(2), shape_k, 1, 0, 0);
        // K: (total_seq_len, num_k_heads, shape_k)
        CUtensorMap k_tensor_map =
            make_tma_2d_desc(k, shape_k, num_k_heads * total_seq_len, k.stride(2), shape_k, 1, 0,
                             ti_align(shape_k * 2, 64));
        // V: (total_seq_len, num_v_heads, shape_v) - load 1D vectors of block_v
        CUtensorMap v_tensor_map =
            make_tma_2d_desc(v, shape_v, num_v_heads * total_seq_len, v.stride(2),
                             gdn_config.block_v, 1, 0, ti_align(gdn_config.block_v * 2, 64));
        // State: (batch, num_v_heads, shape_v, shape_k) - load 2D blocks, no swizzle
        CUtensorMap state_tensor_map =
            is_initial_state
                ? make_tma_4d_desc(initial_state.value(), shape_k, shape_v, num_v_heads, batch_size,
                                   initial_state->stride(-2), initial_state->stride(-3),
                                   initial_state->stride(-4), shape_k, gdn_config.block_v, 1, 1, 0,
                                   ti_align(shape_k * 2, 64))
                : CUtensorMap();
        // Step State: (batch, num_v_heads, shape_v, shape_k) - load 2D blocks, no swizzle
        CUtensorMap final_state_tensor_map =
            make_tma_4d_desc(final_state, shape_k, shape_v, num_v_heads, state_size,
                             final_state.stride(-2), final_state.stride(-3), final_state.stride(-4),
                             shape_k, gdn_config.block_v, 1, 1, 0, ti_align(shape_k * 2, 64));

        uint32_t total_work = ti_ceil_div(shape_v, gdn_config.block_v) * num_v_heads * batch_size;

        // Verify grid divisibility by cluster
        HOST_ASSERT(gdn_config.num_blocks % gdn_config.num_tma_multicast == 0,
                    "Grid dimension must be divisible by cluster dimension!");

        // Launch configuration
        LaunchConfig launch_config;
        launch_config.blockDim =
            dim3(gdn_config.num_math_threads + gdn_config.num_tma_threads, 1, 1);
        launch_config.gridDim = dim3(gdn_config.num_blocks, 1, 1);
        launch_config.stream = stream;
        launch_config.smem_size = static_cast<int>(gdn_config.smem_size);
        launch_config.num_multicast = 1;

        // Build Args
        SM90_BF16_GDN_Recurrent_Runtime::Args args;
        args.shape_k = shape_k;
        args.shape_v = shape_v;
        args.num_v_heads = num_v_heads;
        args.num_k_heads = num_k_heads;
        args.block_v = gdn_config.block_v;
        args.num_blocks = gdn_config.num_blocks;
        args.gdn_type = GDNType::Recurrent;
        args.num_tma_threads = gdn_config.num_tma_threads;
        args.num_math_threads = gdn_config.num_math_threads;
        args.seq_len = seq_len;
        args.is_var_len = false;
        args.is_initial_state = is_initial_state;
        args.is_qk_norm = is_qk_norm;
        args.store_step_state = store_step_state;
        args.use_gate = gate.has_value();
        args.q_tensor_map = q_tensor_map;
        args.k_tensor_map = k_tensor_map;
        args.v_tensor_map = v_tensor_map;
        args.state_tensor_map = state_tensor_map;
        args.final_state_tensor_map = final_state_tensor_map;
        args.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
        args.beta = reinterpret_cast<__nv_bfloat16*>(beta.data_ptr());
        args.gate = gate.has_value() ? reinterpret_cast<__nv_bfloat16*>(gate->data_ptr()) : nullptr;
        args.num_accepted_tokens =
            num_accepted_tokens.has_value() ? num_accepted_tokens->data_ptr<int>() : nullptr;
        args.batch_size = static_cast<int>(batch_size);
        args.cu_seqlens = nullptr;
        args.scale = scale;
        args.compiled_dims = compiled_dims;
        args.launch_config = launch_config;

        // Generate, build, and launch
        const std::string code = LaunchRuntime<SM90_BF16_GDN_Recurrent_Runtime>::generate(args);
        std::shared_ptr<KernelRuntime> runtime =
            compiler->build("sm90_bf16_gdn_recurrent_padded", code);
        LaunchRuntime<SM90_BF16_GDN_Recurrent_Runtime>::launch(runtime, args);
    }
}
