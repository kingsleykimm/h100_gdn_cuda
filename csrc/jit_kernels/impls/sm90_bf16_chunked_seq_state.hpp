#pragma once
#include <gdn_cuda/device.hpp>
#include <gdn_cuda/format.hpp>
#include <jit/compiler.hpp>
#include <jit/runtime.hpp>
#include <jit/utils/common.hpp>
#include <jit/utils/culib.hpp>
#include <jit_kernels/heuristics/common.hpp>

#include "jit_kernels/heuristics/sm90_arch.hpp"

class SM90_BF16_GDN_State_Passing_Runtime : LaunchRuntime<SM90_BF16_GDN_State_Passing_Runtime> {
   public:
    struct Args {
        uint32_t batch_size, chunk_size;
        uint32_t num_k_heads, num_v_heads;
        uint32_t shape_k, block_k, shape_v, block_v;
        uint32_t swizzle_k_mode, swizzle_u_mode, swizzle_w_mode, swizzle_s_mode;
        uint32_t num_math_threads, num_tma_threads;
        uint32_t num_blocks;
        uint32_t seq_len;
        bool is_var_len, is_initial_state, output_final_state, use_gate;

        CUtensorMap k_tensor_map, u_tensor_map, w_tensor_map, initial_state_tensor_map,
            state_tensor_map, final_state_tensor_map;
        int shape_T;
        int* cu_seqlens;
        int* cu_chunks;
        __nv_bfloat16* U_ptr;
        float* gate_ptr;
        uint32_t gate_stride;

        std::string compiled_dims;
        LaunchConfig launch_config;
    };

    static std::string generate_impl(const Args& args) {
        const std::string code = fmt::format(
            R"(
#include <gdn_cuda/kernels/sm90_bf16_chunked_seq_state_update2.cuh>

static void __instantiate_kernel() {{
    auto kernel_ptr = reinterpret_cast<void *>(&sm90_bf16_chunked_seq_state_update<
        {}, {},
        {}, {},
        {}, {}, {}, {},
        {}, {}, {}, {},
        {}, {}, {}, {},
        {}, {}, {}, {}, {}
    >);
}}
            )",
            get_compiled_dim(args.compiled_dims, 'b', args.batch_size), args.chunk_size,
            args.num_k_heads, args.num_v_heads, args.shape_k, args.block_k, args.shape_v,
            args.block_v, args.swizzle_k_mode, args.swizzle_u_mode, args.swizzle_w_mode,
            args.swizzle_s_mode, args.num_math_threads, args.num_tma_threads, args.num_blocks,
            args.launch_config.num_multicast, args.is_initial_state, args.is_var_len, args.seq_len,
            args.output_final_state, args.use_gate);
        return code;
    }

    static void launch_impl(KernelHandle& kernel, const LaunchConfigHandle& launch_config,
                            const Args& args) {
        CUDA_CHECK(launch_kernel(kernel, launch_config, args.state_tensor_map, args.k_tensor_map,
                                 args.w_tensor_map, args.u_tensor_map,
                                 args.initial_state_tensor_map, args.final_state_tensor_map,
                                 args.U_ptr, args.gate_ptr, args.gate_stride, args.batch_size,
                                 args.shape_T, args.cu_seqlens, args.cu_chunks));
    }
};

// API expectations:
// - Purpose: update recurrent chunked state with precomputed U/W and optional gate.
// - Padded mode expects rank-4 k/u/w and rank-5 state; varlen expects rank-3 k/u/w and rank-4
// state.
// - Varlen requires valid cu_seqlens; gate is optional and must be MN-major when provided.
// - DTypes: k/u/w/state/final_state are BF16; gate is FP32.
// - Mutates `state` and writes `final_state` (and may update `u`) on the provided stream.
inline void sm90_bf16_chunked_seq_state_update(
    at::Tensor& k,  // (batch_size, seq_len, num_k_heads, k_head_dim) or (total_seq_len,
                    // num_k_heads, k_head_dim)
    at::Tensor& u, at::Tensor& w, std::optional<at::Tensor>& initial_state,
    at::Tensor& state,  // per chunk state (num_chunks, num_v_heads, shape_v, shape_k)
    std::optional<at::Tensor>& final_state, std::optional<at::Tensor>& gate,
    const std::string& compiled_dims, cudaStream_t stream, std::optional<at::Tensor>& cu_seqlens,
    std::optional<at::Tensor>& cu_chunks, std::optional<int> total_chunks,
    const uint32_t chunk_size = 64) {
    bool is_initial_state = initial_state.has_value();
    bool output_final_state = final_state.has_value();
    HOST_ASSERT(!gate.has_value() || is_mn_major(gate.value()),
                "Gate must satisfy MN-major stride contract");
    HOST_ASSERT(k.scalar_type() == at::kBFloat16 && u.scalar_type() == at::kBFloat16 &&
                    w.scalar_type() == at::kBFloat16 && state.scalar_type() == at::kBFloat16 &&
                    final_state->scalar_type() == at::kBFloat16,
                "sm90_bf16_chunked_seq_state_update expects BF16 tensors for "
                "k/u/w/state/final_state (for TMA loads)");
    bool is_var_len = cu_seqlens.has_value();
    bool use_gate = gate.has_value();
    if (is_var_len) {
        // Variable length sequences: tensors are (total_seq_len, num_heads, head_dim)
        HOST_ASSERT(k.dim() == 3, "Variable length expects 3D tensors (total_seq, heads, dim)");
        HOST_ASSERT(state.dim() == 4,
                    "Varlen expects state tensor to have shape (total_chunks, num_v_heads, "
                    "shape_v, shape_k)");

        at::Tensor& cu_seqlens_tensor = cu_seqlens.value();
        HOST_ASSERT(cu_seqlens_tensor.dim() == 1, "cu_seqlens must be a 1D tensor");

        const uint32_t batch_size = static_cast<uint32_t>(cu_seqlens_tensor.size(0) - 1);
        const uint32_t total_seq_len = static_cast<uint32_t>(k.size(0));
        const uint32_t num_k_heads = static_cast<uint32_t>(k.size(1));
        const uint32_t num_v_heads = static_cast<uint32_t>(u.size(1));
        const uint32_t k_head_dim = static_cast<uint32_t>(k.size(2));
        const uint32_t v_head_dim = static_cast<uint32_t>(u.size(2));
        // Get kernel configuration from heuristics
        // chunk_indices stores pairs (batch_idx, chunk_idx), so actual num_chunks = size / 2
        // Parameters: shape_k, shape_v, num_v_heads, num_k_heads, batch_size, seq_len, num_chunks,
        // chunk_size
        GDNConfig gdn_config = get_seq_state_config<true>(
            k_head_dim, v_head_dim, num_v_heads, num_k_heads, batch_size, 0, chunk_size, use_gate);

        // Create TMA descriptors for all tensors
        // K tensor: (total_seq_len, num_k_heads, k_head_dim)
        // remember to revise these l2 promotion sizes -
        CUtensorMap k_tensor_map =
            make_tma_3d_desc(k, k_head_dim, num_k_heads, total_seq_len, k.stride(1), k.stride(0),
                             k_head_dim, 1, chunk_size, gdn_config.swizzle_k_mode,
                             ti_align(k_head_dim * get_type_size(k.scalar_type()), 64));
        // V tensor: (total_seq_len, num_v_heads, v_head_dim)
        CUtensorMap u_tensor_map =
            make_tma_3d_desc(u, v_head_dim, num_v_heads, total_seq_len, u.stride(1), u.stride(0),
                             gdn_config.block_v, 1, chunk_size, gdn_config.swizzle_v_mode,
                             ti_align(gdn_config.block_v * get_type_size(u.scalar_type()), 64));
        // W tensor: same layout as K (output)
        CUtensorMap w_tensor_map =
            make_tma_3d_desc(w, k_head_dim, num_v_heads, total_seq_len, w.stride(1), w.stride(0),
                             k_head_dim, 1, chunk_size, gdn_config.swizzle_k_mode,
                             ti_align(gdn_config.block_k * get_type_size(w.scalar_type()), 64));
        // State tensor shape: (batch, total_chunks, num_v_heads, v_head_dim, k_head_dim)
        // For 4D TMA, we merge batch and total_chunks into the outermost dimension
        // Note: smem tile dim0 must match TMA_S_BLOCK_N = swizzle_state_mode / sizeof(bf16) used in
        // kernel's split TMA stores
        uint32_t tma_s_block_n = gdn_config.swizzle_state_mode / get_type_size(state.scalar_type());
        uint32_t tma_fin_s_block_n =
            gdn_config.swizzle_state_mode / get_type_size(final_state->scalar_type());

        CUtensorMap initial_state_tensor_map =
            is_initial_state
                ? make_tma_4d_desc(
                      initial_state.value(), initial_state->size(-1), initial_state->size(-2),
                      initial_state->size(-3), batch_size, initial_state->stride(2),
                      initial_state->stride(1), initial_state->stride(0), tma_s_block_n,
                      gdn_config.block_v, 1,
                      1,  // smem_dim_3 = 1 since we load one (batch, chunk) at a time
                      gdn_config.swizzle_state_mode,
                      ti_align(tma_s_block_n * get_type_size(initial_state->scalar_type()), 64))
                : CUtensorMap();

        CUtensorMap state_tensor_map = make_tma_4d_desc(
            state, state.size(-1), state.size(-2), state.size(-3), total_chunks.value(),
            state.stride(-2), state.stride(-3), state.stride(-4), tma_s_block_n, gdn_config.block_v,
            1, 1,  // smem_dim_3 = 1 since we load one (batch, chunk) at a time
            gdn_config.swizzle_state_mode,
            ti_align(tma_s_block_n * get_type_size(state.scalar_type()), 64));
        // Final state tensor shape: (batch, num_v_heads, v_head_dim, k_head_dim)
        CUtensorMap final_state_tensor_map =
            output_final_state
                ? make_tma_4d_desc(
                      final_state.value(), final_state->size(-1), final_state->size(-2),
                      final_state->size(-3), batch_size, final_state->stride(-2),
                      final_state->stride(-3), final_state->stride(-4), tma_fin_s_block_n,
                      gdn_config.block_v, 1, 1, gdn_config.swizzle_state_mode,
                      ti_align(tma_fin_s_block_n * get_type_size(final_state->scalar_type()), 64))
                : CUtensorMap();

        // Beta tensor: (total_seq_len, num_v_heads, v_head_dim)
        // Set up launch configuration - persistent scheduler uses num_sms as grid size
        LaunchConfig launch_config = {
            dim3(gdn_config.num_math_threads + gdn_config.num_tma_threads, 1, 1),
            dim3(gdn_config.num_blocks, 1, 1), stream, static_cast<int>(gdn_config.smem_size),
            1  // num_multicast
        };

        // Build Args struct
        SM90_BF16_GDN_State_Passing_Runtime::Args args;
        args.shape_k = k_head_dim;
        args.shape_v = v_head_dim;
        args.batch_size = batch_size;
        args.chunk_size = chunk_size;
        args.num_k_heads = num_k_heads;
        args.num_v_heads = num_v_heads;
        args.block_k = gdn_config.block_k;
        args.block_v = gdn_config.block_v;
        args.num_blocks = gdn_config.num_blocks;
        args.num_math_threads = gdn_config.num_math_threads;
        args.num_tma_threads = gdn_config.num_tma_threads;
        args.swizzle_k_mode = gdn_config.swizzle_k_mode;
        args.swizzle_u_mode = gdn_config.swizzle_v_mode;
        args.swizzle_s_mode = gdn_config.swizzle_state_mode;
        args.swizzle_w_mode = gdn_config.swizzle_k_mode;
        args.seq_len = 1;  // varlen doesn't use fixed seq_len
        args.is_var_len = true;
        args.use_gate = gate.has_value();
        args.k_tensor_map = k_tensor_map;
        args.u_tensor_map = u_tensor_map;
        args.w_tensor_map = w_tensor_map;
        args.initial_state_tensor_map = initial_state_tensor_map;
        args.state_tensor_map = state_tensor_map;
        args.final_state_tensor_map = final_state_tensor_map;
        args.shape_T = total_seq_len;
        args.cu_seqlens = cu_seqlens_tensor.data_ptr<int>();
        args.compiled_dims = compiled_dims;
        args.launch_config = launch_config;
        args.is_initial_state = is_initial_state;
        args.output_final_state = output_final_state;
        args.cu_chunks = cu_chunks.value().data_ptr<int>();
        args.U_ptr = reinterpret_cast<__nv_bfloat16*>(u.data_ptr());
        args.gate_ptr = gate.has_value() ? gate.value().data_ptr<float>() : nullptr;
        args.gate_stride = gate.has_value() ? gate.value().size(0) : 0;

        // Generate, build, and launch kernel
        const std::string code = LaunchRuntime<SM90_BF16_GDN_State_Passing_Runtime>::generate(args);
        std::shared_ptr<KernelRuntime> runtime =
            compiler->build("sm90_bf16_chunked_seq_state_update_varlen", code);
        LaunchRuntime<SM90_BF16_GDN_State_Passing_Runtime>::launch(runtime, args);

        // Free device memory after kernel completes

    } else {
        // Fixed length sequences: tensors are (batch_size, seq_len, num_heads, head_dim)
        // state shape is (batch_size, num_chunks, num_v_heads, shape_v, block_k)
        HOST_ASSERT(k.dim() == 4, "Fixed length expects 4D tensors (batch, seq, heads, dim)");
        const uint32_t batch_size = static_cast<uint32_t>(k.size(0));
        const uint32_t seq_len = static_cast<uint32_t>(k.size(1));
        const uint32_t num_k_heads = static_cast<uint32_t>(k.size(2));
        const uint32_t num_v_heads = static_cast<uint32_t>(u.size(2));
        const uint32_t k_head_dim = static_cast<uint32_t>(k.size(3));
        const uint32_t v_head_dim = static_cast<uint32_t>(u.size(3));
        uint32_t num_chunks = ti_ceil_div(seq_len, chunk_size);

        // Get kernel configuration from heuristics
        // Parameters: shape_k, shape_v, num_v_heads, num_k_heads, batch_size, seq_len, num_chunks,
        // chunk_size
        GDNConfig gdn_config = get_seq_state_config<false>(
            k_head_dim, v_head_dim, num_v_heads, num_k_heads, batch_size, seq_len, 64, use_gate);

        // Create TMA descriptors - reshape 4D to 3D view for TMA
        // K tensor: viewed as (batch*seq, num_k_heads, k_head_dim)
        CUtensorMap k_tensor_map = make_tma_4d_desc(
            k, k_head_dim, num_k_heads, seq_len, batch_size, k.stride(2), k.stride(1), k.stride(0),
            k_head_dim, 1, chunk_size, 1, gdn_config.swizzle_k_mode,
            ti_align(k_head_dim * get_type_size(k.scalar_type()), 64));
        // U tensor: same layout as V (output)
        CUtensorMap u_tensor_map = make_tma_4d_desc(
            u, v_head_dim, num_v_heads, seq_len, batch_size, u.stride(2), u.stride(1), u.stride(0),
            gdn_config.block_v, 1, chunk_size, 1, gdn_config.swizzle_v_mode,
            ti_align(gdn_config.block_v * get_type_size(u.scalar_type()), 64));
        // W tensor: same layout as K (output)
        CUtensorMap w_tensor_map = make_tma_4d_desc(
            w, k_head_dim, num_v_heads, seq_len, batch_size, w.stride(2), w.stride(1), w.stride(0),
            k_head_dim, 1, chunk_size, 1, gdn_config.swizzle_k_mode,
            ti_align(k_head_dim * get_type_size(w.scalar_type()), 64));

        // Note: smem tile dim0 must match TMA_S_BLOCK_N = swizzle_state_mode / sizeof(bf16) used in
        // kernel's split TMA stores
        uint32_t tma_s_block_n = gdn_config.swizzle_state_mode / get_type_size(state.scalar_type());
        uint32_t tma_fin_s_block_n =
            gdn_config.swizzle_state_mode / get_type_size(final_state->scalar_type());

        CUtensorMap initial_state_tensor_map =
            is_initial_state
                ? make_tma_4d_desc(
                      initial_state.value(), initial_state->size(-1), initial_state->size(-2),
                      initial_state->size(-3), batch_size, initial_state->stride(-2),
                      initial_state->stride(-3), initial_state->stride(-4), tma_s_block_n,
                      gdn_config.block_v, 1,
                      1,  // smem_dim_3 = 1 since we load one (batch, chunk) at a time
                      gdn_config.swizzle_state_mode,
                      ti_align(tma_s_block_n * get_type_size(initial_state->scalar_type()), 64))
                : CUtensorMap();

        CUtensorMap state_tensor_map = make_tma_4d_desc(
            state, state.size(-1), state.size(-2), state.size(-3), num_chunks * batch_size,
            state.stride(-2), state.stride(-3), state.stride(-4), tma_s_block_n, gdn_config.block_v,
            1, 1,  // smem_dim_3 = 1 since we load one (batch, chunk) at a time
            gdn_config.swizzle_state_mode,
            ti_align(tma_s_block_n * get_type_size(state.scalar_type()), 64));

        // Final state tensor shape: (batch, num_v_heads, v_head_dim, k_head_dim)
        CUtensorMap final_state_tensor_map =
            output_final_state
                ? make_tma_4d_desc(
                      final_state.value(), final_state->size(-1), final_state->size(-2),
                      final_state->size(-3), batch_size, final_state->stride(-2),
                      final_state->stride(-3), final_state->stride(-4), tma_fin_s_block_n,
                      gdn_config.block_v, 1, 1, gdn_config.swizzle_state_mode,
                      ti_align(tma_fin_s_block_n * get_type_size(final_state->scalar_type()), 64))
                : CUtensorMap();

        // Set up launch configuration - persistent scheduler uses num_sms as grid size
        LaunchConfig launch_config = {
            dim3(gdn_config.num_math_threads + gdn_config.num_tma_threads, 1, 1),
            dim3(gdn_config.num_blocks, 1, 1), stream, static_cast<int>(gdn_config.smem_size),
            1  // num_multicast
        };

        // Build Args struct
        SM90_BF16_GDN_State_Passing_Runtime::Args args;
        args.shape_k = k_head_dim;
        args.shape_v = v_head_dim;
        args.batch_size = batch_size;
        args.chunk_size = chunk_size;
        args.num_k_heads = num_k_heads;
        args.num_v_heads = num_v_heads;
        args.block_k = gdn_config.block_k;
        args.block_v = gdn_config.block_v;
        args.num_blocks = gdn_config.num_blocks;
        args.num_math_threads = gdn_config.num_math_threads;
        args.num_tma_threads = gdn_config.num_tma_threads;
        args.swizzle_k_mode = gdn_config.swizzle_k_mode;
        args.swizzle_u_mode = gdn_config.swizzle_v_mode;
        args.swizzle_s_mode = gdn_config.swizzle_state_mode;
        args.swizzle_w_mode = gdn_config.swizzle_k_mode;
        args.seq_len = seq_len;
        args.is_var_len = false;
        args.use_gate = gate.has_value();
        args.k_tensor_map = k_tensor_map;
        args.u_tensor_map = u_tensor_map;
        args.w_tensor_map = w_tensor_map;
        args.initial_state_tensor_map = initial_state_tensor_map;
        args.state_tensor_map = state_tensor_map;
        args.final_state_tensor_map = final_state_tensor_map;
        args.shape_T = seq_len;
        args.cu_seqlens = nullptr;
        args.compiled_dims = compiled_dims;
        args.launch_config = launch_config;
        args.is_initial_state = is_initial_state;
        args.output_final_state = output_final_state;
        args.cu_chunks = nullptr;
        args.U_ptr = nullptr;
        args.gate_ptr = gate.has_value() ? gate.value().data_ptr<float>() : nullptr;
        args.gate_stride = gate.has_value() ? gate.value().size(1) : 0;

        // Generate, build, and launch kernel
        const std::string code = LaunchRuntime<SM90_BF16_GDN_State_Passing_Runtime>::generate(args);
        std::shared_ptr<KernelRuntime> runtime =
            compiler->build("sm90_bf16_chunked_seq_state_update_padded", code);
        LaunchRuntime<SM90_BF16_GDN_State_Passing_Runtime>::launch(runtime, args);
    }
}
