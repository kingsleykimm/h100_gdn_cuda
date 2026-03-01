#pragma once
#include <gdn_cuda/device.hpp>
#include <gdn_cuda/format.hpp>
#include <jit/compiler.hpp>
#include <jit/runtime.hpp>
#include <jit/utils/common.hpp>
#include <jit/utils/culib.hpp>
#include <jit_kernels/heuristics/common.hpp>

#include "jit_kernels/heuristics/sm90_arch.hpp"

class SM90_BF16_GDN_Chunked_Compute_O_Runtime
    : LaunchRuntime<SM90_BF16_GDN_Chunked_Compute_O_Runtime> {
   public:
    struct Args {
        uint32_t batch_size, chunk_size, chunk_block;
        uint32_t num_k_heads, num_v_heads;
        uint32_t shape_k, block_k, shape_v, block_v;
        uint32_t swizzle_k_mode, swizzle_u_mode, swizzle_q_mode, swizzle_s_mode;
        uint32_t num_math_threads, num_tma_threads;
        uint32_t num_blocks;
        uint32_t num_stages;
        uint32_t num_tma_multicast;
        uint32_t seq_len;
        bool is_var_len;

        CUtensorMap k_tensor_map, u_tensor_map, q_tensor_map, state_tensor_map, o_tensor_map;

        int shape_T;
        int *cu_seqlens;
        int *cu_chunks;
        int *chunk_indices;
        int chunk_indices_length;
        float scale_factor;
        __nv_bfloat16 *O_ptr;
        float *gate_ptr;
        bool use_gating;
        uint32_t gate_stride;
        std::string compiled_dims;
        LaunchConfig launch_config;
    };

    static std::string generate_impl(const Args &args) {
        // Template order: kBatchSize, kChunkShape, kChunkBlock, kNumKHeads, kNumVHeads,
        //                 SHAPE_K, BLOCK_K, SHAPE_V, BLOCK_V,
        //                 kSwizzleKMode, kSwizzleUMode, kSwizzleQMode, kSwizzleSMode,
        //                 kNumMathThreads, kNumTMAThreads, kNumBlocks, kNumStages,
        //                 kNumTMAMulticast, kIsVarLen, kSeqLen
        const std::string code = fmt::format(
            R"(
#include <gdn_cuda/kernels/sm90_bf16_gdn_chunked_compute_O.cuh>

using namespace gdn_cuda::kernels::sm90_gdn_compute_o_impl;

static void __instantiate_kernel() {{
    auto kernel_ptr = reinterpret_cast<void *>(&sm90_bf16_gdn_chunked_compute_O<
        {}, {}, {},
        {}, {},
        {}, {}, {}, {},
        {}, {}, {}, {},
        {}, {}, {}, {}, {},
        {}, {}, {}
    >);
}}
            )",
            get_compiled_dim(args.compiled_dims, 'b', args.batch_size),  // kBatchSize
            args.chunk_size,                                             // kChunkShape
            args.chunk_block,                                            // kChunkBlock
            args.num_k_heads,                                            // kNumKHeads
            args.num_v_heads,                                            // kNumVHeads
            args.shape_k,                                                // SHAPE_K
            args.block_k,                                                // BLOCK_K
            args.shape_v,                                                // SHAPE_V
            args.block_v,                                                // BLOCK_V
            args.swizzle_k_mode,                                         // kSwizzleKMode
            args.swizzle_u_mode,                                         // kSwizzleUMode
            args.swizzle_q_mode,                                         // kSwizzleQMode
            args.swizzle_s_mode,                                         // kSwizzleSMode
            args.num_math_threads,                                       // kNumMathThreads
            args.num_tma_threads,                                        // kNumTMAThreads
            args.num_blocks,                                             // kNumBlocks
            args.num_stages,                                             // kNumStages
            args.num_tma_multicast,                                      // kNumTMAMulticast
            args.is_var_len,                                             // kIsVarLen
            get_compiled_dim(args.compiled_dims, 't', args.seq_len),     // kSeqLen
            args.use_gating);                                            // kUseGating
        return code;
    }

    static void launch_impl(KernelHandle &kernel, const LaunchConfigHandle &launch_config,
                            const Args &args) {
        // Kernel params: state_tensor_map, k_tensor_map, q_tensor_map, u_tensor_map, o_tensor_map,
        //                O_ptr, batch_size, shape_T, cu_seqlens, chunk_indices_length,
        //                chunk_indices, cu_chunks, scale_factor
        CUDA_CHECK(launch_kernel(
            kernel, launch_config, args.state_tensor_map, args.k_tensor_map, args.q_tensor_map,
            args.u_tensor_map, args.o_tensor_map, args.O_ptr, args.gate_ptr, args.gate_stride,
            static_cast<int>(args.batch_size), args.shape_T, args.cu_seqlens,
            args.chunk_indices_length, args.chunk_indices, args.cu_chunks, args.scale_factor));
    }
};

// API expectations:
// - Purpose: compute final chunked output O from q/k/u and propagated state.
// - Padded mode expects rank-4 q/k/u/o and rank-5 state.
// - Varlen mode expects rank-3 q/k/u/o, rank-4 state, plus cu_seqlens/chunk_indices/cu_chunks.
// - DTypes: q/k/u/o/state are BF16; gate is optional FP32.
// - Mutates output tensor `o` in-place on the provided CUDA stream.
inline void sm90_bf16_gdn_chunked_compute_O(
    at::Tensor &q, at::Tensor &state, at::Tensor &k, at::Tensor &u, at::Tensor &o,
    std::optional<at::Tensor> &gate, std::optional<float> scale, const std::string &compiled_dims,
    cudaStream_t stream, std::optional<at::Tensor> &cu_seqlens,
    std::optional<at::Tensor> &chunk_indices, std::optional<at::Tensor> &cu_chunks,
    std::optional<int> total_chunks, const uint32_t chunk_size = 64) {
    bool is_var_len = cu_seqlens.has_value();

    if (is_var_len) {
        // Variable length sequences: tensors are (total_seq_len, num_heads, head_dim)
        HOST_ASSERT(k.dim() == 3, "Variable length expects 3D tensors (total_seq, heads, dim)");
        HOST_ASSERT(q.dim() == 3, "Variable length expects 3D tensors (total_seq, heads, dim)");
        HOST_ASSERT(u.dim() == 3, "Variable length expects 3D tensors (total_seq, heads, dim)");
        HOST_ASSERT(o.dim() == 3, "Variable length expects 3D tensors (total_seq, heads, dim)");
        HOST_ASSERT(state.dim() == 4,
                    "Variable length state expects 4D tensor (total_chunks, heads, v_dim, k_dim)");

        at::Tensor &cu_seqlens_tensor = cu_seqlens.value();
        HOST_ASSERT(cu_seqlens_tensor.dim() == 1, "cu_seqlens must be a 1D tensor");

        const uint32_t batch_size = static_cast<uint32_t>(cu_seqlens_tensor.size(0) - 1);
        const uint32_t total_seq_len = static_cast<uint32_t>(k.size(0));
        const uint32_t num_k_heads = static_cast<uint32_t>(k.size(1));
        const uint32_t num_v_heads = static_cast<uint32_t>(u.size(1));
        const uint32_t shape_k = static_cast<uint32_t>(k.size(2));
        const uint32_t shape_v = static_cast<uint32_t>(u.size(2));

        HOST_ASSERT(cu_chunks.has_value(), "cu_chunks must be provided for varlen mode");
        HOST_ASSERT(cu_chunks->dim() == 1, "cu_chunks must be a 1D tensor");
        HOST_ASSERT(cu_chunks->size(0) > 0, "cu_chunks must be non-empty");

        // Prepare chunk indices for varlen scheduling.
        // cu_chunks is cumulative; total chunks is its last element, not its length.
        // Get kernel configuration from heuristics
        GDNConfig config =
            get_compute_O_config<true>(shape_k, shape_v, num_v_heads, num_k_heads, batch_size, 0,
                                       total_chunks.value(), gate.has_value(), chunk_size);

        const float scale_factor =
            scale.has_value() ? scale.value() : 1.0f / std::sqrt(static_cast<float>(shape_k));

        // Create TMA descriptors for all tensors
        // K tensor: (total_seq_len, num_k_heads, shape_k) -> load (block_k, chunk_block) tiles
        // TMA box: (block_k cols, chunk_block rows)
        CUtensorMap k_tensor_map = make_tma_3d_desc(
            k, shape_k, num_k_heads, total_seq_len, k.stride(1), k.stride(0), config.block_k, 1,
            config.chunk_block, config.swizzle_k_mode, ti_align(config.block_k * 2, 64));

        // Q tensor: (total_seq_len, num_k_heads, shape_k) -> load (block_k, chunk_size) tiles
        CUtensorMap q_tensor_map = make_tma_3d_desc(
            q, shape_k, num_k_heads, total_seq_len, q.stride(1), q.stride(0), config.block_k, 1,
            chunk_size, config.swizzle_q_mode, ti_align(config.block_k * 2, 64));

        // U tensor: (total_seq_len, num_v_heads, shape_v) -> load (block_v, chunk_block) tiles
        CUtensorMap u_tensor_map = make_tma_3d_desc(
            u, shape_v, num_v_heads, total_seq_len, u.stride(1), u.stride(0), config.block_v, 1,
            config.chunk_block, config.swizzle_u_mode, ti_align(config.block_v * 2, 64));

        // State tensor: (total_chunks, num_v_heads, shape_v, shape_k) -> load (block_k, block_v)
        // tiles For kernel: tma_copy<BLOCK_K, BLOCK_V, ...> with coords (k1*BLOCK_K,
        // v_block*BLOCK_V, 1, v_head, chunk_offset)
        CUtensorMap state_tensor_map = make_tma_4d_desc(
            state, state.size(-1), state.size(-2), state.size(-3), total_chunks.value(),
            state.stride(2), state.stride(1), state.stride(0), config.block_k, config.block_v, 1, 1,
            config.swizzle_state_mode, ti_align(config.block_k * 2, 64));

        // O tensor: (total_seq_len, num_v_heads, shape_v) -> store (block_v, chunk_size) tiles
        CUtensorMap o_tensor_map = make_tma_3d_desc(
            o, shape_v, num_v_heads, total_seq_len, o.stride(1), o.stride(0), config.block_v, 1,
            chunk_size, config.swizzle_u_mode, ti_align(config.block_v * 2, 64));

        // Set up launch configuration
        LaunchConfig launch_config = {dim3(config.num_math_threads + config.num_tma_threads, 1, 1),
                                      dim3(config.num_blocks, 1, 1), stream,
                                      static_cast<int>(config.smem_size), config.num_tma_multicast};

        // Build Args struct
        SM90_BF16_GDN_Chunked_Compute_O_Runtime::Args args;
        args.batch_size = batch_size;
        args.chunk_size = chunk_size;
        args.chunk_block = config.chunk_block;
        args.num_k_heads = num_k_heads;
        args.num_v_heads = num_v_heads;
        args.shape_k = shape_k;
        args.block_k = config.block_k;
        args.shape_v = shape_v;
        args.block_v = config.block_v;
        args.swizzle_k_mode = config.swizzle_k_mode;
        args.swizzle_u_mode = config.swizzle_u_mode;
        args.swizzle_q_mode = config.swizzle_q_mode;
        args.swizzle_s_mode = config.swizzle_state_mode;
        args.num_math_threads = config.num_math_threads;
        args.num_tma_threads = config.num_tma_threads;
        args.num_blocks = config.num_blocks;
        args.num_stages = config.num_stages;
        args.num_tma_multicast = config.num_tma_multicast;
        args.seq_len = 0;  // varlen doesn't use fixed seq_len
        args.is_var_len = true;
        args.k_tensor_map = k_tensor_map;
        args.u_tensor_map = u_tensor_map;
        args.q_tensor_map = q_tensor_map;
        args.state_tensor_map = state_tensor_map;
        args.o_tensor_map = o_tensor_map;
        args.shape_T = static_cast<int>(total_seq_len);
        args.cu_seqlens = cu_seqlens->data_ptr<int>();
        args.cu_chunks = cu_chunks->data_ptr<int>();
        args.chunk_indices = chunk_indices->data_ptr<int>();  // Not used in current scheduler
        args.chunk_indices_length = chunk_indices->size(0) / 2;
        args.scale_factor = scale_factor;
        args.O_ptr = reinterpret_cast<__nv_bfloat16 *>(o.data_ptr());
        args.gate_ptr = gate.has_value() ? gate.value().data_ptr<float>() : nullptr;
        args.gate_stride = gate.has_value() ? gate.value().size(0) : 0;
        args.use_gating = gate.has_value();
        args.compiled_dims = compiled_dims;
        args.launch_config = launch_config;

        // Generate, build, and launch kernel
        const std::string code =
            LaunchRuntime<SM90_BF16_GDN_Chunked_Compute_O_Runtime>::generate(args);
        std::shared_ptr<KernelRuntime> runtime =
            compiler->build("sm90_bf16_gdn_chunked_compute_O_varlen", code);
        LaunchRuntime<SM90_BF16_GDN_Chunked_Compute_O_Runtime>::launch(runtime, args);
    } else {
        // Fixed length (padded) sequences: tensors are (batch_size, seq_len, num_heads, head_dim)
        HOST_ASSERT(k.dim() == 4, "Fixed length expects 4D tensors (batch, seq, heads, dim)");
        HOST_ASSERT(q.dim() == 4, "Fixed length expects 4D tensors (batch, seq, heads, dim)");
        HOST_ASSERT(u.dim() == 4, "Fixed length expects 4D tensors (batch, seq, heads, dim)");
        HOST_ASSERT(o.dim() == 4, "Fixed length expects 4D tensors (batch, seq, heads, dim)");
        HOST_ASSERT(state.dim() == 5,
                    "Fixed length state expects 5D tensor (batch, chunks, heads, v_dim, k_dim)");

        const uint32_t batch_size = static_cast<uint32_t>(k.size(0));
        const uint32_t seq_len = static_cast<uint32_t>(k.size(1));
        const uint32_t num_k_heads = static_cast<uint32_t>(k.size(2));
        const uint32_t num_v_heads = static_cast<uint32_t>(u.size(2));
        const uint32_t shape_k = static_cast<uint32_t>(k.size(3));
        const uint32_t shape_v = static_cast<uint32_t>(u.size(3));
        const uint32_t num_chunks = ti_ceil_div(seq_len, chunk_size);

        // Get kernel configuration from heuristics
        GDNConfig config =
            get_compute_O_config<false>(shape_k, shape_v, num_v_heads, num_k_heads, batch_size,
                                        seq_len, num_chunks, gate.has_value(), chunk_size);

        const float scale_factor =
            scale.has_value() ? scale.value() : 1.0f / std::sqrt(static_cast<float>(shape_k));

        // Create 4D TMA descriptors for padded mode
        // K tensor: (batch, seq_len, num_k_heads, shape_k) -> load (block_k, chunk_block) tiles
        CUtensorMap k_tensor_map =
            make_tma_4d_desc(k, shape_k, num_k_heads, seq_len, batch_size, k.stride(2), k.stride(1),
                             k.stride(0), config.block_k, 1, config.chunk_block, 1,
                             config.swizzle_k_mode, ti_align(config.block_k * 2, 64));

        // Q tensor: (batch, seq_len, num_k_heads, shape_k) -> load (block_k, chunk_size) tiles
        CUtensorMap q_tensor_map =
            make_tma_4d_desc(q, shape_k, num_k_heads, seq_len, batch_size, q.stride(2), q.stride(1),
                             q.stride(0), config.block_k, 1, chunk_size, 1, config.swizzle_q_mode,
                             ti_align(config.block_k * 2, 64));

        // U tensor: (batch, seq_len, num_v_heads, shape_v) -> load (block_v, chunk_block) tiles
        CUtensorMap u_tensor_map =
            make_tma_4d_desc(u, shape_v, num_v_heads, seq_len, batch_size, u.stride(2), u.stride(1),
                             u.stride(0), config.block_v, 1, config.chunk_block, 1,
                             config.swizzle_u_mode, ti_align(config.block_v * 2, 64));

        // State tensor: (batch, num_chunks, num_v_heads, shape_v, shape_k) -> load (block_k,
        // block_v) tiles Merge batch and num_chunks into outermost dimension for 4D TMA
        CUtensorMap state_tensor_map = make_tma_4d_desc(
            state, state.size(-1), state.size(-2), state.size(-3), num_chunks * batch_size,
            state.stride(3), state.stride(2), state.stride(1), config.block_k, config.block_v, 1, 1,
            config.swizzle_state_mode, ti_align(config.block_k * 2, 64));

        // O tensor: (batch, seq_len, num_v_heads, shape_v) -> store (block_v, chunk_size) tiles
        CUtensorMap o_tensor_map =
            make_tma_4d_desc(o, shape_v, num_v_heads, seq_len, batch_size, o.stride(2), o.stride(1),
                             o.stride(0), config.block_v, 1, chunk_size, 1, config.swizzle_u_mode,
                             ti_align(config.block_v * 2, 64));

        // Set up launch configuration
        LaunchConfig launch_config = {dim3(config.num_math_threads + config.num_tma_threads, 1, 1),
                                      dim3(config.num_blocks, 1, 1), stream,
                                      static_cast<int>(config.smem_size), config.num_tma_multicast};

        // Build Args struct
        SM90_BF16_GDN_Chunked_Compute_O_Runtime::Args args;
        args.batch_size = batch_size;
        args.chunk_size = chunk_size;
        args.chunk_block = config.chunk_block;
        args.num_k_heads = num_k_heads;
        args.num_v_heads = num_v_heads;
        args.shape_k = shape_k;
        args.block_k = config.block_k;
        args.shape_v = shape_v;
        args.block_v = config.block_v;
        args.swizzle_k_mode = config.swizzle_k_mode;
        args.swizzle_u_mode = config.swizzle_u_mode;
        args.swizzle_q_mode = config.swizzle_q_mode;
        args.swizzle_s_mode = config.swizzle_state_mode;
        args.num_math_threads = config.num_math_threads;
        args.num_tma_threads = config.num_tma_threads;
        args.num_blocks = config.num_blocks;
        args.num_stages = config.num_stages;
        args.num_tma_multicast = config.num_tma_multicast;
        args.seq_len = seq_len;
        args.is_var_len = false;
        args.k_tensor_map = k_tensor_map;
        args.u_tensor_map = u_tensor_map;
        args.q_tensor_map = q_tensor_map;
        args.state_tensor_map = state_tensor_map;
        args.o_tensor_map = o_tensor_map;
        args.shape_T = static_cast<int>(seq_len);
        args.cu_seqlens = nullptr;
        args.cu_chunks = nullptr;
        args.chunk_indices = nullptr;
        args.chunk_indices_length = 0;
        args.scale_factor = scale_factor;
        args.O_ptr = reinterpret_cast<__nv_bfloat16 *>(o.data_ptr());
        args.gate_ptr = gate.has_value() ? gate.value().data_ptr<float>() : nullptr;
        args.gate_stride = gate.has_value() ? gate.value().size(1) : 0;
        args.use_gating = gate.has_value();
        args.compiled_dims = compiled_dims;
        args.launch_config = launch_config;

        // Generate, build, and launch kernel
        const std::string code =
            LaunchRuntime<SM90_BF16_GDN_Chunked_Compute_O_Runtime>::generate(args);
        std::shared_ptr<KernelRuntime> runtime =
            compiler->build("sm90_bf16_gdn_chunked_compute_O_padded", code);
        LaunchRuntime<SM90_BF16_GDN_Chunked_Compute_O_Runtime>::launch(runtime, args);
    }
}
