
/*

some quick notes - remember that threads must be at least warpgroup-sized for tma loads and stores

remember that the minimum tma payload size is 128 bytes - this should be accounted for when loading
any of the QKV 1d vectors which are small and may not fit that requirement - this also plays in
tandem with the shared memory size
*/

#pragma once
#include <cuda.h>

#include <cstring>
#include <gdn_cuda/device.hpp>
#include <gdn_cuda/format.hpp>
#include <jit/compiler.hpp>
#include <jit/runtime.hpp>
#include <jit/utils/common.hpp>
#include <jit/utils/culib.hpp>
#include <jit_kernels/heuristics/common.hpp>

#include "jit_kernels/heuristics/sm90_arch.hpp"

class SM90_BF16_GDN_Compute_U_W_Runtime : LaunchRuntime<SM90_BF16_GDN_Compute_U_W_Runtime> {
   public:
    struct Args {
        uint32_t shape_k, shape_v, batch_size;
        uint32_t chunk_size;
        uint32_t num_k_heads, num_v_heads;
        uint32_t block_k;
        uint32_t num_blocks;
        uint32_t num_math_threads, num_tma_threads;
        uint32_t swizzle_k_mode, swizzle_v_mode, swizzle_a_mode;
        uint32_t seq_len;

        bool is_var_len;
        bool use_gating;

        CUtensorMap k_tensor_map;
        CUtensorMap v_tensor_map;
        CUtensorMap u_tensor_map;
        CUtensorMap w_tensor_map;

        uint32_t shape_T;
        uint32_t beta_stride;
        uint32_t gate_stride;

        int chunk_indices_length;
        int* chunk_indices;
        int* cu_seqlens;

        __nv_bfloat16* U_ptr;
        __nv_bfloat16* W_ptr;
        __nv_bfloat16* beta_ptr;
        float* gate_ptr;
        std::string compiled_dims;
        LaunchConfig launch_config;
    };

    static std::string generate_impl(const Args& args) {
        const std::string code = fmt::format(
            R"(
#include <gdn_cuda/kernels/sm90_bf16_gdn_compute_u_w.cuh>

using namespace gdn_cuda::kernels::sm90_gdn_compute_u_w_impl;
static void __instantiate_kernel() {{
    auto kernel_ptr = reinterpret_cast<void *>(&sm90_bf16_compute_u_w<
        {}, {}, {}, {},
        {}, {}, {},
        {}, {}, {}, {},
        {}, {}, {},
        {}, {}, {}
    >);
}}
            )",
            args.shape_k, args.shape_v, get_compiled_dim(args.compiled_dims, 'b', args.batch_size),
            args.chunk_size, args.num_v_heads, args.num_k_heads, args.block_k, args.num_blocks,
            args.launch_config.num_multicast, args.num_tma_threads, args.num_math_threads,
            args.swizzle_k_mode, args.swizzle_v_mode, args.swizzle_a_mode,
            get_compiled_dim(args.compiled_dims, 't', args.seq_len), args.is_var_len,
            args.use_gating);
        return code;
    }

    static void launch_impl(KernelHandle& kernel, const LaunchConfigHandle& launch_config,
                            const Args& args) {
        CUDA_CHECK(launch_kernel(kernel, launch_config, args.k_tensor_map, args.v_tensor_map,
                                 args.u_tensor_map, args.w_tensor_map, args.batch_size,
                                 args.shape_T, args.chunk_indices_length, args.chunk_indices,
                                 args.cu_seqlens, args.U_ptr, args.W_ptr, args.beta_ptr,
                                 args.gate_ptr, args.beta_stride, args.gate_stride));
    }
};

// API expectations:
// - Purpose: compute chunked U/W intermediates for GDN.
// - Padded mode expects rank-4 k/v/u/w and rank-3 beta (MN-major).
// - Varlen mode expects rank-3 k/v/u/w, rank-2 beta (MN-major), plus cu_seqlens and chunk_indices.
// - DTypes: k/v/u/w/beta are BF16, gate (optional) is FP32 in MN-major.
// - `u` and `w` are mutated in-place on the provided CUDA stream.
inline void sm90_bf16_compute_u_w(
    at::Tensor& k,  // (batch_size, seq_len, num_k_heads, k_head_dim) or (total_seq_len,
                    // num_k_heads, k_head_dim)
    at::Tensor& v, at::Tensor& u, at::Tensor& w,
    at::Tensor& beta,  // NOTE: (batch_size, num_v_heads), or (len(cu_seqlens) -1, num_v_heads) - we
                       // don't use a seqlen tensor here
    std::optional<at::Tensor>& gate, const std::string& compiled_dims, cudaStream_t stream,
    std::optional<at::Tensor>& cu_seqlens, std::optional<at::Tensor>& chunk_indices,
    const uint32_t chunk_size = 64) {
    bool is_var_len = cu_seqlens.has_value();

    HOST_ASSERT(is_mn_major(beta), "Beta must satisfy MN-major stride contract");
    HOST_ASSERT(!gate.has_value() || is_mn_major(gate.value()),
                "Gate must satisfy MN-major stride contract");
    // HOST_ASSERT(beta.size(-1) >= 8, "Number of beta heads must be at least 8 for TMA load
    // requirements, gmem_stride must be at least 8 * 2 = 16");

    if (is_var_len) {
        // Variable length sequences: tensors are (total_seq_len, num_heads, head_dim)
        HOST_ASSERT(k.dim() == 3 && v.dim() == 3,
                    "Variable length expects 3D tensors (total_seq, heads, dim)");

        // beta shape should be (total_num_tokens, num_v_heads), strides are MN-major
        HOST_ASSERT(beta.dim() == 2, "Beta must be 2D and match v dimensions");
        at::Tensor& cu_seqlens_tensor = cu_seqlens.value();
        HOST_ASSERT(cu_seqlens_tensor.dim() == 1, "cu_seqlens must be a 1D tensor");

        const uint32_t batch_size = static_cast<uint32_t>(cu_seqlens_tensor.size(0) - 1);
        const uint32_t total_seq_len = static_cast<uint32_t>(k.size(0));
        const uint32_t num_k_heads = static_cast<uint32_t>(k.size(1));
        const uint32_t num_v_heads = static_cast<uint32_t>(v.size(1));
        const uint32_t k_head_dim = static_cast<uint32_t>(k.size(2));
        const uint32_t v_head_dim = static_cast<uint32_t>(v.size(2));
        // Prepare chunk indices for varlen scheduling
        HOST_ASSERT(chunk_indices->size(0), "chunk_indices is empty - no valid chunks to process");
        HOST_ASSERT(chunk_indices->size(0) % 2 == 0,
                    "chunk_indices must contain pairs of (batch_idx, chunk_idx)");

        // Get kernel configuration from heuristics
        // chunk_indices stores pairs (batch_idx, chunk_idx), so actual num_chunks = size / 2
        const uint32_t num_chunks = static_cast<uint32_t>(chunk_indices->size(0) / 2);
        GDNConfig gdn_config = get_uw_config<true>(k_head_dim, v_head_dim, 1, 0, num_chunks,
                                                   num_v_heads, gate.has_value(), chunk_size);

        // Create TMA descriptors for all tensors
        // K tensor: (total_seq_len, num_k_heads, k_head_dim)
        // remember to revise these l2 promotion sizes -
        CUtensorMap k_tensor_map =
            make_tma_3d_desc(k, k_head_dim, num_k_heads, total_seq_len, k.stride(1), k.stride(0),
                             gdn_config.block_k, 1, chunk_size, gdn_config.swizzle_k_mode,
                             ti_align(gdn_config.block_k * get_type_size(k.scalar_type()), 64));
        // V tensor: (total_seq_len, num_v_heads, v_head_dim)
        CUtensorMap v_tensor_map =
            make_tma_3d_desc(v, v_head_dim, num_v_heads, total_seq_len, v.stride(1), v.stride(0),
                             gdn_config.block_v, 1, chunk_size, gdn_config.swizzle_v_mode,
                             ti_align(gdn_config.block_v * get_type_size(v.scalar_type()), 64));
        // U tensor: same layout as V (output)
        CUtensorMap u_tensor_map =
            make_tma_3d_desc(u, v_head_dim, num_v_heads, total_seq_len, u.stride(1), u.stride(0),
                             gdn_config.block_v, 1, chunk_size, gdn_config.swizzle_v_mode,
                             ti_align(gdn_config.block_v * get_type_size(u.scalar_type()), 64));
        // W tensor: same layout as K (output)
        CUtensorMap w_tensor_map =
            make_tma_3d_desc(w, k_head_dim, num_v_heads, total_seq_len, w.stride(1), w.stride(0),
                             gdn_config.block_k, 1, chunk_size, gdn_config.swizzle_k_mode,
                             ti_align(gdn_config.block_k * get_type_size(w.scalar_type()), 64));

        // Beta tensor: (total_seq_len, num_v_heads, v_head_dim)
        // Set up launch configuration - persistent scheduler uses num_sms as grid size
        LaunchConfig launch_config = {
            dim3(gdn_config.num_math_threads + gdn_config.num_tma_threads, 1, 1),
            dim3(gdn_config.num_blocks, 1, 1), stream, static_cast<int>(gdn_config.smem_size),
            1  // num_multicast
        };
        // cu_seqlens is already on device as a tensor
        int* d_cu_seqlens = cu_seqlens_tensor.data_ptr<int>();
        int* d_chunk_indices = chunk_indices->data_ptr<int>();

        // Build Args struct
        SM90_BF16_GDN_Compute_U_W_Runtime::Args args;
        args.shape_k = k_head_dim;
        args.shape_v = v_head_dim;
        args.batch_size = 1;
        args.chunk_size = chunk_size;
        args.num_k_heads = num_k_heads;
        args.num_v_heads = num_v_heads;
        args.block_k = gdn_config.block_k;
        args.num_blocks = gdn_config.num_blocks;
        args.num_math_threads = gdn_config.num_math_threads;
        args.num_tma_threads = gdn_config.num_tma_threads;
        args.swizzle_k_mode = gdn_config.swizzle_k_mode;
        args.swizzle_v_mode = gdn_config.swizzle_v_mode;
        args.swizzle_a_mode = gdn_config.swizzle_a_mode;
        args.seq_len = 0;  // varlen doesn't use fixed seq_len
        args.is_var_len = true;
        args.k_tensor_map = k_tensor_map;
        args.v_tensor_map = v_tensor_map;
        args.u_tensor_map = u_tensor_map;
        args.w_tensor_map = w_tensor_map;
        args.shape_T = total_seq_len;
        args.chunk_indices_length = static_cast<int>(
            chunk_indices->size(0) / 2);  // chunk_indices stores pairs (batch_idx, chunk_idx)
        args.chunk_indices = d_chunk_indices;
        args.cu_seqlens = d_cu_seqlens;
        args.compiled_dims = compiled_dims;
        args.launch_config = launch_config;
        args.U_ptr = reinterpret_cast<__nv_bfloat16*>(u.data_ptr());
        args.W_ptr = reinterpret_cast<__nv_bfloat16*>(w.data_ptr());
        args.beta_ptr = reinterpret_cast<__nv_bfloat16*>(beta.data_ptr());
        args.gate_ptr = gate.has_value() ? gate.value().data_ptr<float>() : nullptr;
        args.use_gating = gate.has_value();
        args.beta_stride = beta.size(0);
        args.gate_stride = gate.has_value() ? gate->size(0) : 0;

        // Generate, build, and launch kernel
        const std::string code = LaunchRuntime<SM90_BF16_GDN_Compute_U_W_Runtime>::generate(args);
        std::shared_ptr<KernelRuntime> runtime =
            compiler->build("sm90_bf16_compute_u_w_varlen", code);
        LaunchRuntime<SM90_BF16_GDN_Compute_U_W_Runtime>::launch(runtime, args);
    } else {
        // Fixed length sequences: tensors are (batch_size, seq_len, num_heads, head_dim)
        HOST_ASSERT(k.dim() == 4 && v.dim() == 4,
                    "Fixed length expects 4D tensors (batch, seq, heads, dim)");
        HOST_ASSERT(beta.dim() == 3 && beta.size(0) == v.size(0), "Beta must be 3D");

        const uint32_t batch_size = static_cast<uint32_t>(k.size(0));
        const uint32_t seq_len = static_cast<uint32_t>(k.size(1));
        const uint32_t num_k_heads = static_cast<uint32_t>(k.size(2));
        const uint32_t num_v_heads = static_cast<uint32_t>(v.size(2));
        const uint32_t k_head_dim = static_cast<uint32_t>(k.size(3));
        const uint32_t v_head_dim = static_cast<uint32_t>(v.size(3));

        // Get kernel configuration from heuristics
        GDNConfig gdn_config = get_uw_config<false>(k_head_dim, v_head_dim, batch_size, seq_len, 0,
                                                    num_v_heads, gate.has_value(), chunk_size);

        // Create TMA descriptors - reshape 4D to 3D view for TMA
        // K tensor: viewed as (batch*seq, num_k_heads, k_head_dim)
        CUtensorMap k_tensor_map = make_tma_4d_desc(
            k, k_head_dim, num_k_heads, seq_len, batch_size, k.stride(-2), k.stride(-3),
            k.stride(-4), gdn_config.block_k, 1, chunk_size, 1, gdn_config.swizzle_k_mode,
            ti_align(gdn_config.block_k * get_type_size(k.scalar_type()), 64));
        // V tensor: viewed as (batch*seq, num_v_heads, v_head_dim)
        CUtensorMap v_tensor_map = make_tma_4d_desc(
            v, v_head_dim, num_v_heads, seq_len, batch_size, v.stride(-2), v.stride(-3),
            v.stride(-4), gdn_config.block_v, 1, chunk_size, 1, gdn_config.swizzle_v_mode,
            ti_align(gdn_config.block_v * get_type_size(v.scalar_type()), 64));
        // U tensor: same layout as V (output)
        CUtensorMap u_tensor_map = make_tma_4d_desc(
            u, v_head_dim, num_v_heads, seq_len, batch_size, u.stride(-2), u.stride(-3),
            u.stride(-4), gdn_config.block_v, 1, chunk_size, 1, gdn_config.swizzle_v_mode,
            ti_align(gdn_config.block_v * get_type_size(u.scalar_type()), 64));
        // W tensor: same layout as K (output)
        CUtensorMap w_tensor_map = make_tma_4d_desc(
            w, k_head_dim, num_v_heads, seq_len, batch_size, w.stride(-2), w.stride(-3),
            w.stride(-4), gdn_config.block_k, 1, chunk_size, 1, gdn_config.swizzle_k_mode,
            ti_align(gdn_config.block_k * get_type_size(w.scalar_type()), 64));

        // Set up launch configuration - persistent scheduler uses num_sms as grid size
        LaunchConfig launch_config = {
            dim3(gdn_config.num_math_threads + gdn_config.num_tma_threads, 1, 1),
            dim3(gdn_config.num_blocks, 1, 1), stream, static_cast<int>(gdn_config.smem_size),
            1  // num_multicast
        };

        // Build Args struct
        SM90_BF16_GDN_Compute_U_W_Runtime::Args args;
        args.shape_k = k_head_dim;
        args.shape_v = v_head_dim;
        args.batch_size = batch_size;
        args.chunk_size = chunk_size;
        args.num_k_heads = num_k_heads;
        args.num_v_heads = num_v_heads;
        args.block_k = gdn_config.block_k;
        args.num_blocks = gdn_config.num_blocks;
        args.num_math_threads = gdn_config.num_math_threads;
        args.num_tma_threads = gdn_config.num_tma_threads;
        args.swizzle_k_mode = gdn_config.swizzle_k_mode;
        args.swizzle_v_mode = gdn_config.swizzle_v_mode;
        args.swizzle_a_mode = gdn_config.swizzle_a_mode;
        args.seq_len = seq_len;
        args.is_var_len = false;
        args.k_tensor_map = k_tensor_map;
        args.v_tensor_map = v_tensor_map;
        args.u_tensor_map = u_tensor_map;
        args.w_tensor_map = w_tensor_map;
        args.shape_T = seq_len;
        args.chunk_indices_length = 0;
        args.chunk_indices = nullptr;
        args.cu_seqlens = nullptr;
        args.compiled_dims = compiled_dims;
        args.launch_config = launch_config;
        args.U_ptr = nullptr;
        args.W_ptr = nullptr;
        args.use_gating = gate.has_value();
        args.beta_ptr = reinterpret_cast<__nv_bfloat16*>(beta.data_ptr());
        args.gate_ptr = gate.has_value() ? gate.value().data_ptr<float>() : nullptr;
        args.beta_stride = beta.size(1);  // 2nd dimension is the seq_len (padded for alignment)
        args.gate_stride = gate.has_value() ? gate->size(1) : 0;
        // Generate, build, and launch kernel
        const std::string code = LaunchRuntime<SM90_BF16_GDN_Compute_U_W_Runtime>::generate(args);
        std::shared_ptr<KernelRuntime> runtime =
            compiler->build("sm90_bf16_compute_u_w_padded", code);
        LaunchRuntime<SM90_BF16_GDN_Compute_U_W_Runtime>::launch(runtime, args);
    }
}
