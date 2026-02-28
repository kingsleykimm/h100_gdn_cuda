// the methods here don't depend on the cuda runtime or cuda::std libraries, which is a requirement
// of any JIT kernel

#pragma once
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/config.hpp>
#include <cute/swizzle.hpp>

#include "cute/util/type_traits.hpp"

template <uint32_t kNumBits, typename unit_ptr_t>
struct VEC_LOAD {
    static auto base_init() {
        if constexpr (kNumBits > 0 && kNumBits % 128 == 0) {
            return make_uint4(0, 0, 0, 0);
        } else if constexpr (kNumBits > 0 && kNumBits % 64 == 0) {
            return make_uint2(0, 0);
        } else if constexpr (kNumBits > 0 && kNumBits % 32 == 0) {
            return (uint32_t)0;
        } else if constexpr (kNumBits > 0 && kNumBits % 16 == 0) {
            return (uint16_t)0;
        } else {
            CUTE_STATIC_ASSERT(kNumBits > 0, "Invalid vectorization for kNumBits");
        }
    }
    using ptr_type = decltype(base_init());
    static constexpr size_t SIZE = sizeof(ptr_type) / sizeof(unit_ptr_t);
    float V[SIZE];
};

template <int kNumVec, size_t kNumBits>
struct VEC_LOAD_PTR {
    static auto base_init() {
        if constexpr (kNumBits > 0 && kNumBits * kNumVec == 128) {
            return make_uint4(0, 0, 0, 0);
        } else if constexpr (kNumBits > 0 && kNumBits * kNumVec == 64) {
            return make_uint2(0, 0);
        } else if constexpr (kNumBits > 0 && kNumBits * kNumVec == 32) {
            return (uint32_t)0;
        } else if constexpr (kNumBits > 0 && kNumBits * kNumVec == 16) {
            return (uint16_t)0;
        } else {
            CUTE_STATIC_ASSERT(kNumBits > 0, "Invalid vectorization for kNumBits");
        }
    }
    using ptr_type = decltype(base_init());
};

template <uint32_t kSwizzleMode>
struct CUTE_SWIZZLE {
    static auto get_swizzle() {
        if constexpr (kSwizzleMode == 128) {
            return cute::Swizzle<3, 4, 3>{};
        } else if constexpr (kSwizzleMode == 64) {
            return cute::Swizzle<2, 4, 3>{};
        } else if constexpr (kSwizzleMode == 32) {
            return cute::Swizzle<1, 4, 3>{};
        } else if constexpr (kSwizzleMode == 16 || kSwizzleMode == 0) {
            return cute::Swizzle<0, 4, 3>{};
        }
    }

    using type = decltype(get_swizzle());
};

template <typename from_t, typename to_t>
struct VEC_LOAD_CVT {};

// ============================================================================
// uint16_t conversions (2 bytes)
// ============================================================================

// uint16_t -> __half (1 half)
template <>
struct VEC_LOAD_CVT<uint16_t, __half> {
    __device__ __forceinline__ static void convert(uint16_t val, __half* out) {
        *reinterpret_cast<uint16_t*>(out) = val;
    }
};

// uint16_t -> __nv_bfloat16 (1 bfloat16)
template <>
struct VEC_LOAD_CVT<uint16_t, __nv_bfloat16> {
    __device__ __forceinline__ static void convert(uint16_t val, __nv_bfloat16* out) {
        *reinterpret_cast<uint16_t*>(out) = val;
    }
};

// ============================================================================
// uint32_t conversions (4 bytes)
// ============================================================================

// uint32_t -> float (1 float)
template <>
struct VEC_LOAD_CVT<uint32_t, float> {
    __device__ __forceinline__ static void convert(uint32_t val, float* out) {
        *reinterpret_cast<uint32_t*>(out) = val;
    }
};

// uint32_t -> __half (2 halves)
template <>
struct VEC_LOAD_CVT<uint32_t, __half> {
    __device__ __forceinline__ static void convert(uint32_t val, __half* outs) {
        *reinterpret_cast<uint32_t*>(outs) = val;
    }
};

// uint32_t -> __nv_bfloat16 (2 bfloat16s)
template <>
struct VEC_LOAD_CVT<uint32_t, __nv_bfloat16> {
    __device__ __forceinline__ static void convert(uint32_t val, __nv_bfloat16* outs) {
        *reinterpret_cast<uint32_t*>(outs) = val;
    }
};

// uint32_t -> __nv_fp8_e4m3 (4 fp8s)
template <>
struct VEC_LOAD_CVT<uint32_t, cutlass::float_e4m3_t> {
    __device__ __forceinline__ static void convert(uint32_t val, cutlass::float_e4m3_t* outs) {
        *reinterpret_cast<uint32_t*>(outs) = val;
    }
};

// ============================================================================
// uint2 conversions (8 bytes)
// ============================================================================

// uint2 -> float (2 floats)
template <>
struct VEC_LOAD_CVT<uint2, float> {
    __device__ __forceinline__ static void convert(uint2 val, float* outs) {
        *reinterpret_cast<uint2*>(outs) = val;
    }
};

// uint2 -> __half (4 halves)
template <>
struct VEC_LOAD_CVT<uint2, __half> {
    __device__ __forceinline__ static void convert(uint2 val, __half* outs) {
        *reinterpret_cast<uint2*>(outs) = val;
    }
};

// uint2 -> __nv_bfloat16 (4 bfloat16s)
template <>
struct VEC_LOAD_CVT<uint2, __nv_bfloat16> {
    __device__ __forceinline__ static void convert(uint2 val, __nv_bfloat16* outs) {
        *reinterpret_cast<uint2*>(outs) = val;
    }
};

// uint2 -> __nv_fp8_e4m3 (8 fp8s)
template <>
struct VEC_LOAD_CVT<uint2, cutlass::float_e4m3_t> {
    __device__ __forceinline__ static void convert(uint2 val, cutlass::float_e4m3_t* outs) {
        *reinterpret_cast<uint2*>(outs) = val;
    }
};

// ============================================================================
// uint4 conversions (16 bytes)
// ============================================================================

// uint4 -> float (4 floats)
template <>
struct VEC_LOAD_CVT<uint4, float> {
    __device__ __forceinline__ static void convert(uint4 val, float* outs) {
        *reinterpret_cast<uint4*>(outs) = val;
    }
};

// uint4 -> __half (8 halves)
template <>
struct VEC_LOAD_CVT<uint4, __half> {
    __device__ __forceinline__ static void convert(uint4 val, __half* outs) {
        *reinterpret_cast<uint4*>(outs) = val;
    }
};

// uint4 -> __nv_bfloat16 (8 bfloat16s)
template <>
struct VEC_LOAD_CVT<uint4, __nv_bfloat16> {
    __device__ __forceinline__ static void convert(uint4 val, __nv_bfloat16* outs) {
        *reinterpret_cast<uint4*>(outs) = val;
    }
};

// uint4 -> __nv_fp8_e4m3 (16 fp8s)
template <>
struct VEC_LOAD_CVT<uint4, cutlass::float_e4m3_t> {
    __device__ __forceinline__ static void convert(uint4 val, cutlass::float_e4m3_t* outs) {
        *reinterpret_cast<uint4*>(outs) = val;
    }
};

// ============================================================================
// Reverse conversions - base type into vec_t
// ============================================================================

template <>
struct VEC_LOAD_CVT<__half, uint16_t> {
    __device__ __forceinline__ static void convert(__half* val, uint16_t* outs) {
        *outs = *reinterpret_cast<uint16_t*>(val);
    }
};

template <>
struct VEC_LOAD_CVT<__nv_bfloat16, uint16_t> {
    __device__ __forceinline__ static void convert(__nv_bfloat16* val, uint16_t* outs) {
        *outs = *reinterpret_cast<uint16_t*>(val);
    }
};

template <>
struct VEC_LOAD_CVT<float, uint32_t> {
    __device__ __forceinline__ static void convert(float* val, uint32_t* outs) {
        *outs = *reinterpret_cast<uint32_t*>(val);
    }
};

template <>
struct VEC_LOAD_CVT<__half, uint32_t> {
    __device__ __forceinline__ static void convert(__half* val, uint32_t* outs) {
        *outs = *reinterpret_cast<uint32_t*>(val);
    }
};

template <>
struct VEC_LOAD_CVT<__nv_bfloat16, uint32_t> {
    __device__ __forceinline__ static void convert(__nv_bfloat16* val, uint32_t* outs) {
        *outs = *reinterpret_cast<uint32_t*>(val);
    }
};

// uint2 (64 bit) conversions

template <>
struct VEC_LOAD_CVT<float, uint2> {
    __device__ __forceinline__ static void convert(float* val, uint2* out) {
        // for these conversions, we're only doing one value of the to type, since we need to
        // explicitly place into load
        out[0].x = *reinterpret_cast<uint32_t*>(val);
        out[0].y = *reinterpret_cast<uint32_t*>(val + 1);
    }
};

// 4 bf16 -> uint2
template <>
struct VEC_LOAD_CVT<__nv_bfloat16, uint2> {
    __device__ __forceinline__ static void convert(__nv_bfloat16* val, uint2* out) {
        // for these conversions, we're only doing one value of the to type, since we need to
        // explicitly place into load
        out[0].x = *reinterpret_cast<uint32_t*>(val);
        out[0].y = *reinterpret_cast<uint32_t*>(val + 2);
    }
};

// 4 bf16 -> uint2
template <>
struct VEC_LOAD_CVT<__half, uint2> {
    __device__ __forceinline__ static void convert(__half* val, uint2* out) {
        // for these conversions, we're only doing one value of the to type, since we need to
        // explicitly place into load
        out[0].x = *reinterpret_cast<uint32_t*>(val);
        out[0].y = *reinterpret_cast<uint32_t*>(val + 2);
    }
};

template <>
struct VEC_LOAD_CVT<float, uint4> {
    __device__ __forceinline__ static void convert(float* val, uint4* out) {
        // for these conversions, we're only doing one value of the to type, since we need to
        // explicitly place into load
        out[0].x = *reinterpret_cast<uint32_t*>(val);
        out[0].y = *reinterpret_cast<uint32_t*>(val + 1);
        out[0].z = *reinterpret_cast<uint32_t*>(val + 2);
        out[0].w = *reinterpret_cast<uint32_t*>(val + 3);
    }
};

// 4 bf16 -> uint2
template <>
struct VEC_LOAD_CVT<__nv_bfloat16, uint4> {
    __device__ __forceinline__ static void convert(__nv_bfloat16* val, uint4* out) {
        // for these conversions, we're only doing one value of the to type, since we need to
        // explicitly place into load
        out[0].x = *reinterpret_cast<uint32_t*>(val);
        out[0].y = *reinterpret_cast<uint32_t*>(val + 2);
        out[0].z = *reinterpret_cast<uint32_t*>(val + 4);
        out[0].w = *reinterpret_cast<uint32_t*>(val + 6);
    }
};

// 4 bf16 -> uint2
template <>
struct VEC_LOAD_CVT<__half, uint4> {
    __device__ __forceinline__ static void convert(__half* val, uint4* out) {
        // for these conversions, we're only doing one value of the to type, since we need to
        // explicitly place into load
        out[0].x = *reinterpret_cast<uint32_t*>(val);
        out[0].y = *reinterpret_cast<uint32_t*>(val + 2);
        out[0].z = *reinterpret_cast<uint32_t*>(val + 4);
        out[0].w = *reinterpret_cast<uint32_t*>(val + 6);
    }
};

template <typename from_t>
__forceinline__ __host__ __device__ float to_float(from_t val) {
    if constexpr (cute::is_same_v<from_t, __nv_bfloat16>) {
        return __bfloat162float(val);
    } else if constexpr (cute::is_same_v<from_t, __half>) {
        return __half2float(val);
    } else if constexpr (cute::is_same_v<from_t, float>) {
        return val;
    } else {
        static_assert(sizeof(from_t) == 0, "Unsupported type");
    }
}

template <typename to_t>
__forceinline__ __host__ __device__ to_t from_float(float val) {
    if constexpr (cute::is_same_v<to_t, __nv_bfloat16>) {
        return __float2bfloat16(val);
    } else if constexpr (cute::is_same_v<to_t, __half>) {
        return __float2half(val);
    } else if constexpr (cute::is_same_v<to_t, float>) {
        return val;
    }
}

// Use ti_ prefix to avoid conflict with cutlass::ceil_div
template <typename TA, typename TB>
CUTE_HOST_DEVICE int ti_ceil_div(TA a, TB b) {
    return (a + b - 1) / b;
}

// align a to b
template <typename TA, typename TB>
CUTE_HOST_DEVICE int ti_align(TA a, TB b) {
    return ti_ceil_div(a, b) * b;
}

template <typename TA, typename TB>
CUTE_HOST_DEVICE constexpr int constexpr_ti_ceil_div(TA a, TB b) {
    return (a + b - 1) / b;
}

// align a to b
template <typename TA, typename TB>
CUTE_HOST_DEVICE constexpr int constexpr_ti_align(TA a, TB b) {
    return constexpr_ti_ceil_div(a, b) * b;
}

template <typename TA, typename TB>
__host__ __device__ constexpr TA constexpr_min(TA a, TB b) {
    return (a <= b) ? a : b;
}

template <typename TA, typename TB>
__host__ __device__ TA ti_min(TA a, TB b) {
    return a <= b ? a : b;
}
// Helper to apply swizzle with proper 128-bit normalization
// NVIDIA swizzling operates at 128-bit (16-byte) granularity
// returns the elem offset, NOT byte offset
template <typename SwizzleOp, typename dtype_t>
CUTE_HOST_DEVICE constexpr uint32_t swizzle_offset(SwizzleOp swizzle, uint32_t logical_idx) {
    constexpr uint32_t kBytesPerSwizzleUnit = 16;  // 128 bits
    constexpr uint32_t kElemsPerSwizzleUnit = kBytesPerSwizzleUnit / sizeof(dtype_t);

    uint32_t swizzle_unit = logical_idx / kElemsPerSwizzleUnit;
    uint32_t intra_unit_offset = logical_idx % kElemsPerSwizzleUnit;
    uint32_t physical_unit = swizzle(swizzle_unit);

    return physical_unit * kElemsPerSwizzleUnit + intra_unit_offset;
}

// Overload for 2D indexing (row, col) with row stride
template <typename SwizzleOp, typename dtype_t>
CUTE_HOST_DEVICE constexpr uint32_t swizzle_offset(SwizzleOp swizzle, uint32_t row, uint32_t col,
                                                   uint32_t row_stride) {
    return swizzle_offset<SwizzleOp, dtype_t>(swizzle, row * row_stride + col);
}

// Compute physical element offset for STSM-stored WGMMA output
// This matches the layout produced by stmatrix.sync.aligned.m8n8.x2.shared.b16
// storing WGMMA 64xN accumulators with swizzled layout
//
// The STSM layout has:
// - 4 warps, each handling 16 rows (M=64 / 4 warps)
// - Within each warp's region: data arranged for bank-conflict-free WGMMA access
// - 128-byte swizzle atoms (64 bf16 elements per atom width)
//
// Parameters:
//   mat_row, mat_col: logical matrix position
//   row_stride: number of columns in the matrix (e.g., 64 for 64x64)
//   kSwizzleMode: swizzle size in bytes (e.g., 128)
//
// Returns: physical element offset (NOT byte offset)
template <typename dtype_t, uint32_t kSwizzleMode, uint32_t BLOCK_M>
CUTE_HOST_DEVICE constexpr uint32_t stsm_wgmma_offset(uint32_t mat_row, uint32_t mat_col) {
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kElemsPerBankGroup = kNumBankGroupBytes / sizeof(dtype_t);  // 8 for bf16
    constexpr uint32_t kRowsPerWarp = 16;  // WGMMA M=64 / 4 warps
    constexpr uint32_t kSwizzleAtomWidth =
        kSwizzleMode / sizeof(dtype_t);  // 64 for 128-byte swizzle

    // Which warp's region contains this row
    uint32_t warp_idx = mat_row / kRowsPerWarp;
    uint32_t row_in_warp = mat_row % kRowsPerWarp;

    // Which swizzle atom (column group) this column is in
    uint32_t atom_offset = mat_col / kSwizzleAtomWidth;
    uint32_t col_in_atom = mat_col % kSwizzleAtomWidth;

    // The stmatrix m8n8.x2 layout arranges data in 8x8 tiles
    // Within the swizzle atom, find the 8-column tile and position
    uint32_t tile_col =
        col_in_atom / kElemsPerBankGroup;  // which 8-element tile (0-7 for 64-wide atom)
    uint32_t elem_in_tile = col_in_atom % kElemsPerBankGroup;  // position within tile (0-7)

    // The m8n8 tile layout: 8 rows x 8 cols
    // Within each warp's 16 rows, we have 2 vertical 8-row groups
    uint32_t row_tile = row_in_warp / 8;     // 0 or 1
    uint32_t row_in_tile = row_in_warp % 8;  // 0-7

    // Compute bank_group_index matching the STSM store pattern
    // The STSM uses: bank_group_index = in_atom_offset + lane_idx * 8
    // where lane_idx encodes position within the m8n8 block
    //
    // For reading: we reverse this mapping
    // row (in STSM formula) = bank_group_index / 8 = lane_idx
    // col (in STSM formula) = bank_group_index % 8 = in_atom_offset
    //
    // The lane_idx in stmatrix corresponds to: row_in_tile + row_tile * 8 + tile_col * 2
    // (This is the stmatrix thread-to-position mapping)
    uint32_t stsm_row = row_in_tile + row_tile * 8;  // 0-15 range for rows within warp
    uint32_t stsm_col = tile_col;                    // 0-7 for 8 bank groups

    // Compute byte offset then convert to element offset
    // byte_offset = atom_offset * row_stride * kSwizzleMode +
    //               warp_idx * kRowsPerWarp * kSwizzleMode +
    //               stsm_row * kSwizzleMode +
    //               stsm_col * kNumBankGroupBytes +
    //               elem_in_tile * sizeof(dtype_t)
    uint32_t byte_offset = atom_offset * BLOCK_M * kSwizzleMode +
                           warp_idx * kRowsPerWarp * kSwizzleMode + stsm_row * (kSwizzleMode) +
                           stsm_col * kNumBankGroupBytes + elem_in_tile * sizeof(dtype_t);

    return byte_offset / sizeof(dtype_t);
}

template <uint32_t SMEM_BYTES, uint32_t NUM_THREADS>
__device__ __forceinline__ void zero_smem(void* smem_ptr, int tidx) {
    constexpr int NUM_UINT4 = SMEM_BYTES / sizeof(uint4);
    constexpr int UINT4_PER_THREAD = (NUM_UINT4 + NUM_THREADS - 1) / NUM_THREADS;

    uint4* smem_vec = reinterpret_cast<uint4*>(smem_ptr);
    uint4 zero = make_uint4(0, 0, 0, 0);

#pragma unroll
    for (int i = 0; i < UINT4_PER_THREAD; i++) {
        int idx = tidx + i * NUM_THREADS;
        if (idx < NUM_UINT4) {
            smem_vec[idx] = zero;
        }
    }
}

__device__ __forceinline__ int2 tid_to_accum_row(int tid) {
    return make_int2(tid / 4, tid / 4 + 8);
}

__device__ __forceinline__ int2 get_accum_row_col(int tid, int val_idx) {
    int warp_idx = tid / 32;
    int lane_idx = tid & 0x1f;
    int row_idx = lane_idx / 4 + warp_idx * 16 + (val_idx & 0x3) / 2 * 8;
    int col_chunk = val_idx / 4;
    int col_idx = col_chunk * 8 + (lane_idx & 0x3) * 2 + (val_idx & 0x1);

    return make_int2(row_idx, col_idx);
}

#ifndef DEVICE_ASSERT
#define DEVICE_ASSERT(cond)                                                                \
    do {                                                                                   \
        if (not(cond)) {                                                                   \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
            asm("trap;");                                                                  \
        }                                                                                  \
    } while (0)
#endif

// Torch type conversion functions are in dtype_torch.h to avoid c10 dependency
// Table of finfo attributes for FP32 and BF16
// These are the same as torch.finfo(torch.float32) and torch.finfo(torch.bfloat16)
struct DeviceTypeFInfo {
    float eps;
    float min;
    float max;
};

constexpr DeviceTypeFInfo kernel_finfo_table[] = {

    // this can be expanded later, mainly for testing
    // FP32
    {1.1920928955078125e-07f, -3.4028234663852886e+38f, 3.4028234663852886e+38f},
    // FP16 (not used here, but could be added)
    {6.103515625e-05f, -65504.0f, 65504.0f},
    // BF16
    {0.0078125f, -3.38953139e+38f, 3.38953139e+38f},
    // INT32
    {1.0f, -2147483648.0f, 2147483647.0f},
    // INT16
    {1.0f, -32768, 32767},
    // INT8
    {1.0f, -128, 127},
    // BOOL
    {1.0f, 0, 1},
    // FP8 (E4M3)
    {0.0625f, -448.0, 448.0}};

template <typename T>
inline const DeviceTypeFInfo& get_finfo_from_typename() {
    if constexpr (cute::is_same_v<T, float>) {
        return kernel_finfo_table[0];
    } else if constexpr (cute::is_same_v<T, __nv_bfloat16>) {
        return kernel_finfo_table[2];
    } else if constexpr (cute::is_same_v<T, cutlass::float_e4m3_t>) {
        return kernel_finfo_table[7];
    } else if constexpr (cute::is_same_v<T, uint32_t>) {
        return kernel_finfo_table[3];
    } else {
        printf("DType is not supported yet");
        assert(false);
        return kernel_finfo_table[0];
    }
}

template <typename Func_T>
struct PatternVisitor {
    Func_T func;

    __host__ __device__ explicit PatternVisitor(Func_T&& func) : func(std::forward<Func_T>(func)){};

    __host__ __device__ auto operator[](const uint32_t& i) { return func(i); }
};
