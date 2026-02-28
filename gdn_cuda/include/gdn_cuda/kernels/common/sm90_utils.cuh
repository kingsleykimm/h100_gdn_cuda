/*
Based off of
https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/common/sm90_utils.cuh
 */
#pragma once

#include <cutlass/numeric_types.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/numeric/integer_sequence.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <gdn_cuda/kernels/common/common.hpp>
#include <gdn_cuda/kernels/common/types.hpp>

#include "cute/util/type_traits.hpp"

template <int N_, typename MMA>
struct FP8MMA {
    template <size_t... Idx>
    __forceinline__ __device__ static void call_fma_impl(uint64_t const &desc_a,
                                                         uint64_t const &desc_b, float *d,
                                                         bool scale_d,
                                                         cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    __forceinline__ __device__ static void wgmma(uint64_t const &desc_a, uint64_t const &desc_b,
                                                 float *d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_ / 2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
    // kNumAccum - 128 describes the number of FP32 accum registers used per WGMMA
    // tile, and this is a fixed bound because of per SM register capacities. So
    // we need to tile the M x N shape with these 128 regs.
};

template <int N>
struct FP8MMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 8)
            return MMA_64x8x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 16)
            return MMA_64x16x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 24)
            return MMA_64x24x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 32)
            return MMA_64x32x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 40)
            return MMA_64x40x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 48)
            return MMA_64x48x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 56)
            return MMA_64x56x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 64)
            return MMA_64x64x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 72)
            return MMA_64x72x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 80)
            return MMA_64x80x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 88)
            return MMA_64x88x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 96)
            return MMA_64x96x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 104)
            return MMA_64x104x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 112)
            return MMA_64x112x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 120)
            return MMA_64x120x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 128)
            return MMA_64x128x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 136)
            return MMA_64x136x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 144)
            return MMA_64x144x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 152)
            return MMA_64x152x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 160)
            return MMA_64x160x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 168)
            return MMA_64x168x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 176)
            return MMA_64x176x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 184)
            return MMA_64x184x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 192)
            return MMA_64x192x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 200)
            return MMA_64x200x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 208)
            return MMA_64x208x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 216)
            return MMA_64x216x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 224)
            return MMA_64x224x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 232)
            return MMA_64x232x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 240)
            return MMA_64x240x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 248)
            return MMA_64x248x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 256)  // max of 256 since each thread takes an 8-column
                                 // chunk of the output matrix
            return MMA_64x256x32_F32E4M3E4M3_SS_TN();
    }

    static constexpr auto select_type() { return FP8MMA<N, decltype(select_mma())>(); }

    using type = decltype(select_type());
};

template <int N_, typename MMA>
struct BF16MMA {
    template <size_t... Idx>
    __forceinline__ __device__ static void call_fma_impl(uint64_t const &desc_a,
                                                         uint64_t const &desc_b, float *d,
                                                         bool scale_d,
                                                         cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    __forceinline__ __device__ static void wgmma(uint64_t const &desc_a, uint64_t const &desc_b,
                                                 float *d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_ / 2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 16;
    static constexpr int kNumAccum = M * N / 128;
};

template <Major kMajor>
constexpr cute::SM90::GMMA::Major to_sm90_major() {
    CUTE_STATIC_ASSERT(kMajor == Major::K or kMajor == Major::MN, "Invalid major-ness");
    return kMajor == Major::K ? cute::SM90::GMMA::Major::K : cute::SM90::GMMA::Major::MN;
}

template <int N, Major kMajorA = Major::K, Major kMajorB = Major::K,
          cute::GMMA::ScaleIn kScaleIn = cute::GMMA::ScaleIn::One>
struct BF16MMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        constexpr auto kGMMAMajorA = to_sm90_major<kMajorA>();
        constexpr auto kGMMAMajorB = to_sm90_major<kMajorB>();
        if constexpr (N == 8)
            return MMA_64x8x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 16)
            return MMA_64x16x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 24)
            return MMA_64x24x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 32)
            return MMA_64x32x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 40)
            return MMA_64x40x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 48)
            return MMA_64x48x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 56)
            return MMA_64x56x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 64)
            return MMA_64x64x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 72)
            return MMA_64x72x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 80)
            return MMA_64x80x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 88)
            return MMA_64x88x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 96)
            return MMA_64x96x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 104)
            return MMA_64x104x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 112)
            return MMA_64x112x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 120)
            return MMA_64x120x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 128)
            return MMA_64x128x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 136)
            return MMA_64x136x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 144)
            return MMA_64x144x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 152)
            return MMA_64x152x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 160)
            return MMA_64x160x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 168)
            return MMA_64x168x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 176)
            return MMA_64x176x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 184)
            return MMA_64x184x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 192)
            return MMA_64x192x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 200)
            return MMA_64x200x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 208)
            return MMA_64x208x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 216)
            return MMA_64x216x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 224)
            return MMA_64x224x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 232)
            return MMA_64x232x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 240)
            return MMA_64x240x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 248)
            return MMA_64x248x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
        if constexpr (N == 256)
            return MMA_64x256x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB, kScaleIn>();
    }

    static constexpr auto select_type() { return BF16MMA<N, decltype(select_mma())>(); }

    using type = decltype(select_type());
};

template <int N_, typename MMA>
struct BF16MMARS {
    template <size_t... Idx>
    __forceinline__ __device__ static void call_fma_impl(uint32_t const &a00, uint32_t const &a01,
                                                         uint32_t const &a02, uint32_t const &a03,
                                                         uint64_t const &desc_b, float *d,
                                                         bool scale_d,
                                                         cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(a00, a01, a02, a03, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    __forceinline__ __device__ static void wgmma(uint32_t const &a00, uint32_t const &a01,
                                                 uint32_t const &a02, uint32_t const &a03,
                                                 uint64_t const &desc_b, float *d, bool scale_d) {
        call_fma_impl(a00, a01, a02, a03, desc_b, d, scale_d, cute::make_index_sequence<N_ / 2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 16;
    static constexpr int kNumAccum = M * N / 128;
};

template <int N, Major kMajorB = Major::K>
struct BF16MMASelectorRS {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        constexpr auto kGMMAMajorA =
            cute::SM90::GMMA::Major::K;  // RS atoms require A to be K-major
        constexpr auto kGMMAMajorB = to_sm90_major<kMajorB>();
        if constexpr (N == 8)
            return MMA_64x8x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 16)
            return MMA_64x16x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 24)
            return MMA_64x24x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 32)
            return MMA_64x32x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 40)
            return MMA_64x40x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 48)
            return MMA_64x48x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 56)
            return MMA_64x56x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 64)
            return MMA_64x64x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 72)
            return MMA_64x72x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 80)
            return MMA_64x80x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 88)
            return MMA_64x88x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 96)
            return MMA_64x96x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 104)
            return MMA_64x104x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 112)
            return MMA_64x112x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 120)
            return MMA_64x120x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 128)
            return MMA_64x128x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 136)
            return MMA_64x136x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 144)
            return MMA_64x144x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 152)
            return MMA_64x152x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 160)
            return MMA_64x160x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 168)
            return MMA_64x168x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 176)
            return MMA_64x176x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 184)
            return MMA_64x184x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 192)
            return MMA_64x192x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 200)
            return MMA_64x200x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 208)
            return MMA_64x208x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 216)
            return MMA_64x216x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 224)
            return MMA_64x224x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 232)
            return MMA_64x232x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 240)
            return MMA_64x240x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 248)
            return MMA_64x248x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 256)
            return MMA_64x256x16_F32BF16BF16_RS<kGMMAMajorA, kGMMAMajorB>();
    }

    static constexpr auto select_type() { return BF16MMARS<N, decltype(select_mma())>(); }

    using type = decltype(select_type());
};

template <uint32_t kSwizzleMode>
struct getSwizzleEnum {
    static int get_value() {
        if constexpr (kSwizzleMode == 0) {
            return 0;
        } else if constexpr (kSwizzleMode == 128) {
            return 1;
        } else if constexpr (kSwizzleMode == 64) {
            return 2;
        } else if constexpr (kSwizzleMode == 32) {
            return 3;
        }
    }
};

template <class ClusterShape>
CUTLASS_DEVICE bool is_same_row_or_col(int dst_block_id, dim3 block_id,
                                       ClusterShape cluster_shape) {
    return (((dst_block_id % cute::size<0>(cluster_shape)) == block_id.x) ||
            (((dst_block_id / cute::size<0>(cluster_shape)) == block_id.y)));
}

template <class ClusterShape>
CUTLASS_DEVICE void setup_multicast_wgmma_threads(bool &signalling_thread, uint32_t &dst_cta_id,
                                                  int warp_idx, ClusterShape cluster_shape) {
    auto [signalling_thread_, dst_cta_id_] =
        cutlass::detail::spread_arrivals_to_warpgroup(threadIdx.x % 128, warp_idx);
    signalling_thread = signalling_thread_;
    dst_cta_id = dst_cta_id_;
    signalling_thread &= dst_cta_id < size(cluster_shape);
    signalling_thread &= is_same_row_or_col(dst_cta_id, cute::block_id_in_cluster(), cluster_shape);
}

// following DeepGEMM, custom TMA copy dispatch - only one thread should call in
// a single CTA
// Use void const* instead of CUtensorMap* to avoid cuda.h dependency
template <int BLOCK_INNER, int BLOCK_OUTER, size_t SWIZZLE_SIZE, typename tdtype_t, int NDIM = 2>
__device__ __forceinline__ void tma_copy(void const *tensor_map,
                                         cutlass::arch::ClusterTransactionBarrier *barrier,
                                         tdtype_t *smem_addr, uint32_t inner_idx,
                                         uint32_t outer_idx, int num_multicast,
                                         uint32_t crd2_idx = 0, uint32_t crd3_idx = 0,
                                         uint32_t crd4_idx = 0) {
    int swizzle_atom_size = BLOCK_INNER;
    if (SWIZZLE_SIZE > 0) {
        swizzle_atom_size =
            SWIZZLE_SIZE / cutlass::bits_to_bytes(cutlass::sizeof_bits<tdtype_t>::value);
    }
    // NOTE : Shared memory layout is like this:
    /*
    Shared Memory Layout:
  ┌──────────────────────────────┐
  │  Atom 0: 64 × 32 elements    │  smem_ptr + 0 * 64 * 32
  │  (outer rows 0-63, inner 0-31)│
  ├──────────────────────────────┤
  │  Atom 1: 64 × 32 elements    │  smem_ptr + 1 * 64 * 32
  │  (outer rows 0-63, inner 32-63)│
  ├──────────────────────────────┤
  │  Atom 2: 64 × 32 elements    │  smem_ptr + 2 * 64 * 32
  │  (outer rows 0-63, inner 64-95)│
  ├──────────────────────────────┤
  │  Atom 3: 64 × 32 elements    │  smem_ptr + 3 * 64 * 32
  │  (outer rows 0-63, inner 96-127)│
  └──────────────────────────────┘
    */

    if constexpr (NDIM == 1) {
        if (num_multicast == 1) {
            for (int i = 0; i < BLOCK_INNER / swizzle_atom_size; i++) {
                if (cute::block_rank_in_cluster() == 0) {
                    cute::SM90_TMA_LOAD_MULTICAST_1D::copy(
                        tensor_map, reinterpret_cast<uint64_t *>(barrier), (1 << num_multicast) - 1,
                        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                        smem_addr + i + swizzle_atom_size, inner_idx + i * swizzle_atom_size);
                }
            }
        } else {
            for (int i = 0; i < BLOCK_INNER / swizzle_atom_size; i++) {
                cute::SM90_TMA_LOAD_1D::copy(
                    tensor_map, reinterpret_cast<uint64_t *>(barrier),
                    static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                    smem_addr + i + swizzle_atom_size, inner_idx + i * swizzle_atom_size);
            }
        }
    }

    else if constexpr (NDIM == 2) {
        if (num_multicast == 1) {
            for (int i = 0; i < BLOCK_INNER / swizzle_atom_size; i++) {
                cute::SM90_TMA_LOAD_2D::copy(
                    tensor_map, reinterpret_cast<uint64_t *>(barrier),
                    static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                    smem_addr + i * swizzle_atom_size * BLOCK_OUTER,
                    inner_idx + i * swizzle_atom_size, outer_idx);
            }
        } else {
            for (int i = 0; i < BLOCK_INNER / swizzle_atom_size; i++) {
                if (cute::block_rank_in_cluster() == 0) {
                    cute::SM90_TMA_LOAD_MULTICAST_2D::copy(
                        tensor_map, reinterpret_cast<uint64_t *>(barrier), (1 << num_multicast) - 1,
                        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                        smem_addr + i * swizzle_atom_size * BLOCK_OUTER,
                        inner_idx + i * swizzle_atom_size, outer_idx);
                }
            }
        }
    } else if constexpr (NDIM == 3) {
        if (num_multicast == 1) {
            for (int i = 0; i < BLOCK_INNER / swizzle_atom_size; i++) {
                cute::SM90_TMA_LOAD_3D::copy(
                    tensor_map, reinterpret_cast<uint64_t *>(barrier),
                    static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                    smem_addr + i * swizzle_atom_size * BLOCK_OUTER,
                    inner_idx + i * swizzle_atom_size, outer_idx, crd2_idx);
            }
        } else {
            for (int i = 0; i < BLOCK_INNER / swizzle_atom_size; i++) {
                if (cute::block_rank_in_cluster() == 0) {
                    cute::SM90_TMA_LOAD_MULTICAST_3D::copy(
                        tensor_map, reinterpret_cast<uint64_t *>(barrier), (1 << num_multicast) - 1,
                        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                        smem_addr + i * swizzle_atom_size * BLOCK_OUTER,
                        inner_idx + i * swizzle_atom_size, outer_idx, crd2_idx);
                }
            }
        }
    } else if constexpr (NDIM == 4) {
        if (num_multicast == 1) {
            for (int i = 0; i < BLOCK_INNER / swizzle_atom_size; i++) {
                cute::SM90_TMA_LOAD_4D::copy(
                    tensor_map, reinterpret_cast<uint64_t *>(barrier),
                    static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                    smem_addr + i * swizzle_atom_size * BLOCK_OUTER,
                    inner_idx + i * swizzle_atom_size, outer_idx, crd2_idx, crd3_idx);
            }
        } else {
            for (int i = 0; i < BLOCK_INNER / swizzle_atom_size; i++) {
                if (cute::block_rank_in_cluster() == 0) {
                    cute::SM90_TMA_LOAD_MULTICAST_4D::copy(
                        tensor_map, reinterpret_cast<uint64_t *>(barrier), (1 << num_multicast) - 1,
                        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                        smem_addr + i * swizzle_atom_size * BLOCK_OUTER,
                        inner_idx + i * swizzle_atom_size, outer_idx, crd2_idx, crd3_idx);
                }
            }
        }
    }
}

template <uint32_t kSwizzleMode>
constexpr auto get_layout() {
    CUTE_STATIC_ASSERT(kSwizzleMode == 0 || kSwizzleMode == 16 || kSwizzleMode == 32 ||
                       kSwizzleMode == 64 || kSwizzleMode == 128);
    if constexpr (kSwizzleMode == 0 || kSwizzleMode == 16) {
        return cute::GMMA::LayoutType::INTERLEAVE;
    } else if constexpr (kSwizzleMode == 32) {
        return cute::GMMA::LayoutType::B32;
    } else if constexpr (kSwizzleMode == 64) {
        return cute::GMMA::LayoutType::B64;
    } else if constexpr (kSwizzleMode == 128) {
        return cute::GMMA::LayoutType::B128;
    }
}

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_block_atom_inner_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

template <Major kMajor, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          typename dtype_t>
__device__ __forceinline__ uint32_t get_gmma_desc_stride_k() {
    return kMajor == Major::K ? 1 : get_block_atom_inner_size<BLOCK_MN, kSwizzleMode, dtype_t>();
}

template <Major kMajor, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          typename dtype_t>
__device__ __forceinline__ uint32_t get_gmma_desc_stride_mn() {
    return kMajor == Major::K ? 1 : get_block_atom_inner_size<BLOCK_MN, kSwizzleMode, dtype_t>();
}

template <typename dtype_t>
CUTE_DEVICE cute::GmmaDescriptor make_smem_desc(dtype_t *smem_ptr, uint32_t layout_type,
                                                uint32_t LBO, uint32_t SBO) {
    cute::GmmaDescriptor desc;
    const auto &uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.base_offset_ =
        0;  // assuming that smem_ptr is 1024 aligned - this is a little dangerous
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.leading_byte_offset_ = LBO >> 4;
    desc.bitfield.stride_byte_offset_ = SBO >> 4;
    desc.bitfield.layout_type_ = layout_type;

    return desc;
}

template <Major kMajor, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          typename dtype_t>
__device__ __forceinline__ cute::GmmaDescriptor create_wgmma_desc(dtype_t *smem_ptr) {
    // mn_idx here is the
    const uint32_t k_stride =
        (kMajor == Major::K) ? 1 : get_block_atom_inner_size<BLOCK_MN, kSwizzleMode, dtype_t>();
    const auto &layout_type = get_layout<kSwizzleMode>();

    if constexpr (kMajor == Major::K) {
        CUTE_STATIC_ASSERT(kSwizzleMode > 16, "Invalid swizzle mode");
        CUTE_STATIC_ASSERT(BLOCK_K * sizeof(dtype_t) == kSwizzleMode,
                           "BLOCK_K * sizeof(dtype_t) should be equal to kSwizzleMode");

        // CUTE_STATIC_ASSERT(kSwizzleMode == BLOCK_K * sizeof(dtype_t), "There should only be one
        // atom in the K direction"); SBO, LBO : for K major, stride between atoms on MN, K resp.
        // since we enforce only one atom in K direction, LBO is 0 swizzle atoms are enforced to 8
        // rows
        uint32_t LBO;
        if constexpr (kSwizzleMode <= 16) {
            LBO = BLOCK_MN * 16;
        } else {
            LBO = 1;
        }

        constexpr uint32_t BLOCK_K_ATOM =
            get_block_atom_inner_size<BLOCK_K, kSwizzleMode, dtype_t>();
        const uint32_t SBO = 8 * BLOCK_K_ATOM * sizeof(dtype_t);
        // const uint32_t LBO = BLOCK_MN * kSwizzleMode;

        return make_smem_desc(smem_ptr, static_cast<uint32_t>(layout_type), LBO, SBO);
    } else {
        CUTE_STATIC_ASSERT(kSwizzleMode > 0, "Invalid swizzle mode");

        // from DeepGEMM::
        // Atom size: `kSwizzleMode` (in bytes, on MN) x 8
        // NOTES: `kSwizzleMode == 16` mean non-swizzling but interleaving
        // {SBO, LBO} means the byte stride between atoms on {K, MN} for swizzling
        // {SBO, LBO} means the byte stride between atoms on {MN, K} for
        // non-swizzling
        constexpr uint32_t BLOCK_MN_ATOM =
            get_block_atom_inner_size<BLOCK_MN, kSwizzleMode, dtype_t>();
        const int num_swizzle_atoms = BLOCK_MN / (kSwizzleMode / sizeof(dtype_t));
        // uint32_t SBO = 8 * kSwizzleMode * num_swizzle_atoms;
        uint32_t SBO = 8 * BLOCK_MN_ATOM * sizeof(dtype_t);
        uint32_t LBO = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);  // offset by entire chunk
        if constexpr (kSwizzleMode == 16) {  // no swizzling, but interleave means swap
            uint32_t temp = SBO;
            SBO = LBO;
            LBO = temp;
        }
        return make_smem_desc(smem_ptr, static_cast<uint32_t>(layout_type), LBO, SBO);
    }
}

// closely following cutlass's make_gmma_desc

// inline ptx

__forceinline__ __device__ uint32_t pack_float_2(float a, float b) {
    uint32_t val;
    asm volatile("{ cvt.rn.bf16x2.f32 %0, %1, %2; }" : "=r"(val) : "f"(a), "f"(b) :);
    return val;
}

// store into shared, "l" = long, "f" = float
CUTE_DEVICE void st_shared(const __nv_bfloat16 *ptr, __nv_bfloat16 val) {
    uint16_t val16 = __bfloat16_as_ushort(val);
    asm volatile("{ st.shared.u16 [%0], %1; }"
                 :
                 : "l"(__cvta_generic_to_shared(ptr)), "h"(val16)
                 :);
}

CUTE_DEVICE void st_shared(const float *ptr, float val) {
    asm volatile("{ st.shared.f32 [%0], %1; }" : : "l"(__cvta_generic_to_shared(ptr)), "f"(val) :);
}

CUTE_DEVICE void st_shared(const float2 *ptr, float2 val) {
    asm volatile("{ st.shared.v2.f32 [%0], {%1, %2}; }"
                 :
                 : "l"(__cvta_generic_to_shared(ptr)), "f"(val.x), "f"(val.y)
                 :);
}

CUTE_DEVICE void st_shared(const uint32_t *ptr, uint32_t val) {
    asm volatile("{ st.shared.u32 [%0], %1; }" ::"l"(__cvta_generic_to_shared(ptr)), "r"(val));
}

CUTE_DEVICE void st_shared(const uint2 *ptr, uint2 val) {
    asm volatile("{ st.shared.v2.u32 [%0], {%1, %2}; }" ::"l"(__cvta_generic_to_shared(ptr)),
                 "r"(val.x), "r"(val.y));
}

CUTE_DEVICE void st_shared(const uint4 *ptr, uint4 val) {
    asm volatile("{st.shared.v4.u32 [%0], {%1, %2, %3, %4};}" ::"l"(__cvta_generic_to_shared(ptr)),
                 "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

CUTE_DEVICE float ld_shared(float *ptr) {
    float ret;
    asm volatile("{ ld.shared.f32 %0, [%1]; }"
                 : "=f"(ret)
                 : "l"(__cvta_generic_to_shared(ptr))
                 : "memory");
    return ret;
}

CUTE_DEVICE float2 ld_shared(float2 *ptr) {
    float ret1, ret2;
    asm volatile("{ ld.shared.v2.f32 {%0, %1}, [%2]; }"
                 : "=f"(ret1), "=f"(ret2)
                 : "l"(__cvta_generic_to_shared(ptr))
                 : "memory");
    return make_float2(ret1, ret2);
}

CUTE_DEVICE float4 ld_shared(float4 *ptr) {
    float ret1, ret2, ret3, ret4;
    asm volatile("{ ld.shared.v4.f32 {%0, %1, %2, %3}, [%4]; }"
                 : "=f"(ret1), "=f"(ret2), "=f"(ret3), "=f"(ret4)
                 : "l"(__cvta_generic_to_shared(ptr))
                 : "memory");
    return make_float4(ret1, ret2, ret3, ret4);
}

CUTE_DEVICE __nv_bfloat162 ld_sharedbf16x2(__nv_bfloat16 *ptr) {
    uint32_t ret;
    asm volatile("{ ld.shared.u32 %0, [%1]; }"
                 : "=r"(ret)
                 : "l"(__cvta_generic_to_shared(ptr))
                 : "memory");
    return *reinterpret_cast<__nv_bfloat162 *>(&ret);
}

CUTE_DEVICE __nv_bfloat16 ld_shared(__nv_bfloat16 *ptr) {
    uint16_t ret;
    asm volatile("{ ld.shared.u16 %0, [%1]; }"
                 : "=h"(ret)
                 : "l"(__cvta_generic_to_shared(ptr))
                 : "memory");
    return __short_as_bfloat16(ret);
    ;
}

CUTE_DEVICE uint32_t ld_shared(uint32_t *ptr) {
    uint32_t ret;
    asm volatile("{ ld.shared.u32 %0, [%1]; }"
                 : "=r"(ret)
                 : "l"(__cvta_generic_to_shared(ptr))
                 : "memory");
    return ret;
}

CUTE_DEVICE uint2 ld_shared(uint2 *ptr) {
    uint32_t ret1, ret2;
    asm volatile("{ ld.shared.v2.u32 {%0, %1}, [%2]; }"
                 : "=r"(ret1), "=r"(ret2)
                 : "l"(__cvta_generic_to_shared(ptr))
                 : "memory");
    return make_uint2(ret1, ret2);
}

CUTE_DEVICE uint4 ld_shared(uint4 *ptr) {
    uint32_t ret1, ret2, ret3, ret4;
    asm volatile("{ ld.shared.v4.u32 {%0, %1, %2, %3}, [%4]; }"
                 : "=r"(ret1), "=r"(ret2), "=r"(ret3), "=r"(ret4)
                 : "l"(__cvta_generic_to_shared(ptr))
                 : "memory");
    return make_uint4(ret1, ret2, ret3, ret4);
}

// special STSM matrix instruction (CC 9.0 <=) that loads from rMem to Smem
template <typename dtype_t>
struct custom_SM90_U32x2_STSM_N {
    CUTE_DEVICE static void copy(dtype_t src_0, dtype_t src_1, void *smem_dst) {
        const uint32_t src[2] = {*reinterpret_cast<uint32_t *>(&src_0),
                                 *reinterpret_cast<uint32_t *>(&src_1)};
        asm volatile("{ stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2}; }\n" ::"l"(
                         __cvta_generic_to_shared(smem_dst)),
                     "r"(src[0]), "r"(src[1]));
    }
};

struct custom_SM90_U32x2_STLM_N {
    CUTE_DEVICE static uint2 load(void *smem_dst) {
        uint32_t src_0, src_1;
        asm volatile("{ ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2]; }\n"
                     : "=r"(src_0), "=r"(src_1)
                     : "l"(__cvta_generic_to_shared(smem_dst))
                     : "memory");
        return make_uint2(src_0, src_1);
    }
};

__forceinline__ __host__ __device__ uint32_t ld_volatile_u32(uint32_t *ptr) {
    uint32_t x;
    asm volatile("{ ld.volatile.u32 %0, [%1]; }\n" : "=r"(x) : "l"(ptr) : "memory");
    return x;
}

__forceinline__ __host__ __device__ void st_volatile_u32(uint32_t *ptr, uint32_t val) {
    asm volatile("{ st.volatile.u32 [%1], %0; }\n" ::"r"(val), "l"(ptr) :);
}

__forceinline__ __host__ __device__ uint32_t ld_acquire_u32(uint32_t *ptr) {
    uint32_t x;
    asm volatile("{ ld.acquire.sys.global.u32 %0, [%1]; }\n" : "=r"(x) : "l"(ptr) :);
    return x;
}

__forceinline__ __host__ __device__ void st_release_u32(uint32_t *ptr, uint32_t val) {
    asm volatile("{ st.release.sys.global.u32 [%1], %0; }\n" ::"r"(val), "l"(ptr) :);
}

__forceinline__ __host__ __device__ uint32_t ti_elect_one_sync() {
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %2;\n"
        "@%%px mov.s32 %1, 1;\n"
        "     mov.s32 %0, %%rx;\n"
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

// this will use the noncoherent (texture) cache, which doesn't require maintaining coherency with
// global memory. only unintended for read-only data, much faster

__forceinline__ __host__ __device__ uint4 ld_global_nc_uint4(const uint4 *ptr) {
    uint4 val;

    // L1 no allocate
    // L2 - cache prefetch size of 256B, since we're almost always fetching contiguous chunks of
    // uint4
    asm volatile("{ ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4]; }\n"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(ptr)
                 :);
    return val;
}

__forceinline__ __device__ void st_global_nc_uint4(uint4 val, const uint4 *ptr) {
    asm volatile("{ st.global.L1::no_allocate.v4.u32 [%0], {%1, %2, %3, %4}; }\n" ::"l"(ptr),
                 "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                 :);
}

__forceinline__ __device__ uint16_t ld_global_uint16_dispatch(uint16_t *ptr) {
    uint16_t result;
    asm volatile("{ ld.global.u16 %0, [%1]; \n}" : "=h"(result) : "l"(ptr) :);
    return result;
}

__forceinline__ __device__ uint32_t ld_global_uint32_dispatch(uint32_t *ptr) {
    uint32_t result;
    asm volatile("{ ld.global.u32 %0, [%1]; \n}" : "=r"(result) : "l"(ptr) :);
    return result;
}

__forceinline__ __device__ uint2 ld_global_uint2_dispatch(uint2 *ptr) {
    uint2 result;
    asm volatile("{ ld.global.v2.u32 {%0, %1}, [%2]; \n}"
                 : "=r"(result.x), "=r"(result.y)
                 : "l"(ptr)
                 :);
    return result;
}

__forceinline__ __device__ uint4 ld_global_uint4_dispatch(uint4 *ptr) {
    uint4 result;
    asm volatile("{ ld.global.v4.u32 {%0, %1, %2, %3}, [%4]; \n}"
                 : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
                 : "l"(ptr)
                 :);
    return result;
}
template <typename uint_t>
__forceinline__ __device__ uint_t ld_global_uint_dispatch(uint_t *ptr) {
    if constexpr (cute::is_same_v<uint_t, uint16_t>) {
        return ld_global_uint16_dispatch(ptr);
    } else if constexpr (cute::is_same_v<uint_t, uint32_t>) {
        return ld_global_uint32_dispatch(ptr);
    } else if constexpr (cute::is_same_v<uint_t, uint2>) {
        return ld_global_uint2_dispatch(ptr);
    } else if constexpr (cute::is_same_v<uint_t, uint4>) {
        return ld_global_uint4_dispatch(ptr);
    } else {
        DEVICE_ASSERT(false);
    }
}

__forceinline__ __device__ void st_global_uint16_dispatch(uint16_t val, uint16_t *ptr) {
    asm volatile("{ st.global.u16 [%0], %1; \n}" ::"l"(ptr), "h"(val) :);
}

__forceinline__ __device__ void st_global_uint32_dispatch(uint32_t val, uint32_t *ptr) {
    asm volatile("{ st.global.u32 [%0], %1; \n}" ::"l"(ptr), "r"(val) :);
}

__forceinline__ __device__ void st_global_uint2_dispatch(uint2 val, uint2 *ptr) {
    asm volatile("{ st.global.v2.u32 [%0], {%1, %2}; \n}" ::"l"(ptr), "r"(val.x), "r"(val.y) :);
}

__forceinline__ __device__ void st_global_uint4_dispatch(uint4 val, uint4 *ptr) {
    asm volatile("{ st.global.v4.u32 [%0], {%1, %2, %3, %4}; \n}" ::"l"(ptr), "r"(val.x),
                 "r"(val.y), "r"(val.z), "r"(val.w)
                 :);
}

template <typename uint_t>
__forceinline__ __device__ void st_global_uint_dispatch(uint_t val, uint_t *ptr) {
    if constexpr (cute::is_same_v<uint_t, uint16_t>) {
        st_global_uint16_dispatch(val, ptr);
    } else if constexpr (cute::is_same_v<uint_t, uint32_t>) {
        st_global_uint32_dispatch(val, ptr);
    } else if constexpr (cute::is_same_v<uint_t, uint2>) {
        st_global_uint2_dispatch(val, ptr);
    } else if constexpr (cute::is_same_v<uint_t, uint4>) {
        st_global_uint4_dispatch(val, ptr);
    } else {
        DEVICE_ASSERT(false);
    }
}

// =============== Cluster Operations ================= //

__forceinline__ __device__ uint32_t mapa_shared_addr_cluster(uint32_t *smem_addr,
                                                             uint32_t peer_rank) {
    uint32_t result_u32;
    uint32_t smem_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
    asm volatile("{mapa.shared::cluster.u32 %0, %1, %2; } \n"
                 : "=r"(result_u32)
                 : "r"(smem_u32), "r"(peer_rank)
                 :);
    return result_u32;
}

__forceinline__ __device__ uint32_t atom_add_shared_cluster(uint32_t smem_addr, uint32_t val) {
    uint32_t result;
    asm volatile("{atom.add.shared::cluster.u32 %0, [%1], %2; \n}"
                 : "=r"(result)
                 : "r"(smem_addr), "r"(val)
                 :);
    return result;
}

__forceinline__ __device__ uint32_t ld_shared_cluster_u32(uint32_t smem_addr) {
    uint32_t result;
    asm volatile("{ld.shared::cluster.u32 %0, [%1]; \n}" : "=r"(result) : "r"(smem_addr) :);
    return result;
}

// Load bf16 data from swizzled shared memory into float accumulator registers.
// This assumes a single swizzle atom (BLOCK_INNER * sizeof(bf16) == kSwizzleMode for K-major
// layouts). Template params:
//   BLOCK_INNER: number of elements in the inner (K) dimension
//   kSwizzleMode: swizzle mode in bytes (0 for non-swizzled, 32/64/128 for swizzled)
//   WGMMA_M_PER_WARP: rows per warp (typically MMA::M / 4 = 16)
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode,
          uint32_t WGMMA_M_PER_WARP>
__device__ __forceinline__ void load_swizzled_smem_to_accum(__nv_bfloat16 *smem_ptr, float *accum,
                                                            int warp_idx, int lane_idx) {
    constexpr uint32_t kNumBankGroupBytes = 16;
#pragma unroll
    for (int col_idx = 0; col_idx < BLOCK_INNER / 8; col_idx++) {
        float *shifted_accum = accum + col_idx * 4;
        uint8_t *ptr = nullptr;
        if constexpr (kSwizzleMode > 0) {
            constexpr uint32_t kSwizzleAtomSize =
                kSwizzleMode / sizeof(__nv_bfloat16);  // elements per swizzle atom
            int atom_offset = col_idx / (kSwizzleAtomSize / 8);
            int in_atom_offset = col_idx % (kSwizzleAtomSize / 8);
            int bank_group_index = in_atom_offset + lane_idx * (kSwizzleMode / kNumBankGroupBytes);
            int row = bank_group_index / 8, col = bank_group_index % 8;
            col ^= row % (kSwizzleMode / 16);
            // Reshape layout from (BLOCK_M, kSwizzleMode / kNumBankGroupBytes) to
            // (BLOCK_M * kSwizzleMode / kNumBankGroupBytes / 8, 8) to match TMA 128B line layout
            ptr = reinterpret_cast<uint8_t *>(smem_ptr) +
                  WGMMA_M_PER_WARP * warp_idx * kSwizzleMode +
                  atom_offset * BLOCK_OUTER * kSwizzleMode + row * (kNumBankGroupBytes * 8) +
                  col * kNumBankGroupBytes;
        } else {
            ptr = reinterpret_cast<uint8_t *>(
                smem_ptr + (WGMMA_M_PER_WARP * warp_idx + lane_idx) * BLOCK_INNER + col_idx * 8);
        }
        const uint2 data = custom_SM90_U32x2_STLM_N::load(ptr);
        __nv_bfloat162 bf16_pair1 = *reinterpret_cast<const __nv_bfloat162 *>(&data.x);
        __nv_bfloat162 bf16_pair2 = *reinterpret_cast<const __nv_bfloat162 *>(&data.y);

        shifted_accum[0] = __bfloat162float(bf16_pair1.x);
        shifted_accum[1] = __bfloat162float(bf16_pair1.y);
        shifted_accum[2] = __bfloat162float(bf16_pair2.x);
        shifted_accum[3] = __bfloat162float(bf16_pair2.y);
    }
}

// Store float accumulator registers as bf16 to swizzled shared memory.
// Handles multiple swizzle atoms when BLOCK_INNER > swizzle_atom_size.
// Template params:
//   BLOCK_INNER: number of elements in the inner (N/V) dimension
//   BLOCK_OUTER: number of elements in the outer (M) dimension (for atom stride calculation)
//   kSwizzleMode: swizzle mode in bytes (0 for non-swizzled, 32/64/128 for swizzled)
//   WGMMA_M_PER_WARP: rows per warp (typically MMA::M / 4 = 16)
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode,
          uint32_t WGMMA_M_PER_WARP>
__device__ __forceinline__ void store_accum_to_swizzled_smem(float *accum, __nv_bfloat16 *smem_ptr,
                                                             int warp_idx, int lane_idx) {
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumIter =
        BLOCK_INNER / 8;  // Each iteration handles 8 elements (4 accum values * 2 bf16)
#pragma unroll
    for (int i = 0; i < kNumIter; i++) {
        uint8_t *ptr = nullptr;
        if constexpr (kSwizzleMode > 0) {
            constexpr uint32_t kSwizzleAtomSize =
                kSwizzleMode / sizeof(__nv_bfloat16);  // elements per swizzle atom
            int atom_offset = i / (kSwizzleAtomSize / 8);
            int in_atom_offset = i % (kSwizzleAtomSize / 8);
            int bank_group_index = in_atom_offset + lane_idx * (kSwizzleMode / kNumBankGroupBytes);
            int row = bank_group_index / 8, col = bank_group_index % 8;
            col ^= row % (kSwizzleMode / 16);  //
            ptr = reinterpret_cast<uint8_t *>(smem_ptr) + atom_offset * BLOCK_OUTER * kSwizzleMode +
                  WGMMA_M_PER_WARP * kSwizzleMode * warp_idx + row * (kNumBankGroupBytes * 8) +
                  col * kNumBankGroupBytes;
        } else {
            ptr = reinterpret_cast<uint8_t *>(
                smem_ptr + (WGMMA_M_PER_WARP * warp_idx + lane_idx) * BLOCK_INNER + i * 8);
        }
        custom_SM90_U32x2_STSM_N<__nv_bfloat162>::copy(
            __float22bfloat162_rn({accum[i * 4 + 0], accum[i * 4 + 1]}),
            __float22bfloat162_rn({accum[i * 4 + 2], accum[i * 4 + 3]}), ptr);
    }
    cute::tma_store_fence();
}

// vectorized loads
template <size_t PADDED_SF_K>
struct vec_load_size;

template <>
struct vec_load_size<2> {
    using type = float2;
};

template <>
struct vec_load_size<4> {
    using type = float4;
};
