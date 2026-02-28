/*
Based off of DeepGEMM Scheduler
*/
#include <gdn_cuda/kernels/common/common.hpp>
#include <gdn_cuda/kernels/common/sm90_utils.cuh>
#include <gdn_cuda/kernels/common/types.hpp>

#include "cute/config.hpp"
#include "cutlass/detail/helper_macros.hpp"

// DeepGEMM seems to only do 1D clusters, for simplicity, and then perform a 1D
// multicast Yet we still need to perform swizzling on the non-multicast
// direction in order to ensure maximal use of the loaded A/B tile, this method
// performs an extremely small search to find the best swizzle

template <uint32_t BLOCK_V, uint32_t kNumVHeads, uint32_t kNumKHeads, uint32_t kNumBlocks,
          uint32_t kNumTMAMulticast>
struct RecurrentGDNScheduler {
    int shape_k, shape_v;
    int cur_block_idx;
    int current_iter = -1;
    int num_heads, batch_size, num_v_blocks;
    int num_blocks;
    bool is_valid_work = true;
    int actual_num_blocks;
    bool is_peer_cta_alive = true;

    __device__ __forceinline__ RecurrentGDNScheduler(int shape_v, int batch_size) {
        // calculate grid size and place them in a cluster together, so when writing
        // partial sums it's easier to write through
        this->shape_v = shape_v;
        this->batch_size = batch_size;
        this->num_v_blocks = ti_ceil_div(shape_v, BLOCK_V);
        // Store actual work items count before padding
        this->actual_num_blocks = num_v_blocks * kNumVHeads * batch_size;
        // Pad total work items to multiple of kNumTMAMulticast to ensure whole
        // clusters stay alive
        this->num_blocks = ti_align(actual_num_blocks, kNumTMAMulticast);
    }

    __device__ __forceinline__ bool get_next_block(int &v_block_idx, int &v_head_idx,
                                                   int &k_head_idx, int &batch_index) {
        // kNumBlocks is the total number of blocks allocated for this persistent
        // scheduler however, we can pack multiple head indices into the same SM
        cur_block_idx = (++current_iter) * kNumBlocks + blockIdx.x;

        if (cur_block_idx >= num_blocks) {
            return false;
        }
        // is_peer_cta_alive = (cur_block_idx ^ 1) < actual_num_blocks &&
        // same_seq_slab && same_k_head;
        v_head_idx = cur_block_idx % kNumVHeads;
        k_head_idx =
            (v_head_idx * kNumKHeads) / kNumVHeads;  // equivalent to dividing by kNumTMAMulticast
        int v_head_blocked_idx = cur_block_idx / kNumVHeads;
        v_block_idx = v_head_blocked_idx % num_v_blocks;
        batch_index = v_head_blocked_idx / num_v_blocks;

        return true;
    }

    // Returns the effective multicast count for the current work item.
    // Returns kNumTMAMulticast if all CTAs in the cluster have VALID (non-padded)
    // work, otherwise returns 1 to disable multicast and let each CTA load
    // independently.
    __device__ __forceinline__ uint32_t get_effective_multicast() const {
        if constexpr (kNumTMAMulticast == 1) {
            return 1;
        }

        // CTAs in a cluster have consecutive blockIdx.x values.
        // Multicast is valid only if ALL CTAs in the cluster have valid
        // (non-padded) work. Check if the highest blockIdx.x in our cluster still
        // has a valid work item. IMPORTANT: Check against actual_num_blocks, not
        // num_blocks (which includes padding)
        uint32_t cluster_base = (blockIdx.x / kNumTMAMulticast) * kNumTMAMulticast;
        uint32_t max_block_in_cluster =
            current_iter * kNumBlocks + cluster_base + (kNumTMAMulticast - 1);
        return (max_block_in_cluster < actual_num_blocks) ? kNumTMAMulticast : 1;
    }

    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t &block_idx,
                                                       const uint32_t &block_size,
                                                       const uint32_t &global_size) {
        return block_idx * block_size;
    }
};

// ChunkGDNScheduler - Persistent scheduler for chunked GDN (Gated Delta
// Networks) Handles both varlen (packed sequences) and fixed padded settings
// BLOCK_M = chunk size (always 64)
// kNumHeads = number of query heads
// kNumSMs = number of blocks across the gridDim for persistent scheduling
// kIsVarLen = compile-time flag for varlen vs padded mode
template <uint32_t BLOCK_M, uint32_t kNumVHeads, uint32_t kNumKHeads, uint32_t kNumBlocks,
          uint32_t BLOCK_V, bool kIsVarLen = false>
struct ChunkGDNScheduler {
    // Runtime parameters
    int batch_size;
    int num_chunks;      // total number of chunks across all batches
    int *cu_seqlens;     // cumulative sequence lengths [batch_size + 1], only used in
                         // varlen mode
    int *chunk_indices;  // shape (num_chunks, 2) linearized - pairs of [batch_idx,
                         // chunk_idx]
    int num_v_blocks;
    // For padded mode
    int max_seq_len;

    // Scheduler state
    int current_iter;
    int cur_block_idx;
    int num_blocks;  // total blocks = num_chunks * kNumHeads

    int seq_start;
    int seq_end;
    int seq_len;

    // Varlen mode constructor
    __device__ __forceinline__ ChunkGDNScheduler(int batch_size, int shape_v, int num_chunks,
                                                 int *cu_seqlens, int *chunk_indices)
        : batch_size(batch_size),
          num_chunks(num_chunks),
          cu_seqlens(cu_seqlens),
          chunk_indices(chunk_indices),
          num_v_blocks(ti_ceil_div(shape_v, BLOCK_V)),
          max_seq_len(0),
          current_iter(-1) {
        CUTE_STATIC_ASSERT(kIsVarLen, "Cannot use varlen constructor with padded mode");
        num_blocks = num_chunks * kNumVHeads * num_v_blocks;
    }

    // Padded mode constructor
    __device__ __forceinline__ ChunkGDNScheduler(int batch_size, int shape_v, int max_seq_len)
        : batch_size(batch_size),
          num_chunks(0),
          cu_seqlens(nullptr),
          chunk_indices(nullptr),
          num_v_blocks(ti_ceil_div(shape_v, BLOCK_V)),
          max_seq_len(max_seq_len) {
        CUTE_STATIC_ASSERT(!kIsVarLen, "Cannot use padded constructor with varlen mode");
        // For padded mode, calculate num_chunks from batch_size and max_seq_len
        int chunks_per_batch = ti_ceil_div(max_seq_len, (int)BLOCK_M);
        num_chunks = batch_size * chunks_per_batch;
        num_blocks = num_chunks * kNumVHeads * num_v_blocks;
        current_iter = -1;
    }

    // Returns true if there's more work, false otherwise
    // Sets chunk_idx, batch_idx, head_idx for the current work item
    // Sets seq_end_in_chunk to the row index (0 to BLOCK_M-1) where sequence ends
    // within this chunk, or -1 if no sequence boundary exists in this chunk
    __device__ __forceinline__ bool get_next_block(int &chunk_idx, int &batch_idx, int &head_idx,
                                                   int &seq_end_in_chunk, int &v_block_idx) {
        cur_block_idx = (++current_iter) * kNumBlocks + blockIdx.x;

        if (cur_block_idx >= num_blocks) {
            return false;
        }

        // Layout: iterate over heads within each chunk
        // This keeps chunks that share the same KV data close together
        // perform thread block swizzling so that chunks in the same sequence are
        // close together for locality so fastest changing 'dimension' should be the
        // sequence index, not the head index
        int v_head_blocked_idx = cur_block_idx / kNumVHeads;
        v_block_idx = v_head_blocked_idx % num_v_blocks;
        head_idx = cur_block_idx % kNumVHeads;
        int global_chunk_idx = v_head_blocked_idx / num_v_blocks;
        if constexpr (kIsVarLen) {
            // Load batch_idx and chunk_idx from precomputed chunk_indices
            // chunk_indices is linearized as [batch_idx_0, chunk_idx_0, batch_idx_1,
            // chunk_idx_1, ...]
            int2 indices = __ldg(reinterpret_cast<int2 *>(chunk_indices + global_chunk_idx * 2));
            batch_idx = indices.x;
            chunk_idx = indices.y;

            // Compute sequence boundary for masking in varlen mode
            // Get sequence length for this batch
            int global_seq_start = __ldg(cu_seqlens + batch_idx);
            int global_seq_end = __ldg(cu_seqlens + batch_idx + 1);

            seq_start = global_seq_start + chunk_idx * BLOCK_M;
            seq_end = seq_start + BLOCK_M;
            // Check if sequence ends within this chunk
            if (seq_end > global_seq_end) {
                // seq_end_in_chunk is the last valid row index (0-indexed within chunk)
                seq_end_in_chunk = BLOCK_M - 1 - (seq_end - global_seq_end);
                seq_len = seq_end_in_chunk + 1;
            } else {
                seq_end_in_chunk = -1;  // no boundary in this chunk, or chunk is fully valid
                seq_len = BLOCK_M;
            }
        } else {
            // Padded mode: derive batch_idx and chunk_idx from linear index
            int chunks_per_batch = ti_ceil_div(max_seq_len, (int)BLOCK_M);
            batch_idx = global_chunk_idx / chunks_per_batch;
            chunk_idx = global_chunk_idx % chunks_per_batch;

            // In padded mode, sequence boundary is at max_seq_len for all batches
            seq_start = chunk_idx * BLOCK_M;
            seq_end = seq_start + BLOCK_M;

            if (seq_end > max_seq_len) {
                seq_end_in_chunk = BLOCK_M - (seq_end - max_seq_len);
                seq_len = seq_end_in_chunk - seq_start;
            } else {
                seq_end_in_chunk = -1;
                seq_len = max_seq_len;
            }
        }

        return true;
    }

    // Get the row offset into the global tensor for the start of this chunk
    // For Q tensor: shape (batch_size, seq_len, num_heads, head_dim) or
    // (total_tokens, num_heads, head_dim) For KV tensor: shape (batch_size,
    // seq_len, num_kv_heads, head_dim) or (total_tokens, num_kv_heads, head_dim)
    // Returns offset in units of elements (not bytes)
    template <bool kIsKV>
    __device__ __forceinline__ int get_row_offset(int &batch_idx, int &chunk_idx, int &head_idx,
                                                  int num_kv_heads, int head_dim) {
        if constexpr (kIsVarLen) {
            // Varlen: tensor is (total_tokens, num_heads, head_dim)
            int seq_start = __ldg(cu_seqlens + batch_idx);
            int row_in_global = seq_start + chunk_idx * BLOCK_M;

            if constexpr (kIsKV) {
                // For grouped-query attention, KV has fewer heads
                int kv_head_idx = head_idx / (kNumVHeads / num_kv_heads);
                return row_in_global * num_kv_heads * head_dim + kv_head_idx * head_dim;
            } else {
                return row_in_global * kNumVHeads * head_dim + head_idx * head_dim;
            }
        } else {
            // Padded: tensor is (batch_size, seq_len, num_heads, head_dim)
            int row_in_batch = chunk_idx * BLOCK_M;
            int global_row = batch_idx * max_seq_len + row_in_batch;

            if constexpr (kIsKV) {
                int kv_head_idx = head_idx / (kNumVHeads / num_kv_heads);
                return global_row * num_kv_heads * head_dim + kv_head_idx * head_dim;
            } else {
                return global_row * kNumVHeads * head_dim + head_idx * head_dim;
            }
        }
    }

    // Get just the sequence-local row offset for this chunk (without head/batch
    // stride) Useful for computing masks or local indexing
    __device__ __forceinline__ int get_chunk_row_offset(int chunk_idx) {
        return chunk_idx * BLOCK_M;
    }

    // Get the sequence length for the current batch (useful for mask computation)
    __device__ __forceinline__ int get_seq_len(int batch_idx) {
        if constexpr (kIsVarLen) {
            int seq_start = __ldg(cu_seqlens + batch_idx);
            int seq_end = __ldg(cu_seqlens + batch_idx + 1);
            return seq_end - seq_start;
        } else {
            return max_seq_len;
        }
    }

    // Check if a row within the current chunk is valid (not past sequence end)
    __device__ __forceinline__ bool is_row_valid(int batch_idx, int chunk_idx, int row_in_chunk) {
        int seq_len = get_seq_len(batch_idx);
        int global_row = chunk_idx * BLOCK_M + row_in_chunk;
        return global_row < seq_len;
    }
};
