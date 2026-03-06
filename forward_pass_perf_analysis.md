# GDN Forward Pass — Host-Side Performance Analysis

Analysis of `csrc/apis/gdn_forward.cpp` and the API/kernel-launch code it calls.
Device-side kernel logic is treated as stationary unless a glaring issue is noted.

---

## HIGH — Blocking Device-to-Host Synchronization

### 1. `cudaMemcpy(DeviceToHost)` in varlen `compute_O` path

**File:** `csrc/jit_kernels/impls/sm90_bf16_gdn_chunked_compute_O.hpp:141-143`

```cpp
int total_chunks_host = 0;
CUDA_CHECK(cudaMemcpy(&total_chunks_host,
                      cu_chunks->data_ptr<int>() + (cu_chunks->size(0) - 1),
                      sizeof(int), cudaMemcpyDeviceToHost));
```

This is a **synchronous D2H copy** that stalls the CPU until all prior GPU work on the
device completes. It reads a single `int` from `cu_chunks` to determine grid dimensions.

The same value is already available on the host in the `chunked_forward` caller (via
`total_chunks`), and `bf16_chunked_seq_state_update` already accepts it as a host-side
`std::optional<int>`. The fix is to thread `total_chunks` through to `compute_O` the
same way, eliminating this sync entirely.

**Impact:** On an H100, a single `cudaMemcpy` D2H flushes the command buffer and waits
for all queued kernels to drain. In a pipeline with multiple kernel launches already
queued on the stream, this creates a bubble where the CPU blocks instead of queuing
the next kernel. Expected cost: 5–20 µs stall per call depending on queue depth.

---

## MEDIUM — Redundant Allocations and Extra Kernel Launches

### 2. Two separate `transpose_to_mn_major` calls per `chunked_forward`

**File:** `csrc/apis/gdn_forward.cpp:23, 83`

```cpp
at::Tensor beta_mn     = api::transpose_to_mn_major(beta, stream);         // line 23
at::Tensor gate_mn     = api::transpose_to_mn_major(gate_cumsum, stream);  // line 83
```

Each call allocates a new output tensor (`at::empty_strided`) and launches a JIT
transpose GPU kernel. That's **2 extra GPU kernel launches + 2 temporary allocations**
on every forward call.

For the gate path specifically, the data flows:

```
gate  →  chunk_local_cumsum  →  gate_cumsum  →  transpose_to_mn_major  →  gate_mn
              (kernel 1)           (alloc 1)          (kernel 2)           (alloc 2)
```

If `chunk_local_cumsum` wrote directly into MN-major layout, **one kernel launch and
one intermediate buffer (`gate_cumsum`) would be eliminated**.

### 3. `at::zeros` for default initial state on every call

**File:** `csrc/apis/gdn_forward.cpp:96-107`

```cpp
initial_state.emplace(at::zeros(
    {(long)total_chunks, value.size(-2), aligned_dim, aligned_dim}, value.options()));
```

When no `initial_state` is provided (common during training), this allocates a
potentially large `[chunks, heads, D, D]` tensor and launches a `cudaMemset` to zero
it. For D=128, H=16, chunks=32, that's 128 MB of memset per call.

**Fix options:**
- Have the state-update kernel treat a null initial-state pointer as implicit zeros
  (skip the load, zero-init from registers).
- Pre-allocate and reuse a zeroed state buffer across calls.

### 4. `torch::pad` launches uncontrolled PyTorch internal kernels

**File:** `csrc/apis/gdn_forward.cpp:42-44, 157-159`

When `head_dim % 64 != 0`, three `torch::pad` calls are made for Q, K, V (plus
optionally `initial_state`). Each one:
- Allocates a new padded tensor
- Launches PyTorch's internal padding kernel (which may not land on the user's stream
  depending on PyTorch's stream management)

This is only triggered for non-aligned head dims. If head dims are always multiples of
64 in production, this is a non-issue. But if non-aligned dims are common, consider
padding once into a pre-allocated buffer or fusing padding into the first kernel.

---

## LOW — Minor Host-Side Overhead

### 5. Heuristic re-computation on every transpose call

**File:** `csrc/jit_kernels/heuristics/common.hpp` — `get_transpose_config`

Each `transpose_to_mn_major` call runs a CPU-side loop over ~25 (block_size ×
thread_count) combinations to pick the best kernel config. The loop itself takes
< 1 µs, but it runs twice per `chunked_forward` and the result is deterministic
for a given (mn, k, elem_size) triple. Caching the result in a small LRU map would
eliminate this.

### 6. JIT compiler mutex on every kernel launch (warm path)

**File:** `csrc/jit/compiler.hpp` — `Compiler::build()`

Even on a warm cache hit, the JIT build path takes a mutex lock and does an
`unordered_map` lookup per kernel launch. For the `chunked_forward` call, there are
~5 kernel launches, meaning ~5 mutex acquisitions. This is likely < 1 µs total but
could matter at very small batch sizes where host overhead dominates.

---

## DEVICE-SIDE NOTE — Glaring Issue

### 7. (Device) Recurrent kernel re-reads state from global memory every step

**File:** `csrc/jit_kernels/impls/sm90_bf16_gdn_recurrent.hpp`

While the device code is treated as stationary, this is worth flagging: the recurrent
kernel structure appears to read/write the full `[heads, D, D]` state matrix from/to
global memory on each recurrent step rather than keeping it resident in registers or
shared memory across steps. For D=128, the state is 128×128×2B = 32 KB per head —
fits in shared memory on SM90. If the state is being round-tripped through global
memory on every token, this would be the dominant bottleneck for recurrent inference,
far exceeding any host-side issue listed above.

---

## Summary Table

| # | Severity | Issue | Location | Est. Impact |
|---|----------|-------|----------|-------------|
| 1 | **HIGH** | Blocking `cudaMemcpy` D2H in varlen compute_O | `sm90_bf16_gdn_chunked_compute_O.hpp:141` | 5–20 µs stall/call |
| 2 | MEDIUM | Two transpose kernels + allocs per forward | `gdn_forward.cpp:23,83` | 2 extra kernel launches |
| 3 | MEDIUM | `at::zeros` memset for default initial state | `gdn_forward.cpp:96-107` | Up to 128 MB memset |
| 4 | MEDIUM | `torch::pad` for non-aligned head dims | `gdn_forward.cpp:42-44` | 3+ extra kernels (conditional) |
| 5 | LOW | Heuristic loop re-runs every transpose | heuristics | < 1 µs |
| 6 | LOW | Mutex per JIT cache lookup | compiler | < 1 µs |
| 7 | DEVICE | State round-trip through GMEM in recurrent | `sm90_bf16_gdn_recurrent.hpp` | Potentially dominant for inference |
