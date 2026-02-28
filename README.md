# h100_gdn_cuda
H100 CUDA Kernels for Gated Delta Net


## Requirements;
- CUDA 12.9+ (for nvrtc), CUDA 12.3 for nvcc jit
- GCC 14+
- CMAKE 3.10+
- Python 3.10+
## Build

`cmake -S . -B build` only configures/generates build files. It does not compile.

Use either:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

or the one-command wrapper:

```bash
./run_cmake.sh
```

To build into a different folder (for example `build_cmake`):

```bash
./run_cmake.sh build_cmake
```

## Environment Variables Guide:

- I put an example environment file in the root of the repository called `.example_env`, with explanations of the variables used. You can copy it to `.env` and edit it to your needs.


## Public API Reference

The Python module name is `gdn_cuda` and maps directly to the C++ API in
`gdn_cuda/include/gdn_cuda/api.hpp`.

All APIs use the current CUDA stream from PyTorch.

### High-level APIs

| Function | Purpose | Inputs (shape) | DTypes | Returns |
|---|---|---|---|---|
| `init(library_root, cuda_home)` | Initialize JIT/compiler paths. | Host strings. | N/A | `None` |
| `chunked_forward(query, key, value, beta, gate, state=None, cu_seqlens=None, chunk_indices=None, cu_chunks=None)` | Chunked prefill forward pass. | padded: `q/k (B,T,kh,sk)`, `v (B,T,vh,sv)`, `beta/gate (B,T,vh)`; varlen: `q/k (N,kh,sk)`, `v (N,vh,sv)`, `beta/gate (N,vh)`, `cu_seqlens (B+1)`, `chunk_indices (2*chunks)`, `cu_chunks (B+1)` | `q/k/v/state`: BF16, `beta`: BF16, `gate`: FP32 expected, metadata: INT32 | `(output, final_state)` |
| `recurrent_forward(query, key, value, state, beta, gate, cu_seqlens=None, num_accepted_tokens=None, inference_mode=0, is_qk_norm=False)` | Recurrent decode/spec-verify forward pass. | padded: `q/k/v (B,T,h,d)`, `state (B,vh,sv,sk)`; varlen: `q/k/v (N,h,d)`, `cu_seqlens (B+1)` | recurrent path currently expects BF16 data tensors; metadata INT32 | `(output, new_state)` |
| `fused_gdn_gating(A_log, dt_bias, a, b, is_var_len=False)` | Compute `(beta, gate)` from raw GDN params. | Kernel-defined compatible shapes. | `A_log`: FP32 or BF16 (current impl), others follow kernel path | `(beta, gate)` |

### Low-level kernel wrapper APIs

| Function | Purpose | Inputs (shape) | DTypes | Mutation |
|---|---|---|---|---|
| `bf16_gdn_compute_u_w(k, v, u, w, beta, gate=None, compiled_dims="t", cu_seqlens=None, chunk_indices=None)` | Compute chunked U/W intermediates. | padded rank-4 or varlen rank-3 tensors. | `k/v/u/w/beta`: BF16, `gate`: FP32 MN-major, metadata INT32 | Writes `u`, `w` |
| `bf16_chunked_seq_state_update(k, u, w, state, final_state, gate=None, is_initial_state=False, output_final_state=True, compiled_dims="t", cu_seqlens=None, chunk_size=64)` | Propagate per-chunk recurrent state. | padded rank-4/5 or varlen rank-3/4 tensors. | data BF16, `gate` FP32 MN-major | Writes `state`, `final_state`, may update `u` |
| `bf16_gdn_chunked_compute_O(q, state, k, u, o, gate=None, compiled_dims="t", cu_seqlens=None, chunk_indices=None, cu_chunks=None, chunk_size=64)` | Compute output `O` from propagated state. | padded rank-4/5 or varlen rank-3/4 + metadata | data BF16, `gate` FP32, metadata INT32 | Writes `o` |
| `chunk_local_cumsum_bf16(input, output, batch_size, seq_len, num_heads, head_first=False, cu_seqlens=None, chunk_indices=None)` | Chunk-local inclusive cumsum helper. | rank depends on layout mode | input BF16, output float accumulation path | Writes `output` |
| `transpose_to_mn_major(input, alignment=16)` | Layout helper for MN-major kernel access. | rank >= 2 | FP32 or BF16 | Returns new tensor |

Note: For tensors with head dimensions * type_size that aren't divisible by 32, currently only the chunked_forward API will handle padding - the low-level component kernels will not pad the tensors. To-do is probably to add it to every kernel, it's just a few more lines of code


## Developer Notes

### MN-major layout expectations

Several kernels require matrix-like tensors in MN-major layout. In this codebase:

- `beta` must be MN-major for `bf16_gdn_compute_u_w`.
- `gate` must be MN-major for chunked `compute_u_w`, `seq_state_update`, and `compute_O`.
- Use `transpose_to_mn_major` before invoking low-level wrappers.

### Gate dtype policy

Current behavior in this repository:

- Chunked helper kernels (`chunk_local_cumsum` and related paths) expect gate values in FP32.
- Recurrent launcher path currently passes gate as BF16 pointer.
- `fused_gdn_gating` may emit BF16 gate when `A_log` is BF16.

## Repository Layout Direction (DeepGEMM-style)

I mirrored DeepGEMM's repository structure, as I like their simple, clean JIT infra + header-only kernels, I think it makes the code readable and a lot easier to understand.

## Working with C++ vs Python

I would recommend using run_cmake.sh to build for only cpp testing, it's (slightly) faster, and easier to debug quickly. Note that currently it's still a little unpolished since I customized it for my system. If you want to work with python, use the develop.sh script to build the python bindings.

### Target structure

| Area | Target path | Responsibility |
|---|---|---|
| Python binding entrypoint | `csrc/python_api.cpp` | Pybind module declarations and docstrings only |
| Public C++ API wrappers | `csrc/apis/*.hpp` + `csrc/apis/*.cpp` | Tensor validation, shape adaptation, stream plumbing |
| JIT runtime core | `csrc/jit/*.hpp` | Compiler, cache, runtime handles |
| JIT kernel launch wrappers | `csrc/jit_kernels/impls/*.hpp` | Launch argument packing and runtime dispatch |
| JIT heuristics/config | `csrc/jit_kernels/heuristics/*.hpp` | Launch configuration heuristics |
| CUDA kernels (headers only) | `include/gdn_cuda/kernels/*.cuh` | Device kernel definitions |
| Common CUDA utilities | `include/gdn_cuda/kernels/common/*.hpp/.cuh` | Shared device-side helpers |




## Roadmap:
- [ ] add support for a qk l2 norm that happens in gdn chunked forward, and add this into tests as well
- [ ] write guide on how to use jit and env variables
- [ ] ping pong scheduling on all kernels for heavier tma stores
    - [ ] probably only needed for head_dim > 128, the current 64 x 128
