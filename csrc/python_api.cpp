// Python bindings for gdn_cuda via pybind11.

#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <apis/gdn_forward.hpp>

namespace py = pybind11;

// Helper: get the current CUDA stream as a cudaStream_t
static cudaStream_t current_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

PYBIND11_MODULE(gdn_cuda, m) {
    m.doc() = "GDN CUDA kernels for H100 (SM90a)";

    m.def("init", &gdn_cuda::init, py::arg("library_root"), py::arg("cuda_home"),
          R"(Initialize gdn_cuda JIT runtime paths.

Args:
    library_root: Absolute path to repository root containing include/.
    cuda_home: Absolute path to CUDA toolkit root (for example /usr/local/cuda). Can be found with which nvcc.

Shapes:
    None.

DTypes:
    None.

Returns:
    None.

Raises:
    RuntimeError/host assertion if paths are invalid.

Notes:
    Must be called before launching any operation that may JIT compile kernels.
)");

    // =========================================================================
    // ChunkedForwardWorkspace class
    // =========================================================================

    py::class_<gdn_cuda::ChunkedForwardWorkspace>(
        m, "ChunkedForwardWorkspace",
        R"(Pre-allocated workspace for chunked_forward to eliminate per-call cudaMalloc overhead.

Create once and pass on every call.  Tensors are lazily allocated on the first
call and reused on subsequent calls when the input shapes match.  When shapes
change the workspace automatically reallocates.
)")
        .def(py::init<>());

    // =========================================================================
    // High-level forward passes
    // =========================================================================

    m.def(
        "chunked_forward",
        [](at::Tensor& q, at::Tensor& k, at::Tensor& v, at::Tensor& beta, at::Tensor& gate,
           std::optional<float> scale, std::optional<at::Tensor> initial_state,
           std::optional<at::Tensor> cu_seqlens, std::optional<at::Tensor> chunk_indices,
           std::optional<at::Tensor> cu_chunks, std::optional<int> total_chunks,
           py::object workspace_py) {
            cudaStream_t stream = current_stream();
            gdn_cuda::ChunkedForwardWorkspace* ws_ptr = nullptr;
            if (!workspace_py.is_none()) {
                ws_ptr = workspace_py.cast<gdn_cuda::ChunkedForwardWorkspace*>();
            }
            return gdn_cuda::chunked_forward(q, k, v, beta, gate, scale, initial_state, cu_seqlens,
                                             chunk_indices, cu_chunks, total_chunks, ws_ptr,
                                             stream);
        },
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("beta"), py::arg("gate"),
        py::arg("scale") = py::none(), py::arg("initial_state") = py::none(),
        py::arg("cu_seqlens") = py::none(), py::arg("chunk_indices") = py::none(),
        py::arg("cu_chunks") = py::none(), py::arg("total_chunks") = py::none(),
        py::arg("workspace") = py::none(),
        R"(Chunked prefill forward pass.

Args:
    query: Query tensor.
    key: Key tensor.
    value: Value tensor.
    beta: Beta tensor.
    gate: Gate tensor (pre-cumsum).
    initial_state: Optional initial state tensor.
    cu_seqlens: Optional varlen cumulative sequence offsets.
    chunk_indices: Optional varlen flat pairs [batch_idx, chunk_idx, ...].
    cu_chunks: Optional varlen cumulative chunk counts.
    total_chunks: Optional total number of chunks.
    workspace: Optional ChunkedForwardWorkspace for buffer reuse across calls.
               When provided, intermediate tensors are allocated once and reused
               on subsequent calls with the same shapes, eliminating cudaMalloc overhead.

Shapes:
    padded:
        query/key: (B, T, num_k_heads, shape_k)
        value: (B, T, num_v_heads, shape_v)
        beta/gate: (B, T, num_v_heads)
        state (optional): (B, num_chunks, num_v_heads, shape_v, shape_k)
    varlen:
        query/key: (total_tokens, num_k_heads, shape_k)
        value: (total_tokens, num_v_heads, shape_v)
        beta/gate: (total_tokens, num_v_heads)
        state (optional): (total_chunks, num_v_heads, shape_v, shape_k)
        cu_seqlens: (B + 1)
        chunk_indices: (2 * total_chunks)
        cu_chunks: (B + 1)

DTypes:
    query/key/value/state: BF16
    beta: BF16
    gate: FP32 expected by helper kernels
    cu_seqlens/chunk_indices/cu_chunks: INT32

Returns:
    Tuple[output, final_state]:
        output: same shape as value
        final_state: (B, num_v_heads, shape_v, shape_k)

Raises:
    RuntimeError/host assertion on invalid ranks, dtypes, or missing required varlen tensors.

Notes:
    Uses the current CUDA stream.
)");

    m.def(
        "recurrent_forward",
        [](at::Tensor& q, at::Tensor& k, at::Tensor& v, std::optional<at::Tensor> initial_state,
           at::Tensor& beta, at::Tensor& gate, std::optional<at::Tensor> cu_seqlens,
           std::optional<at::Tensor> num_accepted_tokens, int inference_mode, bool is_qk_norm,
           std::optional<float> scale) {
            cudaStream_t stream = current_stream();
            gdn_cuda::InferenceMode mode = static_cast<gdn_cuda::InferenceMode>(inference_mode);
            return gdn_cuda::recurrent_forward(q, k, v, initial_state, beta, gate, cu_seqlens,
                                               num_accepted_tokens, mode, stream, is_qk_norm,
                                               scale);
        },
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("initial_state") = py::none(),
        py::arg("beta"), py::arg("gate"), py::arg("cu_seqlens") = py::none(),
        py::arg("num_accepted_tokens") = py::none(), py::arg("inference_mode") = 0,
        py::arg("is_qk_norm") = false, py::arg("scale") = py::none(),
        R"(Recurrent decode/spec-verify forward pass.

Args:
    query: Query tensor.
    key: Key tensor.
    value: Value tensor.
    state: Initial state tensor.
    beta: Beta tensor.
    gate: Gate tensor.
    cu_seqlens: Optional varlen cumulative sequence offsets.
    num_accepted_tokens: Optional accepted token counts for speculative decode modes.
    inference_mode: Integer enum value from InferenceMode.
    is_qk_norm: Whether to enable QK normalization path.
    scale: Optional output scale factor. If None, defaults to 1/sqrt(head_dim).

Shapes:
    padded:
        query/key/value: (B, T, heads, dim)
        state: (B, num_v_heads, shape_v, shape_k)
    varlen:
        query/key/value: (total_tokens, heads, dim)
        state: (B, num_v_heads, shape_v, shape_k)
        cu_seqlens: (B + 1)

DTypes:
    query/key/value/state/beta/gate: BF16 in current recurrent launcher path
    cu_seqlens/num_accepted_tokens: INT32

Returns:
    Tuple[output, new_state].

Raises:
    RuntimeError/host assertion on invalid ranks, dtypes, or shape contracts.

Notes:
    Uses the current CUDA stream.
)");

    m.def(
        "fused_gdn_gating",
        [](at::Tensor& A_log, at::Tensor& dt_bias, at::Tensor& a, at::Tensor& b, bool is_var_len) {
            cudaStream_t stream = current_stream();
            return gdn_cuda::fused_gdn_gating(A_log, dt_bias, a, b, is_var_len, stream);
        },
        py::arg("A_log"), py::arg("dt_bias"), py::arg("a"), py::arg("b"),
        py::arg("is_var_len") = false,
        R"(Compute (beta, gate) from raw GDN parameter tensors.

Args:
    A_log: Log-parameter tensor.
    dt_bias: Bias tensor.
    a: Parameter tensor.
    b: Parameter tensor that defines beta output shape.
    is_var_len: If true, executes with varlen seqlen convention.

Shapes:
    Broadcast-compatible with kernel expectation; last dim is interpreted as head axis.

DTypes:
    A_log supports FP32 or BF16 in current implementation.
    Other inputs should match kernel path contracts.

Returns:
    Tuple[beta, gate].

Raises:
    RuntimeError/host assertion on unsupported dtype or shape mismatch.

Notes:
    Uses the current CUDA stream.
)");

    // =========================================================================
    // Individual kernel APIs (for testing/benchmarking)
    // =========================================================================

    m.def(
        "bf16_gdn_compute_u_w",
        [](at::Tensor& k, at::Tensor& v, at::Tensor& u, at::Tensor& w, at::Tensor& beta,
           std::optional<at::Tensor> gate, const std::string& compiled_dims,
           std::optional<at::Tensor> cu_seqlens, std::optional<at::Tensor> chunk_indices) {
            cudaStream_t stream = current_stream();
            gdn_cuda::bf16_gdn_compute_u_w(k, v, u, w, beta, gate, compiled_dims, stream,
                                           cu_seqlens, chunk_indices);
        },
        py::arg("k"), py::arg("v"), py::arg("u"), py::arg("w"), py::arg("beta"),
        py::arg("gate") = py::none(), py::arg("compiled_dims") = "t",
        py::arg("cu_seqlens") = py::none(), py::arg("chunk_indices") = py::none(),
        R"(Launch low-level chunked compute_u_w kernel.

Args:
    k: Key tensor.
    v: Value tensor.
    u: Output U tensor (mutated).
    w: Output W tensor (mutated).
    beta: Beta tensor in MN-major.
    gate: Optional gate tensor in MN-major.
    compiled_dims: Compile-time specialization hint string.
    cu_seqlens: Optional varlen cumulative sequence offsets.
    chunk_indices: Optional varlen chunk index pairs.

Shapes:
    padded: k/v/u/w are rank-4.
    varlen: k/v/u/w are rank-3; cu_seqlens is rank-1; chunk_indices stores flat pairs.

DTypes:
    k/v/u/w/beta: BF16
    gate: FP32
    cu_seqlens/chunk_indices: INT32

Returns:
    None (u and w are written in-place).

Raises:
    RuntimeError/host assertion on invalid contracts.

Notes:
    Uses the current CUDA stream.
)");

    m.def(
        "bf16_chunked_seq_state_update",
        [](at::Tensor& k, at::Tensor& u, at::Tensor& w, std::optional<at::Tensor>& initial_state,
           at::Tensor& state, std::optional<at::Tensor>& final_state,
           std::optional<at::Tensor> gate, const std::string& compiled_dims,
           std::optional<at::Tensor> cu_seqlens, std::optional<at::Tensor> cu_chunks,
           std::optional<int> total_chunks, uint32_t chunk_size) {
            cudaStream_t stream = current_stream();
            gdn_cuda::bf16_chunked_seq_state_update(k, u, w, initial_state, state, final_state,
                                                    gate, compiled_dims, stream, cu_seqlens,
                                                    cu_chunks, total_chunks, chunk_size);
        },
        py::arg("k"), py::arg("u"), py::arg("w"), py::arg("initial_state"), py::arg("state"),
        py::arg("final_state"), py::arg("gate") = py::none(), py::arg("compiled_dims") = "t",
        py::arg("cu_seqlens") = py::none(), py::arg("cu_chunks") = py::none(),
        py::arg("total_chunks") = py::none(), py::arg("chunk_size") = 64u,
        R"(Launch low-level chunked sequential state update kernel.

Args:
    k: Key tensor.
    u: U tensor (may be updated by algorithm).
    w: W tensor.
    initial_state: Initial state tensor.
    state: State tensor input/output.
    final_state: Final state output tensor.
    gate: Optional gate tensor in MN-major.
    is_initial_state: Whether state is pre-initialized.
    output_final_state: Whether to write final_state.
    compiled_dims: Compile-time specialization hint string.
    cu_seqlens: Optional varlen cumulative sequence offsets.
    cu_chunks: Optional varlen cumulative chunk counts.
    total_chunks: Optional total number of chunks.
    chunk_size: Chunk size.

Returns:
    None (state/final_state and potentially u are written in-place).

Raises:
    RuntimeError/host assertion on invalid contracts.

Notes:
    Uses the current CUDA stream.
)");

    m.def(
        "bf16_gdn_chunked_compute_O",
        [](at::Tensor& q, at::Tensor& state, at::Tensor& k, at::Tensor& u, at::Tensor& o,
           std::optional<at::Tensor> gate, std::optional<float> scale,
           const std::string& compiled_dims, std::optional<at::Tensor> cu_seqlens,
           std::optional<at::Tensor> chunk_indices, std::optional<at::Tensor> cu_chunks,
           uint32_t chunk_size) {
            cudaStream_t stream = current_stream();
            gdn_cuda::bf16_gdn_chunked_compute_O(q, state, k, u, o, gate, scale, compiled_dims,
                                                 stream, cu_seqlens, chunk_indices, cu_chunks,
                                                 chunk_size);
        },
        py::arg("q"), py::arg("state"), py::arg("k"), py::arg("u"), py::arg("o"),
        py::arg("gate") = py::none(), py::arg("scale") = py::none(), py::arg("compiled_dims") = "t",
        py::arg("cu_seqlens") = py::none(), py::arg("chunk_indices") = py::none(),
        py::arg("cu_chunks") = py::none(), py::arg("chunk_size") = 64u,
        R"(Launch low-level chunked compute_O kernel.

Args:
    q: Query tensor.
    state: State tensor.
    k: Key tensor.
    u: U tensor.
    o: Output tensor (mutated).
    gate: Optional gate tensor.
    compiled_dims: Compile-time specialization hint string.
    cu_seqlens: Optional varlen cumulative sequence offsets.
    chunk_indices: Optional varlen chunk index pairs.
    cu_chunks: Optional varlen cumulative chunk counts.
    chunk_size: Chunk size.

Returns:
    None (o is written in-place).

Raises:
    RuntimeError/host assertion on invalid contracts.

Notes:
    Uses the current CUDA stream.
)");

    m.def(
        "chunk_local_cumsum_bf16",
        [](at::Tensor& input, at::Tensor& output, int batch_size, int seq_len, int num_heads,
           bool head_first, std::optional<at::Tensor> cu_seqlens,
           std::optional<at::Tensor> chunk_indices) {
            cudaStream_t stream = current_stream();
            gdn_cuda::chunk_local_cumsum_bf16(input, output, batch_size, seq_len, num_heads,
                                              head_first, cu_seqlens, chunk_indices, stream);
        },
        py::arg("input"), py::arg("output"), py::arg("batch_size"), py::arg("seq_len"),
        py::arg("num_heads"), py::arg("head_first") = false, py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none(),
        R"(Chunk-local inclusive cumsum helper.

Args:
    input: Input tensor.
    output: Output tensor.
    batch_size: Logical batch size.
    seq_len: Logical sequence length.
    num_heads: Logical number of heads.
    head_first: Whether input layout is head-first.
    cu_seqlens: Optional varlen cumulative sequence offsets.
    chunk_indices: Optional varlen chunk index pairs.

Returns:
    None (output is written in-place).

Raises:
    RuntimeError/host assertion on invalid contracts.

Notes:
    Uses the current CUDA stream.
)");

    m.def(
        "transpose_to_mn_major",
        [](at::Tensor& input, uint32_t alignment) {
            cudaStream_t stream = current_stream();
            return gdn_cuda::transpose_to_mn_major(input, stream, alignment);
        },
        py::arg("input"), py::arg("alignment") = 16u,
        R"(Transpose last two dims to MN-major layout.

Args:
    input: Rank >= 2 tensor.
    alignment: Byte alignment used to pad MN dimension.

Returns:
    New tensor with MN-major strides and padded MN extent.

Raises:
    RuntimeError/host assertion on unsupported dtype or invalid rank.

Notes:
    Uses the current CUDA stream.
)");

    // =========================================================================
    // InferenceMode enum
    // =========================================================================
    py::enum_<gdn_cuda::InferenceMode>(m, "InferenceMode")
        .value("Prefill", gdn_cuda::InferenceMode::Prefill)
        .value("Decode", gdn_cuda::InferenceMode::Decode)
        .value("SpecVerify", gdn_cuda::InferenceMode::SpecVerify)
        .export_values();
}
