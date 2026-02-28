import math
import os

import pytest
import torch
import torch.nn.functional as F

chunk_gated_delta_rule = pytest.importorskip(
    "fla.ops.gated_delta_rule", reason="FLA is required for parity tests"
).chunk_gated_delta_rule
gdn_cuda = pytest.importorskip(
    "gdn_cuda", reason="gdn_cuda python module is required")
chunk_scaled_dot_kkt_fwd = pytest.importorskip(
    "fla.ops.common.chunk_scaled_dot_kkt", reason="FLA internals are required for component parity tests"
).chunk_scaled_dot_kkt_fwd
recompute_w_u_fwd = pytest.importorskip(
    "fla.ops.gated_delta_rule.wy_fast", reason="FLA internals are required for component parity tests"
).recompute_w_u_fwd
chunk_gated_delta_rule_fwd_h = pytest.importorskip(
    "fla.ops.common.chunk_delta_h", reason="FLA internals are required for component parity tests"
).chunk_gated_delta_rule_fwd_h
chunk_fwd_o = pytest.importorskip(
    "fla.ops.common.chunk_o", reason="FLA internals are required for component parity tests"
).chunk_fwd_o
_fla_utils = pytest.importorskip(
    "fla.ops.utils", reason="FLA utilities are required for component parity tests")
chunk_local_cumsum = _fla_utils.chunk_local_cumsum
solve_tril = _fla_utils.solve_tril

_PADDED_SHAPES_FROM_FLA = [
    (1, 63, 1, 64, 1.0, 1.0, 0.0),
    (2, 500, 3, 60, 1.0, 1.0, 0.0),
    (2, 1000, 3, 64, 0.1, 1.0, 0.5),
    (3, 1024, 4, 100, 1.0, 0.1, 0.0),
    (4, 1024, 4, 128, 0.1, 1.0, 0.0),
    (2, 1500, 4, 128, 0.1, 10.0, 0.0),
    (4, 2048, 8, 64, 0.1, 1.0, 0.0),
]

_VARLEN_SHAPES_FROM_FLA = [
    (4, 60, 0.0, [0, 15]),
    (4, 64, 0.0, [0, 256, 500, 1000]),
    (4, 64, 0.5, [0, 256, 500, 1000]),
    (4, 100, 0.0, [0, 15, 100, 300, 1200, 2000]),
    (4, 60, 0.0, [0, 8192]),
]


def _skip_if_unsupported_dim(head_dim: int) -> None:
    # SM90 chunked forward heuristics require head dims >=64 and divisible by 64.
    pass
    # if head_dim < 64 or head_dim % 64 != 0:
    #     pytest.skip(f"gdn_cuda chunked_forward does not support D={head_dim} (requires D>=64 and D%64==0)")


def _assert_close(name: str, ref: torch.Tensor, out: torch.Tensor, atol: float, rtol: float) -> None:
    try:
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(f"{name} mismatch\n{exc}") from exc


def _make_l2_normalized_key(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k_f32 = torch.randn(*shape, device=device, dtype=torch.float32)
    return F.normalize(k_f32.clone(), p=2, dim=-1).to(dtype)


def _prepare_varlen_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    host = cu_seqlens.cpu().tolist()
    chunk_indices: list[int] = []
    cu_chunks: list[int] = [0]
    running = 0
    for batch_idx in range(len(host) - 1):
        seqlen = host[batch_idx + 1] - host[batch_idx]
        num_chunks = (seqlen + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_indices.extend([batch_idx, chunk_idx])
        running += num_chunks
        cu_chunks.append(running)
    device = cu_seqlens.device
    return (
        torch.tensor(chunk_indices, device=device, dtype=torch.int32),
        torch.tensor(cu_chunks, device=device, dtype=torch.int32),
    )


def _fla_compute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    A = chunk_scaled_dot_kkt_fwd(
        k=k.clone(),
        g=gate_cumsum.clone(),
        beta=beta.clone(),
        cu_seqlens=cu_seqlens if cu_seqlens is not None else None,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A.clone(),
        cu_seqlens=cu_seqlens if cu_seqlens is not None else None,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        A=A.clone(),
        g=gate_cumsum.clone(),
        cu_seqlens=cu_seqlens if cu_seqlens is not None else None,
    )
    return w, u


def _gdn_compute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    beta_mn = gdn_cuda.transpose_to_mn_major(beta.clone())
    gate_mn = gdn_cuda.transpose_to_mn_major(gate_cumsum.clone())
    u = torch.empty_like(v)
    w = torch.empty_like(k)
    gdn_cuda.bf16_gdn_compute_u_w(
        k.clone(),
        v.clone(),
        u,
        w,
        beta_mn,
        gate_mn,
        "t",
        cu_seqlens if cu_seqlens is not None else None,
        chunk_indices if chunk_indices is not None else None,
    )
    return w, u


@pytest.fixture(scope="module", autouse=True)
def _init_runtime() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability(0)[0] < 9:
        pytest.skip("SM90+ GPU is required")
    library_root_path = os.getenv("LIBRARY_ROOT_PATH")
    assert library_root_path is not None, "LIBRARY_ROOT_PATH must be set"
    cuda_home = os.getenv("CUDA_HOME_PATH") or os.getenv(
        "CUDA_HOME") or "/usr/local/cuda"
    gdn_cuda.init(library_root_path, cuda_home)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    ("B", "T", "H", "D", "_scale_from_fla", "gate_logit_normalizer", "mask_p"),
    [
        pytest.param(
            *shape,
            id=(
                f"B{shape[0]}-T{shape[1]}-H{shape[2]}-D{shape[3]}"
                f"-scale{shape[4]}-gate_norm{shape[5]}-mask_p{shape[6]}"
            ),
        )
        for shape in _PADDED_SHAPES_FROM_FLA
    ],
)
def test_chunked_forward_padded_vs_fla(
    dtype: torch.dtype,
    B: int,
    T: int,
    H: int,
    D: int,
    _scale_from_fla: float,
    gate_logit_normalizer: float,
    mask_p: float,
) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    _skip_if_unsupported_dim(D)
    atol, rtol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)

    q = _make_l2_normalized_key((B, T, H, D), device, dtype)
    k = _make_l2_normalized_key((B, T, H, D), device, dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(
        B, T, H, device=device, dtype=torch.float32)).to(dtype)
    gate = F.logsigmoid(torch.rand(
        B, T, H, device=device, dtype=torch.float32))
    gate = gate / gate_logit_normalizer
    gate = gate * (torch.rand_like(gate) > mask_p)

    ref_out, ref_state = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=gate.clone(),
        scale=1.0 / math.sqrt(D),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

    out, final_state = gdn_cuda.chunked_forward(
        q.clone(), k.clone(), v.clone(), beta.clone(), gate.clone(), 1.0 /
        math.sqrt(D), None, None, None, None
    )

    out = out[..., :D]
    final_state = final_state[..., :D]

    _assert_close("padded/output", ref_out, out, atol=atol, rtol=rtol)
    _assert_close("padded/final_state", ref_state.transpose(-1, -
                  2).to(dtype), final_state.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens"),
    [
        pytest.param(
            *shape,
            id=f"H{shape[0]}-D{shape[1]}-mask_p{shape[2]}-cu_seqlens{shape[3]}",
        )
        for shape in _VARLEN_SHAPES_FROM_FLA
    ],
)
def test_chunked_forward_varlen_vs_fla(
    dtype: torch.dtype,
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
) -> None:
    torch.manual_seed(123)
    device = torch.device("cuda")
    chunk_size = 64
    _skip_if_unsupported_dim(D)
    atol, rtol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)

    cu_seqlens_i32 = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
    total_tokens = int(cu_seqlens_i32[-1].item())
    cu_seqlens_i64 = cu_seqlens_i32.to(torch.int64)
    chunk_indices, cu_chunks = _prepare_varlen_indices(
        cu_seqlens_i32, chunk_size)

    q = _make_l2_normalized_key((total_tokens, H, D), device, dtype)
    k = _make_l2_normalized_key((total_tokens, H, D), device, dtype)
    v = torch.randn(total_tokens, H, D, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(
        total_tokens, H, device=device, dtype=torch.float32)).to(dtype)
    gate = F.logsigmoid(torch.rand(
        total_tokens, H, device=device, dtype=torch.float32))
    gate = gate * (torch.rand_like(gate) > mask_p)

    ref_out, ref_state = chunk_gated_delta_rule(
        q=q.unsqueeze(0).clone(),
        k=k.unsqueeze(0).clone(),
        v=v.unsqueeze(0).clone(),
        beta=beta.unsqueeze(0).clone(),
        g=gate.unsqueeze(0).clone(),
        scale=1.0 / math.sqrt(D),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens_i64.clone(),
    )
    ref_out = ref_out.squeeze(0)

    out, final_state = gdn_cuda.chunked_forward(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        gate.clone(),
        1.0 / math.sqrt(D),
        None,  # initial state
        cu_seqlens_i32.clone(),
        chunk_indices.clone(),
        cu_chunks.clone(),
        chunk_indices.size(0) // 2,
    )

    out = out[..., :D]
    final_state = final_state[..., :D]

    _assert_close("varlen/output", ref_out, out, atol=atol, rtol=rtol)
    _assert_close("varlen/final_state", ref_state.transpose(-1, -
                  2).to(dtype), final_state.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    ("B", "T", "H", "D", "_scale_from_fla", "gate_logit_normalizer", "mask_p"),
    [
        pytest.param(
            *shape,
            id=(
                f"B{shape[0]}-T{shape[1]}-H{shape[2]}-D{shape[3]}"
                f"-scale{shape[4]}-gate_norm{shape[5]}-mask_p{shape[6]}"
            ),
        )
        for shape in _PADDED_SHAPES_FROM_FLA
    ],
)
def test_component_compute_u_w_padded_vs_fla_chain(
    dtype: torch.dtype,
    B: int,
    T: int,
    H: int,
    D: int,
    _scale_from_fla: float,
    gate_logit_normalizer: float,
    mask_p: float,
) -> None:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    _skip_if_unsupported_dim(D)
    atol, rtol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)

    k = _make_l2_normalized_key((B, T, H, D), device, dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(
        B, T, H, device=device, dtype=torch.float32)).to(dtype)
    gate = F.logsigmoid(torch.rand(
        B, T, H, device=device, dtype=torch.float32))
    gate = gate / gate_logit_normalizer
    gate = gate * (torch.rand_like(gate) > mask_p)
    gate_cumsum = chunk_local_cumsum(gate.clone(), chunk_size=64)
    w_ref, u_ref = _fla_compute_w_u(k, v, beta, gate_cumsum)

    w_out, u_out = _gdn_compute_w_u(k, v, beta, gate_cumsum)

    _assert_close("compute_u_w/padded/w", w_ref.to(dtype),
                  w_out.to(dtype), atol=atol, rtol=rtol)
    _assert_close("compute_u_w/padded/u", u_ref.to(dtype),
                  u_out.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens"),
    [
        pytest.param(
            *shape,
            id=f"H{shape[0]}-D{shape[1]}-mask_p{shape[2]}-cu_seqlens{shape[3]}",
        )
        for shape in _VARLEN_SHAPES_FROM_FLA
    ],
)
def test_component_compute_u_w_varlen_vs_fla_chain(
    dtype: torch.dtype,
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
) -> None:
    torch.manual_seed(2027)
    device = torch.device("cuda")
    _skip_if_unsupported_dim(D)
    atol, rtol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)

    cu_seqlens_i32 = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
    cu_seqlens_i64 = cu_seqlens_i32.to(torch.int64)
    chunk_indices, _ = _prepare_varlen_indices(cu_seqlens_i32, 64)
    total_tokens = int(cu_seqlens_i32[-1].item())

    k = _make_l2_normalized_key((total_tokens, H, D), device, dtype)
    v = torch.randn(total_tokens, H, D, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(
        total_tokens, H, device=device, dtype=torch.float32)).to(dtype)
    gate = F.logsigmoid(torch.rand(
        total_tokens, H, device=device, dtype=torch.float32))
    gate = gate * (torch.rand_like(gate) > mask_p)

    k_fla = k.unsqueeze(0)
    v_fla = v.unsqueeze(0)
    beta_fla = beta.unsqueeze(0)
    gate_fla = gate.unsqueeze(0)
    gate_cumsum_fla = chunk_local_cumsum(
        gate_fla.clone(), chunk_size=64, cu_seqlens=cu_seqlens_i64.clone())

    w_ref, u_ref = _fla_compute_w_u(
        k_fla, v_fla, beta_fla, gate_cumsum_fla, cu_seqlens=cu_seqlens_i64
    )
    w_out, u_out = _gdn_compute_w_u(
        k, v, beta, gate_cumsum_fla.squeeze(0), cu_seqlens=cu_seqlens_i32, chunk_indices=chunk_indices
    )

    _assert_close("compute_u_w/varlen/w", w_ref.squeeze(0).to(dtype),
                  w_out.to(dtype), atol=atol, rtol=rtol)
    _assert_close("compute_u_w/varlen/u", u_ref.squeeze(0).to(dtype),
                  u_out.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    ("B", "T", "H", "D", "_scale_from_fla", "gate_logit_normalizer", "mask_p"),
    [
        pytest.param(
            *shape,
            id=(
                f"B{shape[0]}-T{shape[1]}-H{shape[2]}-D{shape[3]}"
                f"-scale{shape[4]}-gate_norm{shape[5]}-mask_p{shape[6]}"
            ),
        )
        for shape in _PADDED_SHAPES_FROM_FLA
    ],
)
def test_component_seq_state_padded_vs_fla_chunk_fwd_h(
    dtype: torch.dtype,
    B: int,
    T: int,
    H: int,
    D: int,
    _scale_from_fla: float,
    gate_logit_normalizer: float,
    mask_p: float,
) -> None:
    torch.manual_seed(2028)
    device = torch.device("cuda")
    _skip_if_unsupported_dim(D)
    atol, rtol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)
    num_chunks = (T + 63) // 64

    k = _make_l2_normalized_key((B, T, H, D), device, dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(
        B, T, H, device=device, dtype=torch.float32)).to(dtype)
    gate = F.logsigmoid(torch.rand(
        B, T, H, device=device, dtype=torch.float32))
    gate = gate / gate_logit_normalizer
    gate = gate * (torch.rand_like(gate) > mask_p)
    gate_cumsum = chunk_local_cumsum(gate.clone(), chunk_size=64)

    w_ref, u_ref = _fla_compute_w_u(k, v, beta, gate_cumsum)
    initial_state_fla = torch.zeros(B, H, D, D, device=device, dtype=dtype)
    h_ref, v_new_ref, final_state_ref = chunk_gated_delta_rule_fwd_h(
        k=k.clone(),
        w=w_ref.clone(),
        u=u_ref.clone(),
        g=gate_cumsum.clone(),
        initial_state=initial_state_fla.clone(),
        output_final_state=True,
    )
    u_kernel = u_ref.clone()
    final_state_kernel = torch.empty(B, H, D, D, device=device, dtype=dtype)
    state_kernel = torch.empty(
        B, num_chunks, H, D, D, device=device, dtype=dtype)
    gate_mn = gdn_cuda.transpose_to_mn_major(gate_cumsum.clone(), 128)
    gdn_cuda.bf16_chunked_seq_state_update(
        k.clone(),
        u_kernel,
        w_ref.clone(),
        initial_state_fla.transpose(-1, -2).contiguous().clone(),
        state_kernel,
        final_state_kernel,
        gate_mn.clone(),
        "t",
        None,
        None,
        None,
        64
    )

    _assert_close(
        "seq_state/padded/state",
        h_ref.transpose(-1, -2).to(dtype),
        state_kernel.to(dtype),
        atol=atol,
        rtol=rtol,
    )
    _assert_close(
        "seq_state/padded/final_state",
        final_state_ref.transpose(-1, -2).to(dtype),
        final_state_kernel.to(dtype),
        atol=atol,
        rtol=rtol,
    )
    _assert_close("seq_state/padded/u_mutated", v_new_ref,
                  u_kernel, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens"),
    [
        pytest.param(
            *shape,
            id=f"H{shape[0]}-D{shape[1]}-mask_p{shape[2]}-cu_seqlens{shape[3]}",
        )
        for shape in _VARLEN_SHAPES_FROM_FLA
    ],
)
def test_component_seq_state_varlen_vs_fla_chunk_fwd_h(
    dtype: torch.dtype,
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
) -> None:
    torch.manual_seed(2029)
    device = torch.device("cuda")
    _skip_if_unsupported_dim(D)
    atol, rtol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)

    cu_seqlens_i32 = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
    cu_seqlens_i64 = cu_seqlens_i32.to(torch.int64)
    chunk_indices, cu_chunks = _prepare_varlen_indices(cu_seqlens_i32, 64)
    total_tokens = int(cu_seqlens_i32[-1].item())
    n_seq = len(cu_seqlens) - 1
    total_chunks = int(cu_chunks[-1].item())

    k = _make_l2_normalized_key((total_tokens, H, D), device, dtype)
    v = torch.randn(total_tokens, H, D, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(
        total_tokens, H, device=device, dtype=torch.float32)).to(dtype)
    gate = F.logsigmoid(torch.rand(
        total_tokens, H, device=device, dtype=torch.float32))
    gate = gate * (torch.rand_like(gate) > mask_p)

    k_fla = k.unsqueeze(0)
    v_fla = v.unsqueeze(0)
    beta_fla = beta.unsqueeze(0)
    gate_fla = gate.unsqueeze(0)
    gate_cumsum_fla = chunk_local_cumsum(
        gate_fla.clone(), chunk_size=64, cu_seqlens=cu_seqlens_i64.clone())
    w_ref, u_ref = _fla_compute_w_u(
        k_fla, v_fla, beta_fla, gate_cumsum_fla, cu_seqlens=cu_seqlens_i64
    )

    initial_state_fla = torch.randn(n_seq, H, D, D, device=device, dtype=dtype)
    h_ref, v_new_ref, final_state_ref = chunk_gated_delta_rule_fwd_h(
        k=k_fla.clone(),
        w=w_ref.clone(),
        u=u_ref.clone(),
        g=gate_cumsum_fla.clone(),
        initial_state=initial_state_fla.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_i64.clone(),
    )

    state_kernel = torch.empty(
        total_chunks, H, D, D, device=device, dtype=dtype)
    final_state_kernel = torch.empty(
        n_seq, H, D, D, device=device, dtype=dtype)
    u_kernel = u_ref.squeeze(0).clone()
    gate_mn = gdn_cuda.transpose_to_mn_major(
        gate_cumsum_fla.squeeze(0).clone(), 128)
    gdn_cuda.bf16_chunked_seq_state_update(
        k.clone(),
        u_kernel,
        w_ref.squeeze(0).clone(),
        initial_state_fla.transpose(-1, -2).contiguous().clone(),
        state_kernel,
        final_state_kernel,
        gate_mn.clone(),
        "t",
        cu_seqlens_i32,
        cu_chunks,
        total_chunks,
        64,
    )

    _assert_close(
        "seq_state/varlen/state",
        h_ref.squeeze(0).transpose(-1, -2),
        state_kernel,
        atol=atol,
        rtol=rtol,
    )
    _assert_close(
        "seq_state/varlen/final_state",
        final_state_ref.transpose(-1, -2).to(dtype),
        final_state_kernel,
        atol=atol,
        rtol=rtol,
    )
    _assert_close("seq_state/varlen/u_mutated",
                  v_new_ref.squeeze(0), u_kernel, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    ("B", "T", "H", "D", "_scale_from_fla", "gate_logit_normalizer", "mask_p"),
    [
        pytest.param(
            *shape,
            id=(
                f"B{shape[0]}-T{shape[1]}-H{shape[2]}-D{shape[3]}"
                f"-scale{shape[4]}-gate_norm{shape[5]}-mask_p{shape[6]}"
            ),
        )
        for shape in _PADDED_SHAPES_FROM_FLA
    ],
)
def test_component_compute_o_padded_vs_fla_chunk_fwd_o(
    dtype: torch.dtype,
    B: int,
    T: int,
    H: int,
    D: int,
    _scale_from_fla: float,
    gate_logit_normalizer: float,
    mask_p: float,
) -> None:
    torch.manual_seed(2030)
    device = torch.device("cuda")
    _skip_if_unsupported_dim(D)
    atol, rtol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)

    q = _make_l2_normalized_key((B, T, H, D), device, dtype)
    k = _make_l2_normalized_key((B, T, H, D), device, dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(
        B, T, H, device=device, dtype=torch.float32)).to(dtype)
    gate = F.logsigmoid(torch.rand(
        B, T, H, device=device, dtype=torch.float32))
    gate = gate / gate_logit_normalizer
    gate = gate * (torch.rand_like(gate) > mask_p)
    gate_cumsum = chunk_local_cumsum(gate, chunk_size=64)

    w_ref, u_ref = _fla_compute_w_u(k, v, beta, gate_cumsum)
    h_ref, v_new_ref, _ = chunk_gated_delta_rule_fwd_h(
        k=k.clone(),
        w=w_ref.clone(),
        u=u_ref.clone(),
        g=gate_cumsum.clone(),
        initial_state=torch.zeros(
            B, H, D, D, device=device, dtype=dtype).clone(),
        output_final_state=False,
    )
    o_ref = chunk_fwd_o(
        q=q.clone(),
        k=k.clone(),
        v=v_new_ref.clone(),
        h=h_ref.clone(),
        g=gate_cumsum.clone(),
        scale=1.0 / math.sqrt(D),
    )

    o_out = torch.empty_like(v_new_ref)
    gate_mn = gdn_cuda.transpose_to_mn_major(gate_cumsum.clone(), 128)
    gdn_cuda.bf16_gdn_chunked_compute_O(
        q.clone(),
        h_ref.transpose(-1, -2).contiguous().clone(),
        k.clone(),
        v_new_ref.clone(),
        o_out,
        gate_mn.clone(),
        "t",
        None,
        None,
        None,
        64,
    )

    _assert_close("compute_o/padded/o", o_ref, o_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens"),
    [
        pytest.param(
            *shape,
            id=f"H{shape[0]}-D{shape[1]}-mask_p{shape[2]}-cu_seqlens{shape[3]}",
        )
        for shape in _VARLEN_SHAPES_FROM_FLA
    ],
)
def test_component_compute_o_varlen_vs_fla_chunk_fwd_o(
    dtype: torch.dtype,
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
) -> None:
    torch.manual_seed(2031)
    device = torch.device("cuda")
    _skip_if_unsupported_dim(D)
    atol, rtol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)

    cu_seqlens_i32 = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
    cu_seqlens_i64 = cu_seqlens_i32.to(torch.int64)
    chunk_indices, cu_chunks = _prepare_varlen_indices(cu_seqlens_i32, 64)
    total_tokens = int(cu_seqlens_i32[-1].item())

    q = _make_l2_normalized_key((total_tokens, H, D), device, dtype)
    k = _make_l2_normalized_key((total_tokens, H, D), device, dtype)
    v = torch.randn(total_tokens, H, D, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(
        total_tokens, H, device=device, dtype=torch.float32)).to(dtype)
    gate = F.logsigmoid(torch.rand(
        total_tokens, H, device=device, dtype=torch.float32))
    gate = gate * (torch.rand_like(gate) > mask_p)

    q_fla = q.unsqueeze(0)
    k_fla = k.unsqueeze(0)
    v_fla = v.unsqueeze(0)
    beta_fla = beta.unsqueeze(0)
    gate_fla = gate.unsqueeze(0)
    gate_cumsum_fla = chunk_local_cumsum(
        gate_fla.clone(), chunk_size=64, cu_seqlens=cu_seqlens_i64.clone())
    w_ref, u_ref = _fla_compute_w_u(
        k_fla, v_fla, beta_fla, gate_cumsum_fla, cu_seqlens=cu_seqlens_i64
    )
    h_ref, v_new_ref, _ = chunk_gated_delta_rule_fwd_h(
        k=k_fla.clone(),
        w=w_ref.clone(),
        u=u_ref.clone(),
        g=gate_cumsum_fla.clone(),
        initial_state=torch.zeros(
            len(cu_seqlens) - 1, H, D, D, device=device, dtype=dtype).clone(),
        output_final_state=False,
        cu_seqlens=cu_seqlens_i64.clone(),
    )
    o_ref = chunk_fwd_o(
        q=q_fla.clone(),
        k=k_fla.clone(),
        v=v_new_ref.clone(),
        h=h_ref.clone(),
        g=gate_cumsum_fla.clone(),
        scale=1.0 / math.sqrt(D),
        cu_seqlens=cu_seqlens_i64.clone(),
    ).squeeze(0)

    o_out = torch.empty_like(v)
    gate_mn = gdn_cuda.transpose_to_mn_major(
        gate_cumsum_fla.squeeze(0).clone(), 128)
    gdn_cuda.bf16_gdn_chunked_compute_O(
        q.clone(),
        h_ref.squeeze(0).transpose(-1, -2).contiguous().clone(),
        k.clone(),
        v_new_ref.squeeze(0).clone(),
        o_out,
        gate_mn.clone(),
        "t",
        cu_seqlens_i32.clone(),
        chunk_indices.clone(),
        cu_chunks.clone(),
        64,
    )

    _assert_close("compute_o/varlen/o", o_ref, o_out, atol=atol, rtol=rtol)
