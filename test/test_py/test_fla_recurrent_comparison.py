import math
import os

import pytest
import torch
import torch.nn.functional as F

fused_recurrent_gated_delta_rule = pytest.importorskip(
    "fla.ops.gated_delta_rule",
    reason="FLA is required for recurrent parity tests",
).fused_recurrent_gated_delta_rule
gdn_cuda = pytest.importorskip(
    "gdn_cuda", reason="gdn_cuda python module is required")


_PADDED_RECURRENT_SHAPES = [
    (2, 64, 4, 64, 1.0, 0.0),
    (2, 128, 4, 128, 10.0, 0.0),
    (4, 64, 8, 64, 1.0, 0.5),
]

_VARLEN_RECURRENT_SHAPES = [
    (4, 64, 0.0, [0, 256, 500, 1000]),
    (4, 128, 0.0, [0, 64, 128, 256, 400]),
    (8, 64, 0.5, [0, 15, 100, 300, 1200, 2000]),
]


def _skip_if_unsupported_dim(head_dim: int) -> None:
    # SM90 recurrent heuristics require dims >=64 and divisible by 64.
    if head_dim < 64 or head_dim % 64 != 0:
        pytest.skip(
            f"gdn_cuda recurrent does not support D={head_dim} (requires D>=64 and D%64==0)")


def _assert_close(name: str, ref: torch.Tensor, out: torch.Tensor, atol: float, rtol: float) -> None:
    try:
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(f"{name} mismatch\n{exc}") from exc


def _make_l2_normalized_key(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x_f32 = torch.randn(*shape, device=device, dtype=torch.float32)
    return F.normalize(x_f32.clone(), p=2, dim=-1).to(dtype)


def _make_qk(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    use_qk_l2norm_in_kernel: bool,
) -> torch.Tensor:
    if use_qk_l2norm_in_kernel:
        return torch.randn(*shape, device=device, dtype=dtype)
    return _make_l2_normalized_key(shape, device, dtype)


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
@pytest.mark.parametrize("use_qk_l2norm_in_kernel", [False, True], ids=["qk_l2norm_off", "qk_l2norm_on"])
@pytest.mark.parametrize(
    ("mode_name", "inference_mode", "use_initial_state"),
    [
        pytest.param("prefill", 0, False, id="prefill"),
        pytest.param("decode", 1, True, id="decode"),
    ],
)
@pytest.mark.parametrize(
    ("B", "T", "H", "D", "gate_logit_normalizer", "mask_p"),
    [
        pytest.param(
            *shape,
            id=(
                f"B{shape[0]}-T{shape[1]}-H{shape[2]}-D{shape[3]}"
                f"-gate_norm{shape[4]}-mask_p{shape[5]}"
            ),
        )
        for shape in _PADDED_RECURRENT_SHAPES
    ],
)
def test_recurrent_forward_padded_vs_fla(
    dtype: torch.dtype,
    use_qk_l2norm_in_kernel: bool,
    mode_name: str,
    inference_mode: int,
    use_initial_state: bool,
    B: int,
    T: int,
    H: int,
    D: int,
    gate_logit_normalizer: float,
    mask_p: float,
) -> None:
    del mode_name
    torch.manual_seed(31415 + inference_mode + B + T + H + D)
    device = torch.device("cuda")
    _skip_if_unsupported_dim(D)
    atol, rtol = (6e-2, 6e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)

    q = _make_qk((B, T, H, D), device, dtype, use_qk_l2norm_in_kernel)
    k = _make_qk((B, T, H, D), device, dtype, use_qk_l2norm_in_kernel)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(
        B, T, H, device=device, dtype=torch.float32)).to(dtype)
    gate = F.logsigmoid(torch.rand(
        B, T, H, device=device, dtype=torch.float32))
    gate = gate / gate_logit_normalizer
    gate = gate * (torch.rand_like(gate) > mask_p)
    gate = gate.to(dtype)

    initial_state_fla = None
    if use_initial_state:
        initial_state_fla = torch.randn(B, H, D, D, device=device, dtype=dtype)

    ref_out, ref_state = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=gate.clone(),
        scale=1.0 / math.sqrt(D),
        initial_state=initial_state_fla.clone() if initial_state_fla is not None else None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # gdn recurrent state layout is [B, H, V, K], while FLA uses [B, H, K, V].
    state_gdn = (
        initial_state_fla.transpose(-1, -2).contiguous().clone()
        if initial_state_fla is not None
        else torch.zeros(B, H, D, D, device=device, dtype=dtype)
    )

    out, final_state = gdn_cuda.recurrent_forward(
        q.clone(),
        k.clone(),
        v.clone(),
        state_gdn,
        gate.clone(),
        beta.clone(),
        None,
        None,
        inference_mode,
        use_qk_l2norm_in_kernel,
    )

    _assert_close("recurrent/padded/output",
                  ref_out, out, atol=atol, rtol=rtol)
    _assert_close("recurrent/padded/final_state", ref_state.transpose(-1, -
                  2).to(dtype), final_state, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_qk_l2norm_in_kernel", [False, True], ids=["qk_l2norm_off", "qk_l2norm_on"])
@pytest.mark.parametrize(
    ("mode_name", "inference_mode", "use_initial_state"),
    [
        pytest.param("prefill", 0, False, id="prefill"),
        pytest.param("decode", 1, True, id="decode"),
    ],
)
@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens"),
    [
        pytest.param(
            *shape,
            id=f"H{shape[0]}-D{shape[1]}-mask_p{shape[2]}-cu_seqlens{shape[3]}",
        )
        for shape in _VARLEN_RECURRENT_SHAPES
    ],
)
def test_recurrent_forward_varlen_vs_fla(
    dtype: torch.dtype,
    use_qk_l2norm_in_kernel: bool,
    mode_name: str,
    inference_mode: int,
    use_initial_state: bool,
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
) -> None:
    del mode_name
    torch.manual_seed(27182 + inference_mode + H + D + len(cu_seqlens))
    device = torch.device("cuda")
    _skip_if_unsupported_dim(D)
    atol, rtol = (6e-2, 6e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)

    cu_seqlens_i32 = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
    cu_seqlens_i64 = cu_seqlens_i32.to(torch.int64)
    total_tokens = int(cu_seqlens_i32[-1].item())
    n_seq = len(cu_seqlens) - 1

    q = _make_qk((total_tokens, H, D), device, dtype, use_qk_l2norm_in_kernel)
    k = _make_qk((total_tokens, H, D), device, dtype, use_qk_l2norm_in_kernel)
    v = torch.randn(total_tokens, H, D, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(
        total_tokens, H, device=device, dtype=torch.float32)).to(dtype)
    gate = F.logsigmoid(torch.rand(
        total_tokens, H, device=device, dtype=torch.float32))
    gate = gate * (torch.rand_like(gate) > mask_p)
    gate = gate.to(dtype)

    initial_state_fla = None
    if use_initial_state:
        initial_state_fla = torch.randn(
            n_seq, H, D, D, device=device, dtype=dtype)

    ref_out, ref_state = fused_recurrent_gated_delta_rule(
        q=q.unsqueeze(0).clone(),
        k=k.unsqueeze(0).clone(),
        v=v.unsqueeze(0).clone(),
        beta=beta.unsqueeze(0).clone(),
        g=gate.unsqueeze(0).clone(),
        scale=1.0 / math.sqrt(D),
        initial_state=initial_state_fla.clone() if initial_state_fla is not None else None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens_i64.clone(),
    )
    ref_out = ref_out.squeeze(0)

    state_gdn = (
        initial_state_fla.transpose(-1, -2).contiguous().clone()
        if initial_state_fla is not None
        else torch.zeros(n_seq, H, D, D, device=device, dtype=dtype)
    )

    out, final_state = gdn_cuda.recurrent_forward(
        q.clone(),
        k.clone(),
        v.clone(),
        state_gdn,
        beta.clone(),
        gate.clone(),
        cu_seqlens_i32.clone(),
        None,
        inference_mode,
        use_qk_l2norm_in_kernel,
    )

    _assert_close("recurrent/varlen/output",
                  ref_out, out, atol=atol, rtol=rtol)
    _assert_close("recurrent/varlen/final_state", ref_state.transpose(-1, -
                  2).to(dtype), final_state, atol=atol, rtol=rtol)
