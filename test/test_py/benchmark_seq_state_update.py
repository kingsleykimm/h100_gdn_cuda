from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


_PADDED_SHAPES = [
    # (B, T, H, D, gate_logit_normalizer, mask_p)
    (1, 63, 1, 64, 1.0, 0.0),
    (2, 500, 3, 60, 1.0, 0.0),
    (2, 1000, 3, 64, 1.0, 0.5),
    (3, 1024, 4, 100, 0.1, 0.0),
    (4, 1024, 4, 128, 1.0, 0.0),
    (2, 1500, 4, 128, 10.0, 0.0),
    (4, 2048, 8, 64, 1.0, 0.0),
]

_VARLEN_SHAPES = [
    # (H, D, mask_p, cu_seqlens)
    (4, 60, 0.0, [0, 15]),
    (4, 64, 0.0, [0, 256, 500, 1000]),
    (4, 64, 0.5, [0, 256, 500, 1000]),
    (4, 100, 0.0, [0, 15, 100, 300, 1200, 2000]),
    (4, 60, 0.0, [0, 8192]),
]


def _uniform_cu_seqlens(batch_size: int, seqlen: int) -> list[int]:
    return [idx * seqlen for idx in range(batch_size + 1)]


def _alternating_cu_seqlens(batch_size: int, short_len: int, long_len: int) -> list[int]:
    out = [0]
    acc = 0
    for idx in range(batch_size):
        acc += short_len if idx % 2 == 0 else long_len
        out.append(acc)
    return out


_PADDED_HIGH_BATCH_SHAPES = [
    (8, 512, 8, 64, 1.0, 0.0),
    (16, 512, 8, 64, 1.0, 0.0),
    (32, 256, 8, 64, 1.0, 0.0),
    (64, 128, 8, 64, 1.0, 0.0),
    (8, 512, 8, 128, 1.0, 0.0),
    (16, 256, 8, 128, 1.0, 0.0),
    (32, 128, 8, 128, 1.0, 0.0),
    (64, 64, 8, 128, 1.0, 0.0),
]

_VARLEN_HIGH_BATCH_SHAPES = [
    (4, 64, 0.0, _uniform_cu_seqlens(8, 512)),
    (4, 64, 0.0, _uniform_cu_seqlens(16, 512)),
    (4, 64, 0.0, _uniform_cu_seqlens(32, 256)),
    (4, 64, 0.0, _uniform_cu_seqlens(64, 128)),
    (4, 128, 0.0, _uniform_cu_seqlens(32, 128)),
    (4, 128, 0.0, _alternating_cu_seqlens(64, 16, 192)),
    (4, 64, 0.5, _alternating_cu_seqlens(128, 8, 64)),
]


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    case_id: str
    shape: str
    warmup: int
    iters: int
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


def _chunk_local_cumsum_padded(gate: torch.Tensor, chunk_size: int) -> torch.Tensor:
    out = torch.empty_like(gate, dtype=torch.float32)
    seq_len = gate.size(1)
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        out[:, start:end, :] = torch.cumsum(gate[:, start:end, :], dim=1)
    return out


def _chunk_local_cumsum_varlen(gate: torch.Tensor, cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    out = torch.empty_like(gate, dtype=torch.float32)
    host = cu_seqlens.cpu().tolist()
    for b in range(len(host) - 1):
        seq_start, seq_end = host[b], host[b + 1]
        for start in range(seq_start, seq_end, chunk_size):
            end = min(start + chunk_size, seq_end)
            out[start:end, :] = torch.cumsum(gate[start:end, :], dim=0)
    return out


def _time_cuda_callable(
    fn: Callable[[], object], warmup: int, iters: int, use_cuda_graph: bool
) -> tuple[float, float, float, float]:
    torch.cuda.synchronize()
    replay: Callable[[], object] = fn
    if use_cuda_graph:
        with torch.inference_mode():
            for _ in range(warmup):
                _ = fn()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        captured_out = None
        with torch.inference_mode():
            with torch.cuda.graph(graph):
                captured_out = fn()
        torch.cuda.synchronize()

        _ = captured_out
        replay = graph.replay

        graph_warmup = min(5, max(1, warmup // 4)) if warmup > 0 else 1
        with torch.inference_mode():
            for _ in range(graph_warmup):
                replay()
        torch.cuda.synchronize()
    else:
        with torch.inference_mode():
            for _ in range(warmup):
                _ = fn()
        torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with torch.inference_mode():
        for idx in range(iters):
            starts[idx].record()
            _ = replay()
            ends[idx].record()
    torch.cuda.synchronize()

    latencies = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends)],
        dtype=torch.float64,
    )
    return (
        latencies.median().item(),
        torch.quantile(latencies, 0.95).item(),
        latencies.min().item(),
        latencies.max().item(),
    )


def _fmt_padded_case(shape: tuple[int, int, int, int, float, float]) -> str:
    B, T, H, D, gate_norm, mask_p = shape
    return f"B={B},T={T},H={H},D={D},gate_norm={gate_norm},mask_p={mask_p}"


def _fmt_varlen_case(shape: tuple[int, int, float, list[int]]) -> str:
    H, D, mask_p, cu_seqlens = shape
    batch_size = len(cu_seqlens) - 1
    total_tokens = cu_seqlens[-1]
    return f"B={batch_size},total_tokens={total_tokens},H={H},D={D},mask_p={mask_p}"


def _select_padded_shapes(suite: str) -> list[tuple[int, int, int, int, float, float]]:
    if suite == "base":
        return list(_PADDED_SHAPES)
    if suite == "high_batch":
        return list(_PADDED_HIGH_BATCH_SHAPES)
    return list(_PADDED_SHAPES) + list(_PADDED_HIGH_BATCH_SHAPES)


def _select_varlen_shapes(suite: str) -> list[tuple[int, int, float, list[int]]]:
    if suite == "base":
        return list(_VARLEN_SHAPES)
    if suite == "high_batch":
        return list(_VARLEN_HIGH_BATCH_SHAPES)
    return list(_VARLEN_SHAPES) + list(_VARLEN_HIGH_BATCH_SHAPES)


def _print_results(results: list[BenchmarkResult]) -> None:
    if not results:
        print("No benchmark results were produced.")
        return
    print("\nLatency results (milliseconds):")
    print(f"{'mode':<8} {'shape_case':<56} {'median':>8} {'p95':>8} {'min':>8} {'max':>8}")
    print("-" * 100)
    for res in results:
        print(
            f"{res.mode:<8} {res.case_id:<56} "
            f"{res.median_ms:>8.3f} {res.p95_ms:>8.3f} {res.min_ms:>8.3f} {res.max_ms:>8.3f}"
        )


def _write_csv(path: str, results: list[BenchmarkResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "case_id",
                "shape",
                "warmup",
                "iters",
                "median_ms",
                "p95_ms",
                "min_ms",
                "max_ms",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "mode": row.mode,
                    "case_id": row.case_id,
                    "shape": row.shape,
                    "warmup": row.warmup,
                    "iters": row.iters,
                    "median_ms": f"{row.median_ms:.6f}",
                    "p95_ms": f"{row.p95_ms:.6f}",
                    "min_ms": f"{row.min_ms:.6f}",
                    "max_ms": f"{row.max_ms:.6f}",
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark gdn_cuda seq_state_update latency.")
    parser.add_argument(
        "--mode", choices=["padded", "varlen", "both"], default="both")
    parser.add_argument(
        "--suite", choices=["base", "high_batch", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-cases", type=int, default=0,
                        help="If >0, only run the first N shapes per mode.")
    parser.add_argument("--csv", type=str, default="",
                        help="Optional CSV output path.")
    parser.add_argument("--no-empty-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if torch.cuda.get_device_capability(0)[0] < 9:
        raise RuntimeError(
            "SM90+ GPU is required for gdn_cuda benchmark targets.")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    use_cuda_graph = _env_flag("USE_CUDA_GRAPH", default=False)

    import gdn_cuda as _gdn_cuda

    library_root_path = os.getenv("LIBRARY_ROOT_PATH")
    if library_root_path is None:
        library_root_path = str(Path(__file__).resolve().parents[2])
    cuda_home = os.getenv("CUDA_HOME_PATH") or os.getenv(
        "CUDA_HOME") or "/usr/local/cuda"
    _gdn_cuda.init(library_root_path, cuda_home)
    gdn_cuda = _gdn_cuda

    print("Timing method: CUDA events + warmup + synchronize.")
    print(
        f"CUDA graph replay: {'enabled' if use_cuda_graph else 'disabled'} (USE_CUDA_GRAPH)")
    print("Benchmark target: seq_state_update component only.")

    results: list[BenchmarkResult] = []

    run_padded = args.mode in ("padded", "both")
    run_varlen = args.mode in ("varlen", "both")
    selected_padded = _select_padded_shapes(args.suite)
    selected_varlen = _select_varlen_shapes(args.suite)
    if args.max_cases > 0:
        selected_padded = selected_padded[: args.max_cases]
        selected_varlen = selected_varlen[: args.max_cases]

    if run_padded:
        print(
            f"\nRunning padded seq_state_update benchmarks: {len(selected_padded)} cases")
        for idx, shape in enumerate(selected_padded):
            B, T, H, D, gate_logit_normalizer, mask_p = shape
            shape_text = _fmt_padded_case(shape)
            case_id = shape_text

            torch.manual_seed(args.seed + idx)
            num_chunks = (T + args.chunk_size - 1) // args.chunk_size

            k = _make_l2_normalized_key((B, T, H, D), device, dtype)
            u_base = torch.randn(B, T, H, D, device=device, dtype=dtype) * 0.1
            w = torch.randn(B, T, H, D, device=device, dtype=dtype) * 0.1

            gate = F.logsigmoid(torch.rand(
                B, T, H, device=device, dtype=torch.float32))
            gate = gate / gate_logit_normalizer
            gate = gate * (torch.rand_like(gate) > mask_p)
            gate_cumsum = _chunk_local_cumsum_padded(gate, args.chunk_size)

            initial_state = torch.zeros(
                B, H, D, D, device=device, dtype=dtype)
            initial_state_gdn = initial_state.transpose(
                -1, -2).contiguous().clone()

            u_kernel = u_base.clone()
            state_kernel = torch.empty(
                B, num_chunks, H, D, D, device=device, dtype=dtype)
            final_state_kernel = torch.empty(
                B, H, D, D, device=device, dtype=dtype)
            gate_mn = gdn_cuda.transpose_to_mn_major(
                gate_cumsum.clone(), 128)

            def _run_gdn_padded() -> object:
                return gdn_cuda.bf16_chunked_seq_state_update(
                    k,
                    u_kernel,
                    w,
                    initial_state_gdn,
                    state_kernel,
                    final_state_kernel,
                    gate_mn,
                    "t",
                    None,
                    None,
                    None,
                    args.chunk_size,
                )

            median_ms, p95_ms, min_ms, max_ms = _time_cuda_callable(
                _run_gdn_padded, args.warmup, args.iters, use_cuda_graph
            )
            results.append(
                BenchmarkResult(
                    mode="padded",
                    case_id=case_id,
                    shape=shape_text,
                    warmup=args.warmup,
                    iters=args.iters,
                    median_ms=median_ms,
                    p95_ms=p95_ms,
                    min_ms=min_ms,
                    max_ms=max_ms,
                )
            )
            print(
                f"[done] {case_id} median={median_ms:.3f} ms p95={p95_ms:.3f} ms")

            if not args.no_empty_cache:
                torch.cuda.empty_cache()

    if run_varlen:
        print(
            f"\nRunning varlen seq_state_update benchmarks: {len(selected_varlen)} cases")
        for idx, shape in enumerate(selected_varlen):
            H, D, mask_p, cu_seqlens = shape
            shape_text = _fmt_varlen_case(shape)
            case_id = shape_text

            torch.manual_seed(args.seed + 10000 + idx)
            cu_seqlens_i32 = torch.tensor(
                cu_seqlens, device=device, dtype=torch.int32)
            total_tokens = int(cu_seqlens_i32[-1].item())
            _chunk_indices, cu_chunks = _prepare_varlen_indices(
                cu_seqlens_i32, args.chunk_size)
            total_chunks = int(cu_chunks[-1].item())
            n_seq = len(cu_seqlens) - 1

            k = _make_l2_normalized_key((total_tokens, H, D), device, dtype)
            u_base = torch.randn(total_tokens, H, D,
                                 device=device, dtype=dtype) * 0.1
            w = torch.randn(total_tokens, H, D,
                            device=device, dtype=dtype) * 0.1

            gate = F.logsigmoid(torch.rand(
                total_tokens, H, device=device, dtype=torch.float32))
            gate = gate * (torch.rand_like(gate) > mask_p)
            gate_cumsum = _chunk_local_cumsum_varlen(
                gate, cu_seqlens_i32, args.chunk_size)

            initial_state = torch.zeros(
                n_seq, H, D, D, device=device, dtype=dtype)
            initial_state_gdn = initial_state.transpose(
                -1, -2).contiguous().clone()

            u_kernel = u_base.clone()
            state_kernel = torch.empty(
                total_chunks, H, D, D, device=device, dtype=dtype)
            final_state_kernel = torch.empty(
                n_seq, H, D, D, device=device, dtype=dtype)
            gate_mn = gdn_cuda.transpose_to_mn_major(
                gate_cumsum.clone(), 128)

            def _run_gdn_varlen() -> object:
                return gdn_cuda.bf16_chunked_seq_state_update(
                    k,
                    u_kernel,
                    w,
                    initial_state_gdn,
                    state_kernel,
                    final_state_kernel,
                    gate_mn,
                    "t",
                    cu_seqlens_i32,
                    cu_chunks,
                    total_chunks,
                    args.chunk_size,
                )

            median_ms, p95_ms, min_ms, max_ms = _time_cuda_callable(
                _run_gdn_varlen, args.warmup, args.iters, use_cuda_graph
            )
            results.append(
                BenchmarkResult(
                    mode="varlen",
                    case_id=case_id,
                    shape=shape_text,
                    warmup=args.warmup,
                    iters=args.iters,
                    median_ms=median_ms,
                    p95_ms=p95_ms,
                    min_ms=min_ms,
                    max_ms=max_ms,
                )
            )
            print(
                f"[done] {case_id} median={median_ms:.3f} ms p95={p95_ms:.3f} ms")

            if not args.no_empty_cache:
                torch.cuda.empty_cache()

    _print_results(results)
    if args.csv:
        _write_csv(args.csv, results)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
