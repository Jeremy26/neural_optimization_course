"""Static stats + device latency estimates.

Two related concerns live here:

1. Computing network statistics that don't require running a forward pass --
   parameter breakdown by layer kind, memory footprint at FP32 / FP16 / INT8,
   etc.  These work on both ``nn.Module`` and bare state-dicts.

2. Estimating wall-clock latency on common deployment targets given a FLOP
   count.  These are theoretical bounds (peak TFLOPS * efficiency factor),
   not guarantees -- real latency depends on batch size, kernel fusion,
   memory bandwidth, and a dozen other things the course covers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


# (display, fp32_tflops, fp16_tflops, int8_tops, efficiency)
# TFLOPS / TOPS are vendor peak numbers.  Efficiency is the fraction of peak
# you'd realistically hit on inference workloads at batch sizes >= 8 with a
# well-optimized graph -- typical real-world inference sits at 25-50%.
DEVICES = [
    # Datacenter
    ("NVIDIA H100 (SXM)",   67.0, 989.0, 1979.0, 0.45),
    ("NVIDIA A100 (80GB)",  19.5, 312.0,  624.0, 0.45),
    ("NVIDIA L40S",         91.6, 362.0,  733.0, 0.40),
    ("NVIDIA L4",           30.3, 121.0,  242.0, 0.40),
    ("NVIDIA T4",            8.1,  65.0,  130.0, 0.35),
    # Workstation / consumer
    ("NVIDIA RTX 4090",     82.6, 165.0,  660.0, 0.35),
    ("NVIDIA RTX 3090",     35.6,  35.6,  284.0, 0.30),
    # Edge / embedded
    ("Jetson Orin AGX",      5.3,  85.0,  170.0, 0.35),
    ("Jetson Orin Nano",     0.6,  20.0,   40.0, 0.30),
    ("Coral Edge TPU",       0.0,   0.0,    4.0, 0.30),
    # CPU (single-core estimate -- multi-core scales linearly until memory
    # bandwidth caps; for batch=1 inference single-core dominates anyway).
    ("CPU (Xeon, 1 core)",   0.05,  0.05,   0.10, 0.40),
    ("CPU (M3 Pro, 1 core)", 0.10,  0.10,   0.20, 0.45),
    # Mobile (Apple NPU and equivalents)
    ("iPhone 15 Pro (NPU)",  1.0,    1.0,   35.0, 0.30),
    ("Pixel 8 (TPU)",        0.5,    0.5,   18.0, 0.30),
]


@dataclass
class NetworkStats:
    parameter_count: int
    weight_tensors: int
    leaf_modules: int
    fp32_mb: float          # current footprint at observed dtype mix
    fp16_mb: float
    int8_mb: float
    layer_kind_share: dict[str, int]   # kind -> param count
    biggest_layer: tuple[str, int] | None  # (name, params)
    deepest_path: int       # max dot-depth of any parameter name


def _walk_tensors(obj: Any) -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = []
    if isinstance(obj, nn.Module):
        for k, v in obj.state_dict().items():
            out.append((k, v))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                out.append((k, v))
    return out


def _classify_layer_kind(name: str, tensor: torch.Tensor, full_obj: Any) -> str:
    """Coarse bucket: Conv / Linear / Norm / Embedding / Other."""
    suffix = name.rsplit(".", 1)[-1]
    if suffix in {"running_mean", "running_var", "num_batches_tracked"}:
        return "Norm"
    # Conv weights are 4D (or 3D for Conv1d); Linear is 2D.
    if tensor.dim() == 4:
        return "Conv"
    if tensor.dim() == 3 and "conv" in name.lower():
        return "Conv"
    if tensor.dim() == 2:
        return "Linear"
    if tensor.dim() == 2 and "embed" in name.lower():
        return "Embedding"
    if tensor.dim() == 1 and ("norm" in name.lower() or "bn" in name.lower()):
        return "Norm"
    return "Other"


def network_stats(obj: Any) -> NetworkStats:
    tensors = _walk_tensors(obj)
    params = sum(t.numel() for t in [v for _, v in tensors])
    weight_count = sum(1 for _, t in tensors if t.dim() >= 2 and t.is_floating_point())
    # Group params by parent path -> count "layers".
    parents: dict[str, int] = defaultdict(int)
    kind_share: Counter = Counter()
    deepest = 0
    biggest_name = ""
    biggest_n = 0
    for name, tensor in tensors:
        n = tensor.numel()
        parent = name.rsplit(".", 1)[0] if "." in name else "(root)"
        parents[parent] += n
        deepest = max(deepest, name.count("."))
        kind_share[_classify_layer_kind(name, tensor, obj)] += n
        if n > biggest_n:
            biggest_n = n
            biggest_name = parent
    # Footprints assuming a pure-precision shipout.
    bytes_fp32 = params * 4
    bytes_fp16 = params * 2
    bytes_int8 = params * 1
    return NetworkStats(
        parameter_count=params,
        weight_tensors=weight_count,
        leaf_modules=len(parents),
        fp32_mb=bytes_fp32 / 1e6,
        fp16_mb=bytes_fp16 / 1e6,
        int8_mb=bytes_int8 / 1e6,
        layer_kind_share=dict(kind_share),
        biggest_layer=(biggest_name, biggest_n) if biggest_name else None,
        deepest_path=deepest,
    )


def estimate_latency_ms(flops: int, dtype: str = "fp16") -> list[dict]:
    """Per-device theoretical latency for a single forward pass.

    Returns a list of ``{device, peak_tflops, eff, latency_ms,
    throughput_per_s}`` rows sorted by latency.  ``dtype`` controls which
    column of the TFLOPS table we read from.
    """
    rows = []
    for name, fp32, fp16, int8, eff in DEVICES:
        peak = {"fp32": fp32, "fp16": fp16, "int8": int8}.get(dtype, fp16)
        if peak <= 0:
            continue
        # Effective throughput in FLOPs/sec.
        effective = peak * 1e12 * eff
        latency_s = flops / effective
        rows.append({
            "device": name,
            "peak_tflops": peak,
            "efficiency": eff,
            "latency_ms": latency_s * 1000.0,
            "throughput_per_s": 1.0 / latency_s if latency_s > 0 else 0.0,
        })
    rows.sort(key=lambda r: r["latency_ms"])
    return rows
