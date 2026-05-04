"""Model analysis for the Deployment Health Score visualizer.

The analyzer inspects a PyTorch checkpoint (``.pt`` / ``.pth``) and reports on:

* Floating-point precision used by the parameters
* Sparsity / pruning headroom
* On-disk and in-memory size
* Architectural compatibility with ONNX and TensorRT export

Each check returns a category score in ``[0, 100]`` and a short human-readable
verdict.  The aggregate ``deployment_health_score`` is a weighted average of
those category scores so users get one number to act on.
"""

from __future__ import annotations

import io
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


# Operators / module classes that TensorRT / ONNX commonly struggle with
# without custom plugins.  This is intentionally conservative -- the goal is
# to flag *risk*, not to pretend to be exhaustive.
_ONNX_RISKY_MODULES = {
    "LSTMCell",
    "GRUCell",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "MultiheadAttention",
}
_TRT_RISKY_MODULES = _ONNX_RISKY_MODULES | {
    "Embedding",
    "EmbeddingBag",
    "LayerNorm",  # supported but slow paths in older TRT
}

# Bytes per element for the dtypes we surface.
_DTYPE_BYTES = {
    torch.float32: 4,
    torch.float64: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.bool: 1,
}


@dataclass
class CategoryReport:
    score: float
    verdict: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    file_size_mb: float
    parameter_count: int
    precision: CategoryReport
    pruning: CategoryReport
    size: CategoryReport
    exportability: CategoryReport
    deployment_health_score: float
    overall_verdict: str
    recommendations: list[str]
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_checkpoint(buffer: bytes) -> tuple[Any, str]:
    """Load a checkpoint from raw bytes.

    Tries ``weights_only=True`` first (safe -- no arbitrary code execution),
    then falls back to the legacy unpickler.  The returned ``mode`` indicates
    which path succeeded so the UI can warn the user when an untrusted file is
    deserialized through pickle.
    """
    bio = io.BytesIO(buffer)
    try:
        obj = torch.load(bio, map_location="cpu", weights_only=True)
        return obj, "weights_only"
    except Exception:
        bio.seek(0)
        obj = torch.load(bio, map_location="cpu", weights_only=False)
        return obj, "pickle"


def _iter_tensors(obj: Any) -> list[tuple[str, torch.Tensor]]:
    """Flatten a checkpoint into a list of ``(name, tensor)`` pairs."""
    out: list[tuple[str, torch.Tensor]] = []

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, torch.Tensor):
            out.append((prefix or "tensor", node))
        elif isinstance(node, nn.Module):
            for name, p in node.state_dict().items():
                out.append((f"{prefix}.{name}" if prefix else name, p))
        elif isinstance(node, (dict, OrderedDict)):
            for k, v in node.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(f"{prefix}[{i}]", v)

    walk("", obj)
    return out


def _module_class_counts(obj: Any) -> Counter:
    """If the checkpoint is an ``nn.Module``, count its sub-module classes."""
    counts: Counter = Counter()
    if isinstance(obj, nn.Module):
        for module in obj.modules():
            counts[type(module).__name__] += 1
    return counts


# ---------------------------------------------------------------------------
# Category checks
# ---------------------------------------------------------------------------


def _check_precision(tensors: list[tuple[str, torch.Tensor]]) -> CategoryReport:
    if not tensors:
        return CategoryReport(0.0, "No tensors found in checkpoint.")

    dtype_bytes = 0
    total_elems = 0
    dtype_counts: Counter = Counter()
    for _, t in tensors:
        n = t.numel()
        total_elems += n
        dtype_counts[str(t.dtype)] += n
        dtype_bytes += n * _DTYPE_BYTES.get(t.dtype, t.element_size())

    # Score: FP32 = 30, FP16/BF16 = 75, INT8 / quantized = 100, mixed scaled.
    weighted = 0.0
    for _, t in tensors:
        n = t.numel()
        if t.dtype in (torch.float64,):
            score = 10.0
        elif t.dtype is torch.float32:
            score = 30.0
        elif t.dtype in (torch.float16, torch.bfloat16):
            score = 75.0
        elif t.dtype in (torch.int8, torch.uint8, torch.qint8, torch.quint8):
            score = 100.0
        else:
            score = 50.0
        weighted += score * n
    score = weighted / max(total_elems, 1)

    dominant = max(dtype_counts.items(), key=lambda kv: kv[1])[0]
    if score >= 90:
        verdict = f"Already quantized ({dominant})."
    elif score >= 60:
        verdict = f"Half precision ({dominant}) -- good for GPU inference."
    elif score >= 25:
        verdict = (
            f"Full precision ({dominant}). Quantization to FP16/INT8 could cut "
            "size by 2-4x with minimal accuracy loss."
        )
    else:
        verdict = f"Double precision ({dominant}) -- almost always overkill."

    return CategoryReport(
        score=score,
        verdict=verdict,
        details={
            "dtype_distribution": dict(dtype_counts),
            "bytes_in_memory": dtype_bytes,
        },
    )


def _check_pruning(tensors: list[tuple[str, torch.Tensor]]) -> CategoryReport:
    """Score how much *unrealised* pruning headroom the model has.

    Higher score == more sparsity already present (less work left to do).
    A dense FP model gets a low score because pruning could still pay off.
    """
    weight_tensors = [
        (n, t) for n, t in tensors
        if t.dtype.is_floating_point and t.dim() >= 2
    ]
    if not weight_tensors:
        return CategoryReport(
            50.0,
            "No 2D+ weight tensors found -- pruning analysis skipped.",
        )

    total_elems = 0
    total_zeros = 0
    near_zero_total = 0
    per_layer = []
    for name, t in weight_tensors:
        n = t.numel()
        zeros = int((t == 0).sum().item())
        # "near-zero" = |w| below 1% of the layer's max-abs weight.  This is a
        # cheap proxy for "could be pruned without hurting accuracy".
        max_abs = float(t.abs().max().item()) if n else 0.0
        threshold = max_abs * 0.01
        near_zero = int((t.abs() < threshold).sum().item()) if max_abs > 0 else 0
        total_elems += n
        total_zeros += zeros
        near_zero_total += near_zero
        per_layer.append({
            "name": name,
            "shape": list(t.shape),
            "sparsity": zeros / n if n else 0.0,
            "near_zero_fraction": near_zero / n if n else 0.0,
        })

    sparsity = total_zeros / max(total_elems, 1)
    headroom = near_zero_total / max(total_elems, 1)

    # Score blends realised sparsity with detected headroom.  The intuition is
    # that a model that's already 50% sparse is well-optimised, while a model
    # that's 0% sparse but has 30% near-zero weights has obvious upside.
    if sparsity >= 0.5:
        score = 90.0
    elif sparsity >= 0.2:
        score = 70.0
    elif sparsity >= 0.05:
        score = 55.0
    else:
        # Dense model.  Reward potential headroom modestly so users see *some*
        # signal, but keep the score low to motivate pruning.
        score = max(20.0, 40.0 - 100.0 * (0.05 - sparsity)) - 10.0
        score = max(15.0, score + min(20.0, headroom * 50.0))

    if sparsity >= 0.5:
        verdict = f"Already {sparsity:.0%} sparse -- pruning is well-applied."
    elif sparsity >= 0.05:
        verdict = (
            f"Partially pruned ({sparsity:.1%}). Structured pruning could push "
            "this further."
        )
    elif headroom >= 0.2:
        verdict = (
            f"Dense, but {headroom:.0%} of weights are near zero -- magnitude "
            "pruning is a low-risk win."
        )
    else:
        verdict = (
            "Dense model with little obvious headroom. Try iterative magnitude "
            "pruning or movement pruning."
        )

    # Sort layers by headroom for the UI to surface biggest offenders.
    per_layer.sort(key=lambda d: d["near_zero_fraction"], reverse=True)
    return CategoryReport(
        score=score,
        verdict=verdict,
        details={
            "global_sparsity": sparsity,
            "near_zero_headroom": headroom,
            "top_layers": per_layer[:10],
        },
    )


def _check_size(
    file_size_mb: float, tensors: list[tuple[str, torch.Tensor]]
) -> CategoryReport:
    """Score the raw size.  Not every model needs to be tiny, but a 500MB file
    is almost never deploy-friendly without help."""
    param_count = sum(t.numel() for _, t in tensors)

    # Scoring buckets are tuned for "typical" inference targets.  Edge models
    # under 25MB are excellent; >500MB is rough without sharding/quant.
    if file_size_mb < 25:
        score = 95.0
    elif file_size_mb < 100:
        score = 80.0
    elif file_size_mb < 250:
        score = 60.0
    elif file_size_mb < 500:
        score = 40.0
    elif file_size_mb < 1000:
        score = 25.0
    else:
        score = 10.0

    # Big-but-quantized models should not be punished as hard.  We can't tell
    # here, so we leave the FP penalty to ``_check_precision``.
    if param_count >= 1e9:
        readable = f"{param_count / 1e9:.2f}B"
    elif param_count >= 1e6:
        readable = f"{param_count / 1e6:.2f}M"
    else:
        readable = f"{param_count:,}"

    verdict = f"{file_size_mb:.1f} MB on disk, ~{readable} parameters."
    return CategoryReport(
        score=score,
        verdict=verdict,
        details={"parameter_count": param_count},
    )


def _check_exportability(obj: Any) -> CategoryReport:
    """Heuristic ONNX / TensorRT compatibility.

    We can't actually run ``torch.onnx.export`` here without knowing the input
    signature, so we score on what *we can see*: the module class mix.
    """
    counts = _module_class_counts(obj)

    if not counts:
        return CategoryReport(
            50.0,
            "Checkpoint is a state-dict only -- export feasibility depends on "
            "the model architecture you load it into. ONNX usually works for "
            "vision/MLP backbones; check ops manually for custom layers.",
            details={"is_state_dict": True},
        )

    onnx_risky = sum(c for k, c in counts.items() if k in _ONNX_RISKY_MODULES)
    trt_risky = sum(c for k, c in counts.items() if k in _TRT_RISKY_MODULES)
    total_modules = sum(counts.values())

    onnx_score = 100.0 - 60.0 * (onnx_risky / max(total_modules, 1))
    trt_score = 100.0 - 70.0 * (trt_risky / max(total_modules, 1))
    score = (onnx_score + trt_score) / 2

    verdict_bits = []
    if onnx_risky == 0:
        verdict_bits.append("ONNX export: clean path.")
    else:
        verdict_bits.append(
            f"ONNX export: {onnx_risky} layer(s) need attention "
            "(transformers, RNN cells)."
        )
    if trt_risky == 0:
        verdict_bits.append("TensorRT: no obvious blockers.")
    else:
        verdict_bits.append(
            f"TensorRT: {trt_risky} layer(s) may need plugins or rewrites."
        )

    return CategoryReport(
        score=score,
        verdict=" ".join(verdict_bits),
        details={
            "onnx_risky_modules": onnx_risky,
            "tensorrt_risky_modules": trt_risky,
            "module_classes": dict(counts.most_common(15)),
        },
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


_CATEGORY_WEIGHTS = {
    "precision": 0.30,
    "pruning": 0.20,
    "size": 0.20,
    "exportability": 0.30,
}


def _build_recommendations(
    precision: CategoryReport,
    pruning: CategoryReport,
    size: CategoryReport,
    exportability: CategoryReport,
) -> list[str]:
    recs: list[str] = []
    if precision.score < 60:
        recs.append(
            "Quantize to FP16 (1 line with ``model.half()``) or run dynamic "
            "INT8 quantization for CPU inference."
        )
    if pruning.score < 50:
        headroom = pruning.details.get("near_zero_headroom", 0.0)
        if headroom > 0.15:
            recs.append(
                f"~{headroom:.0%} of weights are effectively zero. Apply "
                "magnitude pruning + a short fine-tune to bake in the savings."
            )
        else:
            recs.append(
                "Try structured pruning (channel / head pruning) -- magnitude "
                "pruning headroom looks limited."
            )
    if size.score < 60:
        recs.append(
            "Combine quantization with knowledge distillation into a smaller "
            "student model to shrink the deployable artifact."
        )
    risky_count = (
        exportability.details.get("onnx_risky_modules", 0)
        + exportability.details.get("tensorrt_risky_modules", 0)
    )
    if exportability.score < 70 and risky_count > 0:
        recs.append(
            "Refactor risky layers (custom attention, RNN cells) into "
            "ONNX-friendly equivalents before exporting."
        )
    elif exportability.details.get("is_state_dict"):
        recs.append(
            "Save the full ``nn.Module`` (not just ``state_dict()``) and re-run "
            "for a precise export-compatibility verdict."
        )
    if not recs:
        recs.append(
            "Model looks deployment-ready. Benchmark latency on your target "
            "hardware to confirm."
        )
    return recs


def analyze(buffer: bytes) -> tuple[AnalysisReport, str]:
    """Full pipeline: bytes in, ``AnalysisReport`` out."""
    obj, load_mode = load_checkpoint(buffer)
    tensors = _iter_tensors(obj)
    file_size_mb = len(buffer) / (1024 * 1024)

    precision = _check_precision(tensors)
    pruning = _check_pruning(tensors)
    size = _check_size(file_size_mb, tensors)
    exportability = _check_exportability(obj)

    score = (
        precision.score * _CATEGORY_WEIGHTS["precision"]
        + pruning.score * _CATEGORY_WEIGHTS["pruning"]
        + size.score * _CATEGORY_WEIGHTS["size"]
        + exportability.score * _CATEGORY_WEIGHTS["exportability"]
    )
    score = max(0.0, min(100.0, score))

    if score >= 80:
        overall = "Deployment-ready. Ship it."
    elif score >= 60:
        overall = "Solid foundation -- a couple of quick wins remain."
    elif score >= 40:
        overall = "Functional, but you're leaving real performance on the table."
    else:
        overall = "Significant optimization work needed before production."

    parameter_count = sum(t.numel() for _, t in tensors)
    report = AnalysisReport(
        file_size_mb=file_size_mb,
        parameter_count=parameter_count,
        precision=precision,
        pruning=pruning,
        size=size,
        exportability=exportability,
        deployment_health_score=score,
        overall_verdict=overall,
        recommendations=_build_recommendations(
            precision, pruning, size, exportability
        ),
        raw={"load_mode": load_mode, "is_module": isinstance(obj, nn.Module)},
    )
    return report, load_mode
