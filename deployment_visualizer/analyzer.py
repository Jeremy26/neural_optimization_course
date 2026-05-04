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
from dataclasses import dataclass, field, replace
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
    architecture: str | None = None
    estimated_flops: int | None = None
    dynamic: Any = None  # benchmarks.DynamicReport, optional (currently unused)
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


_WRAPPER_KEYS = (
    "state_dict", "model_state_dict", "model", "net", "module",
    "weights", "params", "ema_state_dict",
)


def _unwrap(obj: Any) -> Any:
    """Production checkpoints are usually wrapper dicts like
    ``{"state_dict": ..., "optimizer": ..., "epoch": 12}``.  Walk down common
    wrapper keys until we hit either an ``nn.Module`` or a dict whose values
    are mostly tensors -- the thing the rest of the analyzer actually wants
    to inspect."""
    seen = set()
    for _ in range(6):  # bounded depth
        if isinstance(obj, nn.Module):
            return obj
        if not isinstance(obj, dict) or id(obj) in seen:
            return obj
        seen.add(id(obj))
        # If this dict already looks like a state-dict (mostly tensors), stop.
        tensor_count = sum(1 for v in obj.values() if isinstance(v, torch.Tensor))
        if obj and tensor_count / len(obj) >= 0.5:
            return obj
        # Otherwise descend through the first matching wrapper key.
        descended = False
        for k in _WRAPPER_KEYS:
            if k in obj and isinstance(obj[k], (dict, nn.Module)):
                obj = obj[k]
                descended = True
                break
        if not descended:
            return obj
    return obj


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
        return _unwrap(obj), "weights_only"
    except Exception:
        bio.seek(0)
        obj = torch.load(bio, map_location="cpu", weights_only=False)
        return _unwrap(obj), "pickle"


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
            "Weights are at full precision -- there's a 2-4x size and latency "
            "win sitting in quantization, but choosing FP16 vs INT8 (and PTQ "
            "vs QAT) is where most teams trip up."
        )
    if pruning.score < 50:
        headroom = pruning.details.get("near_zero_headroom", 0.0)
        if headroom > 0.15:
            recs.append(
                f"~{headroom:.0%} of weights are effectively zero. Real "
                "savings need the right pruning schedule + fine-tune recipe."
            )
        else:
            recs.append(
                "Magnitude pruning headroom is limited. Structured "
                "(channel / head) pruning could still help -- but it requires "
                "architecture-aware techniques."
            )
    if size.score < 60:
        recs.append(
            "Artifact is large for production deployment. Quantization plus "
            "distillation into a smaller student is the proven combo."
        )
    risky_count = (
        exportability.details.get("onnx_risky_modules", 0)
        + exportability.details.get("tensorrt_risky_modules", 0)
    )
    if exportability.score < 70 and risky_count > 0:
        recs.append(
            f"{risky_count} layer(s) are likely to fight you on ONNX / "
            "TensorRT export. Diagnosing and rewriting them is its own skill."
        )
    if not recs:
        recs.append(
            "Looks deployment-ready on paper -- but real-hardware benchmarking "
            "almost always uncovers more headroom."
        )
    return recs


def _apply_dynamic_to_exportability(
    exportability: CategoryReport, dynamic
) -> CategoryReport:
    """If we actually attempted an ONNX export, the result is ground truth --
    upgrade or downgrade the heuristic verdict accordingly."""
    if dynamic is None or dynamic.onnx_export is None:
        return exportability
    onnx = dynamic.onnx_export
    if onnx["ok"]:
        new_score = max(exportability.score, 90.0)
        verdict = (
            f"ONNX export succeeded ({onnx['size_mb']:.1f} MB, opset "
            f"{onnx['opset']}). " + exportability.verdict
        )
    else:
        new_score = min(exportability.score, 35.0)
        verdict = (
            f"ONNX export failed: {onnx['error']}. Address this before "
            "TensorRT / mobile deployment."
        )
    details = dict(exportability.details)
    details["onnx_attempt"] = onnx
    return CategoryReport(score=new_score, verdict=verdict, details=details)


def _aggregate_score(
    precision: CategoryReport,
    pruning: CategoryReport,
    size: CategoryReport,
    exportability: CategoryReport,
) -> float:
    score = (
        precision.score * _CATEGORY_WEIGHTS["precision"]
        + pruning.score * _CATEGORY_WEIGHTS["pruning"]
        + size.score * _CATEGORY_WEIGHTS["size"]
        + exportability.score * _CATEGORY_WEIGHTS["exportability"]
    )
    return max(0.0, min(100.0, score))


def analyze(buffer: bytes, run_benchmarks: bool = False) -> tuple[AnalysisReport, str, Any]:
    """Static analysis pipeline: bytes in, ``(report, load_mode, obj)`` out.

    The third return value is the deserialized checkpoint object -- callers
    that want to run the slow benchmarks afterwards should pass it to
    ``attach_benchmarks`` to avoid re-loading.
    """
    obj, load_mode = load_checkpoint(buffer)
    tensors = _iter_tensors(obj)
    file_size_mb = len(buffer) / (1024 * 1024)

    precision = _check_precision(tensors)
    pruning = _check_pruning(tensors)
    size = _check_size(file_size_mb, tensors)
    exportability = _check_exportability(obj)

    dynamic = None
    if run_benchmarks:
        try:
            from benchmarks import run_dynamic_analysis

            dynamic = run_dynamic_analysis(obj, file_size_mb)
        except Exception:
            dynamic = None

    exportability = _apply_dynamic_to_exportability(exportability, dynamic)

    score = _aggregate_score(precision, pruning, size, exportability)
    overall = _verdict_for_score(score)

    parameter_count = sum(t.numel() for _, t in tensors)

    # Static fingerprint + FLOPs estimate -- no forward pass required.
    from benchmarks import fingerprint
    from device_estimates import estimate_static_flops

    arch = fingerprint(obj)
    flops_estimate = estimate_static_flops(parameter_count, arch)

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
        architecture=arch,
        estimated_flops=flops_estimate,
        dynamic=dynamic,
        raw={"load_mode": load_mode, "is_module": isinstance(obj, nn.Module)},
    )
    return report, load_mode, obj


def _verdict_for_score(score: float) -> str:
    if score >= 80:
        return "Deployment-ready. Ship it."
    if score >= 60:
        return "Solid foundation -- a couple of quick wins remain."
    if score >= 40:
        return "Functional, but you're leaving real performance on the table."
    return "Significant optimization work needed before production."


def attach_benchmarks(report: AnalysisReport, obj: Any) -> AnalysisReport:
    """Run the slow live benchmarks on an already-loaded checkpoint and return
    a new report with the dynamic results merged in.

    Kept separate from ``analyze`` so the UI can render static results
    immediately and trigger the slow path on demand.
    """
    from benchmarks import run_dynamic_analysis

    try:
        dynamic = run_dynamic_analysis(obj, report.file_size_mb)
    except Exception:
        dynamic = None

    new_export = _apply_dynamic_to_exportability(report.exportability, dynamic)
    new_score = _aggregate_score(
        report.precision, report.pruning, report.size, new_export
    )
    return replace(
        report,
        exportability=new_export,
        deployment_health_score=new_score,
        overall_verdict=_verdict_for_score(new_score),
        recommendations=_build_recommendations(
            report.precision, report.pruning, report.size, new_export
        ),
        dynamic=dynamic,
    )
