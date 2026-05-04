"""Dynamic benchmarks for the Deployment Health Score visualizer.

These checks require an actual ``nn.Module`` (not a state-dict) so we can:

* Sniff a plausible dummy input from the first leaf module.
* Run timed forward passes for latency.
* Estimate FLOPs from layer shapes.
* Actually attempt ONNX export.
* Simulate dynamic quantization and magnitude pruning.

Every entry point is defensive: if anything fails (unsupported ops, weird
input shapes, oversized models) we return ``None`` for that field and the UI
just hides the section. Users should never see a stack trace.
"""

from __future__ import annotations

import copy
import hashlib
import io
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class DynamicReport:
    input_shape: tuple[int, ...] | None = None
    latency_ms_mean: float | None = None
    latency_ms_p95: float | None = None
    peak_activation_mb: float | None = None
    flops: int | None = None
    onnx_export: dict[str, Any] | None = None
    quantization_sim: dict[str, Any] | None = None
    pruning_sim: list[dict[str, Any]] = field(default_factory=list)
    architecture_guess: str | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input sniffing
# ---------------------------------------------------------------------------


def _first_leaf(module: nn.Module) -> nn.Module | None:
    for m in module.modules():
        if not list(m.children()) and any(p.requires_grad or True for p in m.parameters(recurse=False)):
            return m
    return None


def _guess_input_shape(module: nn.Module) -> tuple[int, ...] | None:
    leaf = _first_leaf(module)
    if leaf is None:
        return None
    if isinstance(leaf, nn.Conv2d):
        return (1, leaf.in_channels, 224, 224)
    if isinstance(leaf, nn.Conv1d):
        return (1, leaf.in_channels, 1024)
    if isinstance(leaf, nn.Conv3d):
        return (1, leaf.in_channels, 16, 64, 64)
    if isinstance(leaf, nn.Linear):
        return (1, leaf.in_features)
    if isinstance(leaf, nn.Embedding):
        return (1, 64)  # token-id sequence
    if isinstance(leaf, (nn.LSTM, nn.GRU, nn.RNN)):
        return (1, 32, leaf.input_size)
    return None


def _make_dummy(shape: tuple[int, ...], like_embedding: bool) -> torch.Tensor:
    if like_embedding:
        return torch.randint(0, 100, shape, dtype=torch.long)
    return torch.randn(shape)


# ---------------------------------------------------------------------------
# Latency + memory
# ---------------------------------------------------------------------------


def _bench_latency(
    module: nn.Module, dummy: torch.Tensor, iters: int = 10, warmup: int = 2
) -> tuple[float, float] | None:
    module.eval()
    try:
        with torch.no_grad():
            for _ in range(warmup):
                module(dummy)
            samples = []
            for _ in range(iters):
                t0 = time.perf_counter()
                module(dummy)
                samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        mean = sum(samples) / len(samples)
        p95 = samples[int(0.95 * len(samples)) - 1]
        return mean, p95
    except Exception:
        return None


def _peak_activation_mb(module: nn.Module, dummy: torch.Tensor) -> float | None:
    """Cheap proxy for peak activation memory: sum of intermediate tensor
    sizes captured by forward hooks.  Not exact (no autograd graph, no
    in-place reuse) but gives the right *order of magnitude* for sizing
    decisions."""
    sizes: list[int] = []
    handles = []

    def hook(_module, _inp, output):
        if isinstance(output, torch.Tensor):
            sizes.append(output.numel() * output.element_size())
        elif isinstance(output, (list, tuple)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    sizes.append(o.numel() * o.element_size())

    for m in module.modules():
        if not list(m.children()):
            handles.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():
            module(dummy)
    except Exception:
        for h in handles:
            h.remove()
        return None
    for h in handles:
        h.remove()
    return sum(sizes) / 1e6 if sizes else None


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def _estimate_flops(module: nn.Module, dummy: torch.Tensor) -> int | None:
    """Layer-shape FLOP estimate for the common cases.  Not as accurate as
    fvcore but doesn't add a dependency, and covers Conv / Linear which
    dominate inference cost for most architectures."""
    flops_total = [0]
    handles = []

    def conv_hook(m, inp, out):
        if not isinstance(out, torch.Tensor):
            return
        kernel_ops = 1
        for k in m.kernel_size:
            kernel_ops *= k
        out_elements = out.numel()
        flops_total[0] += int(out_elements * kernel_ops * (m.in_channels // m.groups))

    def linear_hook(m, inp, out):
        if isinstance(inp, tuple) and inp:
            x = inp[0]
            flops_total[0] += int(x.numel() * m.out_features)

    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            module(dummy)
    except Exception:
        for h in handles:
            h.remove()
        return None
    for h in handles:
        h.remove()
    return flops_total[0] if flops_total[0] > 0 else None


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def _try_onnx_export(module: nn.Module, dummy: torch.Tensor) -> dict[str, Any]:
    """Export to a BytesIO via the legacy tracing exporter.

    PyTorch 2.x defaults to the dynamo exporter which pulls in ``onnxscript``;
    we don't want that runtime dep, so force the tracer with ``dynamo=False``
    when the kwarg is supported (older torch silently ignores unknown kwargs
    via ``**kwargs`` -- but we still try/except).
    """
    bio = io.BytesIO()
    common = dict(
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    last_err = None
    for kwargs in ({"dynamo": False, **common}, common):
        bio = io.BytesIO()
        try:
            torch.onnx.export(module, dummy, bio, **kwargs)
            return {
                "ok": True,
                "size_mb": len(bio.getvalue()) / 1e6,
                "opset": 17,
                "error": None,
            }
        except TypeError as exc:
            # ``dynamo`` kwarg not supported on this torch version -- retry.
            last_err = str(exc).splitlines()[0][:300]
            continue
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            last_err = str(exc).splitlines()[0][:300]
            break
    return {"ok": False, "size_mb": None, "opset": 17, "error": last_err}


# ---------------------------------------------------------------------------
# Quantization simulation
# ---------------------------------------------------------------------------


def _state_dict_size_mb(sd: dict) -> float:
    bio = io.BytesIO()
    torch.save(sd, bio)
    return len(bio.getvalue()) / 1e6


def _output_drift(
    a: torch.Tensor, b: torch.Tensor
) -> float | None:
    """L2 distance, normalised by the original output's norm.  0 = identical,
    1 = totally different.  Capped at 1.0."""
    try:
        if a.shape != b.shape or not a.is_floating_point():
            return None
        denom = a.norm().item() + 1e-9
        return float(min((a - b).norm().item() / denom, 1.0))
    except Exception:
        return None


def _try_dynamic_quantization(
    module: nn.Module, dummy: torch.Tensor, baseline_mb: float
) -> dict[str, Any] | None:
    try:
        with torch.no_grad():
            ref = module(dummy)
        qmodule = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(module), {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8
        )
        with torch.no_grad():
            qout = qmodule(dummy)
        size_mb = _state_dict_size_mb(qmodule.state_dict())
        ref_t = ref if isinstance(ref, torch.Tensor) else None
        qout_t = qout if isinstance(qout, torch.Tensor) else None
        drift = _output_drift(ref_t, qout_t) if ref_t is not None else None
        return {
            "size_mb": size_mb,
            "size_reduction_pct": (
                (baseline_mb - size_mb) / baseline_mb * 100.0
                if baseline_mb > 0 else 0.0
            ),
            "output_drift": drift,
            "supported_layers": ["Linear", "LSTM", "GRU"],
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pruning simulation
# ---------------------------------------------------------------------------


def _magnitude_prune(module: nn.Module, ratio: float) -> nn.Module:
    pruned = copy.deepcopy(module)
    with torch.no_grad():
        for m in pruned.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                w = m.weight.data
                if w.numel() == 0:
                    continue
                k = int(ratio * w.numel())
                if k <= 0:
                    continue
                threshold = torch.kthvalue(w.abs().flatten(), k).values
                mask = w.abs() > threshold
                m.weight.data = w * mask
    return pruned


def _try_pruning_sim(
    module: nn.Module, dummy: torch.Tensor, baseline_mb: float
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            ref = module(dummy)
    except Exception:
        return out
    ref_t = ref if isinstance(ref, torch.Tensor) else None

    for ratio in (0.3, 0.5, 0.7):
        try:
            pruned = _magnitude_prune(module, ratio)
            with torch.no_grad():
                pout = pruned(dummy)
            # Storage doesn't shrink without sparse formats, so we report
            # *theoretical* size (what you'd get with sparse storage or a
            # quantized + pruned export).
            theoretical_mb = baseline_mb * (1 - ratio)
            drift = (
                _output_drift(ref_t, pout if isinstance(pout, torch.Tensor) else None)
                if ref_t is not None else None
            )
            out.append({
                "ratio": ratio,
                "theoretical_size_mb": theoretical_mb,
                "output_drift": drift,
            })
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Architecture fingerprinting
# ---------------------------------------------------------------------------


# Crude signatures based on canonical state-dict key prefixes.  The point is
# to give the user a "looks like a ResNet" hint, not to be authoritative.
_ARCH_SIGNATURES = [
    ("ResNet", ["conv1.weight", "bn1.weight", "layer1.0.conv1.weight"]),
    ("VGG", ["features.0.weight", "classifier.0.weight"]),
    ("BERT", ["embeddings.word_embeddings.weight", "encoder.layer.0.attention.self.query.weight"]),
    ("GPT-2", ["wte.weight", "h.0.attn.c_attn.weight"]),
    ("ViT", ["patch_embed.proj.weight", "blocks.0.attn.qkv.weight"]),
    ("YOLO", ["model.0.conv.weight", "model.0.bn.weight"]),
    ("UNet", ["inc.double_conv.0.weight", "down1.maxpool_conv.1.double_conv.0.weight"]),
    ("MobileNet", ["features.0.0.weight", "features.1.conv.0.0.weight"]),
]


def _fingerprint(state_dict: dict) -> str | None:
    keys = set(state_dict.keys())
    for name, sig_keys in _ARCH_SIGNATURES:
        if all(k in keys for k in sig_keys):
            return name
    return None


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def run_dynamic_analysis(obj: Any, baseline_mb: float) -> DynamicReport:
    report = DynamicReport()

    # Architecture fingerprint works on either a module or a state-dict.
    if isinstance(obj, nn.Module):
        report.architecture_guess = _fingerprint(obj.state_dict())
    elif isinstance(obj, dict):
        report.architecture_guess = _fingerprint(obj)

    if not isinstance(obj, nn.Module):
        report.notes.append(
            "Live benchmarks need a full nn.Module. Save with "
            "``torch.save(model, ...)`` (not ``model.state_dict()``) to unlock."
        )
        return report

    module = obj
    module.eval()

    shape = _guess_input_shape(module)
    if shape is None:
        report.notes.append(
            "Couldn't auto-detect an input shape for this model -- skipped "
            "live benchmarks. Wire up a custom dummy input to enable."
        )
        return report

    leaf = _first_leaf(module)
    is_embedding = isinstance(leaf, nn.Embedding)
    try:
        dummy = _make_dummy(shape, like_embedding=is_embedding)
    except Exception:
        report.notes.append("Couldn't build a dummy input for this shape.")
        return report

    report.input_shape = shape

    lat = _bench_latency(module, dummy)
    if lat is not None:
        report.latency_ms_mean, report.latency_ms_p95 = lat

    report.peak_activation_mb = _peak_activation_mb(module, dummy)
    report.flops = _estimate_flops(module, dummy)

    # ONNX export, quant sim, and prune sim each do a deepcopy + forward pass.
    # On multi-GB models that adds up to minutes -- skip the heaviest pieces
    # past a threshold and tell the user we did so.
    if baseline_mb < 800:
        report.onnx_export = _try_onnx_export(module, dummy)
    else:
        report.notes.append(
            f"Model is {baseline_mb:.0f} MB -- skipped ONNX export to keep "
            "the analysis under a minute. Run the analyzer locally for full "
            "results."
        )

    if baseline_mb < 500:
        report.quantization_sim = _try_dynamic_quantization(module, dummy, baseline_mb)
        report.pruning_sim = _try_pruning_sim(module, dummy, baseline_mb)
    else:
        report.notes.append(
            "Skipped quant / prune simulation on this model (deepcopy cost). "
            "Use the CLI for full simulations on large models."
        )

    return report
