"""Turn the analysis report into a list of damning, quantified callouts.

The point of this module is *exposure*, not advice.  Each ``Problem`` names a
specific weakness with a concrete number attached -- "47 MB of memory wasted
on FP32 precision", not "consider quantization".  The UI surfaces these
front-and-center to make the gap between the user's model and a properly
optimized one impossible to ignore.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Problem:
    severity: str  # "critical" | "warning" | "info"
    headline: str
    detail: str


def _fmt_mb(bytes_count: float) -> str:
    if bytes_count >= 1024 * 1024:
        return f"{bytes_count / (1024 * 1024):.1f} MB"
    return f"{bytes_count / 1024:.0f} KB"


def detect_problems(report) -> list[Problem]:
    out: list[Problem] = []

    # ---- Precision waste --------------------------------------------------
    # If the model is mostly FP32, quantify how much memory the user is
    # leaving on the table.  The numbers come straight from the dtype
    # distribution we already collected.
    dist = report.precision.details.get("dtype_distribution", {})
    bytes_in_mem = report.precision.details.get("bytes_in_memory", 0)
    fp32_elements = dist.get("torch.float32", 0) + dist.get("torch.float64", 0)
    total_elements = sum(dist.values()) if dist else 0
    if total_elements and fp32_elements / total_elements > 0.9:
        # Going FP32 -> FP16 halves storage; FP32 -> INT8 quarters it.
        fp16_savings = bytes_in_mem * 0.5
        int8_savings = bytes_in_mem * 0.75
        out.append(Problem(
            severity="critical",
            headline=f"{_fmt_mb(fp16_savings)} of memory you're shipping for nothing",
            detail=(
                f"{fp32_elements / total_elements:.0%} of your weights are FP32. "
                f"Half-precision would already free {_fmt_mb(fp16_savings)}; "
                f"INT8 would free {_fmt_mb(int8_savings)}. Production teams "
                "stopped shipping FP32 weights years ago."
            ),
        ))

    # ---- Pruning headroom -------------------------------------------------
    headroom = report.pruning.details.get("near_zero_headroom", 0.0)
    sparsity = report.pruning.details.get("global_sparsity", 0.0)
    param_count = report.parameter_count
    if sparsity < 0.05 and headroom > 0.1:
        dead_weights = int(param_count * headroom)
        out.append(Problem(
            severity="warning",
            headline=f"{dead_weights:,} weights aren't doing useful work",
            detail=(
                f"{headroom:.0%} of your weights are below 1% of their layer's "
                "max magnitude -- statistically indistinguishable from zero. "
                "You're paying memory and FLOPs to multiply by noise."
            ),
        ))

    # ---- Worst-offender layer --------------------------------------------
    top_layers = report.pruning.details.get("top_layers", [])
    if top_layers:
        worst = top_layers[0]
        if worst["near_zero_fraction"] > 0.5:
            short = worst["name"]
            if len(short) > 50:
                short = "..." + short[-47:]
            out.append(Problem(
                severity="warning",
                headline=(
                    f"Layer ``{short}``: "
                    f"{worst['near_zero_fraction']:.0%} of weights are noise"
                ),
                detail=(
                    f"This layer has shape {tuple(worst['shape'])} -- one of "
                    "the largest in your model -- and most of it is dead "
                    "weight. A real audit would have caught this before "
                    "training finished."
                ),
            ))

    # ---- Size vs production norms ----------------------------------------
    if report.file_size_mb > 50:
        # Roughly: FP16 + 50% prune would put a typical model at ~1/4 of FP32.
        target_size = report.file_size_mb * 0.25
        out.append(Problem(
            severity="warning",
            headline=(
                f"{report.file_size_mb:.0f} MB on disk -- "
                f"~{target_size:.0f} MB is the realistic target"
            ),
            detail=(
                "Comparable architectures ship at a quarter of this size with "
                "standard quant + prune pipelines. Bandwidth, cold-start, "
                "and on-device install size all suffer at this weight."
            ),
        ))

    # ---- Export risk ------------------------------------------------------
    onnx_risky = report.exportability.details.get("onnx_risky_modules", 0)
    trt_risky = report.exportability.details.get("tensorrt_risky_modules", 0)
    onnx_attempt = report.exportability.details.get("onnx_attempt")
    if onnx_attempt and not onnx_attempt.get("ok"):
        out.append(Problem(
            severity="critical",
            headline="ONNX export failed on your model",
            detail=(
                f"Real export attempt produced: ``{onnx_attempt.get('error')}``. "
                "Most engineers discover this the day before the deadline."
            ),
        ))
    elif onnx_risky + trt_risky > 0:
        out.append(Problem(
            severity="warning",
            headline=(
                f"{onnx_risky + trt_risky} layer(s) will fight you at export time"
            ),
            detail=(
                "Custom attention, RNN cells, and a few other op patterns "
                "either fail outright on ONNX / TensorRT or fall back to "
                "slow paths -- with no warning until you're already debugging."
            ),
        ))

    # ---- Sanity: deployment-ready models also get *something* -----------
    if not out:
        out.append(Problem(
            severity="info",
            headline="Looks clean on the surface -- but real hardware is the test",
            detail=(
                "Static analysis can't catch latency cliffs, memory blowups, "
                "or accuracy drift under quantization. The course covers what "
                "to measure when a model looks fine but performs poorly."
            ),
        ))

    return out
