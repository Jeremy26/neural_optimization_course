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
        savings = bytes_in_mem * 0.75
        out.append(Problem(
            severity="critical",
            headline=f"It's hauling {_fmt_mb(savings)} of dead weight",
            detail=(
                "About three-quarters of what it carries today doesn't need "
                "to be there. That's bandwidth you're paying for, memory "
                "you're holding, and latency you're spending -- on every "
                "single inference."
            ),
        ))

    # ---- Pruning headroom -------------------------------------------------
    headroom = report.pruning.details.get("near_zero_headroom", 0.0)
    sparsity = report.pruning.details.get("global_sparsity", 0.0)
    param_count = report.parameter_count
    if sparsity < 0.05 and headroom > 0.1:
        out.append(Problem(
            severity="warning",
            headline="It multiplies by noise on every forward pass",
            detail=(
                f"Roughly {headroom:.0%} of what's inside is statistically "
                "indistinguishable from zero -- but the model still drags it "
                "through every inference, paying time and energy along the way."
            ),
        ))

    # ---- Worst-offender layer --------------------------------------------
    top_layers = report.pruning.details.get("top_layers", [])
    if top_layers:
        worst = top_layers[0]
        if worst["near_zero_fraction"] > 0.5:
            out.append(Problem(
                severity="warning",
                headline="One of the largest parts of the model is mostly noise",
                detail=(
                    f"The heaviest single component is {worst['near_zero_fraction']:.0%} "
                    "dead weight. A real audit would have caught this "
                    "before training finished."
                ),
            ))

    # ---- Size vs production norms ----------------------------------------
    if report.file_size_mb > 50:
        target_size = report.file_size_mb * 0.25
        out.append(Problem(
            severity="warning",
            headline=(
                f"It weighs {report.file_size_mb:.0f} MB. "
                f"A deploy-grade version of this would be ~{target_size:.0f} MB."
            ),
            detail=(
                "Comparable models ship at a quarter of this weight. The "
                "difference shows up in bandwidth, cold starts, install "
                "size -- everywhere your customer feels the model load."
            ),
        ))

    # ---- Export risk ------------------------------------------------------
    onnx_risky = report.exportability.details.get("onnx_risky_modules", 0)
    trt_risky = report.exportability.details.get("tensorrt_risky_modules", 0)
    onnx_attempt = report.exportability.details.get("onnx_attempt")
    if onnx_attempt and not onnx_attempt.get("ok"):
        out.append(Problem(
            severity="critical",
            headline="It won't survive the export step",
            detail=(
                "We tried, and it failed. Most teams discover this the day "
                "before the deadline -- you're discovering it now."
            ),
        ))
    elif onnx_risky + trt_risky > 0:
        out.append(Problem(
            severity="warning",
            headline=(
                f"{onnx_risky + trt_risky} part(s) will resist when you try to ship it"
            ),
            detail=(
                "A few components inside either won't translate cleanly to "
                "production runtimes or will fall back to slow paths -- "
                "without any warning until you're already debugging."
            ),
        ))

    # ---- Sanity: deployment-ready models also get *something* -----------
    if not out:
        out.append(Problem(
            severity="info",
            headline="Looks clean on the surface -- real hardware is the test",
            detail=(
                "What you can see here doesn't catch latency cliffs, memory "
                "blowups, or accuracy drift under deployment. The course "
                "covers what to measure when a model looks fine but stumbles."
            ),
        ))

    return out
