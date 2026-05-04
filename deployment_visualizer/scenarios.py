"""Deployment narratives focused on robotics, autonomous vehicles, and edge.

Each scenario is short and vivid: a target hardware platform, a budget, a
"this is what would happen" line, and a Ships / Tight / Won't ship verdict.
We don't tell the user how to fix it -- that's the course.

All numbers come from static estimates (param count + architecture hint
-> FLOPs, plus a 3x batch=1 penalty against published peak-TFLOPS specs),
so scenarios always render -- no forward pass required.
"""

from __future__ import annotations

from dataclasses import dataclass

from device_estimates import estimate_latency_ms


@dataclass
class Scenario:
    setting: str          # short label: "Self-driving car"
    target: str           # hardware + budget: "DRIVE Orin · 30 FPS"
    hook: str             # italic "If you ran this on..."
    today_line: str       # consequence as-is
    optimized_line: str   # what optimization unlocks
    verdict: str          # "Ships." | "Tight." | "Won't ship."
    severity: str         # "good" | "warning" | "critical"


# Realistic batch=1 latency runs ~3x slower than peak-TFLOPS theoretical;
# we apply that penalty so the verdicts reflect what teams actually live with.
_BATCH1_PENALTY = 3.0


def _device_latency(flops: int, device_name: str, dtype: str) -> float:
    rows = estimate_latency_ms(flops, dtype=dtype)
    for r in rows:
        if r["device"] == device_name:
            return r["latency_ms"] * _BATCH1_PENALTY
    return float("inf")


def _fmt_ms(ms: float) -> str:
    """Avoid the '0 ms' artifact when latency is <0.5 ms."""
    if ms < 1.0:
        return "<1"
    return f"{ms:.0f}"


def _verdict(latency_ms: float, budget_ms: float) -> tuple[str, str]:
    """Strict thresholds: 'Ships.' requires real margin, not just hitting
    the exact budget.  Hitting the budget exactly is 'Tight.' because by
    the time you ship the rest of the stack you'll blow past it."""
    if latency_ms <= budget_ms * 0.5:
        return "Ships.", "good"
    if latency_ms <= budget_ms:
        return "Tight.", "warning"
    return "Won't ship.", "critical"


def _current_dtype(report) -> str:
    dist = report.precision.details.get("dtype_distribution", {})
    keys = " ".join(dist.keys())
    if "float16" in keys or "bfloat16" in keys:
        return "fp16"
    if "int8" in keys or "qint8" in keys:
        return "int8"
    return "fp32"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _self_driving(flops: int, dtype: str) -> Scenario:
    # The AGX-class SoC sits in DRIVE Orin and the high-end robotics platforms
    # alike -- same silicon, different PCBs.
    today = _device_latency(flops, "Jetson Orin AGX", dtype)
    optimized = _device_latency(flops, "Jetson Orin AGX", "int8")
    verdict, sev = _verdict(today, 33.3)
    return Scenario(
        setting="Self-driving stack",
        target="NVIDIA DRIVE Orin (AGX-class) · 30 FPS perception",
        hook="If a self-driving car perceived the world through this model...",
        today_line=(
            f"At {dtype.upper()}, ~{_fmt_ms(today)} ms per frame. "
            + (
                "Inside the loop. The world doesn't move faster than your model."
                if today <= 33.3
                else "By the time you decide to brake, the car has already "
                "moved a meter."
            )
        ),
        optimized_line=(
            f"At INT8 with TensorRT, ~{_fmt_ms(optimized)} ms -- the "
            "perception stack stops being your bottleneck."
        ),
        verdict=verdict,
        severity=sev,
    )


def _delivery_robot(flops: int, dtype: str) -> Scenario:
    today = _device_latency(flops, "Jetson Orin Nano", dtype)
    optimized = _device_latency(flops, "Jetson Orin Nano", "int8")
    verdict, sev = _verdict(today, 100.0)
    fps_now = 1000.0 / today if today > 0 else 0
    return Scenario(
        setting="Delivery robot",
        target="Jetson Orin Nano · 10 FPS closed-loop control",
        hook="If a delivery robot was navigating with this model...",
        today_line=(
            f"At {dtype.upper()}, ~{_fmt_ms(today)} ms per inference "
            f"({fps_now:.0f} FPS). "
            + (
                "Smooth closed-loop control."
                if today <= 100
                else "The robot reacts to where the world *was*, "
                "not where it is."
            )
        ),
        optimized_line=(
            f"At INT8, ~{_fmt_ms(optimized)} ms -- now you can layer in "
            "tracking, language, and a mapper without missing a beat."
        ),
        verdict=verdict,
        severity=sev,
    )


def _drone(flops: int, dtype: str) -> Scenario:
    today = _device_latency(flops, "Jetson Orin Nano", dtype)
    optimized = _device_latency(flops, "Jetson Orin Nano", "int8")
    verdict, sev = _verdict(today, 33.3)
    return Scenario(
        setting="Autonomous drone",
        target="Jetson Orin Nano · 30 FPS obstacle avoidance",
        hook="If a drone was dodging trees in real time with this model...",
        today_line=(
            f"At {dtype.upper()}, ~{_fmt_ms(today)} ms per frame. "
            + (
                "Inside the avoidance window."
                if today <= 33.3
                else "It hits the branch before it has time to dodge it."
            )
        ),
        optimized_line=(
            f"At INT8 with TensorRT, ~{_fmt_ms(optimized)} ms -- thermal "
            "and battery headroom for everything else the flight stack needs."
        ),
        verdict=verdict,
        severity=sev,
    )


def _smart_camera(flops: int) -> Scenario:
    optimized = _device_latency(flops, "Coral Edge TPU", "int8")
    verdict, sev = _verdict(optimized, 100.0)
    fps = 1000.0 / optimized if optimized > 0 else 0
    return Scenario(
        setting="Smart camera",
        target="Coral Edge TPU · INT8 only",
        hook="If a quality-control camera on a factory line ran this...",
        today_line=(
            "The Edge TPU only runs INT8. An FP32 model can't even land "
            "on the device -- you'd be stuck on the ARM host CPU at <1 FPS."
        ),
        optimized_line=(
            f"Compiled to INT8 for Edge TPU: ~{_fmt_ms(optimized)} ms "
            f"({fps:.0f} FPS). Now the camera keeps up with the line."
        ),
        verdict=verdict,
        severity=sev,
    )


def _embedded_arm(flops: int, dtype: str) -> Scenario:
    # ARM Cortex-A series at ~4x slower than a Xeon core is the rough heuristic.
    today = _device_latency(flops, "CPU (Xeon, 1 core)", dtype) * 4
    optimized = _device_latency(flops, "CPU (Xeon, 1 core)", "int8") * 4
    verdict, sev = _verdict(today, 1000.0)
    return Scenario(
        setting="Embedded ARM",
        target="Raspberry Pi / Cortex-A · CPU only",
        hook="If you tried to ship this on a fanless ARM box...",
        today_line=(
            f"At {dtype.upper()}, ~{_fmt_ms(today)} ms per inference. "
            + (
                "Fits the low-rate sensing this kind of box is built for."
                if today <= 1000
                else "Way past real-time -- the box won't keep up with "
                "its own sensors."
            )
        ),
        optimized_line=(
            f"At INT8, ~{_fmt_ms(optimized)} ms -- enough headroom that "
            "the model isn't what defines your loop rate anymore."
        ),
        verdict=verdict,
        severity=sev,
    )


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def build_scenarios(report) -> list[Scenario]:
    flops = report.estimated_flops
    if not flops:
        return []
    dtype = _current_dtype(report)
    return [
        _self_driving(flops, dtype),
        _delivery_robot(flops, dtype),
        _drone(flops, dtype),
        _smart_camera(flops),
        _embedded_arm(flops, dtype),
    ]
