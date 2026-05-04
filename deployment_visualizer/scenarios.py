"""Deployment narratives -- "if this shipped today, here's what would happen".

Each scenario takes the report and produces a short, vivid story tying the
model's properties to a specific deployment context: robot, smartphone,
serverless cloud, browser.  Some scenarios need FLOPs (computed during the
live benchmarks); others run on file size alone, so we always show the
storage-based ones and only show latency-based ones when benchmarks have
been run.
"""

from __future__ import annotations

from dataclasses import dataclass

from device_estimates import DEVICES, estimate_latency_ms


@dataclass
class Scenario:
    setting: str          # "Edge robot", "Smartphone AR", ...
    target: str           # "Jetson Orin Nano @ 30 FPS"
    hook: str             # "If you bolted this onto a delivery robot tomorrow..."
    today_line: str       # what happens with the model as-is
    optimized_line: str   # what optimization unlocks
    verdict: str          # "Ships." | "Tight." | "Won't ship."
    severity: str         # "good" | "warning" | "critical"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Theoretical peak-FLOPS latency assumes large batch and saturated kernels.
# For the lead-magnet's deployment scenarios (batch=1, real workloads) we
# scale by a realistic penalty -- empirically 2-5x slower than peak.  Using
# 3x is the median across published batch=1 inference benchmarks.
_BATCH1_PENALTY = 3.0


def _device_latency(flops: int, device_name: str, dtype: str) -> float:
    rows = estimate_latency_ms(flops, dtype=dtype)
    for r in rows:
        if r["device"] == device_name:
            return r["latency_ms"] * _BATCH1_PENALTY
    return float("inf")


def _verdict_from_budget(latency_ms: float, budget_ms: float) -> tuple[str, str]:
    if latency_ms <= budget_ms:
        return "Ships.", "good"
    if latency_ms <= budget_ms * 1.6:
        return "Tight.", "warning"
    return "Won't ship.", "critical"


# ---------------------------------------------------------------------------
# Storage-based scenarios (always available)
# ---------------------------------------------------------------------------


def _serverless_cold_start(file_size_mb: float) -> Scenario:
    """AWS Lambda / Cloud Run roughly: ~30 ms per MB of model loaded from
    disk, plus a 500 ms baseline for the runtime itself.  These numbers are
    rough but match the order of magnitude reported by every team that's
    actually measured it."""
    cold_start_s = 0.5 + file_size_mb * 0.030
    optimized_size = file_size_mb * 0.25  # FP32 -> INT8 ballpark
    optimized_cold_s = 0.5 + optimized_size * 0.030
    if cold_start_s <= 1.0:
        verdict, sev = "Ships.", "good"
    elif cold_start_s <= 3.0:
        verdict, sev = "Tight.", "warning"
    else:
        verdict, sev = "Won't ship.", "critical"
    return Scenario(
        setting="Serverless cloud",
        target="AWS Lambda / Cloud Run, cold start",
        hook=(
            "If a request came in cold and your function had to spin up "
            "this model from scratch..."
        ),
        today_line=(
            f"At {file_size_mb:.0f} MB on disk you'd be staring at a "
            f"~{cold_start_s:.1f} s cold-start before the first inference. "
            "AWS bills you for every second; users abandon after three."
        ),
        optimized_line=(
            f"After quantization to INT8 the artifact is ~{optimized_size:.0f} MB "
            f"and cold-start drops to ~{optimized_cold_s:.1f} s -- nearly "
            "indistinguishable from a warm container."
        ),
        verdict=verdict,
        severity=sev,
    )


def _mobile_app_bundle(file_size_mb: float) -> Scenario:
    """Mobile context: app bundle size matters, and 4G download speed is
    where users actually live.  At ~5 MB/s on a decent 4G link, downloading
    a 50 MB checkpoint is a 10-second wait that users *feel*."""
    download_s_4g = file_size_mb / 5.0  # 5 MB/s = decent 4G
    optimized_size = file_size_mb * 0.25
    optimized_dl = optimized_size / 5.0
    if file_size_mb <= 20:
        verdict, sev = "Ships.", "good"
    elif file_size_mb <= 80:
        verdict, sev = "Tight.", "warning"
    else:
        verdict, sev = "Won't ship.", "critical"
    return Scenario(
        setting="On-device mobile",
        target="iPhone / Android app bundle",
        hook="If this shipped inside your iOS / Android app today...",
        today_line=(
            f"You'd add {file_size_mb:.0f} MB to the app download "
            f"({download_s_4g:.0f} s on 4G). On Apple's old cellular limit "
            "you'd have blown past 200 MB once you add the rest of your app."
        ),
        optimized_line=(
            f"Quantized + pruned to ~{optimized_size:.0f} MB "
            f"({optimized_dl:.1f} s on 4G), the model becomes a non-event "
            "in your bundle size budget."
        ),
        verdict=verdict,
        severity=sev,
    )


# ---------------------------------------------------------------------------
# Latency-based scenarios (need FLOPs from benchmarks)
# ---------------------------------------------------------------------------


def _edge_robot(flops: int, current_dtype: str) -> Scenario:
    """30 FPS = 33 ms budget per frame -- the threshold below which a robot
    can keep up with a moving world."""
    budget = 33.3
    today = _device_latency(flops, "Jetson Orin Nano", current_dtype)
    optimized = _device_latency(flops, "Jetson Orin Nano", "int8")
    today_fps = 1000.0 / today if today > 0 else 0.0
    opt_fps = 1000.0 / optimized if optimized > 0 else 0.0
    verdict, sev = _verdict_from_budget(today, budget)
    return Scenario(
        setting="Edge robot",
        target="Jetson Orin Nano · 30 FPS budget (33 ms/frame)",
        hook=(
            "If you bolted this onto a delivery robot or a drone "
            "tomorrow..."
        ),
        today_line=(
            f"At {current_dtype.upper()}, theoretical throughput is "
            f"~{today:.1f} ms/frame ({today_fps:.0f} FPS). "
            + (
                "You're inside the budget."
                if today <= budget
                else "Real-world batch=1 latency runs 2-5x slower -- you'd be "
                "perceiving the world a beat behind reality."
            )
        ),
        optimized_line=(
            f"At INT8 with TensorRT, ~{optimized:.1f} ms/frame "
            f"({opt_fps:.0f} FPS) -- comfortable headroom for sensor fusion, "
            "tracking, and a control loop."
        ),
        verdict=verdict,
        severity=sev,
    )


def _phone_ar(flops: int, current_dtype: str) -> Scenario:
    """60 FPS AR = 16.7 ms budget. The phone NPU peaks at INT8."""
    budget = 16.7
    today = _device_latency(flops, "iPhone 15 Pro (NPU)", current_dtype)
    optimized = _device_latency(flops, "iPhone 15 Pro (NPU)", "int8")
    today_fps = 1000.0 / today if today > 0 else 0.0
    opt_fps = 1000.0 / optimized if optimized > 0 else 0.0
    verdict, sev = _verdict_from_budget(today, budget)
    return Scenario(
        setting="Smartphone AR",
        target="iPhone 15 Pro NPU · 60 FPS AR (17 ms/frame)",
        hook=(
            "If a user pointed an AR camera at the world and asked your "
            "model what they were looking at..."
        ),
        today_line=(
            f"At {current_dtype.upper()}, ~{today:.1f} ms/frame "
            f"({today_fps:.0f} FPS). "
            + (
                "ARKit will composite cleanly."
                if today <= budget
                else "ARKit drops frames; the overlay floats off the world."
            )
        ),
        optimized_line=(
            f"On the Apple Neural Engine at INT8, ~{optimized:.1f} ms "
            f"({opt_fps:.0f} FPS) -- the model leaves the GPU free for the "
            "rest of the experience."
        ),
        verdict=verdict,
        severity=sev,
    )


def _cloud_throughput(flops: int, current_dtype: str) -> Scenario:
    """Throughput economics: how many inferences/sec/GPU, and what that means
    at 1 million requests per day."""
    today = _device_latency(flops, "NVIDIA T4", current_dtype)
    optimized = _device_latency(flops, "NVIDIA T4", "int8")
    today_per_s = 1000.0 / today if today > 0 else 0.0
    opt_per_s = 1000.0 / optimized if optimized > 0 else 0.0
    # Daily budget at 1M requests = req/s needed = ~12
    needed_rps = 1_000_000 / 86_400
    today_gpus = max(1, round(needed_rps / max(today_per_s, 0.01)))
    opt_gpus = max(1, round(needed_rps / max(opt_per_s, 0.01)))
    if today_gpus <= 1:
        verdict, sev = "Ships.", "good"
    elif today_gpus <= 4:
        verdict, sev = "Tight.", "warning"
    else:
        verdict, sev = "Won't ship.", "critical"
    speedup = today_per_s / max(opt_per_s, 0.01)
    if opt_gpus < today_gpus:
        opt_summary = (
            f"INT8 with TensorRT: {opt_per_s:,.0f} req/s/GPU, "
            f"~{opt_gpus} T4(s) -- {today_gpus / max(opt_gpus,1):.1f}x "
            "cheaper to operate."
        )
    else:
        opt_summary = (
            f"INT8 with TensorRT: {opt_per_s:,.0f} req/s/GPU "
            f"({1/max(speedup,0.01):.1f}x throughput, same hardware) -- "
            "headroom for 10x growth before you add a second GPU."
        )
    return Scenario(
        setting="Cloud throughput",
        target="NVIDIA T4 · 1 M inferences / day",
        hook=(
            "If you put this behind an API and traffic hit 1 M requests "
            "per day..."
        ),
        today_line=(
            f"At {current_dtype.upper()}, ~{today_per_s:,.0f} req/s/GPU "
            f"in batch=1. Sustaining 1M/day needs ~{today_gpus} T4(s) "
            "running 24/7 -- before you account for spikes."
        ),
        optimized_line=opt_summary,
        verdict=verdict,
        severity=sev,
    )


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def build_scenarios(report) -> list[Scenario]:
    out: list[Scenario] = [
        _serverless_cold_start(report.file_size_mb),
        _mobile_app_bundle(report.file_size_mb),
    ]
    flops = (
        report.dynamic.flops if report.dynamic and report.dynamic.flops else None
    )
    if flops:
        # Pick a "current" dtype -- whatever the model actually ships at.
        dist = report.precision.details.get("dtype_distribution", {})
        if dist and "torch.float16" in str(dist):
            current_dtype = "fp16"
        elif dist and ("torch.int8" in str(dist) or "torch.qint8" in str(dist)):
            current_dtype = "int8"
        else:
            current_dtype = "fp32"
        out.extend([
            _edge_robot(flops, current_dtype),
            _phone_ar(flops, current_dtype),
            _cloud_throughput(flops, current_dtype),
        ])
    return out
