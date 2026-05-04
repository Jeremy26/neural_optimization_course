"""One-click sample model downloads.

We deliberately favor real self-driving / robotics research checkpoints
over generic ImageNet backbones -- the audience for this tool ships
perception stacks, not classification benchmarks.

* YOLOP -- multi-task driving perception (traffic + lanes + drivable
  surface) from hustvl/YOLOP, the canonical AV multi-task baseline.
* HybridNets -- another end-to-end driving perception model.
* LRASPP MobileNet V3 -- real-time semantic segmentation, the kind of
  model self-driving teams ship for drivable-surface masks.
* Faster R-CNN MobileNet V3 320 -- mobile-grade object detection
  covering the COCO classes (cars, persons, traffic signs) at deploy
  speed.

Some of these third-party AV checkpoints serialise the full ``nn.Module``
class instance instead of a clean ``state_dict``.  When that happens the
deserializer falls back to pickle and we surface a clear warning -- the
analyzer's static checks still produce a useful report on the tensors
inside, but the architecture-specific sub-paths (instantiating into a
torchvision model for ONNX export, etc.) won't fire.

Downloads are cached to ``~/.cache/deployment_visualizer/``.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DemoModel:
    key: str
    name: str
    one_liner: str
    domain: str
    size_mb: float
    url: str
    filename: str


DEMOS: list[DemoModel] = [
    DemoModel(
        key="yolop",
        name="YOLOP",
        one_liner=(
            "Multi-task driving perception: traffic-object detection, "
            "drivable-area segmentation, and lane-line segmentation in "
            "one forward pass. Trained on BDD100K."
        ),
        domain="Self-driving · multi-task",
        size_mb=7.9,
        url="https://github.com/hustvl/YOLOP/raw/main/weights/End-to-end.pth",
        filename="yolop_end_to_end.pth",
    ),
    DemoModel(
        key="hybridnets",
        name="HybridNets",
        one_liner=(
            "End-to-end perception network that does what YOLOP does -- "
            "object detection plus drivable-area and lane segmentation -- "
            "with a different backbone choice."
        ),
        domain="Self-driving · multi-task",
        size_mb=30.0,
        url="https://github.com/datvuthanh/HybridNets/releases/download/v1.1/hybridnets.pth",
        filename="hybridnets.pth",
    ),
    DemoModel(
        key="lraspp",
        name="LR-ASPP MobileNet V3",
        one_liner=(
            "Lightweight real-time semantic segmentation -- the architecture "
            "self-driving teams reach for when they need road / lane / "
            "obstacle masks at 30+ FPS on edge hardware."
        ),
        domain="Edge segmentation",
        size_mb=12.0,
        url="https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth",
        filename="lraspp_mobilenet_v3.pth",
    ),
    DemoModel(
        key="frcnn_mobile",
        name="Faster R-CNN (MobileNet V3 320)",
        one_liner=(
            "Mobile-grade object detection on a 320-px input. Detects the "
            "80 COCO categories -- cars, persons, traffic lights, stop signs, "
            "bicycles -- the AV / robotics shortlist."
        ),
        domain="Mobile detection",
        size_mb=74.0,
        url="https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
        filename="fasterrcnn_mobilenet_v3_320.pth",
    ),
]


def cache_dir() -> Path:
    base = os.environ.get("DEPLOYMENT_VIZ_CACHE")
    if base:
        path = Path(base)
    else:
        path = Path.home() / ".cache" / "deployment_visualizer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_path(model: DemoModel) -> Path:
    return cache_dir() / model.filename


def is_cached(model: DemoModel) -> bool:
    return cached_path(model).exists() and cached_path(model).stat().st_size > 1000


def download(model: DemoModel, on_progress=None) -> bytes:
    path = cached_path(model)
    if is_cached(model):
        if on_progress:
            size = path.stat().st_size
            on_progress(size, size)
        return path.read_bytes()

    req = urllib.request.Request(
        model.url,
        headers={"User-Agent": "deployment-visualizer/1.0 (+lead-magnet)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            chunks: list[bytes] = []
            so_far = 0
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                so_far += len(chunk)
                if on_progress:
                    on_progress(so_far, total)
        data = b"".join(chunks)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Couldn't download {model.name}: {exc}. "
            "Check your network or upload a local file instead."
        ) from exc

    try:
        path.write_bytes(data)
    except OSError:
        pass
    return data
