"""One-click sample model downloads.

For users who don't have a checkpoint handy, we offer a few public models
with directly-downloadable weights.  All of these are well-known
deployment targets in the robotics / AV / computer-vision space:

* ResNet-18 -- the canonical vision backbone (PyTorch official URL)
* MobileNet V2 -- a deploy-grade mobile vision model
* DeepLabV3-MobileNet-V3 -- semantic segmentation (used in self-driving
  perception stacks)
* YOLOv5n -- real-time object detection backbone (Ultralytics release)

Downloads are cached to ``~/.cache/deployment_visualizer/`` so the second
click is instant.
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
    domain: str             # "Self-driving perception", "Mobile vision", ...
    size_mb: float          # advertised download size
    url: str
    filename: str           # local filename to cache as


DEMOS: list[DemoModel] = [
    DemoModel(
        key="resnet18",
        name="ResNet-18",
        one_liner="The canonical 11M-parameter vision backbone -- the model every quantization paper benchmarks against.",
        domain="Image classification (ImageNet-1K)",
        size_mb=45.0,
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        filename="resnet18.pth",
    ),
    DemoModel(
        key="mobilenet_v2",
        name="MobileNet V2",
        one_liner="A 3.5M-parameter mobile-first backbone -- designed from the ground up for edge deployment.",
        domain="Mobile-grade image classification",
        size_mb=14.0,
        url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        filename="mobilenet_v2.pth",
    ),
    DemoModel(
        key="deeplab_mobilenet",
        name="DeepLabV3 (MobileNet V3)",
        one_liner="Semantic segmentation on a mobile backbone -- the kind of model self-driving teams ship for drivable-surface masks.",
        domain="Self-driving perception",
        size_mb=42.0,
        url="https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
        filename="deeplab_mobilenet.pth",
    ),
    DemoModel(
        key="yolov5n",
        name="YOLOv5-Nano",
        one_liner="A 1.9M-parameter real-time detector -- the classic AV / robotics object-detection demo.",
        domain="Real-time object detection",
        size_mb=4.0,
        url="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
        filename="yolov5n.pt",
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
    """Fetch the model weights -- from cache when available, otherwise via
    HTTP with optional progress reporting.

    ``on_progress`` is a callable receiving ``(bytes_so_far, total_bytes)``.
    Total may be 0 if the server doesn't announce content-length.
    """
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

    # Persist to cache.
    try:
        path.write_bytes(data)
    except OSError:
        # Cache write failed (read-only home, etc.) -- still usable in memory.
        pass
    return data
