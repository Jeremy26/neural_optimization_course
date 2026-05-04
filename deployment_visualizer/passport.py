"""Model passport: structural inferences about an uploaded model.

We squeeze out everything we can from the checkpoint without running it:

* Architecture family (via fingerprint -- already populated on the report)
* Input shape (sniffed from the first leaf module)
* Output shape (sniffed from the last parameterised module)
* Likely task (classification / segmentation / detection / ...)
* Best-guess training dataset, when the output dimensionality matches a
  well-known one (1000 -> ImageNet, 80 -> COCO, etc.)

The point of this module is to give Tab 1 enough story to *feel* like the
spec sheet of a real machine -- not just a list of layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


# Output-dimension hints.  These are deliberately conservative -- false
# positives here would make the passport feel sloppy.
_DATASET_HINTS = {
    1000: ("ImageNet-1K", "Image classification (1000 categories)"),
    1001: ("ImageNet-1K (with background class)", "Image classification"),
    100:  ("CIFAR-100", "Image classification"),
    10:   ("CIFAR-10 / MNIST / Fashion-MNIST", "Small-scale image classification"),
    21:   ("PASCAL VOC", "Semantic segmentation"),
    19:   ("Cityscapes", "Driving-scene segmentation"),
    80:   ("COCO", "Object detection / instance segmentation"),
    91:   ("COCO (full)", "Object detection / instance segmentation"),
    133:  ("COCO Panoptic", "Panoptic segmentation"),
}


@dataclass
class Passport:
    architecture: str | None
    input_shape: tuple[int, ...] | None
    input_description: str
    output_shape: tuple[int, ...] | None
    output_description: str
    likely_task: str
    likely_dataset: str | None
    sample_classes: list[str]   # short illustrative labels (3-5)


# ---------------------------------------------------------------------------
# Sniffers
# ---------------------------------------------------------------------------


def _first_leaf(obj):
    if isinstance(obj, nn.Module):
        for m in obj.modules():
            if not list(m.children()) and any(True for _ in m.parameters(recurse=False)):
                return m
    return None


def _last_param_module(obj):
    """The last leaf module that has parameters -- usually the
    classification head."""
    if not isinstance(obj, nn.Module):
        return None
    last = None
    for m in obj.modules():
        if not list(m.children()) and any(True for _ in m.parameters(recurse=False)):
            last = m
    return last


def _input_shape_from_module(module):
    """``(C, H, W)`` triple for vision models; sequence-style for others."""
    if isinstance(module, nn.Conv2d):
        return (module.in_channels, 224, 224)
    if isinstance(module, nn.Conv1d):
        return (module.in_channels, 1024)
    if isinstance(module, nn.Conv3d):
        return (module.in_channels, 16, 64, 64)
    if isinstance(module, nn.Linear):
        return (module.in_features,)
    if isinstance(module, nn.Embedding):
        return (64,)  # token sequence
    return None


def _input_shape_from_state_dict(sd):
    """Look for the obvious first conv weight to get input channels."""
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".weight") and v.dim() == 4:
            return (v.shape[1], 224, 224)  # in_channels, H, W (assumed 224)
        if k.endswith(".weight") and v.dim() == 2:
            return (v.shape[1],)
    return None


def _output_shape_from_module(module):
    if isinstance(module, nn.Linear):
        return (module.out_features,)
    if isinstance(module, nn.Conv2d):
        return (module.out_channels, "H/n", "W/n")
    return None


def _output_shape_from_state_dict(sd):
    """Look at the last 2D weight tensor -- usually the classifier."""
    last_2d = None
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.dim() == 2 and k.endswith(".weight"):
            last_2d = (k, v)
    if last_2d is not None:
        return (last_2d[1].shape[0],)
    last_4d = None
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.dim() == 4 and k.endswith(".weight"):
            last_4d = (k, v)
    if last_4d is not None:
        return (last_4d[1].shape[0], "H/n", "W/n")
    return None


def _format_input(shape, arch):
    if shape is None:
        return "Unknown"
    if len(shape) == 3:
        c, h, w = shape
        ch_label = {1: "grayscale", 3: "RGB", 4: "RGBA"}.get(c, f"{c}-channel")
        return f"{ch_label} image @ {h}×{w}"
    if len(shape) == 2:
        c, l = shape
        return f"{c}-channel signal · length {l}"
    if len(shape) == 1:
        return f"{shape[0]}-d feature vector"
    return f"shape {shape}"


def _format_output(shape, arch):
    if shape is None:
        return "Unknown"
    if len(shape) == 1:
        n = shape[0]
        return f"{n:,}-d output (probability vector)"
    if len(shape) >= 3:
        c = shape[0]
        return f"{c}-channel feature map (spatial output)"
    return f"shape {shape}"


def _infer_task(arch: str | None, output_shape, input_shape) -> str:
    arch_l = (arch or "").lower()
    if any(x in arch_l for x in ("yolo",)):
        return "Object detection (real-time)"
    if "unet" in arch_l:
        return "Semantic segmentation"
    if "bert" in arch_l or "gpt" in arch_l:
        return "Natural language" + (" generation" if "gpt" in arch_l else " understanding")
    if "vit" in arch_l:
        return "Image classification (transformer-based)"
    if output_shape is None:
        return "Unknown"
    if len(output_shape) == 1:
        return "Image classification" if input_shape and len(input_shape) == 3 else "Classification"
    if len(output_shape) >= 3:
        return "Semantic segmentation"
    return "Inference"


def _dataset_hint(output_shape) -> tuple[str | None, list[str]]:
    if not output_shape or len(output_shape) != 1:
        return None, []
    n = output_shape[0]
    if n in _DATASET_HINTS:
        ds, _task = _DATASET_HINTS[n]
        # A few illustrative class labels per dataset.  Just enough that the
        # passport feels concrete -- we don't try to be exhaustive.
        labels = {
            "ImageNet-1K": ["golden retriever", "espresso", "convertible", "pirate ship"],
            "ImageNet-1K (with background class)": ["golden retriever", "espresso", "convertible"],
            "CIFAR-100": ["bicycle", "mushroom", "tiger", "telephone"],
            "CIFAR-10 / MNIST / Fashion-MNIST": ["airplane", "cat", "deer", "ship"],
            "PASCAL VOC": ["person", "car", "bicycle", "dog"],
            "Cityscapes": ["road", "sidewalk", "vehicle", "pedestrian"],
            "COCO": ["person", "car", "bicycle", "stop sign"],
            "COCO (full)": ["person", "car", "bicycle", "traffic light"],
            "COCO Panoptic": ["road", "sky", "person", "car"],
        }.get(ds, [])
        return ds, labels
    return None, []


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def build_passport(obj: Any, architecture: str | None) -> Passport:
    if isinstance(obj, nn.Module):
        first = _first_leaf(obj)
        last = _last_param_module(obj)
        in_shape = _input_shape_from_module(first) if first else None
        out_shape = _output_shape_from_module(last) if last else None
    elif isinstance(obj, dict):
        in_shape = _input_shape_from_state_dict(obj)
        out_shape = _output_shape_from_state_dict(obj)
    else:
        in_shape = out_shape = None

    dataset, sample_classes = _dataset_hint(out_shape)
    task = _infer_task(architecture, out_shape, in_shape)
    if dataset:
        # If the output matches a known dataset, refine the task wording.
        _, dataset_task = _DATASET_HINTS.get(out_shape[0], (None, task))
        task = dataset_task

    return Passport(
        architecture=architecture,
        input_shape=in_shape,
        input_description=_format_input(in_shape, architecture),
        output_shape=out_shape,
        output_description=_format_output(out_shape, architecture),
        likely_task=task,
        likely_dataset=dataset,
        sample_classes=sample_classes,
    )
