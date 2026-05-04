"""Deployment compatibility families.

Replicates the spirit of the existing ``exportability`` check across the
runtime ecosystem AV / robotics teams actually ship into: ONNX, TensorRT,
CUDA FP16, INT8 quantization, NVIDIA DLA (the deep-learning accelerator on
Jetson AGX / Orin), and Apple CoreML.

For each family we compute a 0-100 readiness score from:

* the architecture fingerprint when available (``ResNet`` / ``ViT`` / ...)
* the module-class mix when we have a full ``nn.Module``

The verdicts are deliberately vibey, not prescriptive.  We're naming the gap;
the course teaches how to close it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Family:
    name: str
    score: float
    verdict: str        # short, vibey
    blockers: int       # count of layers that won't behave on this runtime


# Module classes that misbehave on each runtime.  These are conservative
# (i.e. lean toward flagging) -- the goal is to expose risk, not to litigate
# every edge case.
_ONNX_RISKY = {
    "MultiheadAttention", "LSTMCell", "GRUCell",
    "TransformerEncoderLayer", "TransformerDecoderLayer",
}
_TRT_RISKY = _ONNX_RISKY | {"LayerNorm", "GroupNorm", "MultiheadAttention"}
_INT8_RISKY = {
    "LayerNorm", "GroupNorm", "BatchNorm3d",
    "GELU", "SiLU", "Mish", "Softmax", "Sigmoid",
    "MultiheadAttention",
}
_DLA_RISKY = {
    "LayerNorm", "GroupNorm", "GELU", "Mish", "SiLU",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "MultiheadAttention", "Embedding",
}
_COREML_RISKY = {"LSTMCell", "GRUCell", "MultiheadAttention"}


# Hand-curated readiness profiles for known architecture families.  When the
# fingerprint matches one of these we lean on the curated profile instead of
# the module-class scan -- because state-dict-only checkpoints don't expose
# their module list.  Each entry is (score, verdict).
_PROFILES = {
    "ResNet": {
        "onnx":     (98, "Clean. The benchmark every other path is measured against."),
        "tensorrt": (95, "Native path. Most teams see 1.5-2x at FP16 with zero work."),
        "cuda":     (98, "Every op has a fast CUDA kernel."),
        "int8":     (90, "Conv + Linear quantize cleanly. The classifier head is the usual asterisk."),
        "dla":      (95, "Maps fully onto Jetson DLA -- one of the few archs that does."),
        "coreml":   (92, "Standard vision-backbone conversion."),
    },
    "VGG": {
        "onnx":     (98, "Among the simplest ONNX exports possible."),
        "tensorrt": (98, "TRT loves VGG. Almost everything fuses."),
        "cuda":     (98, "Every op has a fast CUDA kernel."),
        "int8":     (92, "Quantizes cleanly. The reason VGG keeps showing up in benchmarks."),
        "dla":      (95, "Full DLA support. Old, but it ships."),
        "coreml":   (95, "Trivial to convert."),
    },
    "MobileNet": {
        "onnx":     (96, "Standard mobile-vision export."),
        "tensorrt": (88, "Depthwise conv occasionally hits slow paths -- but ships."),
        "cuda":     (95, "Depthwise + pointwise have native CUDA kernels."),
        "int8":     (95, "MobileNet was *designed* for INT8. This is its happy place."),
        "dla":      (90, "DLA-friendly ops with one or two fallbacks for hard-swish."),
        "coreml":   (97, "Apple ships these on the Neural Engine."),
    },
    "EfficientNet": {
        "onnx":     (92, "SiLU / swish ops sometimes confuse older opsets."),
        "tensorrt": (78, "Squeeze-Excite patterns and SiLU need TRT 8.5+ to stay native."),
        "cuda":     (90, "Mostly fp16-friendly; SiLU has good kernels in modern PyTorch."),
        "int8":     (70, "SiLU and SE blocks make PTQ rough -- QAT is the usual answer."),
        "dla":      (55, "DLA struggles with SiLU and adaptive pooling -- expect fallbacks."),
        "coreml":   (85, "Converts -- but watch the squeeze-excite ops."),
    },
    "DenseNet": {
        "onnx":     (95, "Dense connectivity exports fine."),
        "tensorrt": (88, "Concat-heavy graphs can fragment -- still solid."),
        "cuda":     (95, "Standard ops, all fast."),
        "int8":     (88, "Quantizes well; intermediate concats can need calibration tricks."),
        "dla":      (75, "Mostly DLA-friendly; a few ops fall back."),
        "coreml":   (88, "Vision-style. Converts cleanly."),
    },
    "BERT": {
        "onnx":     (90, "Standard transformer export -- but verbose."),
        "tensorrt": (65, "LayerNorm + Softmax fall back. TensorRT-LLM is where you'd go."),
        "cuda":     (92, "Modern fp16 attention kernels are great."),
        "int8":     (60, "Encoder-only is doable. The course covers QAT vs SmoothQuant trade-offs."),
        "dla":      (20, "DLA doesn't run transformers. You'd ship to GPU."),
        "coreml":   (78, "Apple Neural Engine handles BERT-like models -- with care."),
    },
    "GPT-2": {
        "onnx":     (85, "Decoder export needs KV-cache plumbing. Not turnkey."),
        "tensorrt": (60, "Native TensorRT struggles -- TensorRT-LLM is the path."),
        "cuda":     (90, "FlashAttention v2 etc. are well-supported."),
        "int8":     (55, "Generative decoders are the hardest INT8 case there is."),
        "dla":      (15, "Not a DLA target."),
        "coreml":   (70, "Possible -- but generative on-device is its own beast."),
    },
    "ViT": {
        "onnx":     (88, "Patch embed + transformer encoder export -- but big graphs."),
        "tensorrt": (72, "LayerNorm + Softmax fall back. Speedup capped without rewrite."),
        "cuda":     (92, "fp16 attention is fast on modern GPUs."),
        "int8":     (60, "LayerNorm + GELU resist PTQ. QAT is the usual answer."),
        "dla":      (25, "DLA can't run transformer encoders. GPU only."),
        "coreml":   (80, "ANE handles ViTs -- with the right export."),
    },
    "YOLO": {
        "onnx":     (90, "Standard detection export. The post-process is usually the gotcha."),
        "tensorrt": (88, "TRT plugins exist for the detection head."),
        "cuda":     (95, "Detection backbones are CUDA-friendly."),
        "int8":     (82, "Detection is the canonical INT8 success story."),
        "dla":      (75, "Backbone runs on DLA; head usually falls back to GPU."),
        "coreml":   (82, "Mobile YOLO conversions are well-trodden."),
    },
    "UNet": {
        "onnx":     (94, "Encoder-decoder, skip connections -- standard export."),
        "tensorrt": (90, "Concat skips fragment slightly; still ships."),
        "cuda":     (95, "All standard vision ops."),
        "int8":     (82, "Segmentation quantizes cleanly with the right calibration set."),
        "dla":      (78, "Most ops supported; transposed conv occasionally falls back."),
        "coreml":   (88, "Standard segmentation conversion."),
    },
}


_FAMILY_LABELS = [
    ("onnx",     "ONNX"),
    ("tensorrt", "TensorRT"),
    ("cuda",     "CUDA FP16"),
    ("int8",     "INT8 quant"),
    ("dla",      "Jetson DLA"),
    ("coreml",   "Apple CoreML"),
]


def _from_module_classes(report) -> list[Family]:
    classes = report.exportability.details.get("module_classes", {}) or {}
    total = max(sum(classes.values()), 1)

    def family(name: str, risky: set[str]) -> Family:
        problematic = sum(c for k, c in classes.items() if k in risky)
        score = max(15.0, 100.0 - 75.0 * (problematic / total))
        if problematic == 0:
            verdict = "Clean. No layers known to misbehave."
        else:
            verdict = (
                f"{problematic} layer(s) likely to fall back or fight you. "
                "Real teams discover this the day before the demo."
            )
        return Family(name=name, score=score, verdict=verdict, blockers=problematic)

    return [
        family("ONNX", _ONNX_RISKY),
        family("TensorRT", _TRT_RISKY),
        family("CUDA FP16", set()),  # we don't model CUDA fp16 risk from class names
        family("INT8 quant", _INT8_RISKY),
        family("Jetson DLA", _DLA_RISKY),
        family("Apple CoreML", _COREML_RISKY),
    ]


def _from_profile(arch_name: str) -> list[Family] | None:
    if arch_name not in _PROFILES:
        return None
    profile = _PROFILES[arch_name]
    out = []
    for key, label in _FAMILY_LABELS:
        score, verdict = profile[key]
        out.append(Family(name=label, score=float(score), verdict=verdict, blockers=0))
    return out


def compute_compatibility(report) -> list[Family]:
    """Return one ``Family`` per runtime target.  Prefers the curated profile
    when the fingerprint matched a known architecture, otherwise falls back
    to a module-class scan."""
    arch = report.architecture
    if arch:
        profiled = _from_profile(arch)
        if profiled:
            return profiled
    return _from_module_classes(report)
