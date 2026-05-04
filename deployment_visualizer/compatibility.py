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
# Curated readiness profiles.  Scores reflect the *architecture's* ceiling
# only; the precision penalty applied later in compute_compatibility lowers
# this further when the user's actual checkpoint is at FP32 (which is most
# of them, and which leaves most of the speed on the table).
_PROFILES = {
    "ResNet": {
        "onnx":     (90, "Should export -- ResNet is the easy case."),
        "tensorrt": (75, "TRT will compile, but the speedup ceiling is FP16+INT8."),
        "cuda":     (88, "Standard CUDA path. No exotic ops to break things."),
        "int8":     (70, "Conv + Linear quantize OK. The head needs care."),
        "dla":      (75, "DLA-compatible ops, but you're at the mercy of TRT plugin coverage."),
        "coreml":   (75, "Vision backbone -- usually converts. Pre/post is the gotcha."),
    },
    "VGG": {
        "onnx":     (92, "Trivial export."),
        "tensorrt": (80, "Old, simple, fuses well -- but compute-heavy."),
        "cuda":     (90, "Every op is native."),
        "int8":     (75, "Quantizes cleanly. Still bigger than it needs to be."),
        "dla":      (78, "Full DLA op coverage."),
        "coreml":   (80, "Trivial to convert -- but who ships VGG to mobile?"),
    },
    "MobileNet": {
        "onnx":     (85, "Mobile-vision export -- but watch hard-swish."),
        "tensorrt": (60, "Depthwise convs are not where TRT shines. Latency caps fast."),
        "cuda":     (75, "Depthwise + pointwise have CUDA kernels but aren't fastest."),
        "int8":     (78, "Designed for INT8 -- but PTQ vs QAT is a real choice here."),
        "dla":      (65, "Hard-swish + adaptive pool fall back. Partial DLA."),
        "coreml":   (85, "Apple Neural Engine handles these well."),
    },
    "EfficientNet": {
        "onnx":     (75, "SiLU / Swish trip up older opsets. Modern ones export cleanly."),
        "tensorrt": (55, "Squeeze-Excite + SiLU need TRT 8.5+ to stay native -- still slow."),
        "cuda":     (72, "Mostly fp16-friendly; SE blocks fragment your kernel graph."),
        "int8":     (45, "SiLU + SE blocks make PTQ rough -- you'll need QAT."),
        "dla":      (30, "DLA struggles with SiLU and adaptive pooling. Falls back to GPU."),
        "coreml":   (65, "Converts -- but the SE blocks are slow on ANE."),
    },
    "DenseNet": {
        "onnx":     (82, "Dense connectivity exports fine -- but the graph gets verbose."),
        "tensorrt": (65, "Concat-heavy graphs fragment. Speedup ceiling is real."),
        "cuda":     (80, "Standard ops; concats hit memory bandwidth, not compute."),
        "int8":     (68, "Quantizes OK; intermediate concats need calibration thought."),
        "dla":      (55, "Partial DLA. Several fallbacks."),
        "coreml":   (72, "Converts -- but big and slow on mobile."),
    },
    "BERT": {
        "onnx":     (75, "Transformer export works -- the graph is huge."),
        "tensorrt": (45, "LayerNorm + Softmax fall back. Native TRT is not the path."),
        "cuda":     (78, "fp16 attention kernels exist. FlashAttention is your friend."),
        "int8":     (40, "Encoder-only is doable. Calibration is its own discipline."),
        "dla":      (10, "DLA doesn't run transformers. GPU only."),
        "coreml":   (60, "Apple Neural Engine handles BERT -- with significant care."),
    },
    "GPT-2": {
        "onnx":     (65, "Decoder export needs KV-cache plumbing you have to write."),
        "tensorrt": (35, "Native TensorRT struggles -- TensorRT-LLM is the actual path."),
        "cuda":     (75, "FlashAttention v2 helps, but the autoregressive loop dominates."),
        "int8":     (35, "Generative decoders are the hardest INT8 case there is."),
        "dla":      (5,  "Not a DLA workload. Don't even try."),
        "coreml":   (50, "On-device generation is possible -- but its own beast entirely."),
    },
    "ViT": {
        "onnx":     (72, "Big verbose graph. Patch embed + transformer encoder."),
        "tensorrt": (50, "LayerNorm + Softmax fall back. Speedup capped without rewrite."),
        "cuda":     (78, "fp16 attention is fast on modern GPUs."),
        "int8":     (40, "LayerNorm + GELU resist PTQ. QAT is non-trivial."),
        "dla":      (15, "DLA can't run transformer encoders. GPU only."),
        "coreml":   (60, "ANE handles ViTs -- with a careful export."),
    },
    "YOLO": {
        "onnx":     (75, "Backbone exports cleanly. Detection head is the work."),
        "tensorrt": (65, "TRT plugins exist -- you'll be writing or borrowing them."),
        "cuda":     (80, "Detection backbones are CUDA-friendly. NMS is the hot spot."),
        "int8":     (62, "Detection quantizes -- with the right calibration data."),
        "dla":      (55, "Backbone runs on DLA; head almost always falls back."),
        "coreml":   (60, "Mobile YOLO conversions exist but each model is a project."),
    },
    "UNet": {
        "onnx":     (80, "Encoder-decoder + skip connections export OK."),
        "tensorrt": (68, "Concat skips fragment slightly. Manageable."),
        "cuda":     (82, "All standard vision ops."),
        "int8":     (60, "Segmentation quantizes -- calibration set matters a lot."),
        "dla":      (55, "Most ops supported; transposed conv often falls back."),
        "coreml":   (70, "Standard segmentation conversion path."),
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


def _precision_penalty(report) -> dict[str, int]:
    """How many points to knock off each family based on the model's current
    precision.  An FP32 model isn't TensorRT-ready in any practical sense --
    you'd convert to FP16 or INT8 first or leave most of the speed on the
    table.  Same for DLA and INT8 quant: they assume you've done that work."""
    dist = report.precision.details.get("dtype_distribution", {})
    keys = " ".join(dist.keys())
    if "float16" in keys or "bfloat16" in keys:
        return {"tensorrt": 5, "int8": 15, "dla": 15, "cuda": 0,
                "onnx": 0, "coreml": 5}
    if "qint8" in keys or ("int8" in keys and "uint8" not in keys):
        return {"tensorrt": 0, "int8": 0, "dla": 0, "cuda": 0,
                "onnx": 0, "coreml": 0}
    # FP32 default -- the most common case, and the most punished.
    return {"tensorrt": 25, "int8": 30, "dla": 25, "cuda": 15,
            "onnx": 5, "coreml": 15}


def _apply_precision_penalty(families: list[Family], penalties: dict[str, int]) -> list[Family]:
    out = []
    label_to_key = {label: key for key, label in _FAMILY_LABELS}
    for fam in families:
        key = label_to_key.get(fam.name, "")
        penalty = penalties.get(key, 0)
        new_score = max(10.0, fam.score - penalty)
        out.append(Family(
            name=fam.name,
            score=new_score,
            verdict=fam.verdict,
            blockers=fam.blockers,
        ))
    return out


def compute_compatibility(report) -> list[Family]:
    """Return one ``Family`` per runtime target.  Prefers the curated profile
    when the fingerprint matched a known architecture, otherwise falls back
    to a module-class scan.  Applies a precision penalty so FP32 models
    don't score deploy-ready when in reality they're a quantization step
    away from any of these runtimes paying off."""
    arch = report.architecture
    if arch:
        profiled = _from_profile(arch)
        if profiled is not None:
            return _apply_precision_penalty(profiled, _precision_penalty(report))
    return _apply_precision_penalty(
        _from_module_classes(report), _precision_penalty(report)
    )
