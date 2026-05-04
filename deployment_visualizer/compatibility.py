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
        "onnx":     (90, "Should glide through. The easy case."),
        "tensorrt": (75, "Will compile, but you're capped at a fraction of its real speed."),
        "cuda":     (88, "Standard path. Nothing exotic to trip on."),
        "int8":     (70, "Most of it shrinks cleanly. The output head is the gotcha."),
        "dla":      (75, "Mostly compatible -- but at the mercy of plugin coverage."),
        "coreml":   (75, "Usually lands. Pre/post-processing is the work."),
    },
    "VGG": {
        "onnx":     (92, "Glides through. Boring, in the best way."),
        "tensorrt": (80, "Old and simple -- everything fuses cleanly."),
        "cuda":     (90, "Native, fast, no surprises."),
        "int8":     (75, "Shrinks cleanly. Still big to begin with."),
        "dla":      (78, "Full edge-accelerator support."),
        "coreml":   (80, "Trivial to convert. Question is whether you should."),
    },
    "MobileNet": {
        "onnx":     (85, "Mobile-grade export, with a couple of sharp edges."),
        "tensorrt": (60, "Will run, but not where this runtime shines."),
        "cuda":     (75, "Runs natively. Not the fastest path either."),
        "int8":     (78, "Designed for low precision -- this is its happy place."),
        "dla":      (65, "Partial fit. A handful of operations fall back."),
        "coreml":   (85, "Apple's silicon handles these well."),
    },
    "EfficientNet": {
        "onnx":     (75, "Newer activations trip up older runtimes. Modern ones glide."),
        "tensorrt": (55, "Will compile but stays slow. Native fast paths are limited."),
        "cuda":     (72, "Compatible -- but the graph fragments and loses speed."),
        "int8":     (45, "Resists shrinking cleanly. Needs the harder optimization path."),
        "dla":      (30, "Several activations don't run on the accelerator -- falls back."),
        "coreml":   (65, "Converts. The fancy blocks are slow on Apple silicon."),
    },
    "DenseNet": {
        "onnx":     (82, "Lands -- the graph is just verbose."),
        "tensorrt": (65, "Concat-heavy graphs fragment. The speedup ceiling is real."),
        "cuda":     (80, "Compatible. Memory bandwidth is the limit, not compute."),
        "int8":     (68, "Shrinks acceptably. Calibration takes thought."),
        "dla":      (55, "Partial fit. Several fallbacks."),
        "coreml":   (72, "Converts -- but big and slow on a phone."),
    },
    "BERT": {
        "onnx":     (75, "Lands -- the graph is huge."),
        "tensorrt": (45, "Native runtime won't carry this. Specialized path needed."),
        "cuda":     (78, "Fast attention kernels exist on modern GPUs."),
        "int8":     (40, "Doable, but shrinking it is its own discipline."),
        "dla":      (10, "Transformer-shaped models don't run on the accelerator."),
        "coreml":   (60, "Apple silicon can handle this -- with significant care."),
    },
    "GPT-2": {
        "onnx":     (65, "Generative decoders need plumbing you'd have to write."),
        "tensorrt": (35, "Generic runtime struggles. A specialized one is the answer."),
        "cuda":     (75, "Fast paths exist, but the autoregressive loop dominates."),
        "int8":     (35, "The hardest shrinking case there is."),
        "dla":      (5,  "Not an accelerator workload. Don't even try."),
        "coreml":   (50, "On-device generation is possible -- but its own beast."),
    },
    "ViT": {
        "onnx":     (72, "Lands. Verbose graph for a vision model."),
        "tensorrt": (50, "Will compile. The speedup ceiling is much lower than it should be."),
        "cuda":     (78, "Fast on modern GPUs."),
        "int8":     (40, "Resists low-precision shrinking without serious work."),
        "dla":      (15, "Transformer-shaped models don't run on the accelerator."),
        "coreml":   (60, "Apple silicon handles these -- with a careful export."),
    },
    "YOLO": {
        "onnx":     (75, "Backbone lands cleanly. The detection head is the work."),
        "tensorrt": (65, "Plugins exist -- you'll be writing or borrowing them."),
        "cuda":     (80, "Backbone is fast. The post-processing is the hot spot."),
        "int8":     (62, "Shrinks well -- with the right calibration data."),
        "dla":      (55, "Backbone runs on the accelerator; the head almost never does."),
        "coreml":   (60, "Possible -- but every model is its own project."),
    },
    "UNet": {
        "onnx":     (80, "Encoder-decoder lands. Skip connections are fine."),
        "tensorrt": (68, "Skip connections fragment a bit. Manageable."),
        "cuda":     (82, "Standard vision path."),
        "int8":     (60, "Segmentation shrinks -- calibration set matters a lot."),
        "dla":      (55, "Most of it fits. A few ops fall back to the host."),
        "coreml":   (70, "Standard segmentation conversion path."),
    },
}


_FAMILY_LABELS = [
    ("onnx",     "ONNX"),
    ("tensorrt", "TensorRT"),
    ("cuda",     "CUDA"),
    ("int8",     "Low-precision"),
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
            verdict = "Should glide through cleanly."
        else:
            verdict = (
                f"{problematic} component(s) will resist when you try to ship. "
                "Teams find these the day before the demo."
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
