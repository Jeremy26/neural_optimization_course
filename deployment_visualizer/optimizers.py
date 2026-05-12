"""One-click optimization actions, mirroring the course notebooks.

Each function takes a checkpoint object (``nn.Module`` or state-dict),
applies the canonical course-recipe transformation, and returns the new
object plus a ``Snippet`` showing the actual PyTorch code from the matching
notebook in this repo.

When a state-dict is uploaded but a known architecture is detected, we
quietly load the weights into a torchvision model so the heavier actions
(dynamic quantization, ONNX export) can run on something that actually has
a ``forward()``.
"""

from __future__ import annotations

import copy
import io
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class Snippet:
    title: str
    code: str
    notebook: str   # filename of the source notebook in this repo
    summary: str    # plain-English one-liner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_dict_copy(obj):
    """Always operate on a deep-copied state-dict so we never mutate the
    user's uploaded object."""
    if isinstance(obj, nn.Module):
        return {k: v.detach().clone() for k, v in obj.state_dict().items()}, "module"
    if isinstance(obj, dict):
        return (
            {
                k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in obj.items()
            },
            "state_dict",
        )
    return obj, "unknown"


def _rebuild(orig, sd, kind):
    if kind == "module":
        new_module = copy.deepcopy(orig)
        new_module.load_state_dict(sd, strict=False)
        return new_module
    return sd


def _ensure_module(obj):
    """If obj is a state-dict whose architecture we recognise, instantiate
    the matching torchvision model and load the weights into it.  Returns
    ``(module_or_None, error_or_None)``."""
    if isinstance(obj, nn.Module):
        return obj, None

    from benchmarks import fingerprint, try_load_into_arch, _TORCHVISION_BUILDERS

    arch = fingerprint(obj)
    if not arch:
        return None, (
            "We can't run this without the architecture. Save the full "
            "model with ``torch.save(model, ...)`` (not ``state_dict()``)."
        )

    # Try every torchvision builder whose name starts with the fingerprint.
    candidates = [name for name in _TORCHVISION_BUILDERS if name.startswith(arch)]
    last_err = None
    for cand in candidates:
        module, err = try_load_into_arch(obj, cand)
        if module is not None:
            return module, None
        last_err = err
    return None, last_err or f"Couldn't find a matching {arch} variant."


# ---------------------------------------------------------------------------
# Optimizations
# ---------------------------------------------------------------------------


def make_efficient(obj: Any) -> tuple[Any, Snippet]:
    """Half-precision: cast every FP32 weight to FP16.

    For ``nn.Module`` inputs we call ``.half()`` on a deep copy directly --
    routing the cast tensors through ``load_state_dict`` silently upcasts
    them back to FP32 because the destination parameters are still FP32.
    """
    if isinstance(obj, nn.Module):
        return copy.deepcopy(obj).half(), _SNIPPET_HALF
    if isinstance(obj, dict):
        new_sd = {}
        for k, v in obj.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                new_sd[k] = v.detach().clone().half()
            elif isinstance(v, torch.Tensor):
                new_sd[k] = v.detach().clone()
            else:
                new_sd[k] = v
        return new_sd, _SNIPPET_HALF
    return obj, _SNIPPET_HALF


_SNIPPET_HALF = Snippet(
    title="Half-precision cast",
    summary="The cheapest deployment move. One line, half the carry weight.",
    notebook="Mini_Quantization.ipynb",
    code=(
        "# Half-precision: every weight goes from FP32 to FP16.\n"
        "# Inference accuracy stays effectively identical for most\n"
        "# vision and audio models.\n"
        "model = model.half()\n"
        "torch.save(model.state_dict(), 'model_fp16.pt')\n"
    ),
)


def make_dynamic_quantized(obj: Any) -> tuple[Any, Snippet]:
    """Dynamic INT8 quantization on Linear (+ LSTM / GRU) layers -- the
    canonical Mini_Quantization recipe.

    Quantizes WEIGHTS at conversion time (no calibration data needed) and
    activations dynamically at inference time.  This is the post-training
    quantization path that doesn't require a calibration loop -- distinct
    from static PTQ which does.

    Raises a clean ``RuntimeError`` when the upload is a state-dict whose
    architecture we can't materialise into an nn.Module -- INT8
    quantization fundamentally needs the module's forward graph.
    """
    module, err = _ensure_module(obj)
    if module is None:
        raise RuntimeError(
            "Dynamic INT8 quantization needs the full ``nn.Module``. "
            "Your file is a state-dict only -- save with "
            "``torch.save(model, 'model.pt')`` (not just "
            "``model.state_dict()``) to unlock this step."
        )
    qmodel = torch.quantization.quantize_dynamic(
        copy.deepcopy(module).eval(),
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
        dtype=torch.qint8,
    )
    return qmodel, _SNIPPET_DYNAMIC_QUANT


_SNIPPET_DYNAMIC_QUANT = Snippet(
    title="Dynamic INT8 quantization (no calibration)",
    summary=(
        "Quantizes weights at conversion time and activations dynamically "
        "at inference time. No calibration loop required -- that's static "
        "PTQ, which is a different recipe (Static_Quantization.ipynb)."
    ),
    notebook="Mini_Quantization.ipynb",
    code=(
        "import torch.quantization\n"
        "\n"
        "# Dynamic = weights quantized at conversion, activations at runtime.\n"
        "# No calibration data needed.  Static PTQ is a separate recipe.\n"
        "quantized_model = torch.quantization.quantize_dynamic(\n"
        "    model,\n"
        "    {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},\n"
        "    dtype=torch.qint8,\n"
        ")\n"
    ),
)


def make_lean(obj: Any, ratio: float = 0.3) -> tuple[Any, Snippet]:
    """L1 unstructured pruning at ``ratio`` -- the canonical Mini_Pruning
    recipe.  Uses ``torch.nn.utils.prune.l1_unstructured`` when we have a
    full module; falls back to manual magnitude masking on state-dicts."""
    module, _err = _ensure_module(obj)
    if module is not None:
        try:
            import torch.nn.utils.prune as prune

            new_module = copy.deepcopy(module)
            for m in new_module.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    prune.l1_unstructured(m, "weight", amount=ratio)
                    prune.remove(m, "weight")
            return new_module, _SNIPPET_PRUNE
        except Exception:
            pass

    # Manual fallback for state-dicts: zero out the smallest |w| per tensor.
    sd, kind = _state_dict_copy(obj)
    if kind == "unknown":
        return obj, _SNIPPET_PRUNE
    for k, v in list(sd.items()):
        if (
            isinstance(v, torch.Tensor)
            and v.dtype.is_floating_point
            and v.dim() >= 2
            and v.numel() > 0
        ):
            n_zero = int(ratio * v.numel())
            if n_zero <= 0:
                continue
            threshold = torch.kthvalue(v.abs().flatten(), n_zero).values
            sd[k] = v * (v.abs() > threshold).to(v.dtype)
    return _rebuild(obj, sd, kind), _SNIPPET_PRUNE


_SNIPPET_PRUNE = Snippet(
    title="L1 unstructured pruning",
    summary="The Mini_Pruning recipe -- zero out the lowest-magnitude weights, layer by layer. Fine-tune afterwards to recover.",
    notebook="Mini_Pruning.ipynb",
    code=(
        "import torch.nn.utils.prune as prune\n"
        "\n"
        "for module in model.modules():\n"
        "    if isinstance(module, (torch.nn.Linear,\n"
        "                            torch.nn.Conv2d)):\n"
        "        prune.l1_unstructured(module, 'weight', amount=0.3)\n"
        "        prune.remove(module, 'weight')\n"
    ),
)


def make_structured_pruned(obj: Any, ratio: float = 0.3) -> tuple[Any, Snippet]:
    """L1 structured pruning -- removes whole channels / neurons.
    Course recipe from Mini_Pruning.ipynb."""
    module, err = _ensure_module(obj)
    if module is None:
        return obj, _SNIPPET_PRUNE_STRUCT
    try:
        import torch.nn.utils.prune as prune

        new_module = copy.deepcopy(module)
        for m in new_module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                prune.ln_structured(m, "weight", amount=ratio, n=1, dim=0)
                prune.remove(m, "weight")
        return new_module, _SNIPPET_PRUNE_STRUCT
    except Exception:
        return obj, _SNIPPET_PRUNE_STRUCT


_SNIPPET_PRUNE_STRUCT = Snippet(
    title="L1 structured pruning (channel-level)",
    summary="Removes whole output channels / neurons. More aggressive than unstructured -- and the speedup actually shows up on hardware.",
    notebook="Mini_Pruning.ipynb",
    code=(
        "import torch.nn.utils.prune as prune\n"
        "\n"
        "for module in model.modules():\n"
        "    if isinstance(module, (torch.nn.Linear,\n"
        "                            torch.nn.Conv2d)):\n"
        "        prune.ln_structured(\n"
        "            module, 'weight', amount=0.3, n=1, dim=0,\n"
        "        )\n"
        "        prune.remove(module, 'weight')\n"
    ),
)


def make_compact(obj: Any) -> tuple[Any, Snippet]:
    """Stack half-precision + L1 pruning."""
    obj, _ = make_efficient(obj)
    obj, _ = make_lean(obj, ratio=0.3)
    return obj, _SNIPPET_COMPACT


_SNIPPET_COMPACT = Snippet(
    title="Half precision + L1 pruning, stacked",
    summary="The two cheapest moves combined. Roughly a quarter of the original carry weight after a short fine-tune to recover.",
    notebook="Mini_Pruning.ipynb",
    code=(
        "import torch.nn.utils.prune as prune\n"
        "\n"
        "# Half-precision first.\n"
        "model = model.half()\n"
        "\n"
        "# Then L1 unstructured pruning at 30%.\n"
        "for module in model.modules():\n"
        "    if isinstance(module, (torch.nn.Linear,\n"
        "                            torch.nn.Conv2d)):\n"
        "        prune.l1_unstructured(module, 'weight', amount=0.3)\n"
        "        prune.remove(module, 'weight')\n"
    ),
)


# ---------------------------------------------------------------------------
# ONNX export -- with auto-loading into torchvision models when possible
# ---------------------------------------------------------------------------


def _guess_dummy_input(module: nn.Module) -> torch.Tensor | None:
    """Sniff a plausible dummy tensor from the first leaf module."""
    for m in module.modules():
        if list(m.children()):
            continue
        if isinstance(m, nn.Conv2d):
            return torch.randn(1, m.in_channels, 224, 224)
        if isinstance(m, nn.Conv1d):
            return torch.randn(1, m.in_channels, 1024)
        if isinstance(m, nn.Conv3d):
            return torch.randn(1, m.in_channels, 16, 64, 64)
        if isinstance(m, nn.Linear):
            return torch.randn(1, m.in_features)
        if isinstance(m, nn.Embedding):
            return torch.randint(0, 100, (1, 64), dtype=torch.long)
    return None


def _module_param_dtype(module: nn.Module) -> torch.dtype:
    """Return the dtype of the first floating-point parameter, so we can
    feed a matching dummy input.  Defaults to float32 for modules with no
    floating params (rare, but safe)."""
    for p in module.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def try_export_onnx(obj: Any) -> tuple[Any, Snippet, str | None, bytes | None]:
    """Attempt a real ONNX export.  Auto-loads state-dicts into the matching
    torchvision model when the architecture is recognised.

    Returns ``(unchanged_obj, snippet, error_or_None, onnx_bytes_or_None)``.
    The bytes can be wired straight into ``st.download_button`` when the
    export succeeds.
    """
    module, err = _ensure_module(obj)
    if module is None:
        return obj, _SNIPPET_ONNX, err, None

    dummy = _guess_dummy_input(module)
    if dummy is None:
        return obj, _SNIPPET_ONNX, "Couldn't infer a plausible input shape.", None

    # Match dummy dtype to the model.  After ``make_efficient`` the module
    # is FP16; feeding an FP32 dummy raises
    # ``RuntimeError: expected scalar type Float but found Half``.
    if dummy.is_floating_point():
        dummy = dummy.to(_module_param_dtype(module))

    common = dict(
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    last_err = None
    for kwargs in ({"dynamo": False, **common}, common):
        bio = io.BytesIO()
        try:
            torch.onnx.export(module.eval(), dummy, bio, **kwargs)
            return obj, _SNIPPET_ONNX, None, bio.getvalue()
        except TypeError as exc:
            last_err = str(exc).splitlines()[0][:200]
            continue
        except Exception as exc:
            last_err = str(exc).splitlines()[0][:200]
            break
    return obj, _SNIPPET_ONNX, last_err, None


_SNIPPET_ONNX = Snippet(
    title="ONNX export -- the universal deployment format",
    summary="The first runtime step every deployment team takes. Whether it works tells you a lot about the rest of the journey.",
    notebook="deployment-starter-kit (1).ipynb",
    code=(
        "torch.onnx.export(\n"
        "    model,\n"
        "    input_tensor,\n"
        "    'model.onnx',\n"
        "    input_names=['input'],\n"
        "    output_names=['output'],\n"
        "    opset_version=17,\n"
        "    dynamic_axes={'input':  {0: 'batch_size'},\n"
        "                  'output': {0: 'batch_size'}},\n"
        ")\n"
    ),
)


# ---------------------------------------------------------------------------
# Catalog -- exposed to the UI for the Optimization tab
# ---------------------------------------------------------------------------


@dataclass
class TechniqueOption:
    key: str
    label: str
    description: str
    notebook: str
    apply: callable     # callable(obj) -> (new_obj, Snippet) or (obj, Snippet, err)
    # When True, this technique needs the full ``nn.Module`` -- it can't
    # operate on a bare state-dict.  The UI uses this to gray out the
    # option with a clear "needs full nn.Module" badge when only a
    # state-dict was uploaded.
    requires_module: bool = False
    video_url: str | None = None


TECHNIQUES: list[TechniqueOption] = [
    TechniqueOption(
        key="half",
        label="Half precision (FP16)",
        description=(
            "Cast every weight from 32-bit to 16-bit floats. Half the "
            "memory, half the bandwidth, the same accuracy for almost any "
            "inference workload. Works on both `.pt` and `.pth` uploads."
        ),
        notebook="Mini_Quantization.ipynb",
        apply=make_efficient,
        requires_module=False,
    ),
    TechniqueOption(
        key="dynamic_quant",
        label="Dynamic INT8 quantization",
        description=(
            "Weights drop to INT8 at conversion, activations quantize "
            "dynamically at inference. **No calibration data needed** -- "
            "that's static PTQ, which is a separate recipe. Needs a full "
            "`.pt` file (the module class), not just weights."
        ),
        notebook="Mini_Quantization.ipynb",
        apply=make_dynamic_quantized,
        requires_module=True,
    ),
    TechniqueOption(
        key="prune_unstructured",
        label="L1 unstructured pruning (30%)",
        description=(
            "Zero out the lowest-magnitude 30% of weights, layer by layer. "
            "Compresses well and stacks with quantization. Doesn't speed "
            "up inference on its own (you need sparse storage for that)."
        ),
        notebook="Mini_Pruning.ipynb",
        apply=make_lean,
        requires_module=False,
    ),
    TechniqueOption(
        key="prune_structured",
        label="L1 structured pruning (30%)",
        description=(
            "Remove entire output channels by their L1 norm. Aggressive -- "
            "needs a fine-tune to recover -- but the speedup actually "
            "shows up on hardware. Needs the full `.pt` to know channel "
            "boundaries."
        ),
        notebook="Mini_Pruning.ipynb",
        apply=make_structured_pruned,
        requires_module=True,
    ),
    TechniqueOption(
        key="stack",
        label="Half precision + 30% pruning (stacked)",
        description=(
            "The two cheapest moves combined: FP16 plus magnitude pruning. "
            "Roughly a quarter of the original carry weight after a short "
            "fine-tune. Works on both `.pt` and `.pth`."
        ),
        notebook="Mini_Pruning.ipynb",
        apply=make_compact,
        requires_module=False,
    ),
]
