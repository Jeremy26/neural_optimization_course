"""One-click optimization actions for the deployment visualizer.

Each function takes a checkpoint object (``nn.Module`` or state-dict) and
returns a transformed copy plus a ``Snippet`` that shows the user the actual
PyTorch code that would have done the same thing -- so the tool teaches as
it transforms, mirroring the lesson notebooks shipped in this repo.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class Snippet:
    title: str
    code: str
    notebook: str   # path to the notebook this lesson lives in
    summary: str    # one-line plain-English description


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_float_tensors(obj: Any):
    """Yield ``(setter, tensor)`` pairs for every float tensor reachable in
    a state-dict.  ``setter(new_tensor)`` overwrites the tensor in place.
    ``nn.Module`` callers can rebuild a state-dict and call
    ``module.load_state_dict`` afterwards."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                yield (lambda val, _k=k: obj.__setitem__(_k, val)), v


def _module_to_state_dict_copy(obj):
    """Always operate on a state-dict copy so we never mutate the user's
    uploaded object.  For modules we extract+copy the state-dict, transform
    it, then load it back into a deep-copied module."""
    if isinstance(obj, nn.Module):
        sd = {k: v.detach().clone() for k, v in obj.state_dict().items()}
        return sd, "module"
    if isinstance(obj, dict):
        sd = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in obj.items()
        }
        return sd, "state_dict"
    return obj, "unknown"


def _rebuild(orig, sd, kind):
    if kind == "module":
        new_module = copy.deepcopy(orig)
        new_module.load_state_dict(sd, strict=False)
        return new_module
    return sd


# ---------------------------------------------------------------------------
# Optimizations
# ---------------------------------------------------------------------------


def make_efficient(obj: Any) -> tuple[Any, Snippet]:
    """Cast every float-32 weight to float-16.  This is the single cheapest
    optimization there is -- one line of code, half the memory, usually
    indistinguishable accuracy on inference."""
    sd, kind = _module_to_state_dict_copy(obj)
    if kind == "unknown":
        return obj, _SNIPPET_EFFICIENT
    for k, v in list(sd.items()):
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            sd[k] = v.half()
    return _rebuild(obj, sd, kind), _SNIPPET_EFFICIENT


_SNIPPET_EFFICIENT = Snippet(
    title="Cast every weight to half precision",
    summary="The cheapest optimization there is. One line of code, half the carry weight.",
    notebook="Mini_Quantization.ipynb",
    code=(
        "import torch\n"
        "\n"
        "# Cast every float weight from FP32 to FP16.\n"
        "# A deploy-grade artifact almost never carries FP32 anymore.\n"
        "state_dict = torch.load('model.pt')\n"
        "for k, v in state_dict.items():\n"
        "    if v.dtype == torch.float32:\n"
        "        state_dict[k] = v.half()\n"
        "torch.save(state_dict, 'model_half.pt')\n"
    ),
)


def make_lean(obj: Any, ratio: float = 0.5) -> tuple[Any, Snippet]:
    """Magnitude-prune the smallest ``ratio`` fraction of each 2D+ weight
    tensor by setting them to zero.  Real production pruning would fine-tune
    afterwards to recover accuracy -- we don't (this is a demonstration)."""
    sd, kind = _module_to_state_dict_copy(obj)
    if kind == "unknown":
        return obj, _SNIPPET_LEAN
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
    return _rebuild(obj, sd, kind), _SNIPPET_LEAN


_SNIPPET_LEAN = Snippet(
    title="Drop the smallest 50% of weights in every layer",
    summary="Cut the weights that contribute the least. Real teams fine-tune afterwards to recover the last bit of accuracy.",
    notebook="Mini_Pruning.ipynb",
    code=(
        "import torch\n"
        "\n"
        "# For every 2D+ weight tensor, zero out the smallest 50% by\n"
        "# magnitude. In production you'd fine-tune for a few epochs\n"
        "# afterwards to recover accuracy.\n"
        "state_dict = torch.load('model.pt')\n"
        "for k, v in state_dict.items():\n"
        "    if v.dim() >= 2 and v.is_floating_point():\n"
        "        n = int(0.5 * v.numel())\n"
        "        threshold = torch.kthvalue(v.abs().flatten(), n).values\n"
        "        state_dict[k] = v * (v.abs() > threshold).to(v.dtype)\n"
        "torch.save(state_dict, 'model_pruned.pt')\n"
    ),
)


def make_compact(obj: Any) -> tuple[Any, Snippet]:
    """Apply both efficiency (FP16) and leanness (50% prune) for the
    one-shot 'make this lighter' button."""
    obj, _ = make_efficient(obj)
    obj, _ = make_lean(obj, ratio=0.5)
    return obj, _SNIPPET_COMPACT


_SNIPPET_COMPACT = Snippet(
    title="Combine half precision + pruning",
    summary="Stacking the two cheapest moves. Roughly a quarter of the original carry weight, with one fine-tune pass to recover.",
    notebook="Knowledge_Distillation.ipynb",
    code=(
        "import torch\n"
        "\n"
        "# Stack the two cheapest deployment moves: FP16 + 50% pruning.\n"
        "state_dict = torch.load('model.pt')\n"
        "for k, v in state_dict.items():\n"
        "    if not v.is_floating_point():\n"
        "        continue\n"
        "    if v.dim() >= 2:\n"
        "        n = int(0.5 * v.numel())\n"
        "        threshold = torch.kthvalue(v.abs().flatten(), n).values\n"
        "        v = v * (v.abs() > threshold).to(v.dtype)\n"
        "    if v.dtype == torch.float32:\n"
        "        v = v.half()\n"
        "    state_dict[k] = v\n"
        "torch.save(state_dict, 'model_compact.pt')\n"
    ),
)


def try_export_onnx(obj: Any) -> tuple[Any, Snippet, str | None]:
    """Attempt a real ONNX export.  Returns ``(unchanged_obj, snippet,
    error_or_None)``.  This one doesn't transform the model -- it
    diagnoses whether export would actually work."""
    err: str | None = None
    if not isinstance(obj, nn.Module):
        err = "Need the full nn.Module to attempt a real export."
        return obj, _SNIPPET_EXPORT, err

    import io
    bio = io.BytesIO()
    # Sniff a default input shape from the first leaf module.
    dummy = None
    for m in obj.modules():
        if list(m.children()):
            continue
        if isinstance(m, nn.Conv2d):
            dummy = torch.randn(1, m.in_channels, 224, 224)
            break
        if isinstance(m, nn.Linear):
            dummy = torch.randn(1, m.in_features)
            break
    if dummy is None:
        return obj, _SNIPPET_EXPORT, "Couldn't infer an input shape."

    try:
        torch.onnx.export(
            obj.eval(), dummy, bio,
            opset_version=17, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
        )
        return obj, _SNIPPET_EXPORT, None
    except TypeError:
        # Older torch without dynamo kwarg -- retry without it.
        try:
            torch.onnx.export(
                obj.eval(), dummy, bio,
                opset_version=17, do_constant_folding=True,
            )
            return obj, _SNIPPET_EXPORT, None
        except Exception as exc:
            return obj, _SNIPPET_EXPORT, str(exc).splitlines()[0][:200]
    except Exception as exc:
        return obj, _SNIPPET_EXPORT, str(exc).splitlines()[0][:200]


_SNIPPET_EXPORT = Snippet(
    title="Export to ONNX -- the universal deployment format",
    summary="The first runtime step every deployment team takes. Whether it works tells you a lot about how the rest of the journey will go.",
    notebook="deployment-starter-kit (1).ipynb",
    code=(
        "import torch\n"
        "\n"
        "model.eval()\n"
        "dummy = torch.randn(1, 3, 224, 224)  # match your model's input\n"
        "torch.onnx.export(\n"
        "    model, dummy, 'model.onnx',\n"
        "    opset_version=17,\n"
        "    do_constant_folding=True,\n"
        "    input_names=['input'],\n"
        "    output_names=['output'],\n"
        "    dynamic_axes={'input': {0: 'batch'},\n"
        "                  'output': {0: 'batch'}},\n"
        ")\n"
    ),
)
