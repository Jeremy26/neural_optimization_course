"""Visualization data builders for the Streamlit UI.

Decouples expensive iteration over the checkpoint from the rendering layer.
Each function returns a plain dict / dataframe-friendly structure so the UI
file can stay focused on plotly calls.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import nn


# Soft cap on how many tensors we sample for the weight-distribution panel.
# Sampling 8 layers covers "the big ones" without hammering the page.
_TOP_LAYERS_FOR_HIST = 8
_HIST_BINS = 60
_HIST_MAX_SAMPLES = 200_000


def _module_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters(recurse=False))


def module_tree(obj: Any) -> list[dict]:
    """Flat list of nodes for a plotly treemap.

    Each row has ``id`` (full path), ``parent`` (full path of parent or empty
    string for roots), ``label`` (short name + class), ``params``, ``type``.
    Works for both ``nn.Module`` checkpoints and bare state-dicts; for the
    latter we infer hierarchy from key prefixes.
    """
    if isinstance(obj, nn.Module):
        rows: list[dict] = []
        # Root sentinel so plotly's treemap roots correctly.
        rows.append({
            "id": "model",
            "parent": "",
            "label": f"model ({type(obj).__name__})",
            "params": sum(p.numel() for p in obj.parameters()),
            "type": type(obj).__name__,
        })
        for name, m in obj.named_modules():
            if name == "":
                continue
            params = sum(p.numel() for p in m.parameters())
            if params == 0:
                continue
            parts = name.split(".")
            parent = ".".join(parts[:-1]) if len(parts) > 1 else "model"
            # named_modules returns parents before children; the parent must
            # exist in `rows` for plotly to anchor the child.  We post-fix any
            # missing intermediate IDs by prefixing with "model" only when the
            # parent string is empty.
            if not parent:
                parent = "model"
            rows.append({
                "id": name,
                "parent": parent,
                "label": f"{parts[-1]} ({type(m).__name__})",
                "params": params,
                "type": type(m).__name__,
            })
        return rows

    if isinstance(obj, dict):
        # state-dict: aggregate by parent module path.
        agg: dict[str, int] = defaultdict(int)
        for k, v in obj.items():
            if not isinstance(v, torch.Tensor):
                continue
            parts = k.split(".")
            # Strip the trailing param name (weight/bias/running_mean/etc).
            parent_path = ".".join(parts[:-1])
            agg[parent_path] += v.numel()

        rows = [{
            "id": "model",
            "parent": "",
            "label": "state_dict",
            "params": sum(agg.values()),
            "type": "state_dict",
        }]
        seen = {"model"}
        for path, n in agg.items():
            # Insert ancestors so plotly has a chain.
            parts = path.split(".") if path else []
            running = []
            for part in parts:
                running.append(part)
                pid = ".".join(running)
                if pid in seen:
                    continue
                parent = ".".join(running[:-1]) or "model"
                rows.append({
                    "id": pid,
                    "parent": parent,
                    "label": part,
                    "params": 0,
                    "type": "module",
                })
                seen.add(pid)
            # Set the leaf's params (overwrite the placeholder if it exists).
            for r in rows:
                if r["id"] == path:
                    r["params"] = n
                    break
        return rows

    return []


def _infer_layer_type(layer_path: str, tensor_keys: list[str]) -> str:
    """Cheap classifier: pattern-match on the parameter names attached to
    this module path to guess what kind of layer it is."""
    suffixes = {k.rsplit(".", 1)[-1] for k in tensor_keys}
    if {"running_mean", "running_var"} & suffixes:
        return "BatchNorm"
    if "num_batches_tracked" in suffixes:
        return "BatchNorm"
    # Heuristic: 4D conv weights vs 2D linear weights would need shapes.
    # Without them we keep it generic.
    if "weight" in suffixes and "bias" in suffixes:
        return "Linear/Conv"
    if "weight" in suffixes:
        return "Weight-only"
    return "Module"


def per_layer_costs(obj: Any) -> list[dict]:
    """Bar-chart-ready: each row is a leaf module with params + a coarse
    'cost class' for coloring.  Works on both ``nn.Module`` and state-dicts
    by grouping keys by their parent path."""
    rows: list[dict] = []
    if isinstance(obj, nn.Module):
        for name, m in obj.named_modules():
            if name == "" or list(m.children()):
                continue
            params = _module_params(m)
            if params == 0:
                continue
            rows.append({
                "name": name,
                "type": type(m).__name__,
                "params": params,
            })
    elif isinstance(obj, dict):
        groups: dict[str, list[str]] = defaultdict(list)
        sums: dict[str, int] = defaultdict(int)
        for k, v in obj.items():
            if not isinstance(v, torch.Tensor):
                continue
            parent = k.rsplit(".", 1)[0] if "." in k else "(root)"
            groups[parent].append(k)
            sums[parent] += v.numel()
        for path, n in sums.items():
            if n == 0:
                continue
            rows.append({
                "name": path,
                "type": _infer_layer_type(path, groups[path]),
                "params": n,
            })
    rows.sort(key=lambda r: r["params"], reverse=True)
    return rows


def weight_histograms(obj: Any) -> list[dict]:
    """Sampled magnitude histograms for the largest weight tensors.

    Returns a list of ``{name, type, shape, bin_edges, counts}``.  We sample
    tensors larger than ``_HIST_MAX_SAMPLES`` to keep the UI fast.
    """
    # Build a list of (full_name, tensor, type_label).
    entries: list[tuple[str, torch.Tensor, str]] = []
    if isinstance(obj, nn.Module):
        for name, m in obj.named_modules():
            if list(m.children()):
                continue
            for pname, p in m.named_parameters(recurse=False):
                if pname == "weight" and p.dim() >= 2:
                    entries.append((f"{name}.{pname}" if name else pname, p.detach(), type(m).__name__))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.is_floating_point():
                entries.append((k, v.detach(), "weight"))

    entries.sort(key=lambda e: e[1].numel(), reverse=True)
    out = []
    for name, tensor, kind in entries[:_TOP_LAYERS_FOR_HIST]:
        flat = tensor.abs().flatten().float()
        if flat.numel() > _HIST_MAX_SAMPLES:
            idx = torch.randint(0, flat.numel(), (_HIST_MAX_SAMPLES,))
            flat = flat[idx]
        # Use log-spaced bins so dense-near-zero distributions are readable.
        max_abs = float(flat.max().item()) if flat.numel() else 1.0
        if max_abs <= 0:
            continue
        edges = torch.linspace(0, max_abs, _HIST_BINS + 1)
        counts = torch.histc(flat, bins=_HIST_BINS, min=0.0, max=max_abs)
        out.append({
            "name": name,
            "type": kind,
            "shape": list(tensor.shape),
            "params": tensor.numel(),
            "bin_edges": edges.tolist(),
            "counts": counts.tolist(),
            "max_abs": max_abs,
        })
    return out
