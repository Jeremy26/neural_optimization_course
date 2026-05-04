# Deployment Health Score

A free Streamlit lead-magnet for the Neural Network Optimization course.
Upload a PyTorch checkpoint (`.pt` / `.pth`) and get a one-number verdict on
how deployment-ready it is, plus a category breakdown and a list of
recommended next steps.

## What it checks

| Category | Weight | What we look at |
|---|---|---|
| Precision | 30% | dtype mix across all parameters (FP64 / FP32 / FP16 / BF16 / INT8 / quantized) |
| Pruning | 20% | global sparsity + near-zero weight headroom (\|w\| < 1% of layer max) |
| Size | 20% | on-disk size in MB, plus parameter count |
| Exportability | 30% | flags layer classes that ONNX / TensorRT trip on |

The categories are combined into a single **Deployment Health Score** in
`[0, 100]`.

## Run it locally

```bash
cd deployment_visualizer
pip install -r requirements.txt
streamlit run app.py
```

## Security notes

- Checkpoints are loaded with `weights_only=True` first; the unsafe pickle
  path is only used as a fallback (and the UI warns the user).
- All analysis happens inside the Streamlit process -- nothing is sent to a
  remote server.

## How to extend

`analyzer.py` exposes `analyze(buffer: bytes) -> AnalysisReport`, so you can
plug it into a CLI, a CI check, or a serverless endpoint. Each category check
(`_check_precision`, `_check_pruning`, `_check_size`, `_check_exportability`)
returns a `CategoryReport(score, verdict, details)` -- swap in your own
weights or add categories (e.g. activation sparsity, kernel-fusion
opportunities) without touching the UI.
