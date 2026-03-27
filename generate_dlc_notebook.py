#!/usr/bin/env python3
"""Generate DLC_Deployment.ipynb — minimal, student-friendly version."""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

# ── Title ─────────────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("""\
# Neural Network Deployment — PyTorch → ONNX → TensorRT

We start with a familiar PyTorch model, then show how switching to ONNX Runtime
and TensorRT gives us a massive speed boost — with almost no code change.

| Test | Runtime | Device |
|------|---------|--------|
| Test 1 | PyTorch | CPU |
| Test 2 | PyTorch | GPU |
| Test 3 | ONNX Runtime | GPU |
| Test 4 | TensorRT FP16 | GPU |

> **Requires:** Colab T4 GPU runtime"""))

# ── 0. Setup ──────────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("---\n## 0. Setup"))

nb.cells.append(new_code_cell("""\
!pip install onnx onnxsim onnxruntime-gpu tensorrt pycuda -q"""))

nb.cells.append(new_code_cell("""\
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import time
import os
import glob
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import onnx
import onnxsim
import onnxruntime as ort

print(f"PyTorch  {torch.__version__}")
print(f"ORT      {ort.__version__}")
print(f"CUDA     {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")"""))

# ── 1. Data & Models ──────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("---\n## 1. Download Data & Models"))

nb.cells.append(new_code_cell("""\
# Real Waymo driving frames
!mkdir -p /content/data /content/models
!wget -qq https://optical-flow-data.s3.eu-west-3.amazonaws.com/waymo_images.zip -O /content/data/waymo.zip
!unzip -qq /content/data/waymo.zip -d /content/data/

# Autoware SceneSeg ONNX models (pre-exported, multiple precisions)
!gdown -O /content/models/ 'https://docs.google.com/uc?export=download&id=1l-dniunvYyFKvLD7k16Png3AsVTuMl9f'
!gdown -O /content/models/ 'https://docs.google.com/uc?export=download&id=19gMPt_1z4eujo4jm5XKuH-8eafh-wJC6'
!gdown -O /content/models/ 'https://docs.google.com/uc?export=download&id=1zCworKw4aQ9_hDBkHfj1-sXitAAebl5Y'

for f in sorted(glob.glob('/content/models/*')):
    print(f"{Path(f).name}  —  {os.path.getsize(f)/1e6:.1f} MB")"""))

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("""\
---
## 2. Step 1 — PyTorch

We load a pretrained segmentation model straight from `torchvision`.
This is the starting point every ML engineer knows."""))

nb.cells.append(new_code_cell("""\
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
model.eval()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")"""))

nb.cells.append(new_code_cell("""\
# Load a Waymo frame
frames = sorted(glob.glob('/content/data/**/*.jpg', recursive=True)
              + glob.glob('/content/data/**/*.png', recursive=True))

frame = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)
pil   = Image.fromarray(frame)

preprocess = T.Compose([
    T.Resize((520, 520)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
x = preprocess(pil).unsqueeze(0)   # (1, 3, 520, 520)  stays on CPU

plt.imshow(frame); plt.axis('off'); plt.title('Input: Waymo driving frame'); plt.show()
print(f"Input tensor: {x.shape}")"""))

nb.cells.append(new_code_cell("""\
# Run inference
with torch.no_grad():
    out = model(x)['out']   # (1, 21, H, W)

seg = out.argmax(1).squeeze().numpy()

# PASCAL VOC colour map — DeepLabV3 trained on VOC (21 classes)
COLORS = np.zeros((21, 3), dtype=np.uint8)
for i in range(21):
    r, g, b, c = 0, 0, 0, i
    for j in range(8):
        r |= ((c >> 0) & 1) << (7 - j)
        g |= ((c >> 1) & 1) << (7 - j)
        b |= ((c >> 2) & 1) << (7 - j)
        c >>= 3
    COLORS[i] = [r, g, b]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(frame);          axes[0].set_title('Input frame');        axes[0].axis('off')
axes[1].imshow(COLORS[seg]);    axes[1].set_title('Segmentation (PyTorch)'); axes[1].axis('off')
plt.tight_layout(); plt.show()"""))

nb.cells.append(new_code_cell("""\
def benchmark(fn, name, n_warmup=5, n_runs=50, cuda=False):
    with torch.no_grad():
        for _ in range(n_warmup):
            fn()
    if cuda:
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn()
            if cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    print(f"{name:.<55} {t.mean():6.1f} ms ± {t.std():.1f}  ({1000/t.mean():.0f} FPS)")
    return t

# Test 1 — PyTorch CPU  (1 warmup, 5 runs — CPU is just a reference)
pytorch_cpu = benchmark(lambda: model(x), 'Test 1 · PyTorch CPU', n_warmup=1, n_runs=5)"""))

nb.cells.append(new_code_cell("""\
# Test 2 — PyTorch GPU
model_gpu = model.cuda()
x_gpu     = x.cuda()

pytorch_gpu = benchmark(lambda: model_gpu(x_gpu), 'Test 2 · PyTorch GPU', cuda=True)
print(f"  → GPU is {pytorch_cpu.mean()/pytorch_gpu.mean():.1f}x faster than CPU")"""))

# ── 3. ONNX ───────────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("""\
---
## 3. Step 2 — ONNX Runtime

We already have a pre-exported ONNX file. Loading it is 3 lines.
ONNX Runtime handles everything — preprocessing stays the same."""))

nb.cells.append(new_code_cell("""\
# Export our PyTorch model to ONNX
class Wrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)['out']

ONNX_PATH = '/content/models/deeplabv3.onnx'
with torch.no_grad():
    torch.onnx.export(Wrapper(model.cpu()), x, ONNX_PATH,
                      input_names=['input'], output_names=['output'], opset_version=17)

size_mb = os.path.getsize(ONNX_PATH) / 1e6
print(f"Exported → {ONNX_PATH}  ({size_mb:.1f} MB)")"""))

nb.cells.append(new_code_cell("""\
# Load with ONNX Runtime
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
sess = ort.InferenceSession(ONNX_PATH, providers=providers)

print(f"Provider : {sess.get_providers()[0]}")
print(f"Input    : {sess.get_inputs()[0].name}  {sess.get_inputs()[0].shape}")"""))

nb.cells.append(new_code_cell("""\
# Run inference — pass CPU numpy, ONNX Runtime handles the rest
x_np    = x.numpy()
ort_out = sess.run(None, {'input': x_np})[0]   # (1, 21, H, W)

ort_seg = ort_out.argmax(axis=1).squeeze()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(COLORS[seg]);     axes[0].set_title('PyTorch'); axes[0].axis('off')
axes[1].imshow(COLORS[ort_seg]); axes[1].set_title(f'ONNX Runtime ({sess.get_providers()[0]})'); axes[1].axis('off')
plt.tight_layout(); plt.show()

print(f"Max diff vs PyTorch: {np.abs(out.numpy() - ort_out).max():.5f}")"""))

nb.cells.append(new_code_cell("""\
# Test 3 — ONNX Runtime GPU
onnx_gpu = benchmark(lambda: sess.run(None, {'input': x_np}),
                     'Test 3 · ONNX Runtime GPU', cuda=True)
print(f"  → ORT GPU is {pytorch_gpu.mean()/onnx_gpu.mean():.2f}x vs PyTorch GPU")"""))

# ── 4. TensorRT ───────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("""\
---
## 4. Step 3 — TensorRT

TensorRT compiles the ONNX graph into a GPU-specific engine.
It fuses layers and picks the fastest kernels for your exact GPU."""))

nb.cells.append(new_code_cell("""\
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_MAJOR  = int(trt.__version__.split('.')[0])
print(f"TensorRT {trt.__version__}")"""))

nb.cells.append(new_code_cell("""\
# Build engine from ONNX
builder = trt.Builder(TRT_LOGGER)
network = (builder.create_network() if TRT_MAJOR >= 10
           else builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)))
parser  = trt.OnnxParser(network, TRT_LOGGER)

with open(ONNX_PATH, 'rb') as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
config.set_flag(trt.BuilderFlag.FP16)

# Fixed input shape → no optimization profile needed
print("Building... (1-2 min)")
serialized = builder.build_serialized_network(network, config)
engine     = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(serialized)
print("✓ Engine ready")"""))

nb.cells.append(new_code_cell("""\
# Allocate GPU buffers and run inference
ctx = engine.create_execution_context()

if TRT_MAJOR >= 10:
    in_shape  = tuple(engine.get_tensor_shape('input'))
    out_shape = tuple(engine.get_tensor_shape('output'))
else:
    in_shape  = tuple(engine.get_binding_shape(0))
    out_shape = tuple(engine.get_binding_shape(1))

d_in  = cuda.mem_alloc(int(np.prod(in_shape))  * 4)
d_out = cuda.mem_alloc(int(np.prod(out_shape)) * 4)
stream = cuda.Stream()

def trt_infer(x_np):
    inp = np.ascontiguousarray(x_np, dtype=np.float32)
    out = np.empty(out_shape, dtype=np.float32)
    cuda.memcpy_htod_async(d_in, inp, stream)
    if TRT_MAJOR >= 10:
        ctx.set_tensor_address('input',  int(d_in))
        ctx.set_tensor_address('output', int(d_out))
        ctx.execute_async_v3(stream_handle=stream.handle)
    else:
        ctx.execute_async_v2([int(d_in), int(d_out)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(out, d_out, stream)
    stream.synchronize()
    return out

trt_out = trt_infer(x_np)
trt_seg = trt_out.argmax(axis=1).squeeze()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(COLORS[ort_seg]); axes[0].set_title('ONNX Runtime'); axes[0].axis('off')
axes[1].imshow(COLORS[trt_seg]); axes[1].set_title('TensorRT FP16'); axes[1].axis('off')
plt.tight_layout(); plt.show()"""))

nb.cells.append(new_code_cell("""\
# Test 4 — TensorRT GPU
trt_gpu = benchmark(lambda: trt_infer(x_np), 'Test 4 · TensorRT GPU (FP16)', cuda=True)
print(f"  → TRT is {pytorch_gpu.mean()/trt_gpu.mean():.1f}x vs PyTorch GPU")"""))

# ── 5. Results ────────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("---\n## 5. Results"))

nb.cells.append(new_code_cell("""\
results = {
    'PyTorch\\nCPU':    pytorch_cpu,
    'PyTorch\\nGPU':    pytorch_gpu,
    'ONNX RT\\nGPU':    onnx_gpu,
    'TensorRT\\nFP16':  trt_gpu,
}
colors  = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
labels  = list(results.keys())
avgs    = [t.mean() for t in results.values()]
stds    = [t.std()  for t in results.values()]
p95s    = [np.percentile(t, 95) for t in results.values()]
x_pos   = np.arange(len(labels))"""))

nb.cells.append(new_code_cell("""\
# Bar chart: latency + FPS + speedup
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(x_pos, avgs, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor='k')
for i, (a, p) in enumerate(zip(avgs, p95s)):
    axes[0].plot(i, p, 'v', color='black', markersize=7)
    axes[0].text(i, a + stds[i] + 0.5, f'{a:.1f}ms', ha='center', fontsize=8, fontweight='bold')
axes[0].set_xticks(x_pos); axes[0].set_xticklabels(labels)
axes[0].set_ylabel('Latency (ms)'); axes[0].set_title('Latency  (▼ = P95)')

fps = [1000/a for a in avgs]
axes[1].bar(x_pos, fps, color=colors, alpha=0.85, edgecolor='k')
for i, f in enumerate(fps):
    axes[1].text(i, f + 0.3, f'{f:.0f}', ha='center', fontsize=9, fontweight='bold')
axes[1].set_xticks(x_pos); axes[1].set_xticklabels(labels)
axes[1].set_ylabel('FPS'); axes[1].set_title('Throughput')

spd = [avgs[0]/a for a in avgs]
axes[2].bar(x_pos, spd, color=colors, alpha=0.85, edgecolor='k')
axes[2].axhline(1, color='gray', linestyle='--')
for i, s in enumerate(spd):
    axes[2].text(i, s + 0.05, f'{s:.1f}x', ha='center', fontsize=9, fontweight='bold')
axes[2].set_xticks(x_pos); axes[2].set_xticklabels(labels)
axes[2].set_ylabel('Speedup vs CPU'); axes[2].set_title('Speedup')

plt.suptitle('Deployment Benchmark', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()"""))

nb.cells.append(new_code_cell("""\
# Violin plot — full latency distribution
fig, ax = plt.subplots(figsize=(12, 5))
parts = ax.violinplot(list(results.values()), positions=x_pos, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
for i, p95 in enumerate(p95s):
    ax.hlines(p95, i-0.25, i+0.25, colors='black', linestyles='--', linewidth=1.5)
    ax.text(i+0.27, p95, 'P95', fontsize=7)
ax.set_xticks(x_pos); ax.set_xticklabels(labels)
ax.set_ylabel('Latency (ms)'); ax.set_title('Latency Distribution (width = frequency)')
plt.tight_layout(); plt.show()"""))

nb.cells.append(new_code_cell("""\
# CDF — "what % of frames meet a latency budget?"
fig, ax = plt.subplots(figsize=(12, 5))
for (label, times), color in zip(results.items(), colors):
    s = np.sort(times)
    ax.plot(s, np.arange(1, len(s)+1)/len(s)*100,
            label=label.replace('\\n',' '), color=color, linewidth=2)
ax.axhline(95, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Latency (ms)'); ax.set_ylabel('% of runs below X ms')
ax.set_title('Latency CDF'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""))

nb.cells.append(new_code_cell("""\
# torch.profiler — which GPU ops take the most time?
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    profile_memory=True,
) as prof:
    for _ in range(10):
        model_gpu(x_gpu)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))"""))

nb.cells.append(new_code_cell("""\
# Final summary
print('=' * 60)
print(f'  {"Test":<30} {"ms":>7}  {"FPS":>6}  {"vs CPU":>7}')
print('  ' + '-'*52)
for label, times in results.items():
    lbl = label.replace('\\n', ' ')
    a   = times.mean()
    print(f'  {lbl:<30} {a:>7.1f}  {1000/a:>6.0f}  {avgs[0]/a:>6.1f}x')
print('=' * 60)"""))

# Write
with open('DLC_Deployment.ipynb', 'w') as f:
    nbformat.write(nb, f)
print(f"✓  {len(nb.cells)} cells")
