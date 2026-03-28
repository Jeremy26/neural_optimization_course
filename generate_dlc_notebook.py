#!/usr/bin/env python3
"""Generate DLC_Deployment.ipynb — SceneSeg, minimal."""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

# ── Title ─────────────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("""\
# Neural Network Deployment — PyTorch → ONNX → TensorRT
## Autoware SceneSeg · Waymo Frames · Real Benchmarks

| Test | Runtime | Device |
|------|---------|--------|
| Test 1 | PyTorch | CPU |
| Test 2 | PyTorch | GPU |
| Test 3 | ONNX Runtime | GPU |
| Test 4 | TensorRT FP16 | GPU |

> Colab T4 GPU required"""))

# ── 0. Setup ──────────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("---\n## 0. Setup"))

nb.cells.append(new_code_cell("""\
!pip install onnx onnxruntime-gpu gdown tensorrt pycuda -q"""))

nb.cells.append(new_code_cell("""\
import torch
import torchvision.transforms as T
import numpy as np
import time, os, glob
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import onnxruntime as ort

print(f"PyTorch {torch.__version__} | ORT {ort.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")"""))

# ── 1. Download ───────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("---\n## 1. Download Data & Models"))

nb.cells.append(new_code_cell("""\
!mkdir -p /content/data /content/models

# Waymo driving frames
!wget -qq https://optical-flow-data.s3.eu-west-3.amazonaws.com/waymo_images.zip -O /content/data/waymo.zip
!unzip -qq /content/data/waymo.zip -d /content/data/

# SceneSeg — PyTorch traced model
!gdown '1G2pKrjEGLGY1ouQdNPh11N-5LlmDI7ES' -O /content/models/SceneSeg_traced.pt

# SceneSeg — ONNX models (FP32 + others, pre-exported by Autoware)
!gdown -O /content/models/ 'https://docs.google.com/uc?export=download&id=1l-dniunvYyFKvLD7k16Png3AsVTuMl9f'
!gdown -O /content/models/ 'https://docs.google.com/uc?export=download&id=19gMPt_1z4eujo4jm5XKuH-8eafh-wJC6'
!gdown -O /content/models/ 'https://docs.google.com/uc?export=download&id=1zCworKw4aQ9_hDBkHfj1-sXitAAebl5Y'

for f in sorted(glob.glob('/content/models/*')):
    print(f"  {Path(f).name:<40} {os.path.getsize(f)/1e6:.1f} MB")"""))

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("""\
---
## 2. Step 1 — PyTorch

Load SceneSeg as a traced (TorchScript) model.
`torch.jit.load` — no class definition needed.

SceneSeg segments every pixel into 3 classes:
- 🔴 Background
- 🟣 Foreground objects (cars, pedestrians)
- 🟢 Drivable road"""))

nb.cells.append(new_code_cell("""\
# First, read the input size from the ONNX file
# (the traced model was frozen at the same dimensions)
onnx_files = sorted(glob.glob('/content/models/*.onnx'))
ONNX_PATH  = next(f for f in onnx_files if 'FP32' in f)

_sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
_inp  = _sess.get_inputs()[0]
H = int(_inp.shape[2]) if _inp.shape[2] else 256
W = int(_inp.shape[3]) if _inp.shape[3] else 512
print(f"SceneSeg input size: {H} × {W}")"""))

nb.cells.append(new_code_cell("""\
# Load the PyTorch traced model
model = torch.jit.load('/content/models/SceneSeg_traced.pt', map_location='cpu')
model.eval()
print("Model loaded")"""))

nb.cells.append(new_code_cell("""\
# Load a Waymo frame and preprocess it
frames = sorted(glob.glob('/content/data/**/*.jpg', recursive=True)
              + glob.glob('/content/data/**/*.png', recursive=True))

frame_bgr = cv2.imread(frames[0])
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

preprocess = T.Compose([
    T.Resize((H, W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
x    = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0)  # (1,3,H,W) CPU
x_np = x.numpy()

plt.imshow(cv2.resize(frame_rgb, (W, H))); plt.axis('off')
plt.title('Input: Waymo driving frame'); plt.show()"""))

nb.cells.append(new_code_cell("""\
# Run inference
with torch.no_grad():
    out = model(x)

if isinstance(out, (list, tuple)):
    out = out[0]   # SceneSeg returns (seg, ...) sometimes

# SceneSeg colour palette
COLORS = np.array([[255,93,61],[145,28,255],[220,255,0]], dtype=np.uint8)

seg = out.cpu().numpy()
class_map = np.argmax(seg[0], axis=0) if seg.shape[1] > 1 else (seg[0,0] > 0).astype(int)
class_map = np.clip(class_map, 0, 2)

overlay = cv2.addWeighted(
    cv2.resize(frame_bgr, (W, H)), 0.6,
    cv2.resize(COLORS[class_map], (W, H))[:, :, ::-1], 0.4, 0
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].imshow(cv2.resize(frame_rgb, (W, H))); axes[0].set_title('Input'); axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); axes[1].set_title('SceneSeg (PyTorch)'); axes[1].axis('off')
plt.tight_layout(); plt.show()"""))

nb.cells.append(new_code_cell("""\
def benchmark(fn, name, n_warmup=5, n_runs=50, cuda=False):
    with torch.no_grad():
        for _ in range(n_warmup): fn()
    if cuda: torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn()
            if cuda: torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    print(f"{name:.<55} {t.mean():6.1f} ms ± {t.std():.1f}  ({1000/t.mean():.0f} FPS)")
    return t

# Test 1 — PyTorch CPU  (1 run — CPU is just a reference)
with torch.no_grad():
    model(x)  # warmup
    t0 = time.perf_counter(); model(x)
pytorch_cpu = np.array([(time.perf_counter() - t0) * 1000])
print(f"Test 1 · PyTorch CPU{'.'*34} {pytorch_cpu[0]:.1f} ms  ({1000/pytorch_cpu[0]:.0f} FPS)")"""))

nb.cells.append(new_code_cell("""\
# Test 2 — PyTorch GPU
model_gpu = model.cuda()
x_gpu     = x.cuda()

pytorch_gpu = benchmark(lambda: model_gpu(x_gpu), 'Test 2 · PyTorch GPU', cuda=True)
print(f"  → GPU is {pytorch_cpu[0]/pytorch_gpu.mean():.1f}x faster than CPU")"""))

# ── 3. ONNX ───────────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("""\
---
## 3. Step 2 — ONNX Runtime

Autoware already exported SceneSeg to ONNX. We just load it.
Same input, same output — but faster."""))

nb.cells.append(new_code_cell("""\
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
sess      = ort.InferenceSession(ONNX_PATH, providers=providers)
inp_meta  = sess.get_inputs()[0]

print(f"Provider : {sess.get_providers()[0]}")
print(f"Input    : {inp_meta.name}  {inp_meta.shape}")"""))

nb.cells.append(new_code_cell("""\
ort_out = sess.run(None, {inp_meta.name: x_np})[0]

ort_map = np.argmax(ort_out[0], axis=0) if ort_out.shape[1] > 1 else (ort_out[0,0] > 0).astype(int)
ort_map = np.clip(ort_map, 0, 2)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].imshow(COLORS[class_map]); axes[0].set_title('PyTorch output'); axes[0].axis('off')
axes[1].imshow(COLORS[ort_map]);   axes[1].set_title(f'ONNX Runtime ({sess.get_providers()[0]})'); axes[1].axis('off')
plt.tight_layout(); plt.show()"""))

nb.cells.append(new_code_cell("""\
# Test 3 — ONNX Runtime GPU
onnx_gpu = benchmark(lambda: sess.run(None, {inp_meta.name: x_np}),
                     'Test 3 · ONNX Runtime GPU', cuda=True)
print(f"  → ORT GPU is {pytorch_gpu.mean()/onnx_gpu.mean():.2f}x vs PyTorch GPU")"""))

# ── 4. TensorRT ───────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("""\
---
## 4. Step 3 — TensorRT

TensorRT compiles the ONNX into a GPU engine optimised for your exact hardware.
This is what Autoware runs in production."""))

nb.cells.append(new_code_cell("""\
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_MAJOR  = int(trt.__version__.split('.')[0])
print(f"TensorRT {trt.__version__}")"""))

nb.cells.append(new_code_cell("""\
builder = trt.Builder(TRT_LOGGER)
network = (builder.create_network() if TRT_MAJOR >= 10
           else builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)))
parser  = trt.OnnxParser(network, TRT_LOGGER)

with open(ONNX_PATH, 'rb') as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
config.set_flag(trt.BuilderFlag.FP16)

print("Building engine (1-2 min)...")
engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(
    builder.build_serialized_network(network, config)
)
print("✓ Done")"""))

nb.cells.append(new_code_cell("""\
ctx = engine.create_execution_context()

if TRT_MAJOR >= 10:
    in_shape  = tuple(engine.get_tensor_shape(engine.get_tensor_name(0)))
    out_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
    in_name   = engine.get_tensor_name(0)
    out_name  = engine.get_tensor_name(1)
else:
    in_shape  = tuple(engine.get_binding_shape(0))
    out_shape = tuple(engine.get_binding_shape(1))

d_in   = cuda.mem_alloc(int(np.prod(in_shape))  * 4)
d_out  = cuda.mem_alloc(int(np.prod(out_shape)) * 4)
stream = cuda.Stream()

def trt_infer(x_np):
    inp = np.ascontiguousarray(x_np, dtype=np.float32)
    out = np.empty(out_shape, dtype=np.float32)
    cuda.memcpy_htod_async(d_in, inp, stream)
    if TRT_MAJOR >= 10:
        ctx.set_tensor_address(in_name,  int(d_in))
        ctx.set_tensor_address(out_name, int(d_out))
        ctx.execute_async_v3(stream_handle=stream.handle)
    else:
        ctx.execute_async_v2([int(d_in), int(d_out)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(out, d_out, stream)
    stream.synchronize()
    return out

trt_out = trt_infer(x_np)
trt_map = np.clip(np.argmax(trt_out[0], axis=0) if trt_out.shape[1] > 1 else (trt_out[0,0]>0).astype(int), 0, 2)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].imshow(COLORS[ort_map]); axes[0].set_title('ONNX Runtime'); axes[0].axis('off')
axes[1].imshow(COLORS[trt_map]); axes[1].set_title('TensorRT FP16'); axes[1].axis('off')
plt.tight_layout(); plt.show()"""))

nb.cells.append(new_code_cell("""\
# Test 4 — TensorRT GPU
trt_gpu = benchmark(lambda: trt_infer(x_np), 'Test 4 · TensorRT GPU (FP16)', cuda=True)
print(f"  → TRT is {pytorch_gpu.mean()/trt_gpu.mean():.1f}x vs PyTorch GPU")"""))

# ── 5. Results ────────────────────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("---\n## 5. Results"))

nb.cells.append(new_code_cell("""\
results = {
    'PyTorch\\nCPU':   pytorch_cpu,
    'PyTorch\\nGPU':   pytorch_gpu,
    'ONNX RT\\nGPU':   onnx_gpu,
    'TensorRT\\nFP16': trt_gpu,
}
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
labels = list(results.keys())
avgs   = [t.mean() for t in results.values()]
stds   = [t.std()  for t in results.values()]
p95s   = [np.percentile(t, 95) for t in results.values()]
xp     = np.arange(len(labels))"""))

nb.cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(xp, avgs, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor='k')
for i,(a,p) in enumerate(zip(avgs,p95s)):
    axes[0].plot(i, p, 'v', color='black', ms=7)
    axes[0].text(i, a+stds[i]+.5, f'{a:.1f}ms', ha='center', fontsize=8, fontweight='bold')
axes[0].set_xticks(xp); axes[0].set_xticklabels(labels)
axes[0].set_ylabel('ms'); axes[0].set_title('Latency  (▼ = P95)')

fps = [1000/a for a in avgs]
axes[1].bar(xp, fps, color=colors, alpha=0.85, edgecolor='k')
for i,f in enumerate(fps):
    axes[1].text(i, f+.3, f'{f:.0f}', ha='center', fontsize=9, fontweight='bold')
axes[1].set_xticks(xp); axes[1].set_xticklabels(labels)
axes[1].set_ylabel('FPS'); axes[1].set_title('Throughput')

spd = [avgs[0]/a for a in avgs]
axes[2].bar(xp, spd, color=colors, alpha=0.85, edgecolor='k')
axes[2].axhline(1, color='gray', linestyle='--')
for i,s in enumerate(spd):
    axes[2].text(i, s+.05, f'{s:.1f}x', ha='center', fontsize=9, fontweight='bold')
axes[2].set_xticks(xp); axes[2].set_xticklabels(labels)
axes[2].set_ylabel('Speedup vs CPU'); axes[2].set_title('Speedup')

plt.suptitle('SceneSeg Deployment Benchmark', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()"""))

nb.cells.append(new_code_cell("""\
# Violin — full distribution
fig, ax = plt.subplots(figsize=(12, 5))
parts = ax.violinplot(list(results.values()), positions=xp, showmedians=True)
for i,pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
for i,p95 in enumerate(p95s):
    ax.hlines(p95, i-.25, i+.25, colors='k', linestyles='--', linewidth=1.5)
    ax.text(i+.27, p95, 'P95', fontsize=7)
ax.set_xticks(xp); ax.set_xticklabels(labels)
ax.set_ylabel('ms'); ax.set_title('Latency Distribution')
plt.tight_layout(); plt.show()"""))

nb.cells.append(new_code_cell("""\
# CDF
fig, ax = plt.subplots(figsize=(12, 5))
for (label, times), color in zip(results.items(), colors):
    s = np.sort(times)
    ax.plot(s, np.arange(1,len(s)+1)/len(s)*100,
            label=label.replace('\\n',' '), color=color, linewidth=2)
ax.axhline(95, color='gray', linestyle='--')
ax.set_xlabel('Latency (ms)'); ax.set_ylabel('% of runs')
ax.set_title('CDF — % of frames faster than X ms')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()"""))

nb.cells.append(new_code_cell("""\
# torch.profiler — kernel breakdown
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
print('='*58)
print(f'  {"Test":<28} {"ms":>7}  {"FPS":>6}  {"vs CPU":>7}')
print('  '+'-'*50)
for label,times in results.items():
    a = times.mean()
    print(f'  {label.replace(chr(10)," "):<28} {a:>7.1f}  {1000/a:>6.0f}  {avgs[0]/a:>6.1f}x')
print('='*58)"""))

with open('DLC_Deployment.ipynb', 'w') as f:
    nbformat.write(nb, f)
print(f"✓ {len(nb.cells)} cells")
