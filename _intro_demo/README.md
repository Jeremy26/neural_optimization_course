# Course Intro — Jetson Orin Before/After Demo

Internal tooling for the course intro video. **Not part of the student-facing course.**

The goal: 30-second teaser showing SceneSeg running two ways on a real Jetson Orin —
PyTorch eager (slow) vs TensorRT FP16 (fast) — over Waymo driving footage.

---

## What's in here

| File | Purpose |
|------|---------|
| `setup_orin.sh` | Lock the Orin into a reproducible benchmark state (power mode, clocks) |
| `build_engines.sh` | Build TRT engines (FP16 via `trtexec` + INT8 via `build_int8_engine.py`) |
| `build_int8_engine.py` | Standalone INT8 builder with PTQ calibration from a folder of images |
| `compare_pt_fp16.py` | Stage 1 — PyTorch FP32 vs PyTorch FP16 (eager `.half()`) |
| `compare_runtimes.py` | Stage 2 — PyTorch FP32 vs TensorRT FP16 |
| `compare_trt_int8.py` | Stage 3 — PyTorch FP32 vs TensorRT INT8 |
| `monitor_orin.sh` | Capture `tegrastats` while a demo runs (for power/temp overlays) |

---

## Prerequisites on the Orin

- JetPack 6.x with TensorRT 10.x and CUDA 12.x
- Python 3.10+ with `torch`, `torchvision`, `pycuda`, `tensorrt`, `opencv-python`, `numpy`, `tqdm`
- The SceneSeg files (already in `/content/models` from the Workshop):
  - `SceneSeg_traced.pt`
  - `SceneSeg_FP32.onnx`
- Waymo frames unzipped into a folder of `.jpg` files

Place all of these in `~/scene_seg_demo/` for the scripts below to find them.

---

## Workflow

```bash
# 1. Lock Orin into max-perf state (one-time per boot)
sudo ./setup_orin.sh

# 2. Build the TRT engines (~2 min on AGX Orin)
./build_engines.sh

# 3. (Optional) Start tegrastats logging in another terminal
./monitor_orin.sh

# 4. Record the top/bottom comparison MP4 (PyTorch GPU vs TRT FP16 GPU)
python3 compare_runtimes.py \
    --pt     SceneSeg_traced.pt \
    --engine SceneSeg_fp16.engine \
    --frames waymo_frames \
    --out    orin_demo.mp4 \
    --duration 30

# Or run live (loop mode is live-only, no MP4). Needs DISPLAY + GUI cv2:
python3 compare_runtimes.py \
    --pt     SceneSeg_traced.pt \
    --engine SceneSeg_fp16.engine \
    --frames waymo_frames \
    --display --loop
```

Output: top panel = PyTorch eager on the GPU (the "before"), bottom panel =
TensorRT FP16 on the GPU (the "after"). Each panel has a live FPS + latency
overlay.

---

## Why these specific Orin flags

Default Orin power mode throttles aggressively for thermal/power budget. For benchmarks
you want **MAXN power mode** (`nvpmodel -m 0`) and **clocks locked to max** (`jetson_clocks`).
This eliminates run-to-run variance — important for a video where you'll show stable FPS
numbers, not numbers that change every recording.

For the engine build, we use `--fp16` (not `--int8`) for the demo because:
- FP16 needs zero calibration (just a flag flip), works the same as the Workshop notebook
- The before/after is already dramatic at FP16 (5–10× over PyTorch eager)
- INT8 needs a proper calibrator on Orin — possible but extra work for a demo

If you later want INT8, see the comments in `build_engines.sh`.

---

## Notes for filming

- Run the demo a few times before recording — the **first run is always slowest** (CUDA init, kernel autotune).
- Don't record over a thermal throttle. Watch `tegrastats` — if `CPU@` or `GPU@` temps cross ~75°C, give it a break.
- The script writes a side-by-side MP4. You can also run `compare_runtimes.py` with `--display` to preview live on a connected screen instead of rendering to file (handy for a "live demo" framing).
