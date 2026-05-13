"""Top/bottom comparison of SceneSeg on Orin:
   TOP    — PyTorch eager mode (GPU) — the "before"
   BOTTOM — TensorRT FP16 engine (GPU) — the "after"

Both consume the same driving frames, frame by frame.
Each panel shows running latency + FPS in an overlay.

Default output is an MP4. Pass --display to also open a live window.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)

# Class colors — matches the Workshop / Mini_ONNX
COLORS = np.array([
    [240,  40,  40],   # background
    [180,  60, 200],   # foreground (cars / pedestrians)
    [ 80, 200,  80],   # drivable road
], dtype=np.uint8)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(bgr: np.ndarray, H: int, W: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr)


def load_pytorch(pt_path: str, device: str):
    model = torch.jit.load(pt_path, map_location=device)
    model.eval()
    return model


def load_engine(engine_path: str):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


class TRTRunner:
    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.create_execution_context()
        self.in_name = engine.get_tensor_name(0)
        self.out_name = engine.get_tensor_name(1)
        self.in_shape = tuple(engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(engine.get_tensor_shape(self.out_name))
        self.d_in = cuda.mem_alloc(int(np.prod(self.in_shape)) * 4)
        self.d_out = cuda.mem_alloc(int(np.prod(self.out_shape)) * 4)
        self.stream = cuda.Stream()

    def __call__(self, x_np: np.ndarray) -> np.ndarray:
        out = np.empty(self.out_shape, dtype=np.float32)
        x = np.ascontiguousarray(x_np, dtype=np.float32)
        cuda.memcpy_htod_async(self.d_in, x, self.stream)
        self.ctx.set_tensor_address(self.in_name, int(self.d_in))
        self.ctx.set_tensor_address(self.out_name, int(self.d_out))
        self.ctx.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(out, self.d_out, self.stream)
        self.stream.synchronize()
        return out


def to_segmentation_image(logits: np.ndarray, target_size: tuple) -> np.ndarray:
    seg = np.argmax(logits[0], axis=0).astype(np.uint8)
    rgb = COLORS[seg]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, target_size, interpolation=cv2.INTER_NEAREST)


def overlay_stats(img: np.ndarray, label: str, latency_ms: float, fps: float):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, label, (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{latency_ms:5.1f} ms   {fps:5.1f} FPS",
                (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (90, 255, 90) if fps > 30 else (90, 180, 255),
                2, cv2.LINE_AA)


def try_open_window(title: str) -> bool:
    if not os.environ.get("DISPLAY") and sys.platform != "win32":
        print("DISPLAY not set — skipping live window.")
        return False
    try:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        return True
    except cv2.error as e:
        print(f"cv2 has no GUI backend ({e}). Skipping live window.")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="SceneSeg_traced.pt")
    ap.add_argument("--engine", required=True, help="Built TRT FP16 .engine")
    ap.add_argument("--frames", required=True, help="Folder of .jpg frames")
    ap.add_argument("--out", default="orin_demo.mp4",
                    help="MP4 output path (default: ./orin_demo.mp4)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--loop", action="store_true",
                    help="Run forever (live mode only, no MP4)")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--display", action="store_true",
                    help="Open a live OpenCV window in addition to writing MP4")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch device: {device}")

    pt_model = load_pytorch(args.pt, device)
    trt_runner = TRTRunner(load_engine(args.engine))
    print("Models loaded")

    frame_paths = sorted(Path(args.frames).glob("*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No .jpg frames in {args.frames}")
    print(f"{len(frame_paths)} frames available")

    H, W = args.height, args.width
    panel_w, panel_h = W, H
    out_w = panel_w
    out_h = panel_h * 2 + 4

    writer = None
    if args.out and not args.loop:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, args.fps, (out_w, out_h))
    elif args.loop and args.out:
        print("--loop ignores --out; no MP4 will be written.")

    win_title = "Orin demo — PyTorch (top) vs TensorRT FP16 (bottom)"
    show_window = args.display and try_open_window(win_title)

    pt_window = deque(maxlen=15)
    trt_window = deque(maxlen=15)

    n_frames = None if args.loop else args.fps * args.duration

    print("Warming up...")
    x_np = preprocess_bgr(cv2.imread(str(frame_paths[0])), H, W)
    x_pt = torch.from_numpy(x_np).to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = pt_model(x_pt).cpu().numpy()
        _ = trt_runner(x_np)
    if device == "cuda":
        torch.cuda.synchronize()

    target = []
    if writer is not None: target.append(args.out)
    if show_window:        target.append("live window")
    print(f"Running {'forever' if n_frames is None else n_frames} frames -> "
          f"{', '.join(target) or '(no output sinks)'}.  Press 'q' to quit.")

    i = 0
    while n_frames is None or i < n_frames:
        frame_path = frame_paths[i % len(frame_paths)]
        bgr = cv2.imread(str(frame_path))
        x_np = preprocess_bgr(bgr, H, W)
        x_pt = torch.from_numpy(x_np).to(device)

        # PyTorch eager
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_logits = pt_model(x_pt).cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        pt_ms = (time.perf_counter() - t0) * 1000
        pt_window.append(pt_ms)

        # TRT FP16
        t0 = time.perf_counter()
        trt_logits = trt_runner(x_np)
        trt_ms = (time.perf_counter() - t0) * 1000
        trt_window.append(trt_ms)

        pt_seg = to_segmentation_image(pt_logits, (panel_w, panel_h))
        trt_seg = to_segmentation_image(trt_logits, (panel_w, panel_h))

        bg = cv2.resize(bgr, (panel_w, panel_h))
        pt_panel = cv2.addWeighted(bg, 0.4, pt_seg, 0.6, 0)
        trt_panel = cv2.addWeighted(bg, 0.4, trt_seg, 0.6, 0)

        overlay_stats(pt_panel, "PyTorch (before)",
                      np.mean(pt_window), 1000 / np.mean(pt_window))
        overlay_stats(trt_panel, "TensorRT FP16 (after)",
                      np.mean(trt_window), 1000 / np.mean(trt_window))

        gutter = np.zeros((4, panel_w, 3), dtype=np.uint8)
        composite = np.vstack([pt_panel, gutter, trt_panel])

        if show_window:
            cv2.imshow(win_title, composite)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if writer is not None:
            writer.write(composite)

        if (i + 1) % args.fps == 0:
            total = "∞" if n_frames is None else n_frames
            print(f"  {i+1:4}/{total}  "
                  f"PT {np.mean(pt_window):.1f} ms  "
                  f"TRT {np.mean(trt_window):.1f} ms  "
                  f"speedup {np.mean(pt_window)/np.mean(trt_window):.1f}x")

        i += 1

    print(f"\nDone. Final speedup: "
          f"{np.mean(pt_window)/np.mean(trt_window):.1f}x  "
          f"(PT {np.mean(pt_window):.1f} ms  TRT {np.mean(trt_window):.1f} ms)")
    if writer is not None:
        writer.release()
        print(f"Saved: {args.out}")
    if show_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
