"""Live OAK-D demo: SceneSeg on Orin, vertical split.
   TOP    — PyTorch eager (GPU)
   BOTTOM — TensorRT FP16

Frames come from the OAK-D RGB camera via depthai. Press 'q' to quit.
Optional --record PATH writes the composite to MP4 while playing.
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

import depthai as dai


# Stage 2 palette (matches compare_runtimes.py)
COLORS = np.array([
    [160,  60, 200],   # background — purple
    [ 40, 100, 230],   # foreground (cars / pedestrians) — dominant blue
    [118, 185,   0],   # drivable road — Nvidia green
], dtype=np.uint8)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(bgr, H, W):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr)


def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(open(engine_path, "rb").read())


class TRTRunner:
    def __init__(self, engine):
        self.ctx = engine.create_execution_context()
        self.in_name = engine.get_tensor_name(0)
        self.out_name = engine.get_tensor_name(1)
        self.in_shape = tuple(engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(engine.get_tensor_shape(self.out_name))
        self.d_in = cuda.mem_alloc(int(np.prod(self.in_shape)) * 4)
        self.d_out = cuda.mem_alloc(int(np.prod(self.out_shape)) * 4)
        self.stream = cuda.Stream()

    def __call__(self, x_np):
        out = np.empty(self.out_shape, dtype=np.float32)
        x = np.ascontiguousarray(x_np, dtype=np.float32)
        cuda.memcpy_htod_async(self.d_in, x, self.stream)
        self.ctx.set_tensor_address(self.in_name, int(self.d_in))
        self.ctx.set_tensor_address(self.out_name, int(self.d_out))
        self.ctx.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(out, self.d_out, self.stream)
        self.stream.synchronize()
        return out


def to_seg_image(logits, target_size):
    seg = np.argmax(logits[0], axis=0).astype(np.uint8)
    rgb = COLORS[seg]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, target_size, interpolation=cv2.INTER_NEAREST)


def overlay_stats(img, label, latency_ms, fps):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, label, (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{latency_ms:5.1f} ms   {fps:5.1f} FPS",
                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (90, 255, 90) if fps > 30 else (90, 180, 255),
                2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="SceneSeg_traced.pt")
    ap.add_argument("--engine", required=True, help="Built TRT FP16 .engine")
    ap.add_argument("--record", default=None,
                    help="Optional MP4 path; if set, also record while playing")
    ap.add_argument("--cam-fps", type=int, default=30,
                    help="OAK-D requested capture FPS (default 30)")
    ap.add_argument("--cam-w", type=int, default=1280, help="Camera output width")
    ap.add_argument("--cam-h", type=int, default=720,  help="Camera output height")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch device: {device}")
    pt_model = torch.jit.load(args.pt, map_location=device).eval()
    trt_runner = TRTRunner(load_engine(args.engine))
    print("Models loaded")

    H, W = args.height, args.width
    panel_w, panel_h = W, H
    out_w = panel_w
    out_h = panel_h * 2 + 4

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, args.cam_fps, (out_w, out_h))

    win_title = "OAK-D live — PyTorch (top) vs TensorRT FP16 (bottom)"
    if not os.environ.get("DISPLAY") and sys.platform != "win32":
        raise SystemExit("DISPLAY not set — live demo needs a desktop session.")
    cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)

    pt_window = deque(maxlen=15)
    trt_window = deque(maxlen=15)

    print("Connecting to OAK-D...")
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput((args.cam_w, args.cam_h),
                                    dai.ImgFrame.Type.BGR888i,
                                    fps=args.cam_fps)
        q = cam_out.createOutputQueue()
        pipeline.start()

        print("Waiting for first frame...")
        first = q.get().getCvFrame()
        x_np = preprocess_bgr(first, H, W)
        x_pt = torch.from_numpy(x_np).to(device)
        for _ in range(3):
            with torch.no_grad():
                _ = pt_model(x_pt).cpu().numpy()
            _ = trt_runner(x_np)
        if device == "cuda":
            torch.cuda.synchronize()
        print("Running. Press 'q' to quit.")

        i = 0
        while pipeline.isRunning():
            bgr = q.get().getCvFrame()
            x_np = preprocess_bgr(bgr, H, W)
            x_pt = torch.from_numpy(x_np).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                pt_logits = pt_model(x_pt).cpu().numpy()
            if device == "cuda": torch.cuda.synchronize()
            pt_window.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            trt_logits = trt_runner(x_np)
            trt_window.append((time.perf_counter() - t0) * 1000)

            pt_seg  = to_seg_image(pt_logits,  (panel_w, panel_h))
            trt_seg = to_seg_image(trt_logits, (panel_w, panel_h))
            bg = cv2.resize(bgr, (panel_w, panel_h))
            top = cv2.addWeighted(bg, 0.4, pt_seg, 0.6, 0)
            bot = cv2.addWeighted(bg, 0.4, trt_seg, 0.6, 0)

            overlay_stats(top, "PyTorch (before)",
                          np.mean(pt_window), 1000 / np.mean(pt_window))
            overlay_stats(bot, "TensorRT FP16 (after)",
                          np.mean(trt_window), 1000 / np.mean(trt_window))

            gutter = np.zeros((4, panel_w, 3), dtype=np.uint8)
            composite = np.vstack([top, gutter, bot])

            cv2.imshow(win_title, composite)
            if writer is not None:
                writer.write(composite)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if (i + 1) % 30 == 0:
                print(f"  PT {np.mean(pt_window):.1f} ms  "
                      f"TRT {np.mean(trt_window):.1f} ms  "
                      f"speedup {np.mean(pt_window)/np.mean(trt_window):.1f}x")
            i += 1

    if writer is not None:
        writer.release()
        print(f"Saved: {args.record}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
