"""Top/bottom comparison of SceneSeg on Orin:
   TOP    — PyTorch FP32 (eager GPU) — the "before"
   BOTTOM — PyTorch FP16 (eager GPU, .half()) — the "after"

Shows the gain from precision reduction alone, without TensorRT.
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

# Stage 1 palette — red background, blue foreground, Nvidia green road
COLORS = np.array([
    [220,  70,  50],   # background — warm red
    [ 40, 100, 230],   # foreground (cars / pedestrians) — dominant blue
    [118, 185,   0],   # drivable road — Nvidia green
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


def to_segmentation_image(logits: np.ndarray, target_size: tuple) -> np.ndarray:
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


def try_open_window(title):
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
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="orin_pt_fp16.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch device: {device}")

    # Two copies of the same model — one FP32, one FP16
    pt_fp32 = torch.jit.load(args.pt, map_location=device).eval()
    pt_fp16 = torch.jit.load(args.pt, map_location=device).eval().half()
    print("Both models loaded (FP32 + FP16)")

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

    win_title = "Orin demo — PyTorch FP32 (top) vs PyTorch FP16 (bottom)"
    show_window = args.display and try_open_window(win_title)

    fp32_window = deque(maxlen=15)
    fp16_window = deque(maxlen=15)

    n_frames = None if args.loop else args.fps * args.duration

    print("Warming up...")
    x_np = preprocess_bgr(cv2.imread(str(frame_paths[0])), H, W)
    x_32 = torch.from_numpy(x_np).to(device)
    x_16 = x_32.half()
    for _ in range(5):
        with torch.no_grad():
            _ = pt_fp32(x_32).cpu().numpy()
            _ = pt_fp16(x_16).cpu().numpy()
    if device == "cuda":
        torch.cuda.synchronize()

    target = []
    if writer is not None: target.append(args.out)
    if show_window:        target.append("live window")
    print(f"Running {'forever' if n_frames is None else n_frames} frames -> "
          f"{', '.join(target) or '(no output sinks)'}.  Press 'q' to quit.")

    i = 0
    while n_frames is None or i < n_frames:
        bgr = cv2.imread(str(frame_paths[i % len(frame_paths)]))
        x_np = preprocess_bgr(bgr, H, W)
        x_32 = torch.from_numpy(x_np).to(device)
        x_16 = x_32.half()

        t0 = time.perf_counter()
        with torch.no_grad():
            fp32_logits = pt_fp32(x_32).cpu().numpy()
        if device == "cuda": torch.cuda.synchronize()
        fp32_ms = (time.perf_counter() - t0) * 1000
        fp32_window.append(fp32_ms)

        t0 = time.perf_counter()
        with torch.no_grad():
            fp16_logits = pt_fp16(x_16).float().cpu().numpy()
        if device == "cuda": torch.cuda.synchronize()
        fp16_ms = (time.perf_counter() - t0) * 1000
        fp16_window.append(fp16_ms)

        seg32 = to_segmentation_image(fp32_logits, (panel_w, panel_h))
        seg16 = to_segmentation_image(fp16_logits, (panel_w, panel_h))

        bg = cv2.resize(bgr, (panel_w, panel_h))
        top = cv2.addWeighted(bg, 0.4, seg32, 0.6, 0)
        bot = cv2.addWeighted(bg, 0.4, seg16, 0.6, 0)

        overlay_stats(top, "PyTorch FP32 (before)",
                      np.mean(fp32_window), 1000 / np.mean(fp32_window))
        overlay_stats(bot, "PyTorch FP16 (after)",
                      np.mean(fp16_window), 1000 / np.mean(fp16_window))

        gutter = np.zeros((4, panel_w, 3), dtype=np.uint8)
        composite = np.vstack([top, gutter, bot])

        if show_window:
            cv2.imshow(win_title, composite)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if writer is not None:
            writer.write(composite)

        if (i + 1) % args.fps == 0:
            total = "∞" if n_frames is None else n_frames
            print(f"  {i+1:4}/{total}  "
                  f"FP32 {np.mean(fp32_window):.1f} ms  "
                  f"FP16 {np.mean(fp16_window):.1f} ms  "
                  f"speedup {np.mean(fp32_window)/np.mean(fp16_window):.1f}x")
        i += 1

    print(f"\nDone. Final speedup: "
          f"{np.mean(fp32_window)/np.mean(fp16_window):.1f}x  "
          f"(FP32 {np.mean(fp32_window):.1f} ms  FP16 {np.mean(fp16_window):.1f} ms)")
    if writer is not None:
        writer.release()
        print(f"Saved: {args.out}")
    if show_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
