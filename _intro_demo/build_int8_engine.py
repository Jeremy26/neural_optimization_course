"""Build an INT8 TensorRT engine from SceneSeg_FP32.onnx with PTQ calibration.

Calibration data: ~50 frames from a folder of driving images (default:
./downtown/front_images_downtown/). The calibrator preprocesses them
identically to compare_runtimes.py (ImageNet normalization, 1×3×320×640).

Usage:
    python3 build_int8_engine.py \\
        --onnx SceneSeg_FP32.onnx \\
        --calib-dir downtown/front_images_downtown \\
        --out SceneSeg_int8.engine
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(bgr, H, W):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr)


class FolderCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_folder, cache_file, H, W, max_images=50):
        super().__init__()
        paths = sorted(Path(image_folder).glob("*.jpg"))
        if not paths:
            paths = sorted(Path(image_folder).glob("*.png"))
        if not paths:
            raise SystemExit(f"No .jpg/.png in {image_folder}")
        self.paths = paths[:max_images]
        self.cache_file = cache_file
        self.H, self.W = H, W
        self.batch_size = 1
        self.idx = 0
        # Pre-allocate device buffer for one batch
        self.nbytes = 1 * 3 * H * W * 4   # FP32
        self.d_in = cuda.mem_alloc(self.nbytes)
        print(f"Calibrator loaded {len(self.paths)} images "
              f"from {image_folder}")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx >= len(self.paths):
            return None
        bgr = cv2.imread(str(self.paths[self.idx]))
        if bgr is None:
            print(f"WARN: failed to read {self.paths[self.idx]}, skipping")
            self.idx += 1
            return self.get_batch(names)
        x = preprocess_bgr(bgr, self.H, self.W)
        cuda.memcpy_htod(self.d_in, x.tobytes())
        self.idx += 1
        if self.idx % 10 == 0 or self.idx == len(self.paths):
            print(f"  calibrating with image {self.idx}/{len(self.paths)}")
        return [int(self.d_in)]

    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            print(f"Reusing calibration cache: {self.cache_file}")
            return Path(self.cache_file).read_bytes()
        return None

    def write_calibration_cache(self, cache):
        Path(self.cache_file).write_bytes(cache)
        print(f"Wrote calibration cache: {self.cache_file}")


def build(onnx_path, calib_dir, cache_path, engine_path, H, W, workspace_mb):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("ONNX parse failed")

    input_name = network.get_input(0).name
    print(f"ONNX input tensor: {input_name}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_mb * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()
    shape = (1, 3, H, W)
    profile.set_shape(input_name, shape, shape, shape)
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)

    config.int8_calibrator = FolderCalibrator(calib_dir, cache_path, H, W)

    print("Building INT8 engine (this takes several minutes — calibration + autotune)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("Engine build returned None — check log above")

    Path(engine_path).write_bytes(serialized)
    print(f"Wrote: {engine_path} ({Path(engine_path).stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="SceneSeg_FP32.onnx")
    ap.add_argument("--calib-dir", default="downtown/front_images_downtown")
    ap.add_argument("--cache", default="int8_calib.cache")
    ap.add_argument("--out", default="SceneSeg_int8.engine")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--workspace-mb", type=int, default=4096)
    args = ap.parse_args()

    if not Path(args.onnx).exists():
        raise SystemExit(f"ONNX not found: {args.onnx}")
    if not Path(args.calib_dir).is_dir():
        raise SystemExit(f"Calibration folder not found: {args.calib_dir}")

    build(args.onnx, args.calib_dir, args.cache, args.out,
          args.height, args.width, args.workspace_mb)


if __name__ == "__main__":
    main()
