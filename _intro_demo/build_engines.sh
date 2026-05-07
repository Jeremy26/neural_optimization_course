#!/bin/bash
# Build TensorRT engines from SceneSeg_FP32.onnx on Jetson Orin.
# Output: SceneSeg_fp16.engine (and optionally _int8.engine)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/scene_seg_demo}"
ONNX="$MODELS_DIR/SceneSeg_FP32.onnx"

if [ ! -f "$ONNX" ]; then
    echo "ERROR: $ONNX not found"
    echo "Set MODELS_DIR or place SceneSeg_FP32.onnx in ~/scene_seg_demo/"
    exit 1
fi

echo "==> Building FP16 engine..."
# --fp16        : enable FP16 tensor cores (Orin has them on the GPU, not on DLA)
# --workspace   : 4GB scratch memory for the builder to try kernel variants (Orin has plenty)
# --shapes      : bind input shape; SceneSeg expects 1×3×320×640 — adjust if your ONNX differs
# --useCudaGraph: optional, can give a small extra speedup at inference
trtexec \
    --onnx="$ONNX" \
    --fp16 \
    --memPoolSize=workspace:4096 \
    --shapes=image:1x3x320x640 \
    --saveEngine="$MODELS_DIR/SceneSeg_fp16.engine" \
    --useCudaGraph

echo ""
echo "==> FP16 engine built: $MODELS_DIR/SceneSeg_fp16.engine"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL — INT8 build with calibration
# Needs a calibration cache file. Two ways to produce one:
#
# (a) trtexec auto-calibration (uses random data — accuracy will drop noticeably,
#     fine for a "look how fast" demo but not for production):
#
#   trtexec --onnx=$ONNX --int8 --memPoolSize=workspace:4096 \
#           --shapes=image:1x3x320x640 \
#           --saveEngine=$MODELS_DIR/SceneSeg_int8.engine
#
# (b) Proper PTQ calibration: write a Python calibrator that feeds 50–100 real
#     Waymo frames through the network, then call trtexec with --calib=cache.
#     See the Workshop notebook (Step 4 — TensorRT INT8) for the calibrator code,
#     ported to native TRT API for aarch64 / Jetson.
#
# For the demo video, FP16 is enough. INT8 is a "follow up" course topic.
# ─────────────────────────────────────────────────────────────────────────────

echo "==> Engine summary:"
ls -lh "$MODELS_DIR"/*.engine 2>/dev/null || true
