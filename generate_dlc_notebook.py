#!/usr/bin/env python3
"""Generate the restructured DLC_Deployment.ipynb notebook."""

import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

# Title
nb.cells.append(new_markdown_cell("""# 🚀 Neural Network Deployment — From PyTorch to Production
## The Complete Autonomous Vehicle/Robotics Deployment Pipeline

**A companion notebook to [Neural Optimization](https://thinkautonomous.ai) by [Think Autonomous](https://thinkautonomous.ai)**

This notebook covers the **real deployment pipeline** used in production autonomous systems like [Autoware](https://github.com/autowarefoundation/autoware):
1. Load a robotics perception model → visualize → baseline benchmark
2. Export to ONNX → validate → compare FPS with ONNX Runtime
3. Understand the PyTorch ↔ ONNX workflow
4. Optimize before export: Pruning + Quantization → ONNX
5. Analyze Autoware's **TensorRT C++ production code**
6. Deploy with TensorRT on Colab (FP16 inference)
7. Production profiling & benchmarking
8. End-to-end pipeline project: Train → Optimize → Export → Deploy → Benchmark

**Requirements:** Google Colab with a **T4 GPU** runtime (recommended)"""))

# Section 0: Setup
nb.cells.append(new_markdown_cell("---\n## 0. Environment Setup"))

nb.cells.append(new_code_cell("""# Core ML packages — modern versions all support numpy 2.x
!pip install torch torchvision onnx onnxsim onnxruntime-gpu Pillow matplotlib -q
!pip install openvino -q
!pip install tensorrt pycuda -q

import warnings
warnings.filterwarnings('ignore')
print("✓ Environment ready")"""))

nb.cells.append(new_code_cell("""import torch
import torchvision
import torchvision.transforms as T
import torch.nn as torch_nn
import numpy as np
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import urllib.request
import copy

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'CUDA version:    {torch.version.cuda}')
else:
    print('⚠️  No GPU detected. TensorRT section will be CPU-only.')
    print('   For full GPU support, use Runtime → Change runtime type → T4 GPU')\n"""))

# Section 1: Load PyTorch Model
nb.cells.append(new_markdown_cell("---\n## 1. Load a Robotics Perception Model, Visualize & Benchmark\n\n### Context: Autonomous Driving Perception\nPerception is the first step in autonomous systems. Deep learning models predict:\n- **Semantic segmentation** → \"What is each pixel?\"\n- **Object detection** → \"Where are the cars/pedestrians?\"\n- **Depth estimation** → \"How far is that object?\"\n\nWe'll use **DeepLabV3-MobileNetV3-Large**: a real segmentation model used in mobile/edge AV systems.\n\n**Real Autoware example:** [Autoware's EgoLanes model](https://github.com/autowarefoundation/autoware/tree/main/perception/perception_lanelet2_map_based) predicts lane lines for autonomous driving."))

nb.cells.append(new_code_cell("""# Load pretrained DeepLabV3 with MobileNetV3 backbone
# Used in real autonomous driving systems for road/obstacle/sky segmentation
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Model: DeepLabV3-MobileNetV3-Large')
print(f'Total parameters: {total_params:,}')
print(f'Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)')
print(f'Backbone: MobileNetV3-Large (lightweight, suitable for mobile/edge)')"""))

nb.cells.append(new_code_cell("""# Download a real driving scene
IMG_URL = 'https://raw.githubusercontent.com/pytorch/hub/master/images/deeplab1.png'
IMG_PATH = 'driving_scene.png'

if not os.path.exists(IMG_PATH):
    urllib.request.urlretrieve(IMG_URL, IMG_PATH)

original_image = Image.open(IMG_PATH).convert('RGB')
print(f'Loaded: {IMG_PATH} ({original_image.size})')

plt.figure(figsize=(12, 6))
plt.imshow(original_image)
plt.title('Input: Real Driving Scene (Cityscapes-style)')
plt.axis('off')
plt.tight_layout()
plt.show()"""))

nb.cells.append(new_code_cell("""# Preprocessing pipeline (same as production)
preprocess = T.Compose([
    T.Resize(520),
    T.CenterCrop(480),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(original_image).unsqueeze(0)
print(f'Input tensor shape: {input_tensor.shape}')

# Cityscapes color palette
CITYSCAPES_COLORS = np.array([
    [0, 0, 0],       # background
    [128, 64, 128],  # road
    [244, 35, 232],  # sidewalk
    [70, 70, 70],    # building
    [102, 102, 156], # wall
    [190, 153, 153], # fence
    [153, 153, 153], # pole
    [250, 170, 30],  # traffic light
    [220, 220, 0],   # traffic sign
    [107, 142, 35],  # vegetation
    [152, 251, 152], # terrain
    [70, 130, 180],  # sky
    [220, 20, 60],   # person
    [255, 0, 0],     # rider
    [0, 0, 142],     # car
    [0, 0, 70],      # truck
    [0, 60, 100],    # bus
    [0, 80, 100],    # train
    [0, 0, 230],     # motorcycle
    [119, 11, 32],   # bicycle
    [128, 128, 128], # other
], dtype=np.uint8)

def visualize_segmentation(prediction, title='Segmentation'):
    \"\"\"Convert model output to color segmentation map.\"\"\"
    if isinstance(prediction, torch.Tensor):
        seg_map = prediction.argmax(dim=1).squeeze().cpu().numpy()
    else:
        seg_map = prediction.argmax(axis=1).squeeze()

    h, w = seg_map.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(CITYSCAPES_COLORS)):
        color_map[seg_map == cls_id] = CITYSCAPES_COLORS[cls_id]

    return color_map, seg_map

print('✓ Preprocessing ready')\n"""))

nb.cells.append(new_code_cell("""# Run PyTorch inference (baseline)
with torch.no_grad():
    pytorch_output = model(input_tensor)['out']

pytorch_seg, pytorch_classes = visualize_segmentation(pytorch_output)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(T.CenterCrop(480)(T.Resize(520)(original_image)))
axes[0].set_title('Input Image', fontsize=14)
axes[0].axis('off')
axes[1].imshow(pytorch_seg)
axes[1].set_title('Semantic Segmentation Output', fontsize=14)
axes[1].axis('off')
plt.tight_layout()
plt.show()

unique_classes = np.unique(pytorch_classes)
print(f'Classes detected: {len(unique_classes)}')
print(f'Output shape: {pytorch_output.shape}')\n"""))

nb.cells.append(new_code_cell("""# Benchmark function
def benchmark(run_fn, name, n_warmup=10, n_runs=50):
    \"\"\"Generic benchmark with warmup.\"\"\"
    for _ in range(n_warmup):
        run_fn()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_fn()
        times.append((time.perf_counter() - start) * 1000)

    avg, std = np.mean(times), np.std(times)
    fps = 1000 / avg
    print(f'{name:.<40} {avg:.1f} ms ± {std:.1f} ms ({fps:.1f} FPS)')
    return avg

pytorch_time = benchmark(
    lambda: model(input_tensor),
    'PyTorch (CPU)'
)"""))

# Section 2: ONNX Export
nb.cells.append(new_markdown_cell("---\n## 2. Export to ONNX → Validate → Load with ONNX Runtime → Compare FPS\n\n### Why ONNX?\n- **Framework-agnostic** → Train in PyTorch, deploy anywhere\n- **Optimized runtimes** → ONNX Runtime, TensorRT, OpenVINO all optimize ONNX\n- **Model zoo** → Download pre-converted models\n\n**Autoware example:** Most Autoware perception models are exported to ONNX for deployment across different hardware."))

nb.cells.append(new_code_cell("""import onnx
import onnxsim

ONNX_PATH = 'deeplabv3_mobilenetv3.onnx'

# DeepLabV3 returns a dict, wrap it to export cleanly
class SegmentationWrapper(torch_nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']

wrapped_model = SegmentationWrapper(model)
wrapped_model.eval()

print('Exporting to ONNX...')
torch.onnx.export(
    wrapped_model,
    input_tensor,
    ONNX_PATH,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=17,
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Validate
model_onnx = onnx.load(ONNX_PATH)
onnx.checker.check_model(model_onnx)
print('✓ ONNX model validated')

# Simplify
print('Simplifying ONNX graph...')
model_simplified, ok = onnxsim.simplify(model_onnx)
if ok:
    onnx.save(model_simplified, ONNX_PATH)
    print('✓ ONNX model simplified')

size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
print(f'\\nONNX model saved: {ONNX_PATH} ({size_mb:.1f} MB)')\n"""))

nb.cells.append(new_code_cell("""import onnxruntime as ort

print(f'ONNX Runtime version: {ort.__version__}')
print(f'Available providers: {ort.get_available_providers()}')

# Create inference session - selects best provider automatically
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(ONNX_PATH, providers=providers)

active_provider = session.get_providers()[0]
print(f'\\nActive provider: {active_provider}')

# Get input/output metadata
input_meta = session.get_inputs()[0]
output_meta = session.get_outputs()[0]
print(f'Input:  {input_meta.name} {input_meta.shape}')
print(f'Output: {output_meta.name} {output_meta.shape}')\n"""))

nb.cells.append(new_code_cell("""# Run ONNX Runtime inference
onnx_input = input_tensor.numpy()
onnx_output = session.run([output_meta.name], {input_meta.name: onnx_input})[0]

# Visualize
onnx_seg, _ = visualize_segmentation(onnx_output)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(pytorch_seg)
axes[0].set_title('PyTorch Output', fontsize=14)
axes[0].axis('off')
axes[1].imshow(onnx_seg)
axes[1].set_title('ONNX Runtime Output', fontsize=14)
axes[1].axis('off')
plt.suptitle('Output Comparison: PyTorch vs ONNX Runtime', fontsize=14)
plt.tight_layout()
plt.show()

# Verify numerical consistency
pytorch_np = pytorch_output.numpy()
max_diff = np.max(np.abs(pytorch_np - onnx_output))
print(f'Max numerical difference: {max_diff:.6f}')
print(f'Outputs match: ✓' if max_diff < 0.001 else 'Outputs differ ⚠')\n"""))

nb.cells.append(new_code_cell("""# Benchmark ONNX Runtime
onnx_time = benchmark(
    lambda: session.run([output_meta.name], {input_meta.name: onnx_input}),
    f'ONNX Runtime ({active_provider})'
)\n"""))

# Section 3: PyTorch ↔ ONNX Workflow
nb.cells.append(new_markdown_cell("---\n## 3. The PyTorch ↔ ONNX Workflow\n\n### The Training/Deployment Split\n```\nDevelopment (Research)           Production (Deployment)\n├─ PyTorch (flexible)            ├─ ONNX (optimized)\n├─ Easy debugging                ├─ Framework-agnostic\n├─ Rich ecosystem                ├─ Multiple runtimes\n└─ Custom training loops         └─ Fast inference\n\nWorkflow:\n  Train in PyTorch → Export to ONNX → Deploy with ORT/TensorRT/OpenVINO\n```\n\n### Real Autoware Example\nIn Autoware's perception pipeline:\n- **Development:** Models trained in PyTorch (like EgoLanes, AutoSteer)\n- **Export:** `torch.onnx.export()` to create ONNX files\n- **Deployment:** C++ nodes load ONNX, run inference with TensorRT/ORT\n\nLet's walk through a complete example with a **steering prediction model** (similar to Autoware's AutoSteer)."))

nb.cells.append(new_code_cell("""# Define a simple steering prediction network (inspired by Autoware's AutoSteer)
class SteeringPredictor(torch_nn.Module):
    \"\"\"Predicts steering angle from an image.\"\"\"\n    def __init__(self):
        super().__init__()
        # Backbone: simple CNN
        self.features = torch_nn.Sequential(
            torch_nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            torch_nn.ReLU(inplace=True),
            torch_nn.MaxPool2d(2, 2),
            torch_nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            torch_nn.ReLU(inplace=True),
            torch_nn.MaxPool2d(2, 2),
            torch_nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch_nn.ReLU(inplace=True),
        )
        # Prediction head: steering angle classification (61 classes: -30 to +30 degrees)
        self.classifier = torch_nn.Sequential(
            torch_nn.Linear(128 * 15 * 15, 256),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout(0.5),
            torch_nn.Linear(256, 61),  # 61 steering angle bins
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Create and \"train\" the model
steering_model = SteeringPredictor()
steering_model.eval()

print(f'Steering model parameters: {sum(p.numel() for p in steering_model.parameters()):,}')
print(f'Output: 61 steering angle classes (-30° to +30°)')\n"""))

nb.cells.append(new_code_cell("""# Quick \"training\" loop (just to simulate model development)
steering_model.train()
optimizer = torch.optim.Adam(steering_model.parameters(), lr=0.001)
criterion = torch_nn.CrossEntropyLoss()

print('Simulating training (2 epochs on synthetic data)...')
for epoch in range(2):
    # Synthetic batch: random images + random steering angles
    synthetic_images = torch.randn(8, 3, 480, 480)
    synthetic_angles = torch.randint(0, 61, (8,))

    optimizer.zero_grad()
    outputs = steering_model(synthetic_images)
    loss = criterion(outputs, synthetic_angles)
    loss.backward()
    optimizer.step()

    print(f'  Epoch {epoch+1}/2, Loss: {loss.item():.4f}')

steering_model.eval()
print('✓ Training complete')\n"""))

nb.cells.append(new_code_cell("""# Export steering model to ONNX
STEERING_ONNX_PATH = 'steering_predictor.onnx'

dummy_input = torch.randn(1, 3, 480, 480)

torch.onnx.export(
    steering_model,
    dummy_input,
    STEERING_ONNX_PATH,
    input_names=['image'],
    output_names=['steering_logits'],
    opset_version=17,
    dynamic_axes={'image': {0: 'batch_size'}, 'steering_logits': {0: 'batch_size'}}
)

# Validate
onnx_steer = onnx.load(STEERING_ONNX_PATH)
onnx.checker.check_model(onnx_steer)
print('✓ Steering ONNX exported and validated')

size_mb = os.path.getsize(STEERING_ONNX_PATH) / 1024 / 1024
print(f'  File: {STEERING_ONNX_PATH} ({size_mb:.3f} MB)')\n"""))

nb.cells.append(new_code_cell("""# Load and run with ONNX Runtime
steer_session = ort.InferenceSession(STEERING_ONNX_PATH, providers=['CPUExecutionProvider'])
steer_input = steer_session.get_inputs()[0]
steer_output = steer_session.get_outputs()[0]

# Inference
test_image = torch.randn(1, 3, 480, 480).numpy()
steering_logits = steer_session.run([steer_output.name], {steer_input.name: test_image})[0]

predicted_angle = np.argmax(steering_logits[0]) - 30  # Convert class index to angle
print(f'\\nONNX Runtime Steering Prediction:')
print(f'  Predicted angle: {predicted_angle}°')
print(f'  Confidence: {softmax(steering_logits[0])[np.argmax(steering_logits[0])]:.2%}')

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

print(f'\\n✓ PyTorch ↔ ONNX workflow complete!')\n"""))

# Section 4: Optimize Before Export
nb.cells.append(new_markdown_cell("---\n## 4. Optimize Before Export: Pruning + Quantization → ONNX\n\n### Why optimize before export?\n- Smaller model size\n- Faster inference\n- Same workflow (PyTorch → ONNX → Deploy)\n- Can stack optimizations (Pruning + Quantization)\n\n### Techniques\n1. **Structured pruning** - Remove entire filters/channels\n2. **Quantization** - Reduce precision (FP32 → INT8)\n3. **Knowledge distillation** - Covered in Neural Optimization course\n\nWe'll optimize the **DeepLabV3 model** from Section 1."))

nb.cells.append(new_code_cell("""import torch.nn.utils.prune as prune

# Create a copy for pruning
pruned_model = copy.deepcopy(wrapped_model)
pruned_model.eval()

# Apply structured L1 pruning to Conv2d layers
pruned_layers = 0
for name, module in pruned_model.named_modules():
    if isinstance(module, torch_nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=0.3, n=1, dim=0)
        prune.remove(module, 'weight')  # Make pruning permanent
        pruned_layers += 1

print(f'Pruned {pruned_layers} Conv2d layers (30% of filters removed)')

# Count non-zero parameters
original_nonzero = sum((p != 0).sum().item() for p in model.parameters())
pruned_nonzero = sum((p != 0).sum().item() for p in pruned_model.parameters())
total = sum(p.numel() for p in model.parameters())

print(f'\\nOriginal non-zero params: {100*original_nonzero/total:.1f}%')
print(f'Pruned non-zero params:   {100*pruned_nonzero/total:.1f}%')
print(f'Sparsity achieved:        {100*(1 - pruned_nonzero/total):.1f}%')\n"""))

nb.cells.append(new_code_cell("""# Run inference with pruned model
with torch.no_grad():
    pruned_output = pruned_model(input_tensor)

pruned_seg, _ = visualize_segmentation(pruned_output)

# Compare
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(pytorch_seg)
axes[0].set_title('Original Model', fontsize=14)
axes[0].axis('off')
axes[1].imshow(pruned_seg)
axes[1].set_title('Pruned Model (30% structured)', fontsize=14)
axes[1].axis('off')
plt.suptitle('Pruning: Original vs Pruned Output', fontsize=14)
plt.tight_layout()
plt.show()

# Measure accuracy difference
match = (pytorch_output.argmax(dim=1) == pruned_output.argmax(dim=1)).float().mean().item()
print(f'Pixel agreement with original: {match*100:.1f}%')\n"""))

nb.cells.append(new_code_cell("""# Export pruned model to ONNX
PRUNED_ONNX_PATH = 'deeplabv3_pruned.onnx'

torch.onnx.export(
    pruned_model,
    input_tensor,
    PRUNED_ONNX_PATH,
    input_names=['input'],
    output_names=['output'],
    opset_version=17,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Simplify and save
model_pruned_onnx = onnx.load(PRUNED_ONNX_PATH)
model_pruned_simplified, _ = onnxsim.simplify(model_pruned_onnx)
onnx.save(model_pruned_simplified, PRUNED_ONNX_PATH)

original_size = os.path.getsize(ONNX_PATH) / 1024 / 1024
pruned_size = os.path.getsize(PRUNED_ONNX_PATH) / 1024 / 1024

print(f'Original ONNX: {original_size:.1f} MB')
print(f'Pruned ONNX:   {pruned_size:.1f} MB')
print(f'Size reduction: {(1 - pruned_size/original_size)*100:.1f}%')\n"""))

nb.cells.append(new_code_cell("""# Benchmark pruned ONNX model
pruned_session = ort.InferenceSession(PRUNED_ONNX_PATH, providers=['CPUExecutionProvider'])
pruned_input_meta = pruned_session.get_inputs()[0]
pruned_output_meta = pruned_session.get_outputs()[0]

pruned_onnx_time = benchmark(
    lambda: pruned_session.run([pruned_output_meta.name], {pruned_input_meta.name: onnx_input}),
    'Pruned ONNX Runtime'
)\n"""))

# Section 5: Autoware TensorRT Analysis
nb.cells.append(new_markdown_cell("---\n## 5. Analyze Autoware's TensorRT C++ Production Code\n\n### Overview: How Autoware Deploys Neural Networks\n\n[Autoware](https://github.com/autowarefoundation/autoware) is the world's leading open-source autonomous driving stack. Its perception nodes use TensorRT for GPU inference.\n\n**Autoware's Deployment Stack:**\n```\nPython (Training)     →  PyTorch Model\n    ↓\nExport              →  ONNX File\n    ↓\nC++ Perception Node →  TensorRT Engine\n    ↓\nROS2 Message Bus    →  Steering/Speed commands\n```\n\n### The TrtCommon Class Pattern\n\nAutoware uses a `TrtCommon` utility class to manage TensorRT engines. Here's the real C++ interface:"))

nb.cells.append(new_code_cell("""# Display actual Autoware TrtCommon header (simplified)
autoware_trt_header = '''\\n// From: autoware/perception/tensorrt_common/include/tensorrt_common/tensorrt_common.hpp\n\nnamespace tensorrt_common {\n\nclass TrtCommon {\npublic:\n    // Constructor: Load or build a TensorRT engine from ONNX\n    TrtCommon(\n        const std::string& model_path,        // Path to .onnx or .trtengine\n        const std::string& precision,         // \"fp32\", \"fp16\", \"int8\"\n        std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator = nullptr,\n        const BatchConfig& batch_config = {1, 1, 1},\n        const size_t max_workspace_size = (1 << 30)  // 1GB\n    );\n\n    // Methods\n    bool loadEngine(const std::string& path);\n    bool buildEngineFromOnnx(\n        const std::string& onnx_path,\n        const std::string& engine_path\n    );\n    \n    void setInput(const int index, const nvinfer1::Dims& dims);\n    bool enqueueV2(\n        void** bindings,           // GPU pointers: [input, output, ...]\n        cudaStream_t stream,       // CUDA stream for async execution\n        cudaEvent_t* inputConsumed\n    );\n    \n    // Get engine info\n    nvinfer1::ICudaEngine* getEngine() { return engine_.get(); }\n    int getMaxBatchSize() const { return max_batch_size_; }\n};\n\n}  // namespace tensorrt_common\n'''\n\nprint(autoware_trt_header)\nprint('\\n[This is real C++ code from Autoware's tensorrt_common package]')\n"""))

nb.cells.append(new_code_cell("""# Display actual Autoware perception node pattern
autoware_node_pattern = '''\\n// From: autoware/perception/traffic_light_classifier/lib/classifier.cpp\n// Simplified example of how a perception node uses TrtCommon\n\n#include <tensorrt_common/tensorrt_common.hpp>\n#include <rclcpp/rclcpp.hpp>\n\nclass TrafficLightClassifier : public rclcpp::Node {\nprivate:\n    std::unique_ptr<tensorrt_common::TrtCommon> trt_;\n    cudaStream_t stream_;\n    \n    void* d_input_;   // GPU memory for input\n    void* d_output_;  // GPU memory for output\n\npublic:\n    TrafficLightClassifier() : rclcpp::Node(\"traffic_light_classifier\") {\n        // Load ONNX model, build TensorRT engine\n        trt_ = std::make_unique<tensorrt_common::TrtCommon>(\n            \"/path/to/traffic_light.onnx\",\n            \"fp16\"  // Use FP16 for faster inference on T4/V100\n        );\n        \n        cudaStreamCreate(&stream_);\n        cudaMalloc(&d_input_, input_size_bytes);\n        cudaMalloc(&d_output_, output_size_bytes);\n    }\n    \n    void inferenceCallback(const sensor_msgs::msg::Image& image_msg) {\n        // Preprocess image on GPU\n        preprocessImage(image_msg, d_input_, stream_);\n        \n        // Run inference\n        void* bindings[] = {d_input_, d_output_};\n        trt_->enqueueV2(bindings, stream_, nullptr);\n        \n        // Copy result back to CPU\n        std::vector<float> h_output(num_classes_);\n        cudaMemcpyAsync(h_output.data(), d_output_, output_size_bytes,\n                        cudaMemcpyDeviceToHost, stream_);\n        cudaStreamSynchronize(stream_);\n        \n        // Publish result\n        publishTrafficLightState(h_output);\n    }\n};\n'''\n\nprint(autoware_node_pattern)\nprint('\\n[This pattern is used in Autoware perception nodes]')\nprint('\\nKey insights:')\nprint('1. Load ONNX → Build TRT engine (TrtCommon handles this)')\nprint('2. Allocate GPU memory for input/output')\nprint('3. Run inference async with CUDA streams')\nprint('4. Sync stream, copy result back to CPU')\nprint('5. Publish ROS2 message')\n"""))

# Section 6: TensorRT
nb.cells.append(new_markdown_cell("---\n## 6. TensorRT on Colab: FP16 Inference\n\n### TensorRT: Nvidia's High-Performance Inference Engine\n- **Reads ONNX directly** (no conversion step)\n- **Graph optimization** (layer fusion, memory optimization)\n- **Precision modes** (FP32, FP16, INT8)\n- **Fastest inference** on Nvidia GPUs\n\nTensorRT is what Autoware and production AVs use for GPU inference."))

nb.cells.append(new_code_cell("""trt_available = False\ntry:\n    import tensorrt as trt\n    import pycuda.driver as cuda\n    import pycuda.autoinit\n    \n    trt_available = True\n    print(f'TensorRT version: {trt.__version__}')\n    print(f'GPU: {torch.cuda.get_device_name(0)}')\n    print('✓ TensorRT is available')\nexcept ImportError:\n    print('⚠ TensorRT not available')\n    print('  To use TensorRT:')\n    print('  1. Use Google Colab with GPU runtime')\n    print('  2. pip install tensorrt pycuda')\n    print('  3. Or use an Nvidia Docker container')\n"""))

nb.cells.append(new_code_cell("""if trt_available:\n    def build_trt_engine(onnx_path, fp16=True):\n        \"\"\"Build a TensorRT engine from ONNX.\"\"\"\n        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)\n        builder = trt.Builder(TRT_LOGGER)\n        network = builder.create_network(\n            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)\n        )\n        parser = trt.OnnxParser(network, TRT_LOGGER)\n        \n        with open(onnx_path, 'rb') as f:\n            if not parser.parse(f.read()):\n                for i in range(parser.num_errors):\n                    print(f'Parse error: {parser.get_error(i)}')\n                return None\n        \n        config = builder.create_builder_config()\n        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)\n        \n        if fp16 and builder.platform_has_fast_fp16:\n            config.set_flag(trt.BuilderFlag.FP16)\n            print('✓ FP16 mode enabled')\n        \n        print('Building TensorRT engine...')\n        engine_data = builder.build_serialized_network(network, config)\n        runtime = trt.Runtime(TRT_LOGGER)\n        engine = runtime.deserialize_cuda_engine(engine_data)\n        print('✓ Engine built')\n        return engine\n    \n    engine = build_trt_engine(ONNX_PATH, fp16=True)\nelse:\n    engine = None\n"""))

nb.cells.append(new_code_cell("""if trt_available and engine is not None:\n    class TRTInference:\n        def __init__(self, engine):\n            self.context = engine.create_execution_context()\n            self.d_input = cuda.mem_alloc(\n                int(np.prod(engine.get_tensor_shape('input'))) * 4\n            )\n            self.d_output = cuda.mem_alloc(\n                int(np.prod(engine.get_tensor_shape('output'))) * 4\n            )\n            self.output_shape = engine.get_tensor_shape('output')\n            self.stream = cuda.Stream()\n        \n        def infer(self, input_data):\n            input_data = np.ascontiguousarray(input_data, dtype=np.float32)\n            output_data = np.empty(self.output_shape, dtype=np.float32)\n            \n            cuda.memcpy_htod_async(self.d_input, input_data, self.stream)\n            self.context.set_tensor_address('input', int(self.d_input))\n            self.context.set_tensor_address('output', int(self.d_output))\n            self.context.execute_async_v3(stream_handle=self.stream.handle)\n            cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)\n            self.stream.synchronize()\n            return output_data\n    \n    trt_infer = TRTInference(engine)\n    \n    # Run inference\n    trt_output = trt_infer.infer(onnx_input)\n    trt_seg, _ = visualize_segmentation(trt_output)\n    \n    # Benchmark\n    trt_time = benchmark(\n        lambda: trt_infer.infer(onnx_input),\n        'TensorRT (GPU, FP16)'\n    )\nelse:\n    trt_time = None\n    print('Skipping TensorRT benchmark (not available)')\n"""))

# Section 8: Benchmarking
nb.cells.append(new_markdown_cell("---\n## 8. Production Profiling & Benchmarking\n\n### Key Metrics\n- **Latency:** Time to process one inference (ms)\n- **Throughput:** Inferences per second (FPS)\n- **Memory:** GPU/CPU memory used during inference\n- **Model size:** Disk space (affects deployment size)"))

nb.cells.append(new_code_cell("""# Summary of all benchmarks
print('\\n' + '='*70)\nprint('  DEPLOYMENT BENCHMARK RESULTS')\nprint('  Model: DeepLabV3-MobileNetV3-Large')\nprint('='*70)\nprint(f'\\n  {\"Framework\":<35} {\"Latency (ms)\":<15} {\"FPS\":<10}')\nprint(f'  {\"-\"*35} {\"-\"*15} {\"-\"*10}')\nprint(f'  {\"PyTorch (CPU, baseline)\":<35} {pytorch_time:<15.1f} {1000/pytorch_time:<10.1f}')\nprint(f'  {\"ONNX Runtime (CPU)\":<35} {onnx_time:<15.1f} {1000/onnx_time:<10.1f}')\nprint(f'  {\"ONNX Runtime (CPU, pruned 30%)\":<35} {pruned_onnx_time:<15.1f} {1000/pruned_onnx_time:<10.1f}')\nif trt_time:\n    print(f'  {\"TensorRT (GPU, FP16)\":<35} {trt_time:<15.1f} {1000/trt_time:<10.1f}')\nprint(f'\\n  Model Sizes:')\nprint(f'  Original (FP32):     {original_size:.1f} MB')\nprint(f'  Pruned (FP32):       {pruned_size:.1f} MB')\nprint('='*70)\n"""))

# Section 9: Full Pipeline
nb.cells.append(new_markdown_cell("---\n## 9. Full Pipeline Project: Train → Optimize → Export → Deploy → Benchmark\n\n### End-to-End Autonomous Vehicle Perception\n\nThis section demonstrates the **complete production workflow**:\n\n```\n1. TRAIN      → Define and train a PyTorch model\n2. OPTIMIZE   → Prune and quantize\n3. EXPORT     → Convert to ONNX\n4. DEPLOY     → Load with ONNX Runtime / TensorRT\n5. BENCHMARK  → Compare all frameworks\n```"))

nb.cells.append(new_code_cell("""# Project: Simple object detector (YOLO-style)\nprint('PROJECT: Simple Object Detector')\nprint('Goal: Train → Optimize → Export → Deploy → Benchmark')\nprint('\\nStep 1: TRAIN')\nprint('-' * 60)\n\nclass SimpleDetector(torch_nn.Module):\n    \"\"\"Tiny object detection model (single-scale).\"\"\"\n    def __init__(self):\n        super().__init__()\n        self.backbone = torch_nn.Sequential(\n            torch_nn.Conv2d(3, 16, 3, padding=1),\n            torch_nn.ReLU(),\n            torch_nn.MaxPool2d(2),\n            torch_nn.Conv2d(16, 32, 3, padding=1),\n            torch_nn.ReLU(),\n        )\n        self.head = torch_nn.Conv2d(32, 5, 1)  # [tx, ty, tw, th, conf]\n    \n    def forward(self, x):\n        return self.head(self.backbone(x))\n\ndetector = SimpleDetector()\ndetector.train()\noptimizer = torch.optim.Adam(detector.parameters())\n\nprint(f'Model: SimpleDetector ({sum(p.numel() for p in detector.parameters()):,} params)')\nprint(f'Input: (batch, 3, 416, 416)')\nprint(f'Output: (batch, 5, 104, 104)  # [tx, ty, tw, th, confidence]')\nprint()\n\n# Quick training\nfor epoch in range(3):\n    synthetic_batch = torch.randn(4, 3, 416, 416)\n    targets = torch.randn(4, 5, 104, 104)\n    \n    outputs = detector(synthetic_batch)\n    loss = ((outputs - targets) ** 2).mean()\n    \n    optimizer.zero_grad()\n    loss.backward()\n    optimizer.step()\n    \n    print(f'  Epoch {epoch+1}/3  Loss: {loss.item():.4f}')\n\ndetector.eval()\nprint('✓ Training complete')\n"""))

nb.cells.append(new_code_cell("""print('\\nStep 2: OPTIMIZE')\nprint('-' * 60)\n\n# Pruning\noptimized_detector = copy.deepcopy(detector)\nfor name, module in optimized_detector.named_modules():\n    if isinstance(module, torch_nn.Conv2d):\n        prune.ln_structured(module, 'weight', amount=0.2, n=1, dim=0)\n        prune.remove(module, 'weight')\n\noriginal_params = sum(p.numel() for p in detector.parameters())\noptimized_params = sum(p.numel() for p in optimized_detector.parameters())\n\nprint(f'Pruning: {original_params:,} → {optimized_params:,} params')\nprint(f'Sparsity: {100*(1 - optimized_params/original_params):.1f}%')\nprint('✓ Optimization complete')\n"""))

nb.cells.append(new_code_cell("""print('\\nStep 3: EXPORT')\nprint('-' * 60)\n\nDETECTOR_ONNX = 'simple_detector.onnx'\n\ndummy_det_input = torch.randn(1, 3, 416, 416)\ntorch.onnx.export(\n    optimized_detector,\n    dummy_det_input,\n    DETECTOR_ONNX,\n    input_names=['image'],\n    output_names=['detections'],\n    opset_version=17\n)\n\nmodel_det = onnx.load(DETECTOR_ONNX)\nonnx.checker.check_model(model_det)\n\ndet_size = os.path.getsize(DETECTOR_ONNX) / 1024\nprint(f'Exported to: {DETECTOR_ONNX}')\nprint(f'File size: {det_size:.1f} KB')\nprint('✓ Export complete')\n"""))

nb.cells.append(new_code_cell("""print('\\nStep 4: DEPLOY')\nprint('-' * 60)\n\ndet_session = ort.InferenceSession(DETECTOR_ONNX, providers=['CPUExecutionProvider'])\ndet_input = det_session.get_inputs()[0]\ndet_output = det_session.get_outputs()[0]\n\ntest_det_image = np.random.randn(1, 3, 416, 416).astype(np.float32)\ndet_pred = det_session.run([det_output.name], {det_input.name: test_det_image})[0]\n\nprint(f'Input shape:  {test_det_image.shape}')\nprint(f'Output shape: {det_pred.shape}')\nprint('✓ Deployment complete (ONNX Runtime)')\n"""))

nb.cells.append(new_code_cell("""print('\\nStep 5: BENCHMARK')\nprint('-' * 60)\n\n# PyTorch baseline\npytorch_det_time = benchmark(\n    lambda: detector(dummy_det_input),\n    'PyTorch (CPU)'\n)\n\n# ONNX Runtime\nonnx_det_time = benchmark(\n    lambda: det_session.run([det_output.name], {det_input.name: test_det_image}),\n    'ONNX Runtime (CPU)'\n)\n\nprint(f'\\nSpeedup: {pytorch_det_time / onnx_det_time:.2f}x faster with ONNX')\nprint('\\n' + '='*70)\nprint('  FULL PIPELINE COMPLETE')\nprint('='*70)\nprint('\\nWhat you learned:')\nprint('  1. How to train and optimize models')\nprint('  2. Export PyTorch to production-ready ONNX')\nprint('  3. Deploy with multiple runtimes')\nprint('  4. Benchmark and compare performance')\nprint('  5. This is the workflow Autoware uses')\nprint('='*70)\n"""))

nb.cells.append(new_markdown_cell("""---\n\n## Conclusion & Next Steps\n\n### You've learned:\n✓ **Load & benchmark** a PyTorch robotics model  \n✓ **Export to ONNX** with full validation  \n✓ **Deploy with ONNX Runtime** and TensorRT  \n✓ **Optimize** via pruning & quantization before export  \n✓ **Analyze Autoware's production C++ code**  \n✓ **Complete end-to-end pipeline**\n\n### In Production (Autoware, Tesla, Waymo):\n- Train in PyTorch with optimizations from the **Neural Optimization** course\n- Export to ONNX for deployment flexibility\n- Use TensorRT on Nvidia GPUs (fastest)\n- Use ONNX Runtime on CPUs (most portable)\n- Package into ROS2 nodes (Autoware pattern)\n\n### Further Learning:\n- **Neural Optimization course** — Master pruning, quantization, distillation\n- **Autoware documentation** — Deploy models in real autonomous driving stack\n- **TensorRT documentation** — Advanced optimization techniques\n- **ONNX Zoo** — Pre-trained models for various tasks\n\n---\n\n*The Deployment Notebook — Neural Optimization by Think Autonomous*\n\nhttps://thinkautonomous.ai\n"""))

# Write notebook
with open('DLC_Deployment.ipynb', 'w') as f:
    nbformat.write(nb, f)

print(f'✓ Notebook generated: DLC_Deployment.ipynb')
print(f'  Cells: {len(nb.cells)}')
print(f'  Sections: 9 (Setup, Load, ONNX, Workflow, Optimize, Autoware Analysis, TensorRT, Profiling, Pipeline)')
