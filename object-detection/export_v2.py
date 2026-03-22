#!/usr/bin/env python3
"""Export YOLOv8l best.pt to ONNX with dynamic axes and FP16."""
import torch

# Monkey-patch torch.load for cross-version compat
_orig = torch.load
def _patched(*a, **k):
    k.setdefault("weights_only", False)
    return _orig(*a, **k)
torch.load = _patched

from ultralytics import YOLO
from pathlib import Path

# Try v2 (YOLOv8l) first, fall back to v1 (YOLOv8m)
v2_weights = Path("runs/detect/train_v2/weights/best.pt")
v1_weights = Path("runs/detect/train/weights/best.pt")

if v2_weights.exists():
    weights = v2_weights
    print(f"Exporting v2 (YOLOv8l): {weights}")
elif v1_weights.exists():
    weights = v1_weights
    print(f"Exporting v1 (YOLOv8m): {weights}")
else:
    raise FileNotFoundError("No weights found")

model = YOLO(str(weights))

# Export FP16 ONNX with dynamic axes
out = model.export(
    format="onnx",
    dynamic=True,
    half=False,       # FP32 for max compatibility (FP16 can cause issues on some runtimes)
    opset=17,
    simplify=True,
)
print(f"Exported: {out}")

# Also export FP16 version (smaller, faster on L4 GPU)
out_fp16 = model.export(
    format="onnx",
    dynamic=True,
    half=True,
    opset=17,
    simplify=True,
)
print(f"Exported FP16: {out_fp16}")

# Show sizes
import os
for f in [out, out_fp16]:
    if f and os.path.exists(f):
        size = os.path.getsize(f) / (1024*1024)
        print(f"  {f}: {size:.1f} MB")
