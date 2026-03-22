#!/usr/bin/env python3
"""Export YOLO11m best.pt to ONNX format."""
import os
os.chdir(os.path.expanduser("~/object-detection"))

from ultralytics import YOLO

weights = "runs/detect/runs/detect/train_v6/weights/best.pt"
print(f"Loading {weights}")
m = YOLO(weights)

print("Exporting to ONNX with dynamic axes...")
m.export(format="onnx", imgsz=1280, opset=17, simplify=True, dynamic=True)
print("EXPORT DONE")

onnx_path = weights.replace(".pt", ".onnx")
size_mb = os.path.getsize(onnx_path) / 1024 / 1024
print(f"ONNX size: {size_mb:.1f} MB")
