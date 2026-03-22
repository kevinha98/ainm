"""Export v5x (YOLOv8x) to ONNX with dynamic axes. Run on GCP VM."""
from ultralytics import YOLO
import os

model_path = os.path.expanduser("~/object-detection/runs/detect/train_v5x/weights/best.pt")
print(f"Loading model from {model_path}")
model = YOLO(model_path)
print("Exporting to ONNX...")
model.export(format="onnx", dynamic=True, imgsz=1280, simplify=True)
print("Export complete!")
out = model_path.replace(".pt", ".onnx")
size = os.path.getsize(out) / 1024 / 1024
print(f"ONNX file: {out} ({size:.1f} MB)")
