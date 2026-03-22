"""Export v5x as FP16 ONNX and check dynamic axes. Run on GCP VM."""
from ultralytics import YOLO
import onnx
import os

model_path = os.path.expanduser("~/object-detection/runs/detect/train_v5x/weights/best.pt")
dst_dir = os.path.expanduser("~/object-detection/runs/detect/train_v5x/weights/")

print(f"Loading model from {model_path}")
model = YOLO(model_path)

# Export FP16 ONNX
print("Exporting FP16 ONNX...")
model.export(format="onnx", dynamic=True, imgsz=1280, simplify=True, half=True)

# The export creates best.onnx (overwrites), rename it
onnx_path = model_path.replace(".pt", ".onnx")
fp16_path = os.path.join(dst_dir, "best_v5x_fp16.onnx")
os.rename(onnx_path, fp16_path)

size = os.path.getsize(fp16_path) / 1024 / 1024
print(f"FP16 ONNX: {fp16_path} ({size:.1f} MB)")

# Verify dynamic axes
m = onnx.load(fp16_path)
print("Input shapes:")
for inp in m.graph.input:
    dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {dims}")
print("DONE")
