#!/bin/bash
cd ~/object-detection
ls -la runs/detect/train_v6/weights/
python -c "
from ultralytics import YOLO
m = YOLO('runs/detect/train_v6/weights/best.pt')
results = m.val()
print('VAL mAP50:', results.box.map50)
print('VAL mAP50-95:', results.box.map)
m.export(format='onnx', imgsz=1280, opset=17, simplify=True)
print('EXPORT DONE')
ls -la runs/detect/train_v6/weights/best.onnx
"
