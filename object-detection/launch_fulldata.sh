#!/bin/bash
cd /home/AD10209/object-detection

# Download pretrained weights if missing
for w in yolov8m.pt yolo11m.pt yolo11l.pt; do
    if [ ! -f "$w" ]; then
        echo "Downloading $w..."
        python3 -c "from ultralytics import YOLO; YOLO('$w')"
    fi
done

# Prepare full dataset
echo "Preparing full dataset..."
python3 vm_train_v7_fulldata.py --model prepare

# Train v3f (YOLOv8m on full data) — highest priority
echo "Starting v3f training..."
nohup python3 vm_train_v7_fulldata.py --model v3f > train_v3f.log 2>&1 &
echo "v3f started, PID: $!"
echo "Monitor: tail -f train_v3f.log"
