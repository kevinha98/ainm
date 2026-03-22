#!/bin/bash
cd /home/AD10209/object-detection
rm -f training.log
nohup python3 vm_train.py --imgsz 1280 --epochs 100 --batch 4 --patience 20 --skip-prepare > training.log 2>&1 &
echo "PID=$!"
sleep 3
head -5 training.log
