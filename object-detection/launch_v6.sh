#!/bin/bash
cd ~/object-detection
cp /tmp/vm_train_v6.py ~/object-detection/vm_train_v6.py
python3 -c "import shutil; shutil.rmtree('runs/detect/train_v6', ignore_errors=True); shutil.rmtree('runs/detect/runs', ignore_errors=True)"
nohup python3 vm_train_v6.py --skip-prepare --epochs 300 --patience 60 --batch 2 > train_v6.log 2>&1 &
BGPID=$!
echo "PID=$BGPID"
sleep 45
tail -30 train_v6.log 2>/dev/null
echo "---GPU---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
