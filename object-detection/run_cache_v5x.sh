#!/bin/bash
cd /home/AD10209/object-detection
nohup python3 cache_v5x_vm.py > cache_v5x_log.txt 2>&1 &
echo "STARTED PID: $!"
sleep 3
cat cache_v5x_log.txt
