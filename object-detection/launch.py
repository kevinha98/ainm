import subprocess, sys, os, signal
os.chdir("/home/AD10209/object-detection")
# Kill any existing training
subprocess.run("pkill -f vm_train.py", shell=True)
# Clear old log
open("training.log", "w").close()
# Start training with batch=4, skip-prepare since YOLO data already exists
proc = subprocess.Popen(
    [sys.executable, "vm_train.py", "--imgsz", "1280", "--epochs", "100",
     "--batch", "4", "--patience", "20", "--skip-prepare"],
    stdout=open("training.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print(f"Started PID={proc.pid}")
