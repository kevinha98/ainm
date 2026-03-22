#!/usr/bin/env python3
"""Bootstrap: extract data, install deps, convert format, and start training."""
import zipfile
import subprocess
import sys
from pathlib import Path

WORK = Path.home() / "object-detection"

def run(cmd, **kw):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kw)

def main():
    # Step 1: Extract
    zp = WORK / "NM_NGD_coco_dataset.zip"
    dest = WORK / "data" / "coco"
    if not (dest / "train" / "images").exists():
        print("Step 1: Extracting COCO dataset...")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(dest)
        imgs = list((dest / "train" / "images").glob("*.jpg"))
        print(f"  Extracted {len(imgs)} images")
    else:
        imgs = list((dest / "train" / "images").glob("*.jpg"))
        print(f"Step 1: Already extracted ({len(imgs)} images)")

    # Step 2: Install ultralytics
    print("Step 2: Installing ultralytics==8.1.0...")
    run(f"{sys.executable} -m pip install ultralytics==8.1.0 pycocotools -q")

    # Step 3: Convert COCO -> YOLO
    yolo_dir = WORK / "data" / "yolo"
    if not (yolo_dir / "dataset.yaml").exists():
        print("Step 3: Converting COCO -> YOLO format...")
        run(f"cd {WORK} && {sys.executable} vm_train.py --skip-prepare=false --epochs 0 2>/dev/null || true")
        # If vm_train doesn't support epochs=0, do manual conversion
        if not (yolo_dir / "dataset.yaml").exists():
            run(f"cd {WORK} && {sys.executable} prepare_data.py --annotations data/coco/train/annotations.json --images-dir data/coco/train/images --copy-images")
    else:
        print("Step 3: YOLO dataset already prepared")

    # Step 4: Start training in tmux
    print("Step 4: Starting training in tmux...")
    train_cmd = (
        f"cd {WORK} && {sys.executable} vm_train.py "
        f"--skip-prepare --imgsz 1280 --epochs 100 --batch 8 --patience 20 "
        f"2>&1 | tee {WORK}/training.log"
    )
    run(f'tmux kill-session -t train 2>/dev/null || true')
    run(f'tmux new-session -d -s train "{train_cmd}"')
    print("Training started in tmux session 'train'!")
    print("Monitor: tmux attach -t train")
    print(f"Log: tail -f {WORK}/training.log")

if __name__ == "__main__":
    main()
