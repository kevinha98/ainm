#!/usr/bin/env python3
"""Train 3 models on FULL dataset (248 images, no val holdout) on GCP VM.

Strategy: Retrain v3 (YOLOv8m), v6 (YOLO11m), and new v7 (YOLO11l) on all data.
Since we evaluate via competition, val loss doesn't matter — train on everything.

Usage on VM:
    # First upload this + prepare_data_full.py to VM
    # Run data prep
    python prepare_data_full.py
    
    # Then train all models sequentially
    nohup python vm_train_v7_fulldata.py --model v3f > train_v3f.log 2>&1 &
    # After v3f done:
    nohup python vm_train_v7_fulldata.py --model v6f > train_v6f.log 2>&1 &
    # After v6f done:
    nohup python vm_train_v7_fulldata.py --model v7 > train_v7.log 2>&1 &
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

NUM_CATEGORIES = 356
WORK_DIR = Path.home() / "object-detection"


def prepare_full_dataset():
    """Create YOLO dataset with ALL 248 images in both train and val."""
    ann_path = WORK_DIR / "data" / "coco" / "train" / "annotations.json"
    img_dir = WORK_DIR / "data" / "coco" / "train" / "images"
    out_dir = WORK_DIR / "data" / "yolo_full"

    coco = json.load(open(ann_path, "r"))
    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    all_ids = sorted(images.keys())
    print(f"Full dataset: {len(all_ids)} images, {len(coco['annotations'])} annotations")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_id in all_ids:
            info = images[img_id]
            fname = info["file_name"]
            src = img_dir / fname
            dst = out_dir / "images" / split / fname
            if not dst.exists():
                shutil.copy2(src, dst)

            anns = anns_by_image.get(img_id, [])
            lines = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                xc = max(0.0, min(1.0, (x + w / 2) / info["width"]))
                yc = max(0.0, min(1.0, (y + h / 2) / info["height"]))
                wn = max(0.0, min(1.0, w / info["width"]))
                hn = max(0.0, min(1.0, h / info["height"]))
                if wn < 1e-6 or hn < 1e-6:
                    continue
                lines.append(f"{ann['category_id']} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            (out_dir / "labels" / split / f"{Path(fname).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))

    # dataset.yaml
    names_lines = [f'  {cid}: "{categories[cid].replace(chr(34), chr(92)+chr(34))}"'
                   for cid in sorted(categories.keys())]
    yaml = f"""path: {out_dir}
train: images/train
val: images/val
nc: {NUM_CATEGORIES}
names:
{chr(10).join(names_lines)}
"""
    (out_dir / "dataset.yaml").write_text(yaml)
    print(f"Dataset ready: {out_dir / 'dataset.yaml'}")
    return out_dir / "dataset.yaml"


def patch_torch():
    import torch
    _ol = torch.load
    def _pl(*a, **k):
        if 'weights_only' not in k:
            k['weights_only'] = False
        return _ol(*a, **k)
    torch.load = _pl


def train_v3f(data_yaml):
    """Retrain YOLOv8m on full data — same hyperparams as v3 that worked well."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    patch_torch()
    from ultralytics import YOLO

    model = YOLO("yolov8m.pt")
    print("Training v3f: YOLOv8m on FULL dataset (248 images)")

    model.train(
        data=str(data_yaml), imgsz=1280, epochs=300, batch=4, patience=50,
        project="runs/detect", name="train_v3f", exist_ok=True,
        mosaic=1.0, mixup=0.2, copy_paste=0.3, hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.3, scale=0.9, shear=2.0, fliplr=0.5, erasing=0.1,
        close_mosaic=30,
        cls=1.0, box=7.5, dfl=1.5,
        optimizer="AdamW", lr0=0.0005, lrf=0.01, weight_decay=0.0005,
        warmup_epochs=3, cos_lr=True,
        label_smoothing=0.1, dropout=0.1,
        save=True, val=True, plots=True, amp=True, workers=8,
    )

    best = Path("runs/detect/train_v3f/weights/best.pt")
    if best.exists():
        YOLO(str(best)).export(format="onnx", imgsz=1280, dynamic=True, simplify=True)
        print(f"Exported: {best.with_suffix('.onnx')}")


def train_v6f(data_yaml):
    """Retrain YOLO11m on full data."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    patch_torch()
    from ultralytics import YOLO

    model = YOLO("yolo11m.pt")
    print("Training v6f: YOLO11m on FULL dataset (248 images)")

    model.train(
        data=str(data_yaml), imgsz=1280, epochs=300, batch=-1, patience=60,
        project="runs/detect", name="train_v6f", exist_ok=True,
        mosaic=1.0, mixup=0.2, copy_paste=0.3, hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.3, scale=0.5, shear=2.0, fliplr=0.5, erasing=0.1,
        close_mosaic=10,
        cls=1.0, box=7.5, dfl=1.5,
        optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
        warmup_epochs=3, cos_lr=True,
        label_smoothing=0.1, dropout=0.1,
        save=True, val=True, plots=True, amp=True, workers=8,
    )

    best = Path("runs/detect/train_v6f/weights/best.pt")
    if best.exists():
        YOLO(str(best)).export(format="onnx", imgsz=1280, dynamic=True, simplify=True)
        print(f"Exported: {best.with_suffix('.onnx')}")


def train_v7(data_yaml):
    """Train YOLO11l (larger) on full data — new architecture, more capacity."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    patch_torch()
    from ultralytics import YOLO

    model = YOLO("yolo11l.pt")
    print("Training v7: YOLO11l on FULL dataset (248 images)")

    model.train(
        data=str(data_yaml), imgsz=1280, epochs=400, batch=2, patience=80,
        project="runs/detect", name="train_v7", exist_ok=True,
        mosaic=1.0, mixup=0.3, copy_paste=0.4, hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.3, scale=0.7, shear=2.0, fliplr=0.5, erasing=0.15,
        close_mosaic=20,
        cls=1.0, box=7.5, dfl=1.5,
        optimizer="AdamW", lr0=0.0005, lrf=0.01, weight_decay=0.0005,
        warmup_epochs=5, cos_lr=True,
        label_smoothing=0.1, dropout=0.15,
        save=True, val=True, plots=True, amp=True, workers=8,
    )

    best = Path("runs/detect/train_v7/weights/best.pt")
    if best.exists():
        YOLO(str(best)).export(format="onnx", imgsz=1280, dynamic=True, simplify=True)
        print(f"Exported: {best.with_suffix('.onnx')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["v3f", "v6f", "v7", "prepare"], required=True)
    args = parser.parse_args()

    if args.model == "prepare":
        prepare_full_dataset()
    else:
        data_yaml = WORK_DIR / "data" / "yolo_full" / "dataset.yaml"
        if not data_yaml.exists():
            data_yaml = prepare_full_dataset()
        {"v3f": train_v3f, "v6f": train_v6f, "v7": train_v7}[args.model](data_yaml)
