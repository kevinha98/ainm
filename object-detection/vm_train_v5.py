#!/usr/bin/env python3
"""V5 training — YOLOv8x for maximum accuracy + fine-tuned v4 fallback.

Strategy: Two-pronged training to maximize accuracy.
  Track A: YOLOv8x (68.2M params) at imgsz=1280 — 56% more params than v8l
  Track B: Fine-tune v4's best.pt with aggressive augmentation (faster)

T4 VRAM budget (15GB):
  - v8x@1280, batch=1: ~10-12GB → fits
  - v8l@1280, batch=2: ~12-14GB → fits (v4 config)

Key changes from v4:
  - YOLOv8x: 68.2M params (vs 43.7M for v8l, 25.9M for v8m)
  - More aggressive augmentation: copy_paste=0.5, mixup=0.3
  - Longer training: epochs=400, patience=80
  - Multi-scale training: scale=0.9 (random resize ±45%)
  - close_mosaic=15 (fine-tune without mosaic for last 15 epochs)

Usage on VM:
    # Track A: Full YOLOv8x training
    python vm_train_v5.py --skip-prepare --track x --epochs 400 --batch 1

    # Track B: Fine-tune v4 (faster, ~100 epochs)
    python vm_train_v5.py --skip-prepare --track finetune --epochs 150 --batch 2

    # Export to ONNX after training
    python vm_train_v5.py --export-only --track x
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

NUM_CATEGORIES = 356
VAL_SPLIT = 0.15
RANDOM_SEED = 42
WORK_DIR = Path.home() / "object-detection"


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    xc = max(0.0, min(1.0, (x + w / 2) / img_w))
    yc = max(0.0, min(1.0, (y + h / 2) / img_h))
    wn = max(0.0, min(1.0, w / img_w))
    hn = max(0.0, min(1.0, h / img_h))
    return xc, yc, wn, hn


def prepare_yolo_dataset(annotations_path, images_dir, output_dir, val_split=VAL_SPLIT):
    print("Loading COCO annotations...")
    with open(annotations_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    print(f"  {len(images)} images, {len(coco['annotations'])} annotations, {len(categories)} categories")

    ids = sorted(images.keys())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(ids)
    split_idx = int(len(ids) * (1 - val_split))
    train_ids, val_ids = ids[:split_idx], ids[split_idx:]
    print(f"  Split: {len(train_ids)} train, {len(val_ids)} val")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        labels_dir = output_dir / "labels" / split_name
        images_out = output_dir / "images" / split_name
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_out.mkdir(parents=True, exist_ok=True)

        n_anns = 0
        for img_id in split_ids:
            img_info = images[img_id]
            fname = img_info["file_name"]
            src = images_dir / fname
            dst = images_out / fname
            if not dst.exists():
                try:
                    dst.symlink_to(src)
                except (PermissionError, NotImplementedError):
                    shutil.copy2(src, dst)

            anns = anns_by_image.get(img_id, [])
            lines = []
            for ann in anns:
                xc, yc, wn, hn = coco_to_yolo_bbox(ann["bbox"], img_info["width"], img_info["height"])
                if wn < 1e-6 or hn < 1e-6:
                    continue
                lines.append(f"{ann['category_id']} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
                n_anns += 1

            label_file = labels_dir / f"{Path(fname).stem}.txt"
            label_file.write_text("\n".join(lines) + ("\n" if lines else ""))

        print(f"  {split_name}: {len(split_ids)} images, {n_anns} annotations")

    names_lines = []
    for cid in sorted(categories.keys()):
        name = categories[cid].replace('"', '\\"')
        names_lines.append(f'  {cid}: "{name}"')
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val
nc: {NUM_CATEGORIES}
names:
{chr(10).join(names_lines)}
"""
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  dataset.yaml: {yaml_path}")
    return yaml_path


def train_v8x(data_yaml, epochs=400, batch=1, patience=80):
    """Track A: Train YOLOv8x from COCO pretrained."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({mem_gb:.1f} GB)")

    model = YOLO("yolov8x.pt")
    print(f"\nV5 Track A: YOLOv8x — 68.2M params")
    print(f"  imgsz=1280, epochs={epochs}, batch={batch}, patience={patience}")

    results = model.train(
        data=str(data_yaml),
        imgsz=1280,
        epochs=epochs,
        batch=batch,
        patience=patience,
        project="runs/detect",
        name="train_v5x",
        exist_ok=True,

        # === Augmentation (more aggressive than v4) ===
        mosaic=1.0,
        mixup=0.3,         # Up from 0.2
        copy_paste=0.5,    # Up from 0.3
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.3,
        scale=0.9,         # ±45% random resize
        shear=2.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.15,      # Up from 0.1
        close_mosaic=15,    # Fine-tune without mosaic longer

        # === Loss weights ===
        cls=1.0,
        box=7.5,
        dfl=1.5,

        # === Optimizer ===
        optimizer="AdamW",
        lr0=0.0002,         # Lower for larger model
        lrf=0.005,
        weight_decay=0.001,
        warmup_epochs=5,
        cos_lr=True,

        # === Regularization ===
        label_smoothing=0.1,
        dropout=0.2,         # More dropout for bigger model

        # === Training ===
        save=True,
        val=True,
        plots=True,
        verbose=True,
        amp=True,
        workers=8,
    )

    best_pt = Path("runs/detect/train_v5x/weights/best.pt")
    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"\nbest.pt: {best_pt} ({size_mb:.1f} MB)")
    return best_pt


def train_finetune_v4(data_yaml, epochs=150, batch=2, patience=40):
    """Track B: Fine-tune v4's best.pt (YOLOv8l) with different augmentation."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    from ultralytics import YOLO

    v4_best = Path("runs/detect/train_v4/weights/best.pt")
    if not v4_best.exists():
        print(f"ERROR: {v4_best} not found — cannot fine-tune")
        return None

    model = YOLO(str(v4_best))
    print(f"\nV5 Track B: Fine-tune v4 (YOLOv8l)")
    print(f"  imgsz=1280, epochs={epochs}, batch={batch}")

    results = model.train(
        data=str(data_yaml),
        imgsz=1280,
        epochs=epochs,
        batch=batch,
        patience=patience,
        project="runs/detect",
        name="train_v5ft",
        exist_ok=True,

        # === Augmentation (different mix to diversify from v4) ===
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.5,      # Aggressive copy-paste for rare classes
        hsv_h=0.03,          # Slightly different color jitter
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=8.0,         # More rotation
        translate=0.25,
        scale=0.8,
        shear=3.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.2,
        close_mosaic=10,

        # === Loss weights ===
        cls=1.2,             # Boost classification weight
        box=7.5,
        dfl=1.5,

        # === Optimizer (lower LR for fine-tuning) ===
        optimizer="AdamW",
        lr0=0.00005,         # Very low for fine-tuning
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,

        # === Regularization ===
        label_smoothing=0.1,
        dropout=0.15,

        # === Training ===
        save=True,
        val=True,
        plots=True,
        verbose=True,
        amp=True,
        workers=8,
    )

    best_pt = Path("runs/detect/train_v5ft/weights/best.pt")
    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"\nbest.pt: {best_pt} ({size_mb:.1f} MB)")
    return best_pt


def export_onnx(best_pt):
    """Export to ONNX for sandbox deployment."""
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    from ultralytics import YOLO
    model = YOLO(str(best_pt))
    model.export(format="onnx", opset=17, dynamic=True, simplify=True)
    onnx_path = best_pt.with_suffix(".onnx")
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"ONNX exported: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["x", "finetune"], default="x",
                        help="x=YOLOv8x from scratch, finetune=from v4 best.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    if args.export_only:
        track_name = "v5x" if args.track == "x" else "v5ft"
        best = Path(f"runs/detect/train_{track_name}/weights/best.pt")
        if not best.exists():
            print(f"ERROR: {best} not found")
            return
        export_onnx(best)
        return

    # Prepare dataset
    data_dir = WORK_DIR / "data" / "coco" / "train"
    yolo_dir = WORK_DIR / "data" / "yolo"
    ann = data_dir / "annotations.json"
    img_dir = data_dir / "images"

    if not args.skip_prepare:
        if not ann.exists():
            print(f"ERROR: {ann} not found")
            return
        data_yaml = prepare_yolo_dataset(ann, img_dir, yolo_dir)
    else:
        data_yaml = yolo_dir / "dataset.yaml"

    # Train
    if args.track == "x":
        best_pt = train_v8x(data_yaml, epochs=args.epochs, batch=args.batch, patience=args.patience)
    else:
        best_pt = train_finetune_v4(data_yaml, epochs=args.epochs, batch=args.batch, patience=args.patience)

    if best_pt and best_pt.exists():
        export_onnx(best_pt)


if __name__ == "__main__":
    main()
