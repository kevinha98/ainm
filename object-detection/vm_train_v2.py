#!/usr/bin/env python3
"""Training script for YOLOv8l with improvements.

Improvements over v1 (YOLOv8m):
  - YOLOv8l: Larger model (43M vs 26M params) → higher capacity
  - nc=357: Includes unknown_product class 356 (matches task spec recommendation)
  - imgsz=1280: High-res training for shelf images
  - More aggressive augmentation for small dataset (248 images)
  - close_mosaic=20: Disable mosaic for last 20 epochs (better fine-tuning)
  - patience=25: More patience for larger model
  - epochs=120: More epochs for larger model to converge

Usage on VM:
    python vm_train_v2.py
    python vm_train_v2.py --model yolov8x.pt --epochs 150  # even larger
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

NUM_CATEGORIES = 356  # 0-355 (355 products + unknown_product at 355)
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

    # Write dataset.yaml with nc=357
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


def train(data_yaml, model_name="yolov8l.pt", imgsz=1280, epochs=120, batch=2, patience=25):
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    import ultralytics
    print(f"\nultralytics version: {ultralytics.__version__}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({mem_gb:.1f} GB)")

    from ultralytics import YOLO
    model = YOLO(model_name)

    print(f"\nTraining: {model_name}, imgsz={imgsz}, epochs={epochs}, batch={batch}")
    results = model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        project="runs/detect",
        name="train_v2",
        exist_ok=True,
        # Augmentation — more aggressive for small dataset
        mosaic=1.0,
        mixup=0.3,           # Increased from 0.15
        copy_paste=0.3,       # Increased from 0.2
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,          # Shelves are always upright
        translate=0.2,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.1,          # Random erasing for robustness
        close_mosaic=20,      # Disable mosaic for last 20 epochs
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,      # More warmup for larger model
        cos_lr=True,
        # Save
        save=True,
        val=True,
        plots=True,
        verbose=True,
    )

    best_pt = Path("runs/detect/train_v2/weights/best.pt")
    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"\nbest.pt: {best_pt} ({size_mb:.1f} MB)")
    return best_pt


def export_onnx(weights_path):
    """Export best.pt to ONNX for sandbox compatibility."""
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    onnx_path = model.export(format="onnx", dynamic=True, half=False, opset=17, simplify=True)
    print(f"ONNX exported: {onnx_path}")

    # Copy to weights dir
    import shutil
    dest = weights_path.parent / "best.onnx"
    shutil.copy2(onnx_path, dest)
    print(f"Copied to: {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8l.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--export-only", type=Path, default=None)
    args = parser.parse_args()

    # Export only mode
    if args.export_only:
        export_onnx(args.export_only)
        return

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    import os
    os.chdir(WORK_DIR)
    print(f"Working directory: {WORK_DIR}")

    # Data paths
    coco_dir = WORK_DIR / "data" / "coco" / "train"
    annotations = coco_dir / "annotations.json"
    images_dir = coco_dir / "images"
    yolo_dir = WORK_DIR / "data" / "yolo_v2"

    if not annotations.exists():
        print(f"ERROR: annotations not found at {annotations}")
        print("Upload and extract the COCO dataset first")
        return

    # Prepare dataset
    if not args.skip_prepare:
        data_yaml = prepare_yolo_dataset(annotations, images_dir, yolo_dir)
    else:
        data_yaml = yolo_dir / "dataset.yaml"

    # Train
    best_pt = train(data_yaml, args.model, args.imgsz, args.epochs, args.batch, args.patience)

    # Export to ONNX
    if best_pt.exists():
        export_onnx(best_pt)

    print("\n=== DONE ===")
    print(f"Best weights: {best_pt}")
    print("Download with: gcloud compute scp obj-detect-train:~/object-detection/runs/detect/train_v2/weights/best.onnx .")


if __name__ == "__main__":
    main()
