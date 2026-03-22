#!/usr/bin/env python3
"""Self-contained training script for GCP VM.

Upload this single file + the COCO zip to the VM, and it handles everything:
  - Extracts data
  - Converts COCO → YOLO format  
  - Trains YOLOv8m (or specified model)
  - Evaluates on validation set
  - Saves best.pt ready for download

Usage on VM:
    pip install ultralytics==8.1.0 pycocotools
    python vm_train.py                              # Default: YOLOv8m, 80 epochs
    python vm_train.py --model yolov8l.pt --epochs 100 --imgsz 1280
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────────
NUM_CATEGORIES = 356
VAL_SPLIT = 0.15
RANDOM_SEED = 42
WORK_DIR = Path.home() / "object-detection"


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] pixels → YOLO [x_center, y_center, w, h] normalized."""
    x, y, w, h = bbox
    xc = max(0.0, min(1.0, (x + w / 2) / img_w))
    yc = max(0.0, min(1.0, (y + h / 2) / img_h))
    wn = max(0.0, min(1.0, w / img_w))
    hn = max(0.0, min(1.0, h / img_h))
    return xc, yc, wn, hn


def prepare_yolo_dataset(annotations_path, images_dir, output_dir, val_split=VAL_SPLIT):
    """Convert COCO annotations to YOLO format with train/val split."""
    print("Loading COCO annotations...")
    with open(annotations_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    print(f"  {len(images)} images, {len(coco['annotations'])} annotations, {len(categories)} categories")

    # Split
    ids = sorted(images.keys())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(ids)
    split_idx = int(len(ids) * (1 - val_split))
    train_ids, val_ids = ids[:split_idx], ids[split_idx:]
    print(f"  Split: {len(train_ids)} train, {len(val_ids)} val")

    # Clean output
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

            # Copy or symlink image
            if not dst.exists():
                try:
                    dst.symlink_to(src)
                except (PermissionError, NotImplementedError):
                    shutil.copy2(src, dst)

            # Write YOLO labels
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

    # Write dataset.yaml — use double quotes for names (some contain apostrophes)
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


def train(data_yaml, model_name="yolov8m.pt", imgsz=1280, epochs=80, batch=4, patience=15):
    """Train YOLOv8 model."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Monkey-patch torch.load for PyTorch 2.6+ compatibility with ultralytics 8.1.0
    # PyTorch 2.6+ defaults to weights_only=True, but ultralytics 8.1.0 pickles full models
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    import ultralytics
    print(f"\nultralytics version: {ultralytics.__version__}")
    if not ultralytics.__version__.startswith("8.1."):
        print(f"WARNING: Sandbox uses 8.1.0, you have {ultralytics.__version__}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({mem_gb:.1f} GB)")
    else:
        print("WARNING: No GPU — training will be slow!")

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
        name="train",
        exist_ok=True,
        mosaic=0.8,
        mixup=0.15,
        copy_paste=0.2,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0,
        translate=0.2,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        save=True,
        val=True,
        plots=True,
        verbose=True,
    )

    best_pt = Path("runs/detect/train/weights/best.pt")
    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"\nbest.pt: {best_pt} ({size_mb:.1f} MB)")
        if size_mb > 420:
            print("WARNING: > 420 MB! Export FP16 or use smaller model.")
    return best_pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    work = Path.cwd()

    # Auto-detect data paths
    if not args.annotations:
        for candidate in [
            work / "data" / "coco" / "train" / "annotations.json",
            work / "data" / "coco" / "annotations.json",
            work / "annotations.json",
        ]:
            if candidate.exists():
                args.annotations = candidate
                break
    if not args.images_dir:
        for candidate in [
            work / "data" / "coco" / "train" / "images",
            work / "data" / "coco" / "images",
        ]:
            if candidate.exists():
                args.images_dir = candidate
                break

    yolo_dir = work / "data" / "yolo"
    data_yaml = yolo_dir / "dataset.yaml"

    if not args.skip_prepare:
        if not args.annotations or not args.annotations.exists():
            print(f"ERROR: annotations.json not found. Provide --annotations path.")
            return
        data_yaml = prepare_yolo_dataset(args.annotations, args.images_dir, yolo_dir)
    elif not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. Run without --skip-prepare first.")
        return

    best_pt = train(data_yaml, args.model, args.imgsz, args.epochs, args.batch, args.patience)
    print(f"\nDone! Download with:")
    print(f"  gcloud compute scp <VM>:~/{best_pt.relative_to(Path.home()) if str(best_pt).startswith(str(Path.home())) else best_pt} ./best.pt --zone=europe-west4-a")


if __name__ == "__main__":
    main()
