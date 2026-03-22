#!/usr/bin/env python3
"""V3 training — Maximum accuracy for NorgesGruppen competition.

Key improvements over v1:
  - 300 epochs with patience=50 (v1 stopped at 87/100)
  - Higher copy_paste=0.3 (improve rare class representation)  
  - Multi-scale training scale=0.9 (huge range: 0.1x to 1.9x)
  - close_mosaic=30 (refine predictions in last 30 epochs)
  - Lower lr0=0.0005 for more stable convergence
  - label_smoothing=0.1 (helps classification of similar products)
  - cls=1.0 (boost classification loss weight — our main weakness)
  - Retina loss (focal loss) for better handling of class imbalance

Usage on VM:
    pip install ultralytics==8.1.0 pycocotools
    python vm_train_v3.py
    # Then download best.pt and export to ONNX
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


def train(data_yaml, model_name="yolov8m.pt", imgsz=1280, epochs=300, batch=4, patience=50):
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

    print(f"\nV3 Training: {model_name}, imgsz={imgsz}, epochs={epochs}, batch={batch}, patience={patience}")
    print("Key changes: cls=1.0, label_smoothing=0.1, copy_paste=0.3, scale=0.9, close_mosaic=30")

    results = model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        project="runs/detect",
        name="train_v3",
        exist_ok=True,

        # === Augmentation (aggressive for 248 images) ===
        mosaic=1.0,           # Always use mosaic (v1: 0.8)
        mixup=0.2,            # Slightly more mixup (v1: 0.15)
        copy_paste=0.3,       # More copy-paste for rare classes (v1: 0.2)
        hsv_h=0.02,           # Slightly more hue variation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,          # Small rotation — shelf products can be slightly tilted
        translate=0.3,        # More translation to handle edge objects (v1: 0.2)
        scale=0.9,            # VERY aggressive multi-scale: 0.1x to 1.9x (v1: 0.5)
        shear=2.0,            # Small shear for perspective variation
        fliplr=0.5,
        flipud=0.0,
        erasing=0.1,          # Random erasing — forces model to rely on partial features
        close_mosaic=30,      # Refine in last 30 epochs (v1: default 10)

        # === Loss weights (classification is our weakness) ===
        cls=1.0,              # Higher classification loss weight (default 0.5)
        box=7.5,              # Default
        dfl=1.5,              # Default
        
        # === Optimizer (more conservative for better convergence) ===
        optimizer="AdamW",
        lr0=0.0005,           # Lower initial LR (v1: 0.001)
        lrf=0.005,            # Lower final LR (v1: 0.01)
        weight_decay=0.001,   # Higher weight decay (v1: 0.0005)
        warmup_epochs=5,      # Longer warmup (v1: 3)
        cos_lr=True,

        # === Regularization ===
        label_smoothing=0.1,  # Helps with similar-looking product classes
        dropout=0.1,          # Light dropout for generalization

        # === Training ===
        save=True,
        val=True,
        plots=True,
        verbose=True,
        amp=True,             # Mixed precision for speed
        
        # === Resume from v1 best weights for faster convergence ===
        # pretrained=True is default — starts from COCO pretrained
        # We'll load our v1 best.pt as starting point instead
    )

    best_pt = Path("runs/detect/train_v3/weights/best.pt")
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
    parser.add_argument("--model", default="yolov8m.pt", help="Base model or resume weights")
    parser.add_argument("--resume-from", default=None, help="Resume from existing best.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--export-only", action="store_true", help="Just export existing best.pt to ONNX")
    args = parser.parse_args()

    if args.export_only:
        best = Path("runs/detect/train_v3/weights/best.pt")
        if not best.exists():
            best = Path("runs/detect/train/weights/best.pt")
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
    model = args.resume_from if args.resume_from else args.model
    best_pt = train(data_yaml, model, args.imgsz, args.epochs, args.batch, args.patience)

    # Export
    if best_pt.exists():
        export_onnx(best_pt)


if __name__ == "__main__":
    main()
