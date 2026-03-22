#!/usr/bin/env python3
"""V4 training — YOLOv8l for maximum accuracy.

Key changes from v3 (YOLOv8m):
  - YOLOv8l: 43.7M params (vs 25.9M) — 69% more capacity
  - Resume from v3 best.pt is NOT possible (architecture mismatch) — train from COCO pretrained
  - batch=2 to fit T4 VRAM with larger model at imgsz=1280
  - Same aggressive augmentation + cls=1.0 from v3

Usage on VM:
    pip install ultralytics==8.1.0 pycocotools
    python vm_train_v4.py --skip-prepare --epochs 300 --patience 50 --batch 2
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


def train(data_yaml, model_name="yolov8l.pt", imgsz=1280, epochs=300, batch=2, patience=50):
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

    print(f"\nV4 Training: {model_name}, imgsz={imgsz}, epochs={epochs}, batch={batch}, patience={patience}")
    print("YOLOv8l — 43.7M params, cls=1.0, label_smoothing=0.1, aggressive augmentation")

    results = model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        project="runs/detect",
        name="train_v4",
        exist_ok=True,

        # === Augmentation (same aggressive config as v3) ===
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.3,
        scale=0.9,
        shear=2.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.1,
        close_mosaic=30,

        # === Loss weights ===
        cls=1.0,
        box=7.5,
        dfl=1.5,
        
        # === Optimizer ===
        optimizer="AdamW",
        lr0=0.0003,           # Even lower for larger model
        lrf=0.005,
        weight_decay=0.001,
        warmup_epochs=5,
        cos_lr=True,

        # === Regularization ===
        label_smoothing=0.1,
        dropout=0.15,          # Slightly more dropout for larger model

        # === Training ===
        save=True,
        val=True,
        plots=True,
        verbose=True,
        amp=True,
    )

    best_pt = Path("runs/detect/train_v4/weights/best.pt")
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
    parser.add_argument("--model", default="yolov8l.pt", help="Base model")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    if args.export_only:
        best = Path("runs/detect/train_v4/weights/best.pt")
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
    best_pt = train(data_yaml, args.model, args.imgsz, args.epochs, args.batch, args.patience)

    # Export
    if best_pt.exists():
        export_onnx(best_pt)


if __name__ == "__main__":
    main()
