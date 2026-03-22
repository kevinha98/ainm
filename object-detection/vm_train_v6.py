#!/usr/bin/env python3
"""V6 training — YOLO11l for better architecture + auto-batch.

Why YOLO11l over YOLOv8x:
  - Newer C3k2/SPPF/C2PSA architecture = better feature extraction
  - 25.3M params (vs 68.2M for v8x) = batch=3-4 on T4 = much better generalization
  - Typically matches or beats v8x accuracy on same hardware budget
  - ONNX ~100MB = fits perfectly alongside v3 (100MB) in 420MB limit

Usage on VM:
    nohup python vm_train_v6.py --skip-prepare > train_v6.log 2>&1 &
    
    # Export after training
    python vm_train_v6.py --export-only
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


def train_yolo11l(data_yaml, epochs=300, batch=-1, patience=60):
    """Train YOLO11l with auto-batch for optimal T4 utilization."""
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

    model = YOLO("yolo11m.pt")
    print(f"\nV6: YOLO11m — ~20M params, auto-batch")
    print(f"  imgsz=1280, epochs={epochs}, batch={batch}, patience={patience}")

    results = model.train(
        data=str(data_yaml),
        imgsz=1280,
        epochs=epochs,
        batch=batch,            # auto-batch: maximize batch size for T4
        patience=patience,
        project="runs/detect",
        name="train_v6",
        exist_ok=True,

        # === Augmentation (proven settings from v3/v4) ===
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.3,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.1,
        close_mosaic=10,

        # === Loss weights ===
        cls=1.0,
        box=7.5,
        dfl=1.5,

        # === Optimizer ===
        optimizer="AdamW",
        lr0=0.001,           # Higher LR OK for smaller model + bigger batch
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,

        # === Regularization ===
        label_smoothing=0.1,
        dropout=0.1,

        # === Training ===
        save=True,
        val=True,
        plots=True,
        verbose=True,
        amp=True,
        workers=8,
    )

    best_pt = Path("runs/detect/train_v6/weights/best.pt")
    if best_pt.exists():
        print(f"\nBest model: {best_pt}")
        model_best = YOLO(str(best_pt))
        model_best.export(format="onnx", imgsz=1280, simplify=True)
        print("ONNX export complete!")
    return results


def export_only():
    """Export trained model to ONNX."""
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    from ultralytics import YOLO
    best_pt = Path("runs/detect/train_v6/weights/best.pt")
    if not best_pt.exists():
        print(f"ERROR: {best_pt} not found")
        return
    print(f"Exporting {best_pt} to ONNX...")
    model = YOLO(str(best_pt))
    model.export(format="onnx", imgsz=1280, simplify=True)
    onnx_path = best_pt.with_suffix(".onnx")
    print(f"Done: {onnx_path} ({onnx_path.stat().st_size / 1024**2:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="V6: YOLO11l training")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=-1, help="-1 for auto-batch")
    parser.add_argument("--patience", type=int, default=60)
    args = parser.parse_args()

    import os
    os.chdir(WORK_DIR)

    if args.export_only:
        export_only()
        return

    data_yaml = WORK_DIR / "data" / "yolo_v2" / "dataset.yaml"
    if not args.skip_prepare or not data_yaml.exists():
        data_yaml = prepare_yolo_dataset(
            WORK_DIR / "data" / "coco" / "train" / "annotations.json",
            WORK_DIR / "data" / "coco" / "train" / "images",
            WORK_DIR / "data" / "yolo_v2",
        )

    train_yolo11l(data_yaml, epochs=args.epochs, batch=args.batch, patience=args.patience)


if __name__ == "__main__":
    main()
