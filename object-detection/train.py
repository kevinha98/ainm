"""Training script for the NorgesGruppen object detection competition.

Designed for Google Colab (free T4 GPU) but works on any machine.
CRITICAL: Pin ultralytics==8.1.0 to match the sandbox exactly.

Usage (local):
    python train.py --data data/yolo/dataset.yaml

Usage (Colab — run these cells first):
    # Cell 1: Install
    !pip install ultralytics==8.1.0

    # Cell 2: Mount Drive & upload data
    from google.colab import drive
    drive.mount('/content/drive')
    # Upload NM_NGD_coco_dataset.zip to Drive, then:
    !unzip /content/drive/MyDrive/NM_NGD_coco_dataset.zip -d /content/data/coco

    # Cell 3: Convert COCO → YOLO
    !python prepare_data.py --annotations /content/data/coco/annotations.json \\
        --images-dir /content/data/coco/images --output-dir /content/data/yolo --copy-images

    # Cell 4: Train
    !python train.py --data /content/data/yolo/dataset.yaml --epochs 80

    # Cell 5: Copy best weights to Drive
    !cp runs/detect/train/weights/best.pt /content/drive/MyDrive/best.pt
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for NorgesGruppen competition")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base model (default: yolov8m.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project directory")
    parser.add_argument("--name", type=str, default="train", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Verify dataset exists
    if not args.data.exists():
        print(f"ERROR: Dataset YAML not found: {args.data}")
        print("Run prepare_data.py first to convert COCO → YOLO format.")
        return

    # Version check
    import ultralytics
    print(f"ultralytics version: {ultralytics.__version__}")
    if not ultralytics.__version__.startswith("8.1."):
        print(f"WARNING: Sandbox uses ultralytics==8.1.0, you have {ultralytics.__version__}")
        print("Weights may fail to load in sandbox. Pin with: pip install ultralytics==8.1.0")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected — training will be very slow.")
        print("Consider using Google Colab for free T4 GPU.")

    from ultralytics import YOLO

    # Load model
    if args.resume:
        last_pt = Path(args.project) / args.name / "weights" / "last.pt"
        if last_pt.exists():
            print(f"Resuming from {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"No checkpoint found at {last_pt}, starting fresh")
            model = YOLO(args.model)
    else:
        model = YOLO(args.model)

    print(f"\nTraining config:")
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.data}")
    print(f"  Image sz: {args.imgsz}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  Patience: {args.patience}")
    print(f"  Device:   {device}")
    print()

    # Train
    results = model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Augmentation
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,          # No rotation — shelves are always upright
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,           # No vertical flip — products don't appear upside down
        # Training params
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        # Saving
        save=True,
        save_period=-1,        # Save only best + last
        val=True,
        plots=True,
        verbose=True,
    )

    # Report results
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        size_mb = best_weights.stat().st_size / (1024 * 1024)
        print(f"\nBest weights: {best_weights} ({size_mb:.1f} MB)")

        if size_mb > 420:
            print("WARNING: Weights exceed 420 MB limit!")
            print("Export to FP16: model.export(format='onnx', half=True)")
            print("Or try a smaller model (yolov8s.pt)")
    else:
        print("\nWARNING: best.pt not found — training may have failed")

    print("\nNext steps:")
    print(f"  1. Copy {best_weights} to project root as best.pt")
    print(f"  2. Run: python evaluate_local.py")
    print(f"  3. Run: python package.py --weights best.pt")
    print(f"  4. Upload submissions/submission.zip")


if __name__ == "__main__":
    main()
