"""Local evaluation that mirrors the competition scoring.

Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

Detection mAP: IoU ≥ 0.5, category ignored (all preds treated as one class)
Classification mAP: IoU ≥ 0.5 AND correct category_id

Usage:
    # Evaluate fine-tuned model on validation set
    python evaluate_local.py --weights best.pt --data data/yolo/dataset.yaml

    # Evaluate with COCO annotations directly
    python evaluate_local.py --weights best.pt --images data/coco/images --annotations data/coco/annotations.json

    # Detection-only evaluation (pretrained COCO model)
    python evaluate_local.py --weights yolov8m.pt --detection-only
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two COCO-format boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap(precisions: list, recalls: list) -> float:
    """Compute AP using 101-point interpolation (COCO style)."""
    if not precisions:
        return 0.0

    # Add sentinel values
    precisions = [0.0] + precisions + [0.0]
    recalls = [0.0] + recalls + [1.0]

    # Monotone decreasing precision
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = 0.0
        for r, pr in zip(recalls, precisions):
            if r >= t:
                p = max(p, pr)
        ap += p
    return ap / 101


def evaluate_mAP(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
    ignore_category: bool = False,
) -> dict:
    """Compute mAP@0.5, optionally ignoring category (detection-only).

    Args:
        predictions: List of {image_id, category_id, bbox, score}
        ground_truth: List of {image_id, category_id, bbox}
        iou_threshold: IoU threshold for matching (default 0.5)
        ignore_category: If True, treat all predictions as same category (detection mAP)

    Returns:
        Dict with mAP, per-category APs, and summary stats
    """
    # Group by image
    gt_by_image = defaultdict(list)
    for gt in ground_truth:
        gt_by_image[gt["image_id"]].append(gt)

    pred_by_image = defaultdict(list)
    for pred in predictions:
        pred_by_image[pred["image_id"]].append(pred)

    if ignore_category:
        # Detection-only: merge all into one category
        all_gt = [{"image_id": g["image_id"], "category_id": 0, "bbox": g["bbox"]} for g in ground_truth]
        all_pred = [{"image_id": p["image_id"], "category_id": 0, "bbox": p["bbox"], "score": p["score"]}
                    for p in predictions]
        categories = {0}
        gt_by_cat_image = defaultdict(lambda: defaultdict(list))
        for g in all_gt:
            gt_by_cat_image[0][g["image_id"]].append(g)
        pred_by_cat = defaultdict(list)
        for p in all_pred:
            pred_by_cat[0].append(p)
    else:
        categories = set(g["category_id"] for g in ground_truth)
        gt_by_cat_image = defaultdict(lambda: defaultdict(list))
        for g in ground_truth:
            gt_by_cat_image[g["category_id"]][g["image_id"]].append(g)
        pred_by_cat = defaultdict(list)
        for p in predictions:
            pred_by_cat[p["category_id"]].append(p)

    # Compute AP per category
    aps = {}
    for cat_id in categories:
        cat_preds = sorted(pred_by_cat.get(cat_id, []), key=lambda x: -x["score"])
        cat_gt = gt_by_cat_image[cat_id]
        n_gt = sum(len(gts) for gts in cat_gt.values())

        if n_gt == 0:
            continue

        # Track which GT boxes are matched
        matched = defaultdict(set)
        tp_list = []
        fp_list = []

        for pred in cat_preds:
            img_id = pred["image_id"]
            img_gts = cat_gt.get(img_id, [])

            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(img_gts):
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_threshold and best_gt_idx not in matched[img_id]:
                tp_list.append(1)
                fp_list.append(0)
                matched[img_id].add(best_gt_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

        # Compute precision/recall curve
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        recalls = (tp_cumsum / n_gt).tolist()
        precisions = (tp_cumsum / (tp_cumsum + fp_cumsum)).tolist()

        aps[cat_id] = compute_ap(precisions, recalls)

    mAP = np.mean(list(aps.values())) if aps else 0.0

    return {
        "mAP": float(mAP),
        "num_categories_evaluated": len(aps),
        "num_predictions": len(predictions),
        "num_ground_truth": len(ground_truth),
    }


def load_coco_ground_truth(annotations_path: Path, image_ids: set = None) -> list[dict]:
    """Load ground truth from COCO annotations.json."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    gt = []
    for ann in coco["annotations"]:
        if image_ids and ann["image_id"] not in image_ids:
            continue
        gt.append({
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
        })
    return gt


def run_yolo_inference(weights: Path, images_dir: Path, detection_only: bool = False) -> list[dict]:
    """Run YOLOv8 inference and return predictions in COCO format."""
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights))
    predictions = []

    valid_ext = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in valid_ext)

    print(f"Running inference on {len(image_paths)} images (device={device})...")
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(image_paths)}")

        image_id = int(img_path.stem.split("_")[-1])
        results = model(str(img_path), device=device, verbose=False, conf=0.25, iou=0.45, max_det=300)

        for r in results:
            if r.boxes is None:
                continue
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
                cat_id = 0 if detection_only else int(r.boxes.cls[j].item())
                predictions.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "score": round(float(r.boxes.conf[j].item()), 3),
                })

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with competition-equivalent scoring")
    parser.add_argument("--weights", type=Path, required=True, help="Model weights (.pt)")
    parser.add_argument("--images", type=Path, default=None, help="Images directory")
    parser.add_argument("--annotations", type=Path, default=None, help="COCO annotations.json")
    parser.add_argument("--data", type=Path, default=None, help="YOLO dataset.yaml (alternative to --images)")
    parser.add_argument("--detection-only", action="store_true", help="Evaluate detection only (ignore categories)")
    parser.add_argument("--predictions-json", type=Path, default=None,
                        help="Pre-computed predictions JSON (skip inference)")
    args = parser.parse_args()

    # Resolve images dir and annotations
    if args.data and not args.images:
        # Parse dataset.yaml to find val images
        import yaml
        with open(args.data, "r") as f:
            ds = yaml.safe_load(f)
        base = Path(ds["path"])
        args.images = base / ds["val"]
        # For COCO annotations, we need the original
        from config import ANNOTATIONS_FILE
        args.annotations = ANNOTATIONS_FILE

    if not args.images:
        from config import IMAGES_DIR, ANNOTATIONS_FILE
        args.images = IMAGES_DIR
        args.annotations = args.annotations or ANNOTATIONS_FILE

    if not args.annotations:
        from config import ANNOTATIONS_FILE
        args.annotations = ANNOTATIONS_FILE

    if not args.annotations.exists():
        print(f"ERROR: Annotations not found: {args.annotations}")
        return

    # Get image IDs from the images directory
    valid_ext = {".jpg", ".jpeg", ".png"}
    image_ids = set()
    for p in args.images.iterdir():
        if p.suffix.lower() in valid_ext:
            image_ids.add(int(p.stem.split("_")[-1]))

    print(f"Evaluating on {len(image_ids)} images")

    # Load ground truth
    gt = load_coco_ground_truth(args.annotations, image_ids)
    print(f"Ground truth: {len(gt)} annotations")

    # Get predictions
    if args.predictions_json:
        with open(args.predictions_json, "r") as f:
            predictions = json.load(f)
        predictions = [p for p in predictions if p["image_id"] in image_ids]
    else:
        predictions = run_yolo_inference(args.weights, args.images, args.detection_only)

    print(f"Predictions: {len(predictions)}")

    # Compute scores
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    det_result = evaluate_mAP(predictions, gt, iou_threshold=0.5, ignore_category=True)
    print(f"\nDetection mAP@0.5:       {det_result['mAP']:.4f}")

    if not args.detection_only:
        cls_result = evaluate_mAP(predictions, gt, iou_threshold=0.5, ignore_category=False)
        print(f"Classification mAP@0.5:  {cls_result['mAP']:.4f}")
        print(f"  Categories evaluated:  {cls_result['num_categories_evaluated']}")

        combined = 0.7 * det_result["mAP"] + 0.3 * cls_result["mAP"]
        print(f"\nCombined Score:          {combined:.4f}")
        print(f"  (0.7 × {det_result['mAP']:.4f} + 0.3 × {cls_result['mAP']:.4f})")
    else:
        print(f"\nMax possible score (detection-only): {0.7 * det_result['mAP']:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
