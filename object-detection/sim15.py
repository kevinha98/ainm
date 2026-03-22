"""Run 15-image simulation with detailed per-image diagnostics."""
import json
import time
import builtins
from pathlib import Path
from collections import defaultdict

from evaluate_local import evaluate_mAP, load_coco_ground_truth, compute_iou
from run import predict_yolo, list_images

# Write output to file directly (bypass PowerShell buffering)
_outf = open("sim15_result.txt", "w", encoding="utf-8")
_orig_print = builtins.print
def print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    kwargs['file'] = _outf
    kwargs['flush'] = True
    _orig_print(*args, **kwargs)

N_IMAGES = 15
IMG_DIR = Path("data/coco/train/images")
ANN_PATH = Path("data/coco/train/annotations.json")
WEIGHTS = Path("best.onnx")

# Pick 15 varied images (spread across the dataset)
all_imgs = sorted(list_images(IMG_DIR))
step = max(1, len(all_imgs) // N_IMAGES)
subset = [all_imgs[i * step] for i in range(N_IMAGES) if i * step < len(all_imgs)]
subset = subset[:N_IMAGES]
subset_ids = set(int(p.stem.split("_")[-1]) for p in subset)
print(f"Evaluating {len(subset_ids)} images: {sorted(subset_ids)}")

# Load ground truth
gt = load_coco_ground_truth(ANN_PATH, subset_ids)
print(f"Ground truth: {len(gt)} annotations across {len(subset_ids)} images")

# Build a temp dir with just these images (avoid modifying run.py)
import tempfile
tmp = Path(tempfile.mkdtemp())
for p in subset:
    # Use hard link or copy
    dst = tmp / p.name
    dst.write_bytes(p.read_bytes())

# Run inference
t0 = time.time()
preds = predict_yolo(tmp, WEIGHTS, detection_only=False)
elapsed = time.time() - t0
print(f"\nInference done: {len(preds)} preds in {elapsed:.1f}s ({elapsed/len(subset):.1f}s/img)")

# Group by image
gt_by_img = defaultdict(list)
for g in gt:
    gt_by_img[g["image_id"]].append(g)

pred_by_img = defaultdict(list)
for p in preds:
    pred_by_img[p["image_id"]].append(p)

# Per-image analysis
print("\n" + "=" * 80)
print("PER-IMAGE ANALYSIS")
print("=" * 80)

total_gt = 0
total_matched_det = 0
total_matched_cls = 0
total_preds = 0
total_fp = 0

for img_id in sorted(subset_ids):
    img_gt = gt_by_img[img_id]
    img_preds = pred_by_img.get(img_id, [])
    img_preds_sorted = sorted(img_preds, key=lambda x: -x["score"])

    n_gt = len(img_gt)
    n_pred = len(img_preds)

    # Match predictions to GT (detection: ignore category)
    gt_matched_det = set()
    gt_matched_cls = set()
    fp_count = 0

    for pred in img_preds_sorted:
        best_iou = 0
        best_idx = -1
        for idx, g in enumerate(img_gt):
            iou = compute_iou(pred["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= 0.5 and best_idx not in gt_matched_det:
            gt_matched_det.add(best_idx)
            if pred["category_id"] == img_gt[best_idx]["category_id"]:
                gt_matched_cls.add(best_idx)
        elif best_iou < 0.5:
            fp_count += 1

    det_recall = len(gt_matched_det) / n_gt if n_gt > 0 else 0
    cls_recall = len(gt_matched_cls) / n_gt if n_gt > 0 else 0
    missed = n_gt - len(gt_matched_det)

    total_gt += n_gt
    total_matched_det += len(gt_matched_det)
    total_matched_cls += len(gt_matched_cls)
    total_preds += n_pred
    total_fp += fp_count

    status = "PERFECT" if missed == 0 else f"MISSED {missed}"
    print(f"\nimg_{img_id:05d}: GT={n_gt:3d} | Preds={n_pred:3d} | "
          f"Det={len(gt_matched_det)}/{n_gt} ({det_recall:.0%}) | "
          f"Cls={len(gt_matched_cls)}/{n_gt} ({cls_recall:.0%}) | "
          f"FP={fp_count} | {status}")

    # Show missed GT objects
    if missed > 0:
        for idx, g in enumerate(img_gt):
            if idx not in gt_matched_det:
                print(f"   MISSED: cat={g['category_id']} bbox={g['bbox']}")

    # Show misclassified (detected but wrong category)
    misclassified = gt_matched_det - gt_matched_cls
    if misclassified:
        for idx in sorted(misclassified):
            g = img_gt[idx]
            # Find which pred matched this GT
            for pred in img_preds_sorted:
                iou = compute_iou(pred["bbox"], g["bbox"])
                if iou >= 0.5:
                    print(f"   MISCLASS: GT cat={g['category_id']} → Pred cat={pred['category_id']} "
                          f"(score={pred['score']:.2f}, iou={iou:.2f})")
                    break

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
det_recall = total_matched_det / total_gt if total_gt > 0 else 0
cls_recall = total_matched_cls / total_gt if total_gt > 0 else 0
precision = total_matched_det / total_preds if total_preds > 0 else 0

print(f"Total GT objects:       {total_gt}")
print(f"Total predictions:      {total_preds}")
print(f"Detection matches:      {total_matched_det}/{total_gt} ({det_recall:.1%})")
print(f"Classification matches: {total_matched_cls}/{total_gt} ({cls_recall:.1%})")
print(f"False positives:        {total_fp}")
print(f"Precision (det):        {precision:.1%}")
print(f"Avg preds/image:        {total_preds/len(subset_ids):.0f}")
print(f"Time/image:             {elapsed/len(subset):.1f}s")

# Compute actual mAP scores
det_score = evaluate_mAP(preds, gt, iou_threshold=0.5, ignore_category=True)
cls_score = evaluate_mAP(preds, gt, iou_threshold=0.5, ignore_category=False)
combined = 0.7 * det_score["mAP"] + 0.3 * cls_score["mAP"]

print(f"\nDetection  mAP@0.5: {det_score['mAP']:.4f}")
print(f"Classif.   mAP@0.5: {cls_score['mAP']:.4f}")
print(f"Combined Score:     {combined:.4f}")
print(f"  = 0.7 × {det_score['mAP']:.4f} + 0.3 × {cls_score['mAP']:.4f}")

# Cleanup
import shutil as _sh
_sh.rmtree(tmp, ignore_errors=True)
_outf.close()
