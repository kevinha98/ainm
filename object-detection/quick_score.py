"""Quick score estimate using actual run.py WBF+SAHI pipeline."""
import shutil
import tempfile
import time
from pathlib import Path

from evaluate_local import evaluate_mAP, load_coco_ground_truth
from run import predict_yolo

N_IMAGES = 1  # CPU is slow with WBF+SAHI

img_dir = Path("data/coco/train/images")
all_imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
subset = all_imgs[:N_IMAGES]
subset_ids = set(int(p.stem.split("_")[-1]) for p in subset)
print(f"Evaluating on {len(subset_ids)} images: {sorted(subset_ids)}")

gt = load_coco_ground_truth(Path("data/coco/train/annotations.json"), subset_ids)
print(f"Ground truth: {len(gt)} annotations")

tmp = Path(tempfile.mkdtemp())
for p in subset:
    shutil.copy2(p, tmp / p.name)

t0 = time.time()
preds = predict_yolo(tmp, Path("best.onnx"), detection_only=False)
elapsed = time.time() - t0
print(f"Inference: {len(preds)} predictions in {elapsed:.1f}s")

det = evaluate_mAP(preds, gt, iou_threshold=0.5, ignore_category=True)
cls = evaluate_mAP(preds, gt, iou_threshold=0.5, ignore_category=False)
combined = 0.7 * det["mAP"] + 0.3 * cls["mAP"]

print()
print("=" * 50)
print(f"Detection  mAP@0.5: {det['mAP']:.4f}")
print(f"Classif.   mAP@0.5: {cls['mAP']:.4f}")
print(f"Combined Score:     {combined:.4f}")
print(f"  = 0.7 x {det['mAP']:.4f} + 0.3 x {cls['mAP']:.4f}")
print(f"Categories evaluated: {cls['num_categories_evaluated']}")
print(f"Dets per image: {len(preds) / N_IMAGES:.0f}")
print(f"Time per image: {elapsed / N_IMAGES:.1f}s (CPU)")
print("=" * 50)

shutil.rmtree(tmp, ignore_errors=True)
