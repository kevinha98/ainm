"""Compare old vs new run.py scoring on a small image subset."""
import json
import time
import shutil
import tempfile
from pathlib import Path

from evaluate_local import evaluate_mAP, load_coco_ground_truth

N_IMAGES = 3

# Prepare subset
img_dir = Path("data/coco/train/images")
all_imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
subset = all_imgs[:N_IMAGES]
subset_ids = set(int(p.stem.split("_")[-1]) for p in subset)
print(f"Evaluating on {len(subset_ids)} images: {sorted(subset_ids)}")

gt = load_coco_ground_truth(Path("data/coco/train/annotations.json"), subset_ids)
print(f"Ground truth: {len(gt)} annotations\n")

tmp = Path(tempfile.mkdtemp())
for p in subset:
    shutil.copy2(p, tmp / p.name)


def score_run(label, run_module_name):
    """Import and run a specific run.py variant."""
    import importlib
    mod = importlib.import_module(run_module_name)

    t0 = time.time()
    preds = mod.predict_yolo(tmp, Path("best.onnx"), detection_only=False)
    elapsed = time.time() - t0

    det = evaluate_mAP(preds, gt, iou_threshold=0.5, ignore_category=True)
    cls = evaluate_mAP(preds, gt, iou_threshold=0.5, ignore_category=False)
    combined = 0.7 * det["mAP"] + 0.3 * cls["mAP"]

    print(f"{'=' * 55}")
    print(f"  {label}")
    print(f"  Detection  mAP@0.5: {det['mAP']:.4f}")
    print(f"  Classif.   mAP@0.5: {cls['mAP']:.4f}")
    print(f"  Combined Score:     {combined:.4f}")
    print(f"  Predictions: {len(preds)}, Time: {elapsed:.1f}s")
    print(f"{'=' * 55}\n")
    return combined


# Test old version
old_score = score_run("OLD run.py (NMS, single-scale SAHI)", "run")

# Test new version
new_score = score_run("NEW run_v2.py (WBF, multi-scale SAHI)", "run_v2")

delta = new_score - old_score
print(f"Score delta: {delta:+.4f} ({'improvement' if delta > 0 else 'regression'})")

shutil.rmtree(tmp, ignore_errors=True)
