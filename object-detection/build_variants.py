"""Generate multiple run.py variants for A/B testing on competition.

Competition results show:
  sigma=0.949, wbf=0.4989, skip=0.004705 → 0.9208
  sigma=0.85,  wbf=0.50,   skip=0.02     → 0.9213

Lower sigma = more aggressive soft-NMS = fewer false positives on unseen data.
Higher skip_box = filters more low-quality WBF fused boxes.

This suggests local eval (29 images) overfits — competition test set is larger/harder.
Strategy: push toward fewer, higher-confidence detections.
"""
import shutil, re, zipfile
from pathlib import Path

BASE = Path("C:/ainm/object-detection")
RUN_PY = BASE / "run.py"
WEIGHTS = [BASE / "best.onnx", BASE / "best_v4.onnx", BASE / "best_v6.onnx"]
OUT_DIR = BASE / "submissions"

# Read base run.py
base_code = RUN_PY.read_text(encoding="utf-8")

# Parameter variants to test (name, sigma, wbf_iou, skip_box, snms_iou, score_thresh, max_dets)
VARIANTS = [
    # v1: sigma=0.75, moderate filtering — more aggressive than 0.85
    ("sigma75", 0.75, 0.50, 0.01, 0.35, 1e-5, 400),
    # v2: sigma=0.60, strong filtering — fewer boxes, higher precision
    ("sigma60", 0.60, 0.50, 0.015, 0.35, 1e-4, 350),
    # v3: sigma=0.85 + conf=0.01 — replicate winner but with conf fix
    ("sigma85_conf01", 0.85, 0.50, 0.02, 0.303, 6e-6, 450),
]


def make_variant(name, sigma, wbf_iou, skip_box, snms_iou, score_thresh, max_dets):
    code = base_code

    # Replace all _infer_full WBF params
    code = re.sub(
        r"wbf_fuse\(all_passes, img_w, img_h, iou_thresh=[0-9.]+, skip_box_thresh=[0-9e.\-]+,",
        f"wbf_fuse(all_passes, img_w, img_h, iou_thresh={wbf_iou}, skip_box_thresh={skip_box},",
        code
    )
    # Replace all _infer_full/medium/fast soft-NMS params
    code = re.sub(
        r"_soft_nms_class_agnostic\(fused, iou_thresh=[0-9.]+, sigma=[0-9.]+, score_thresh=[0-9e.\-]+\)",
        f"_soft_nms_class_agnostic(fused, iou_thresh={snms_iou}, sigma={sigma}, score_thresh={score_thresh})",
        code
    )
    code = re.sub(
        r"_soft_nms_class_agnostic\(dets, iou_thresh=[0-9.]+, sigma=[0-9.]+, score_thresh=[0-9e.\-]+\)",
        f"_soft_nms_class_agnostic(dets, iou_thresh={snms_iou}, sigma={sigma}, score_thresh={score_thresh})",
        code
    )
    # Also fix the exception fallback path soft-NMS
    code = re.sub(
        r"_soft_nms_class_agnostic\(nms_per_image\(all_dets, iou_thresh=0\.5\), iou_thresh=[0-9.]+, sigma=[0-9.]+, score_thresh=[0-9e.\-]+\)",
        f"_soft_nms_class_agnostic(nms_per_image(all_dets, iou_thresh=0.5), iou_thresh={snms_iou}, sigma={sigma}, score_thresh={score_thresh})",
        code
    )
    # Replace _infer_medium WBF params
    code = re.sub(
        r"wbf_fuse\(all_passes, img_w, img_h,\s*iou_thresh=[0-9.]+, skip_box_thresh=[0-9e.\-]+,",
        f"wbf_fuse(all_passes, img_w, img_h,\n                         iou_thresh={wbf_iou}, skip_box_thresh={skip_box},",
        code
    )
    # Replace MAX_DETS_PER_IMAGE
    code = re.sub(
        r"MAX_DETS_PER_IMAGE\s*=\s*\d+",
        f"MAX_DETS_PER_IMAGE = {max_dets}",
        code
    )

    # Build zip
    zip_path = OUT_DIR / f"submission_{name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("run.py", code)
        for w in WEIGHTS:
            zf.write(w, w.name)

    # Verify
    z = zipfile.ZipFile(zip_path)
    c = z.read("run.py").decode()
    has_sigma = f"sigma={sigma}" in c
    has_wbf = f"iou_thresh={wbf_iou}" in c
    has_skip = f"skip_box_thresh={skip_box}" in c
    has_max = f"MAX_DETS_PER_IMAGE = {max_dets}" in c
    size_mb = zip_path.stat().st_size / 1024 / 1024

    status = "OK" if all([has_sigma, has_wbf, has_skip, has_max]) else "FAIL"
    print(f"  {name}: {size_mb:.1f} MB | sigma={sigma} wbf={wbf_iou} skip={skip_box} max_dets={max_dets} | {status}")
    if status == "FAIL":
        print(f"    sigma={has_sigma} wbf={has_wbf} skip={has_skip} max={has_max}")
    return zip_path


print("Building submission variants...")
for v in VARIANTS:
    make_variant(*v)
print("\nDone! Submit in order of most promising:")
print("  1. sigma85_conf01 — closest to winner (0.9213) + conf fix")
print("  2. sigma75 — more aggressive, fewer false positives")
print("  3. sigma60 — most aggressive, high precision")
