"""Sweep v2 — WBF IoU=0.30 is the new baseline (scored 0.8409 on test).

Tests next improvements on top of the proven config.
Uses 10 images for better test-set correlation.

Usage:
    python sweep_v2.py --n-images 10
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from evaluate_local import evaluate_mAP, load_coco_ground_truth
from sweep_fast import (
    cache_all_passes, wbf_fuse, hard_nms_class_agnostic,
    soft_nms_class_agnostic, run_pipeline, evaluate_config,
)

YOLO_VAL_DIR = Path("data/yolo/images/val")
ANN_PATH = Path("data/coco/train/annotations.json")
WEIGHTS = Path("best.onnx")


def get_v2_configs():
    """New baseline = WBF IoU=0.30, the 0.8409 config."""
    base = {
        "conf_full": 0.08, "conf_sahi": 0.10,
        "wbf_iou": 0.30, "wbf_skip": 0.01, "wbf_conf_type": "max",
        "nms_type": "hard", "nms_iou": 0.45, "nms_sigma": 0.5,
        "max_dets": 200, "passes": [1, 2, 3],
    }

    configs = []
    configs.append(("BASELINE_v2 (wbf=0.30)", {**base}))

    # ── A. Pass combinations (biggest lever in v1) ──
    for passes, label in [
        ([1], "full_only"),
        ([1, 3], "full+960"),
        ([1, 2], "full+640"),
    ]:
        configs.append((f"passes={label}", {**base, "passes": passes}))

    # ── B. WBF IoU fine-tuning around 0.30 ──
    for t in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32, 0.35, 0.40]:
        configs.append((f"wbf_iou={t:.2f}", {**base, "wbf_iou": t}))

    # ── C. Soft-NMS (showed +2.7% in v1) ──
    for sigma in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        configs.append((f"soft_nms_s={sigma}", {**base, "nms_type": "soft_gaussian", "nms_sigma": sigma}))
    configs.append(("soft_nms_linear", {**base, "nms_type": "soft_linear"}))

    # ── D. Hard NMS IoU tuning ──
    for t in [0.35, 0.40, 0.50, 0.55]:
        configs.append((f"nms_iou={t:.2f}", {**base, "nms_iou": t}))

    # ── E. Confidence thresholds ──
    for f, s in [(0.03, 0.05), (0.05, 0.07), (0.05, 0.10), (0.06, 0.08),
                 (0.10, 0.12), (0.12, 0.15)]:
        configs.append((f"conf={f:.2f}/{s:.2f}", {**base, "conf_full": f, "conf_sahi": s}))

    # ── F. Max detections ──
    for m in [150, 250, 300, 400]:
        configs.append((f"max_dets={m}", {**base, "max_dets": m}))

    # ── G. POWER COMBOS ──

    # G1: full_only + various conf thresholds
    for conf in [0.03, 0.05, 0.08, 0.10, 0.15]:
        configs.append((f"full_only_conf={conf:.2f}",
            {**base, "passes": [1], "conf_full": conf}))

    # G2: full+960 + WBF tuning
    for wbf_iou in [0.20, 0.25, 0.30, 0.35]:
        configs.append((f"full+960_wbf={wbf_iou:.2f}",
            {**base, "passes": [1, 3], "wbf_iou": wbf_iou}))

    # G3: full+960 + soft-NMS
    for sigma in [0.5, 1.0]:
        for wbf_iou in [0.25, 0.30]:
            configs.append((f"full+960_snms={sigma}_wbf={wbf_iou:.2f}",
                {**base, "passes": [1, 3], "nms_type": "soft_gaussian",
                 "nms_sigma": sigma, "wbf_iou": wbf_iou}))

    # G4: all passes + soft-NMS + WBF tuning
    for sigma in [0.5, 1.0]:
        for wbf_iou in [0.20, 0.25, 0.30, 0.35]:
            configs.append((f"all_snms={sigma}_wbf={wbf_iou:.2f}",
                {**base, "nms_type": "soft_gaussian", "nms_sigma": sigma,
                 "wbf_iou": wbf_iou}))

    # G5: full+960 + lower conf + soft-NMS
    for sigma in [0.5, 1.0]:
        for f, s in [(0.05, 0.07), (0.06, 0.08)]:
            configs.append((f"full+960_snms={sigma}_c={f}/{s}",
                {**base, "passes": [1, 3], "nms_type": "soft_gaussian",
                 "nms_sigma": sigma, "conf_full": f, "conf_sahi": s}))

    # G6: full_only + soft-NMS
    for sigma in [0.5, 1.0]:
        configs.append((f"full_only_snms={sigma}",
            {**base, "passes": [1], "nms_type": "soft_gaussian", "nms_sigma": sigma}))

    # G7: all passes + soft-NMS + lower conf + WBF fine-tune
    for sigma in [0.5, 1.0]:
        for wbf_iou in [0.25, 0.30]:
            for f, s in [(0.05, 0.07), (0.06, 0.08)]:
                configs.append((f"all_snms={sigma}_wbf={wbf_iou}_c={f}/{s}",
                    {**base, "nms_type": "soft_gaussian", "nms_sigma": sigma,
                     "wbf_iou": wbf_iou, "conf_full": f, "conf_sahi": s}))

    # G8: max_dets combos with best single-param winners
    for m in [250, 300]:
        configs.append((f"full+960_max={m}",
            {**base, "passes": [1, 3], "max_dets": m}))
        configs.append((f"snms=0.5_max={m}",
            {**base, "nms_type": "soft_gaussian", "nms_sigma": 0.5, "max_dets": m}))

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=10)
    args = parser.parse_args()

    valid_ext = {".jpg", ".jpeg", ".png"}
    img_paths = sorted(p for p in YOLO_VAL_DIR.iterdir() if p.suffix.lower() in valid_ext)
    n = min(args.n_images, len(img_paths))
    if n < len(img_paths):
        step = max(1, len(img_paths) // n)
        img_paths = [img_paths[i * step] for i in range(n) if i * step < len(img_paths)]

    img_ids = set(int(p.stem.split("_")[-1]) for p in img_paths)
    gt = load_coco_ground_truth(ANN_PATH, img_ids)
    print(f"Images: {len(img_paths)} | GT annotations: {len(gt)}")

    import torch
    _ol = torch.load
    def _pl(*a, **k):
        if 'weights_only' not in k: k['weights_only'] = False
        return _ol(*a, **k)
    torch.load = _pl
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(WEIGHTS), task="detect")
    model(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
    print(f"Model loaded on {device}")

    print("\nCaching raw detections (conf=0.01, one-time)...")
    t0 = time.time()
    cache = cache_all_passes(img_paths, model, device)
    print(f"Caching done in {time.time()-t0:.0f}s")

    configs = get_v2_configs()
    print(f"\nSweeping {len(configs)} configurations...")
    print("=" * 100)

    results = []
    for label, config in configs:
        t0 = time.time()
        r = evaluate_config(cache, gt, config, label)
        elapsed = time.time() - t0
        results.append(r)
        print(f"  {label:50s} | Score={r['combined']:.4f} | Det={r['det_mAP']:.4f} | "
              f"Cls={r['cls_mAP']:.4f} | N={r['n_preds']:4d} | {elapsed:.2f}s")

    results.sort(key=lambda r: r["combined"], reverse=True)
    baseline = next((r for r in results if "BASELINE" in r["label"]), results[0])

    print(f"\n{'='*100}")
    print(f"TOP 25 CONFIGURATIONS (baseline = {baseline['combined']:.4f})")
    print(f"{'='*100}")
    for i, r in enumerate(results[:25]):
        delta = r["combined"] - baseline["combined"]
        print(f"  #{i+1:2d} {r['label']:50s} | {r['combined']:.4f} ({delta:+.4f}) | "
              f"Det={r['det_mAP']:.4f} | Cls={r['cls_mAP']:.4f}")

    print(f"\n{'='*100}")
    print("BOTTOM 5")
    print(f"{'='*100}")
    for i, r in enumerate(results[-5:]):
        delta = r["combined"] - baseline["combined"]
        print(f"  #{len(results)-4+i:2d} {r['label']:50s} | {r['combined']:.4f} ({delta:+.4f})")

    with open("sweep_v2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to sweep_v2_results.json")

    w = results[0]
    print(f"\nWINNER: {w['label']} -> {w['combined']:.4f} (baseline={baseline['combined']:.4f})")


if __name__ == "__main__":
    main()
