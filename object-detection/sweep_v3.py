"""Sweep V3 — TTA, multi-scale, and model ensemble.

Current best: full_only + conf=0.03 → 0.8893 on test (24.6s / 270s budget).
Massive headroom for heavier inference.

New axes to sweep:
  A. Image size: 1280 vs 1536 vs 1920
  B. Horizontal flip TTA: original + flipped, WBF-fused
  C. Multi-scale TTA: run at two sizes, WBF-fused
  D. Model ensemble: v3 + v4, WBF-fused
  E. Power combos: flip + multi-scale + ensemble

Score = 0.7 × det_mAP@0.5 + 0.3 × cls_mAP@0.5

Usage:
    python sweep_v3.py --n-images 10
"""
import argparse
import json
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
from PIL import Image, ImageOps
from evaluate_local import evaluate_mAP, load_coco_ground_truth
from sweep_fast import (
    extract_dets, wbf_fuse, hard_nms_class_agnostic,
    soft_nms_class_agnostic, _iou, YOLO_VAL_DIR, ANN_PATH
)

WEIGHTS_V3 = Path("best.onnx")
WEIGHTS_V4 = Path("best_v4.onnx")


def load_model(weights_path):
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load
    from ultralytics import YOLO
    print(f"Loading {weights_path} for ONNX Runtime inference...")
    model = YOLO(str(weights_path), task="detect")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded on {device}")
    return model, device


def run_full_inference(model, img_path, device, imgsz=1280, conf=0.01):
    """Run full-image inference at given imgsz."""
    results = model(str(img_path), device=device, verbose=False,
                    conf=conf, iou=0.7, max_det=600, imgsz=imgsz, augment=False)
    return extract_dets(results)


def run_flip_inference(model, img_path, device, imgsz=1280, conf=0.01):
    """Run inference on horizontally-flipped image, mirror boxes back."""
    img = Image.open(img_path)
    img_w, img_h = img.size
    flipped = ImageOps.mirror(img)
    flip_arr = np.array(flipped)
    img.close()

    results = model(flip_arr, device=device, verbose=False,
                    conf=conf, iou=0.7, max_det=600, imgsz=imgsz, augment=False)
    raw_dets = extract_dets(results)

    # Mirror x-coordinates back
    for d in raw_dets:
        x, y, w, h = d["bbox"]
        d["bbox"] = [round(img_w - x - w, 1), y, w, h]
    return raw_dets


def cache_v3_passes(img_paths, model, device):
    """Cache v3 model at multiple scales + flipped versions."""
    cache = {}
    for idx, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        img_w, img_h = img.size
        image_id = int(img_path.stem.split("_")[-1])
        img.close()

        t0 = time.time()

        # Original at 1280 (our current best)
        full_1280 = run_full_inference(model, img_path, device, imgsz=1280)
        # Original at 1536
        full_1536 = run_full_inference(model, img_path, device, imgsz=1536)
        # Original at 1920
        full_1920 = run_full_inference(model, img_path, device, imgsz=1920)
        # Flipped at 1280
        flip_1280 = run_flip_inference(model, img_path, device, imgsz=1280)
        # Flipped at 1536
        flip_1536 = run_flip_inference(model, img_path, device, imgsz=1536)

        cache[image_id] = {
            "full_1280": full_1280,
            "full_1536": full_1536,
            "full_1920": full_1920,
            "flip_1280": flip_1280,
            "flip_1536": flip_1536,
            "img_w": img_w,
            "img_h": img_h,
        }
        elapsed = time.time() - t0
        print(f"  [{idx+1}/{len(img_paths)}] img_{image_id:05d}: "
              f"1280={len(full_1280)}, 1536={len(full_1536)}, 1920={len(full_1920)}, "
              f"flip1280={len(flip_1280)}, flip1536={len(flip_1536)} | {elapsed:.1f}s")

    return cache


def cache_v4_passes(img_paths, model, device):
    """Cache v4 model at 1280 + flip."""
    cache = {}
    for idx, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        img_w, img_h = img.size
        image_id = int(img_path.stem.split("_")[-1])
        img.close()

        full_1280 = run_full_inference(model, img_path, device, imgsz=1280)
        flip_1280 = run_flip_inference(model, img_path, device, imgsz=1280)

        cache[image_id] = {
            "full_1280": full_1280,
            "flip_1280": flip_1280,
            "img_w": img_w,
            "img_h": img_h,
        }
        print(f"  v4 [{idx+1}/{len(img_paths)}] img_{image_id:05d}: "
              f"1280={len(full_1280)}, flip={len(flip_1280)}")
    return cache


def apply_conf_and_nms(pass_list, img_w, img_h, conf=0.03, wbf_iou=0.30, nms_iou=0.45, max_dets=300):
    """Apply conf filter, WBF (if >1 pass), then class-agnostic NMS."""
    filtered = []
    for pass_dets in pass_list:
        fd = [d for d in pass_dets if d["score"] >= conf]
        if fd:
            filtered.append(fd)

    if not filtered:
        return []

    if len(filtered) == 1:
        # Single pass — just NMS, no WBF needed
        dets = hard_nms_class_agnostic(filtered[0], iou_thresh=nms_iou)
    else:
        # Multiple passes — WBF then NMS
        try:
            fused = wbf_fuse(filtered, img_w, img_h,
                             iou_thresh=wbf_iou, skip_box_thresh=0.01,
                             conf_type='max')
            dets = hard_nms_class_agnostic(fused, iou_thresh=nms_iou)
        except Exception:
            all_dets = []
            for f in filtered:
                all_dets.extend(f)
            dets = hard_nms_class_agnostic(all_dets, iou_thresh=nms_iou)

    if len(dets) > max_dets:
        dets.sort(key=lambda x: x["score"], reverse=True)
        dets = dets[:max_dets]
    return dets


def run_config(cache_v3, cache_v4, config):
    """Run a configuration on cached data."""
    passes_keys = config["passes"]  # e.g. ["full_1280"], ["full_1280", "flip_1280"], etc.
    conf = config.get("conf", 0.03)
    wbf_iou = config.get("wbf_iou", 0.30)
    nms_iou = config.get("nms_iou", 0.45)
    max_dets = config.get("max_dets", 300)
    use_v4 = config.get("use_v4", False)
    v4_passes = config.get("v4_passes", [])

    all_preds = []
    for image_id in cache_v3:
        data = cache_v3[image_id]
        img_w, img_h = data["img_w"], data["img_h"]

        pass_list = []
        for key in passes_keys:
            if key in data:
                pass_list.append(data[key])

        # Add v4 model passes if requested
        if use_v4 and cache_v4 and image_id in cache_v4:
            v4_data = cache_v4[image_id]
            for key in v4_passes:
                if key in v4_data:
                    pass_list.append(v4_data[key])

        dets = apply_conf_and_nms(pass_list, img_w, img_h,
                                   conf=conf, wbf_iou=wbf_iou,
                                   nms_iou=nms_iou, max_dets=max_dets)
        for d in dets:
            d["image_id"] = image_id
            all_preds.append(d)

    return all_preds


def evaluate_config(cache_v3, cache_v4, gt, config, label=""):
    t0 = time.time()
    preds = run_config(cache_v3, cache_v4, config)
    # Filter GT to only images we have predictions for
    pred_img_ids = set(cache_v3.keys())
    gt_filtered = [g for g in gt if g["image_id"] in pred_img_ids]
    det = evaluate_mAP(preds, gt_filtered, iou_threshold=0.5, ignore_category=True)
    cls = evaluate_mAP(preds, gt_filtered, iou_threshold=0.5, ignore_category=False)
    combined = 0.7 * det["mAP"] + 0.3 * cls["mAP"]
    elapsed = time.time() - t0
    return {"label": label, "combined": combined, "det_mAP": det["mAP"],
            "cls_mAP": cls["mAP"], "n_preds": len(preds), "time": elapsed}


def get_v3_configs():
    """Generate all configs to sweep."""
    configs = []

    # ── A. BASELINE: full_only @ 1280, conf=0.03 (our 0.8893 submission) ──
    configs.append(("BASELINE (full1280_c=0.03)", {
        "passes": ["full_1280"], "conf": 0.03
    }))

    # ── B. SINGLE SCALE VARIANTS ──
    for imgsz in [1536, 1920]:
        configs.append((f"full_{imgsz}_c=0.03", {
            "passes": [f"full_{imgsz}"], "conf": 0.03
        }))
        configs.append((f"full_{imgsz}_c=0.05", {
            "passes": [f"full_{imgsz}"], "conf": 0.05
        }))

    # ── C. HORIZONTAL FLIP TTA ──
    # Original + flipped at same scale, WBF-fused
    for imgsz in [1280, 1536]:
        for wbf in [0.25, 0.30, 0.35, 0.40, 0.50]:
            configs.append((f"tta_{imgsz}_wbf={wbf}", {
                "passes": [f"full_{imgsz}", f"flip_{imgsz}"],
                "conf": 0.03, "wbf_iou": wbf
            }))

    # TTA with higher conf
    for imgsz in [1280, 1536]:
        configs.append((f"tta_{imgsz}_c=0.05_wbf=0.35", {
            "passes": [f"full_{imgsz}", f"flip_{imgsz}"],
            "conf": 0.05, "wbf_iou": 0.35
        }))

    # ── D. MULTI-SCALE (no flip) ──
    # Two scales fused with WBF
    for wbf in [0.25, 0.30, 0.35, 0.40, 0.50]:
        configs.append((f"ms_1280+1536_wbf={wbf}", {
            "passes": ["full_1280", "full_1536"], "conf": 0.03, "wbf_iou": wbf
        }))
    for wbf in [0.30, 0.40, 0.50]:
        configs.append((f"ms_1280+1920_wbf={wbf}", {
            "passes": ["full_1280", "full_1920"], "conf": 0.03, "wbf_iou": wbf
        }))
    # Three scales
    for wbf in [0.30, 0.40, 0.50]:
        configs.append((f"ms_1280+1536+1920_wbf={wbf}", {
            "passes": ["full_1280", "full_1536", "full_1920"],
            "conf": 0.03, "wbf_iou": wbf
        }))

    # ── E. MULTI-SCALE + FLIP TTA ──
    # Two scales + their flips (4 passes total)
    for wbf in [0.30, 0.35, 0.40, 0.50]:
        configs.append((f"ms+tta_1280+1536_wbf={wbf}", {
            "passes": ["full_1280", "full_1536", "flip_1280", "flip_1536"],
            "conf": 0.03, "wbf_iou": wbf
        }))

    # ── F. CONF SWEEP on best TTA configs ──
    for conf in [0.01, 0.02, 0.05, 0.08]:
        configs.append((f"tta_1280_c={conf}_wbf=0.35", {
            "passes": ["full_1280", "flip_1280"],
            "conf": conf, "wbf_iou": 0.35
        }))

    # ── G. NMS VARIANTS on TTA ──
    for nms_iou in [0.35, 0.40, 0.50, 0.55]:
        configs.append((f"tta_1280_nms={nms_iou}", {
            "passes": ["full_1280", "flip_1280"],
            "conf": 0.03, "wbf_iou": 0.35, "nms_iou": nms_iou
        }))

    # ── H. MAX DETS ──
    for md in [200, 400, 500]:
        configs.append((f"tta_1280_maxd={md}", {
            "passes": ["full_1280", "flip_1280"],
            "conf": 0.03, "wbf_iou": 0.35, "max_dets": md
        }))

    return configs


def get_ensemble_configs():
    """Configs that use v4 model too."""
    configs = []

    # v4 alone
    configs.append(("v4_full_1280_c=0.03", {
        "passes": [], "use_v4": True, "v4_passes": ["full_1280"],
        "conf": 0.03
    }))

    # v3 + v4 ensemble
    for wbf in [0.30, 0.35, 0.40, 0.50]:
        configs.append((f"v3+v4_wbf={wbf}", {
            "passes": ["full_1280"], "use_v4": True, "v4_passes": ["full_1280"],
            "conf": 0.03, "wbf_iou": wbf
        }))

    # v3 + v4 both with flip TTA
    for wbf in [0.35, 0.40, 0.50]:
        configs.append((f"v3+v4_tta_wbf={wbf}", {
            "passes": ["full_1280", "flip_1280"],
            "use_v4": True, "v4_passes": ["full_1280", "flip_1280"],
            "conf": 0.03, "wbf_iou": wbf
        }))

    # v3 multi-scale + v4
    for wbf in [0.35, 0.40]:
        configs.append((f"v3ms+v4_wbf={wbf}", {
            "passes": ["full_1280", "full_1536"],
            "use_v4": True, "v4_passes": ["full_1280"],
            "conf": 0.03, "wbf_iou": wbf
        }))

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--skip-v4", action="store_true", help="Skip v4 model caching")
    args = parser.parse_args()

    # Select val images
    all_imgs = sorted(YOLO_VAL_DIR.glob("*.jpg"))
    if not all_imgs:
        all_imgs = sorted(YOLO_VAL_DIR.glob("*.png"))
    n = min(args.n_images, len(all_imgs))
    np.random.seed(42)
    indices = np.random.choice(len(all_imgs), n, replace=False)
    img_paths = [all_imgs[i] for i in sorted(indices)]
    print(f"Using {n} images from {len(all_imgs)} available")

    # Load ground truth
    gt = load_coco_ground_truth(str(ANN_PATH))

    # Cache v3 model
    model_v3, device = load_model(WEIGHTS_V3)
    # Warm up
    model_v3(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)

    print(f"\nCaching v3 model passes (5 passes per image)...")
    t0 = time.time()
    cache_v3 = cache_v3_passes(img_paths, model_v3, device)
    print(f"V3 caching done in {time.time()-t0:.0f}s\n")

    # Cache v4 model
    cache_v4 = None
    if not args.skip_v4 and WEIGHTS_V4.exists():
        model_v4, _ = load_model(WEIGHTS_V4)
        model_v4(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
        print(f"\nCaching v4 model passes...")
        t0 = time.time()
        cache_v4 = cache_v4_passes(img_paths, model_v4, device)
        print(f"V4 caching done in {time.time()-t0:.0f}s\n")
        del model_v4
    else:
        print("Skipping v4 model\n")

    del model_v3

    # Get configs
    configs = get_v3_configs()
    if cache_v4:
        configs.extend(get_ensemble_configs())

    print(f"Sweeping {len(configs)} configurations...")
    print("=" * 100)

    results = []
    for label, config in configs:
        res = evaluate_config(cache_v3, cache_v4, gt, config, label)
        results.append(res)
        print(f"  {label:52s} | Score={res['combined']:.4f} | "
              f"Det={res['det_mAP']:.4f} | Cls={res['cls_mAP']:.4f} | "
              f"N={res['n_preds']:5d} | {res['time']:.2f}s")

    # Sort and display top 25
    results.sort(key=lambda x: x["combined"], reverse=True)
    baseline_score = next((r["combined"] for r in results if "BASELINE" in r["label"]), results[-1]["combined"])

    print("\n" + "=" * 100)
    print(f"TOP 25 CONFIGURATIONS (baseline = {baseline_score:.4f})")
    print("=" * 100)
    for i, r in enumerate(results[:25]):
        delta = r["combined"] - baseline_score
        print(f"  #{i+1:2d} {r['label']:52s} | {r['combined']:.4f} ({delta:+.4f}) | "
              f"Det={r['det_mAP']:.4f} | Cls={r['cls_mAP']:.4f}")

    print("\n" + "=" * 100)
    print("BOTTOM 5")
    print("=" * 100)
    for r in results[-5:]:
        delta = r["combined"] - baseline_score
        print(f"  {r['label']:52s} | {r['combined']:.4f} ({delta:+.4f})")

    # Save
    out_path = Path("sweep_v3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\nWINNER: {results[0]['label']} -> {results[0]['combined']:.4f} (baseline={baseline_score:.4f})")


if __name__ == "__main__":
    main()
