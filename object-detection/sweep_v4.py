"""Sweep V4 — Fine-grained ensemble tuning + soft-NMS.

Current best: v3@1280 + v3@1536 + v4@1280, WBF=0.35, hard NMS → 0.9244 local, 0.9047 test.
Gap to leader: 0.0208 (0.9047 → 0.9255).

New axes:
  A. v4 at 1536 (never tested — could add diversity)
  B. v3 at 1408 (intermediate scale for denser coverage)
  C. Soft-NMS instead of hard NMS (sweep v2 showed +1.4% cls_mAP)
  D. Fine WBF IoU: 0.30-0.42 in steps of 0.02
  E. WBF conf_type: 'max', 'avg', 'box_and_model_avg'
  F. Soft-NMS sigma tuning
  G. 4-5 scale ensembles

Score = 0.7 × det_mAP@0.5 + 0.3 × cls_mAP@0.5

Usage:
    python sweep_v4.py --n-images 10
    python sweep_v4.py --n-images 15 --skip-1408  # skip intermediate scale
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
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
    print(f"Loading {weights_path}...")
    model = YOLO(str(weights_path), task="detect")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  → {device}")
    return model, device


def run_full_inference(model, img_path, device, imgsz=1280, conf=0.01):
    results = model(str(img_path), device=device, verbose=False,
                    conf=conf, iou=0.7, max_det=600, imgsz=imgsz, augment=False)
    return extract_dets(results)


def cache_model_passes(img_paths, model, device, scales, model_name="v3"):
    """Cache model inference at multiple scales."""
    cache = {}
    for idx, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        img_w, img_h = img.size
        image_id = int(img_path.stem.split("_")[-1])
        img.close()

        t0 = time.time()
        entry = {"img_w": img_w, "img_h": img_h}

        parts = []
        for s in scales:
            key = f"full_{s}"
            dets = run_full_inference(model, img_path, device, imgsz=s)
            entry[key] = dets
            parts.append(f"{s}={len(dets)}")

        cache[image_id] = entry
        elapsed = time.time() - t0
        print(f"  {model_name} [{idx+1}/{len(img_paths)}] img_{image_id:05d}: "
              f"{', '.join(parts)} | {elapsed:.1f}s")

    return cache


def apply_pipeline(pass_list, img_w, img_h, conf=0.03, wbf_iou=0.35,
                   nms_type="hard", nms_iou=0.45, nms_sigma=0.5,
                   max_dets=300, conf_type='max'):
    """WBF → NMS pipeline with configurable NMS type."""
    filtered = []
    for pass_dets in pass_list:
        fd = [d for d in pass_dets if d["score"] >= conf]
        if fd:
            filtered.append(fd)

    if not filtered:
        return []

    if len(filtered) == 1:
        dets = filtered[0]
    else:
        try:
            dets = wbf_fuse(filtered, img_w, img_h,
                            iou_thresh=wbf_iou, skip_box_thresh=0.005,
                            conf_type=conf_type)
        except Exception:
            dets = []
            for f in filtered:
                dets.extend(f)

    # Apply NMS
    if nms_type == "soft":
        dets = soft_nms_class_agnostic(dets, iou_thresh=nms_iou, sigma=nms_sigma)
    else:
        dets = hard_nms_class_agnostic(dets, iou_thresh=nms_iou)

    if len(dets) > max_dets:
        dets.sort(key=lambda x: x["score"], reverse=True)
        dets = dets[:max_dets]
    return dets


def run_config(cache_v3, cache_v4, config):
    """Run a configuration on cached data."""
    v3_passes = config.get("v3_passes", ["full_1280"])
    v4_passes = config.get("v4_passes", [])
    conf = config.get("conf", 0.03)
    wbf_iou = config.get("wbf_iou", 0.35)
    nms_type = config.get("nms_type", "hard")
    nms_iou = config.get("nms_iou", 0.45)
    nms_sigma = config.get("nms_sigma", 0.5)
    max_dets = config.get("max_dets", 300)
    conf_type = config.get("conf_type", "max")

    all_preds = []
    for image_id in cache_v3:
        v3_data = cache_v3[image_id]
        img_w, img_h = v3_data["img_w"], v3_data["img_h"]

        pass_list = []
        for key in v3_passes:
            if key in v3_data:
                pass_list.append(v3_data[key])

        if cache_v4 and image_id in cache_v4:
            v4_data = cache_v4[image_id]
            for key in v4_passes:
                if key in v4_data:
                    pass_list.append(v4_data[key])

        dets = apply_pipeline(pass_list, img_w, img_h, conf=conf,
                              wbf_iou=wbf_iou, nms_type=nms_type,
                              nms_iou=nms_iou, nms_sigma=nms_sigma,
                              max_dets=max_dets, conf_type=conf_type)
        for d in dets:
            d["image_id"] = image_id
            all_preds.append(d)

    return all_preds


def evaluate_config(cache_v3, cache_v4, gt, config, label=""):
    t0 = time.time()
    preds = run_config(cache_v3, cache_v4, config)
    pred_img_ids = set(cache_v3.keys())
    gt_filtered = [g for g in gt if g["image_id"] in pred_img_ids]
    det = evaluate_mAP(preds, gt_filtered, iou_threshold=0.5, ignore_category=True)
    cls = evaluate_mAP(preds, gt_filtered, iou_threshold=0.5, ignore_category=False)
    combined = 0.7 * det["mAP"] + 0.3 * cls["mAP"]
    elapsed = time.time() - t0
    return {"label": label, "combined": combined, "det_mAP": det["mAP"],
            "cls_mAP": cls["mAP"], "n_preds": len(preds), "time": elapsed}


def get_configs(has_1408=True, has_v4_1536=True):
    """Generate all sweep v4 configs."""
    configs = []

    # ── A. BASELINE: current best (v3@1280+1536 + v4@1280, WBF=0.35, hard NMS)
    configs.append(("BASELINE_hard", {
        "v3_passes": ["full_1280", "full_1536"],
        "v4_passes": ["full_1280"],
        "conf": 0.03, "wbf_iou": 0.35, "nms_type": "hard"
    }))

    # ── B. SOFT-NMS on current best (the big hypothesis)
    for sigma in [0.3, 0.5, 0.7]:
        configs.append((f"BASELINE_soft_s={sigma}", {
            "v3_passes": ["full_1280", "full_1536"],
            "v4_passes": ["full_1280"],
            "conf": 0.03, "wbf_iou": 0.35, "nms_type": "soft", "nms_sigma": sigma
        }))

    # ── C. FINE WBF TUNING on current best (hard NMS)
    for wbf in [0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42]:
        configs.append((f"wbf={wbf}_hard", {
            "v3_passes": ["full_1280", "full_1536"],
            "v4_passes": ["full_1280"],
            "conf": 0.03, "wbf_iou": wbf, "nms_type": "hard"
        }))

    # ── D. FINE WBF + SOFT-NMS
    for wbf in [0.30, 0.32, 0.34, 0.36, 0.38, 0.40]:
        configs.append((f"wbf={wbf}_soft", {
            "v3_passes": ["full_1280", "full_1536"],
            "v4_passes": ["full_1280"],
            "conf": 0.03, "wbf_iou": wbf, "nms_type": "soft", "nms_sigma": 0.5
        }))

    # ── E. WBF conf_type variations
    for ct in ["avg", "box_and_model_avg"]:
        configs.append((f"conftype={ct}_wbf=0.35", {
            "v3_passes": ["full_1280", "full_1536"],
            "v4_passes": ["full_1280"],
            "conf": 0.03, "wbf_iou": 0.35, "conf_type": ct
        }))
        configs.append((f"conftype={ct}_wbf=0.35_soft", {
            "v3_passes": ["full_1280", "full_1536"],
            "v4_passes": ["full_1280"],
            "conf": 0.03, "wbf_iou": 0.35, "conf_type": ct,
            "nms_type": "soft", "nms_sigma": 0.5
        }))

    # ── F. NMS IoU tuning
    for nms_iou in [0.35, 0.40, 0.50, 0.55]:
        configs.append((f"nms_iou={nms_iou}_hard", {
            "v3_passes": ["full_1280", "full_1536"],
            "v4_passes": ["full_1280"],
            "conf": 0.03, "wbf_iou": 0.35, "nms_iou": nms_iou, "nms_type": "hard"
        }))
        configs.append((f"nms_iou={nms_iou}_soft", {
            "v3_passes": ["full_1280", "full_1536"],
            "v4_passes": ["full_1280"],
            "conf": 0.03, "wbf_iou": 0.35, "nms_iou": nms_iou,
            "nms_type": "soft", "nms_sigma": 0.5
        }))

    # ── G. v4@1536 (new scale!)
    if has_v4_1536:
        # v3 duo + v4@1536 only
        configs.append(("v3ms+v4@1536_wbf=0.35", {
            "v3_passes": ["full_1280", "full_1536"],
            "v4_passes": ["full_1536"],
            "conf": 0.03, "wbf_iou": 0.35
        }))
        # v3 duo + v4 duo (4 passes)
        for wbf in [0.30, 0.35, 0.38, 0.40]:
            configs.append((f"v3ms+v4duo_wbf={wbf}", {
                "v3_passes": ["full_1280", "full_1536"],
                "v4_passes": ["full_1280", "full_1536"],
                "conf": 0.03, "wbf_iou": wbf
            }))
        # v3 duo + v4 duo + soft NMS
        for wbf in [0.35, 0.38]:
            configs.append((f"v3ms+v4duo_wbf={wbf}_soft", {
                "v3_passes": ["full_1280", "full_1536"],
                "v4_passes": ["full_1280", "full_1536"],
                "conf": 0.03, "wbf_iou": wbf,
                "nms_type": "soft", "nms_sigma": 0.5
            }))

    # ── H. v3@1408 intermediate scale
    if has_1408:
        # v3 trio (1280+1408+1536) + v4@1280
        for wbf in [0.32, 0.35, 0.38, 0.40]:
            configs.append((f"v3trio+v4_wbf={wbf}", {
                "v3_passes": ["full_1280", "full_1408", "full_1536"],
                "v4_passes": ["full_1280"],
                "conf": 0.03, "wbf_iou": wbf
            }))
        # v3 trio + v4@1280 + soft NMS
        configs.append(("v3trio+v4_wbf=0.35_soft", {
            "v3_passes": ["full_1280", "full_1408", "full_1536"],
            "v4_passes": ["full_1280"],
            "conf": 0.03, "wbf_iou": 0.35,
            "nms_type": "soft", "nms_sigma": 0.5
        }))

        if has_v4_1536:
            # v3 trio + v4 duo (5 passes — max ensemble)
            for wbf in [0.35, 0.38, 0.40]:
                configs.append((f"v3trio+v4duo_wbf={wbf}", {
                    "v3_passes": ["full_1280", "full_1408", "full_1536"],
                    "v4_passes": ["full_1280", "full_1536"],
                    "conf": 0.03, "wbf_iou": wbf
                }))
            # Same + soft NMS
            configs.append(("v3trio+v4duo_wbf=0.35_soft", {
                "v3_passes": ["full_1280", "full_1408", "full_1536"],
                "v4_passes": ["full_1280", "full_1536"],
                "conf": 0.03, "wbf_iou": 0.35,
                "nms_type": "soft", "nms_sigma": 0.5
            }))

    # ── I. Confidence threshold sweep on best combo
    for conf in [0.01, 0.02, 0.04, 0.05]:
        configs.append((f"conf={conf}_wbf=0.35", {
            "v3_passes": ["full_1280", "full_1536"],
            "v4_passes": ["full_1280"],
            "conf": conf, "wbf_iou": 0.35
        }))

    # ── J. Max dets sweep
    for md in [200, 400, 500]:
        configs.append((f"maxdets={md}", {
            "v3_passes": ["full_1280", "full_1536"],
            "v4_passes": ["full_1280"],
            "conf": 0.03, "wbf_iou": 0.35, "max_dets": md
        }))

    # ── K. v3 only at 1280 (single model reference)
    configs.append(("v3_1280_only", {
        "v3_passes": ["full_1280"],
        "conf": 0.03
    }))

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--skip-1408", action="store_true", help="Skip v3@1408")
    parser.add_argument("--skip-v4-1536", action="store_true", help="Skip v4@1536")
    args = parser.parse_args()

    has_1408 = not args.skip_1408
    has_v4_1536 = not args.skip_v4_1536

    # Select val images (same seed as v3 for comparability)
    all_imgs = sorted(YOLO_VAL_DIR.glob("*.jpg"))
    if not all_imgs:
        all_imgs = sorted(YOLO_VAL_DIR.glob("*.png"))
    n = min(args.n_images, len(all_imgs))
    np.random.seed(42)
    indices = np.random.choice(len(all_imgs), n, replace=False)
    img_paths = [all_imgs[i] for i in sorted(indices)]
    print(f"Using {n}/{len(all_imgs)} val images")

    gt = load_coco_ground_truth(str(ANN_PATH))

    # ── Cache v3 passes ──
    v3_scales = [1280, 1536]
    if has_1408:
        v3_scales = [1280, 1408, 1536]

    model_v3, device = load_model(WEIGHTS_V3)
    model_v3(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)

    print(f"\nCaching v3 @ {v3_scales}...")
    t0 = time.time()
    cache_v3 = cache_model_passes(img_paths, model_v3, device, v3_scales, "v3")
    print(f"V3 done in {time.time()-t0:.0f}s\n")
    del model_v3

    # ── Cache v4 passes ──
    v4_scales = [1280]
    if has_v4_1536:
        v4_scales = [1280, 1536]

    cache_v4 = None
    if WEIGHTS_V4.exists():
        model_v4, _ = load_model(WEIGHTS_V4)
        model_v4(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)

        print(f"Caching v4 @ {v4_scales}...")
        t0 = time.time()
        cache_v4 = cache_model_passes(img_paths, model_v4, device, v4_scales, "v4")
        print(f"V4 done in {time.time()-t0:.0f}s\n")
        del model_v4

    # ── Run sweep ──
    configs = get_configs(has_1408=has_1408, has_v4_1536=has_v4_1536)
    print(f"Sweeping {len(configs)} configurations...")
    print("=" * 110)

    results = []
    for label, config in configs:
        res = evaluate_config(cache_v3, cache_v4, gt, config, label)
        results.append(res)
        marker = " ★" if res["combined"] > 0.925 else ""
        print(f"  {label:45s} | Score={res['combined']:.4f} | "
              f"Det={res['det_mAP']:.4f} | Cls={res['cls_mAP']:.4f} | "
              f"N={res['n_preds']:5d}{marker}")

    # Sort results
    results.sort(key=lambda x: x["combined"], reverse=True)
    baseline_score = next((r["combined"] for r in results if "BASELINE_hard" == r["label"]), results[-1]["combined"])

    print("\n" + "=" * 110)
    print(f"TOP 20 (baseline = {baseline_score:.4f})")
    print("=" * 110)
    for i, r in enumerate(results[:20]):
        delta = r["combined"] - baseline_score
        star = " ★★★" if delta > 0.005 else (" ★★" if delta > 0.002 else (" ★" if delta > 0 else ""))
        print(f"  #{i+1:2d} {r['label']:45s} | {r['combined']:.4f} ({delta:+.4f})"
              f" | Det={r['det_mAP']:.4f} | Cls={r['cls_mAP']:.4f}{star}")

    # Save results
    out_path = Path("sweep_v4_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    winner = results[0]
    print(f"\nWINNER: {winner['label']} → {winner['combined']:.4f} "
          f"(Δ={winner['combined'] - baseline_score:+.4f})")
    print(f"  Det={winner['det_mAP']:.4f} | Cls={winner['cls_mAP']:.4f}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
