"""Sweep V5 — Thorough evaluation on ALL 29 val images.

Test results showed only 33% transfer of v4 local gains:
  Local: 0.9244 → 0.9426 (+0.018)
  Test:  0.9047 → 0.9108 (+0.006)

Hypothesis: 10-image evaluation is noisy. Need ALL 29 images for reliability.

Strategy:
  1. Cache ALL 29 val images at all scales
  2. Sweep sigma more carefully (the big lever)
  3. Test lower conf thresholds (more detections = more recall)
  4. Test higher max_det limits
  5. Test WBF skip_box_thresh variations
  6. Try conf_type variations again with full sample
  7. Test 3-pass vs 4-pass vs 5-pass tradeoffs

Score = 0.7 * det_mAP@0.5 + 0.3 * cls_mAP@0.5
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
from sweep_v4 import load_model, WEIGHTS_V3, WEIGHTS_V4


def cache_model(img_paths, model, device, scales, name="model"):
    """Cache inference at multiple scales for all images."""
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
            results = model(str(img_path), device=device, verbose=False,
                            conf=0.01, iou=0.7, max_det=600, imgsz=s, augment=False)
            dets = extract_dets(results)
            entry[key] = dets
            parts.append(f"{s}={len(dets)}")

        cache[image_id] = entry
        elapsed = time.time() - t0
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  {name} [{idx+1}/{len(img_paths)}] img_{image_id:05d}: "
                  f"{', '.join(parts)} | {elapsed:.1f}s")
    return cache


def apply_pipeline(pass_list, img_w, img_h, conf=0.03, wbf_iou=0.35,
                   nms_type="soft", nms_sigma=1.5, nms_iou=0.45,
                   max_dets=300, conf_type='max', skip_box_thresh=0.005):
    """WBF + NMS pipeline."""
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
                            iou_thresh=wbf_iou, skip_box_thresh=skip_box_thresh,
                            conf_type=conf_type)
        except Exception:
            dets = []
            for f in filtered:
                dets.extend(f)

    if nms_type == "soft":
        dets = soft_nms_class_agnostic(dets, iou_thresh=nms_iou, sigma=nms_sigma)
    else:
        dets = hard_nms_class_agnostic(dets, iou_thresh=nms_iou)

    if len(dets) > max_dets:
        dets.sort(key=lambda x: x["score"], reverse=True)
        dets = dets[:max_dets]
    return dets


def run_and_eval(cache_v3, cache_v4, gt, config, label=""):
    """Run config and evaluate."""
    v3p = config.get("v3_passes", ["full_1280"])
    v4p = config.get("v4_passes", [])

    all_preds = []
    for image_id in cache_v3:
        v3_data = cache_v3[image_id]
        img_w, img_h = v3_data["img_w"], v3_data["img_h"]

        pass_list = []
        for key in v3p:
            if key in v3_data:
                pass_list.append(v3_data[key])
        if cache_v4 and image_id in cache_v4:
            for key in v4p:
                if key in cache_v4[image_id]:
                    pass_list.append(cache_v4[image_id][key])

        dets = apply_pipeline(pass_list, img_w, img_h,
                              conf=config.get("conf", 0.03),
                              wbf_iou=config.get("wbf_iou", 0.35),
                              nms_type=config.get("nms_type", "soft"),
                              nms_sigma=config.get("nms_sigma", 1.5),
                              nms_iou=config.get("nms_iou", 0.45),
                              max_dets=config.get("max_dets", 300),
                              conf_type=config.get("conf_type", "max"),
                              skip_box_thresh=config.get("skip_box_thresh", 0.005))
        for d in dets:
            d["image_id"] = image_id
            all_preds.append(d)

    pred_img_ids = set(cache_v3.keys())
    gt_filtered = [g for g in gt if g["image_id"] in pred_img_ids]
    det = evaluate_mAP(all_preds, gt_filtered, iou_threshold=0.5, ignore_category=True)
    cls = evaluate_mAP(all_preds, gt_filtered, iou_threshold=0.5, ignore_category=False)
    combined = 0.7 * det["mAP"] + 0.3 * cls["mAP"]
    return {"label": label, "combined": combined, "det_mAP": det["mAP"],
            "cls_mAP": cls["mAP"], "n_preds": len(all_preds)}


def get_configs():
    """Comprehensive configurations to sweep."""
    configs = []
    base3 = ["full_1280", "full_1536"]
    trio3 = ["full_1280", "full_1408", "full_1536"]
    v4_single = ["full_1280"]
    v4_duo = ["full_1280", "full_1536"]

    # ── A. Submitted config (0.9108 test) — reference
    configs.append(("SUBMITTED_5pass_s1.5", {
        "v3_passes": trio3, "v4_passes": v4_duo,
        "conf": 0.03, "wbf_iou": 0.35, "nms_sigma": 1.5
    }))

    # ── B. Previous best (0.9047 test) — hard NMS reference
    configs.append(("PREV_3pass_hard", {
        "v3_passes": base3, "v4_passes": v4_single,
        "conf": 0.03, "wbf_iou": 0.35, "nms_type": "hard"
    }))

    # ── C. Sigma sweep: 0.3 to 3.0 on 5-pass
    for sigma in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0]:
        configs.append((f"5p_s={sigma}", {
            "v3_passes": trio3, "v4_passes": v4_duo,
            "conf": 0.03, "wbf_iou": 0.35, "nms_sigma": sigma
        }))

    # ── D. Sigma sweep on 3-pass (existing best runtime config)
    for sigma in [0.5, 0.7, 1.0, 1.5, 2.0]:
        configs.append((f"3p_s={sigma}", {
            "v3_passes": base3, "v4_passes": v4_single,
            "conf": 0.03, "wbf_iou": 0.35, "nms_sigma": sigma
        }))

    # ── E. Confidence threshold sweep
    for conf in [0.01, 0.02, 0.03, 0.05]:
        configs.append((f"5p_c={conf}", {
            "v3_passes": trio3, "v4_passes": v4_duo,
            "conf": conf, "wbf_iou": 0.35, "nms_sigma": 1.5
        }))

    # ── F. WBF IoU sweep
    for wbf in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        configs.append((f"5p_wbf={wbf}", {
            "v3_passes": trio3, "v4_passes": v4_duo,
            "conf": 0.03, "wbf_iou": wbf, "nms_sigma": 1.5
        }))

    # ── G. skip_box_thresh sweep
    for sbt in [0.001, 0.005, 0.01, 0.02]:
        configs.append((f"5p_sbt={sbt}", {
            "v3_passes": trio3, "v4_passes": v4_duo,
            "conf": 0.03, "wbf_iou": 0.35, "nms_sigma": 1.5,
            "skip_box_thresh": sbt
        }))

    # ── H. conf_type sweep
    for ct in ["max", "avg", "box_and_model_avg"]:
        configs.append((f"5p_ct={ct}", {
            "v3_passes": trio3, "v4_passes": v4_duo,
            "conf": 0.03, "wbf_iou": 0.35, "nms_sigma": 1.5, "conf_type": ct
        }))

    # ── I. NMS IoU sweep
    for nms_iou in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        configs.append((f"5p_niou={nms_iou}", {
            "v3_passes": trio3, "v4_passes": v4_duo,
            "conf": 0.03, "wbf_iou": 0.35, "nms_sigma": 1.5, "nms_iou": nms_iou
        }))

    # ── J. max_dets sweep
    for md in [200, 300, 400, 500]:
        configs.append((f"5p_md={md}", {
            "v3_passes": trio3, "v4_passes": v4_duo,
            "conf": 0.03, "wbf_iou": 0.35, "nms_sigma": 1.5, "max_dets": md
        }))

    # ── K. 4-pass configs (drop v4@1536 to save time)
    for sigma in [0.7, 1.0, 1.5]:
        configs.append((f"4p_s={sigma}", {
            "v3_passes": trio3, "v4_passes": v4_single,
            "conf": 0.03, "wbf_iou": 0.35, "nms_sigma": sigma
        }))

    # ── L. Single model reference
    configs.append(("v3_only_1280", {
        "v3_passes": ["full_1280"], "conf": 0.03, "nms_type": "hard"
    }))

    # ── M. Combined best: optimize multiple params together
    for sigma in [1.0, 1.5]:
        for wbf in [0.35, 0.40]:
            for conf in [0.02, 0.03]:
                configs.append((f"combo_s={sigma}_w={wbf}_c={conf}", {
                    "v3_passes": trio3, "v4_passes": v4_duo,
                    "conf": conf, "wbf_iou": wbf, "nms_sigma": sigma
                }))

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=29,
                        help="Number of val images (default: ALL 29)")
    args = parser.parse_args()

    all_imgs = sorted(YOLO_VAL_DIR.glob("*.jpg"))
    n = min(args.n_images, len(all_imgs))
    if n == len(all_imgs):
        img_paths = all_imgs
        print(f"Using ALL {n} val images (full evaluation)")
    else:
        np.random.seed(42)
        indices = np.random.choice(len(all_imgs), n, replace=False)
        img_paths = [all_imgs[i] for i in sorted(indices)]
        print(f"Using {n}/{len(all_imgs)} val images")

    gt = load_coco_ground_truth(str(ANN_PATH))

    # Cache v3 at 3 scales
    model_v3, device = load_model(WEIGHTS_V3)
    model_v3(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
    print(f"\nCaching v3 @ [1280, 1408, 1536] on {n} images...")
    t0 = time.time()
    cache_v3 = cache_model(img_paths, model_v3, device, [1280, 1408, 1536], "v3")
    print(f"V3 done in {time.time()-t0:.0f}s")
    del model_v3

    # Cache v4 at 2 scales
    cache_v4 = None
    if WEIGHTS_V4.exists():
        model_v4, _ = load_model(WEIGHTS_V4)
        model_v4(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
        print(f"\nCaching v4 @ [1280, 1536] on {n} images...")
        t0 = time.time()
        cache_v4 = cache_model(img_paths, model_v4, device, [1280, 1536], "v4")
        print(f"V4 done in {time.time()-t0:.0f}s")
        del model_v4

    configs = get_configs()
    print(f"\nSweeping {len(configs)} configs on {n} images...")
    print("=" * 115)

    results = []
    for label, config in configs:
        res = run_and_eval(cache_v3, cache_v4, gt, config, label)
        results.append(res)
        print(f"  {label:42s} | Score={res['combined']:.4f} | "
              f"Det={res['det_mAP']:.4f} | Cls={res['cls_mAP']:.4f} | N={res['n_preds']:5d}")

    results.sort(key=lambda x: x["combined"], reverse=True)
    submitted = next((r for r in results if "SUBMITTED" in r["label"]), None)
    prev = next((r for r in results if "PREV" in r["label"]), None)

    print("\n" + "=" * 115)
    print(f"TOP 20 (submitted={submitted['combined']:.4f}, prev_hard={prev['combined']:.4f})")
    print("=" * 115)
    for i, r in enumerate(results[:20]):
        delta = r["combined"] - submitted["combined"]
        print(f"  #{i+1:2d} {r['label']:42s} | {r['combined']:.4f} ({delta:+.4f})"
              f" | Det={r['det_mAP']:.4f} | Cls={r['cls_mAP']:.4f}")

    out_path = Path("sweep_v5_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    winner = results[0]
    print(f"\nWINNER: {winner['label']} -> {winner['combined']:.4f}")
    print(f"  Det={winner['det_mAP']:.4f} | Cls={winner['cls_mAP']:.4f}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
