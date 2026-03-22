"""Fast post-processing sweep — caches inference once with LOW conf, then sweeps post-proc params.

Key insight: Cache raw inference with very low conf (0.01) so we have ALL potential detections.
Then apply different conf thresholds, WBF params, NMS variants, etc. as post-processing.
This avoids re-running slow ONNX inference for each configuration.

Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

Usage:
    python sweep_fast.py              # Full sweep on 5 images
    python sweep_fast.py --n-images 3 # Fewer images for faster iteration
"""
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from evaluate_local import evaluate_mAP, load_coco_ground_truth

# ── Paths ───────────────────────────────────────────────────────────────
YOLO_VAL_DIR = Path("data/yolo/images/val")
ANN_PATH = Path("data/coco/train/annotations.json")
WEIGHTS = Path("best.onnx")

# ── Reuse helpers from run.py ───────────────────────────────────────────
def _iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix = max(0, min(ax2, bx2) - max(ax, bx))
    iy = max(0, min(ay2, by2) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def extract_dets(results, offset_x=0, offset_y=0, clip_w=None, clip_h=None):
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            x1 += offset_x; y1 += offset_y; x2 += offset_x; y2 += offset_y
            if clip_w: x1, x2 = max(0, min(x1, clip_w)), max(0, min(x2, clip_w))
            if clip_h: y1, y2 = max(0, min(y1, clip_h)), max(0, min(y2, clip_h))
            w, h = x2 - x1, y2 - y1
            if w < 3 or h < 3:
                continue
            dets.append({
                "category_id": int(r.boxes.cls[i].item()),
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "score": round(float(r.boxes.conf[i].item()), 4),
            })
    return dets


def get_slices(img_w, img_h, slice_size=640, overlap=0.25):
    step = int(slice_size * (1 - overlap))
    slices = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            sw = min(slice_size, img_w - x)
            sh = min(slice_size, img_h - y)
            if sw < slice_size * 0.3 or sh < slice_size * 0.3:
                continue
            slices.append((x, y, sw, sh))
    return slices


# ── Cache raw detections ────────────────────────────────────────────────
def cache_all_passes(img_paths, model, device):
    """Run inference ONCE with very low conf, store all raw detections per pass."""
    from PIL import Image

    CACHE_CONF = 0.01  # Very low — keep everything, filter later
    cache = {}

    for idx, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        img_w, img_h = img.size
        image_id = int(img_path.stem.split("_")[-1])

        # Pass 1: Full image @ 1280
        results = model(str(img_path), device=device, verbose=False,
                       conf=CACHE_CONF, iou=0.7, max_det=600, imgsz=1280, augment=False)
        pass_full = extract_dets(results)

        # Pass 2: SAHI 640
        pass_640 = []
        for cx, cy, cw, ch in get_slices(img_w, img_h, 640, 0.25):
            crop = img.crop((cx, cy, cx + cw, cy + ch))
            results = model(np.array(crop), device=device, verbose=False,
                           conf=CACHE_CONF, iou=0.7, max_det=400, imgsz=640, augment=False)
            pass_640.extend(extract_dets(results, offset_x=cx, offset_y=cy, clip_w=img_w, clip_h=img_h))

        # Pass 3: SAHI 960
        pass_960 = []
        for cx, cy, cw, ch in get_slices(img_w, img_h, 960, 0.25):
            crop = img.crop((cx, cy, cx + cw, cy + ch))
            results = model(np.array(crop), device=device, verbose=False,
                           conf=CACHE_CONF, iou=0.7, max_det=400, imgsz=960, augment=False)
            pass_960.extend(extract_dets(results, offset_x=cx, offset_y=cy, clip_w=img_w, clip_h=img_h))

        img.close()
        cache[image_id] = {
            "pass_full": pass_full,
            "pass_640": pass_640,
            "pass_960": pass_960,
            "img_w": img_w,
            "img_h": img_h,
        }
        n_total = len(pass_full) + len(pass_640) + len(pass_960)
        print(f"  [{idx+1}/{len(img_paths)}] img_{image_id:05d}: {n_total} raw dets "
              f"(full={len(pass_full)}, 640={len(pass_640)}, 960={len(pass_960)})")

    return cache


# ── WBF ─────────────────────────────────────────────────────────────────
def wbf_fuse(passes_dets, img_w, img_h, iou_thresh=0.50, skip_box_thresh=0.01, conf_type='max'):
    from ensemble_boxes import weighted_boxes_fusion

    boxes_list, scores_list, labels_list = [], [], []
    for pass_dets in passes_dets:
        boxes, scores, labels = [], [], []
        for d in pass_dets:
            x, y, w, h = d["bbox"]
            x1 = max(0.0, x / img_w)
            y1 = max(0.0, y / img_h)
            x2 = min(1.0, (x + w) / img_w)
            y2 = min(1.0, (y + h) / img_h)
            if x2 <= x1 or y2 <= y1: continue
            boxes.append([x1, y1, x2, y2])
            scores.append(d["score"])
            labels.append(d["category_id"])
        if boxes:
            boxes_list.append(np.array(boxes, dtype=np.float32))
            scores_list.append(np.array(scores, dtype=np.float32))
            labels_list.append(np.array(labels, dtype=np.int32))

    if not boxes_list:
        return []

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_thresh, skip_box_thr=skip_box_thresh, conf_type=conf_type,
    )

    result = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x1, y1, x2, y2 = box
        result.append({
            "category_id": int(label),
            "bbox": [round(x1*img_w,1), round(y1*img_h,1), round((x2-x1)*img_w,1), round((y2-y1)*img_h,1)],
            "score": round(float(score), 4),
        })
    return result


# ── NMS variants ────────────────────────────────────────────────────────
def hard_nms_class_agnostic(dets, iou_thresh=0.45):
    if not dets: return dets
    dets_sorted = sorted(dets, key=lambda x: x["score"], reverse=True)
    kept, absorbed = [], []
    for d in dets_sorted:
        merged = False
        for ki, k in enumerate(kept):
            if _iou(d["bbox"], k["bbox"]) >= iou_thresh:
                absorbed[ki].append(d)
                merged = True
                break
        if not merged:
            kept.append(d.copy())
            absorbed.append([d])
    for ki, k in enumerate(kept):
        if len(absorbed[ki]) > 1:
            cat_scores = {}
            for ab in absorbed[ki]: cat_scores[ab["category_id"]] = cat_scores.get(ab["category_id"], 0) + ab["score"]
            kept[ki]["category_id"] = max(cat_scores, key=cat_scores.get)
    return kept


def soft_nms_class_agnostic(dets, iou_thresh=0.45, sigma=0.5, method='gaussian', score_thresh=0.01):
    if not dets: return dets
    dets = [d.copy() for d in dets]
    dets.sort(key=lambda x: x["score"], reverse=True)
    kept, absorbed = [], []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        cur_abs = [best.copy()]
        remaining = []
        for d in dets:
            iou = _iou(best["bbox"], d["bbox"])
            if iou >= iou_thresh:
                cur_abs.append(d.copy())
            if method == 'gaussian':
                d["score"] *= np.exp(-(iou ** 2) / sigma)
            elif method == 'linear' and iou > iou_thresh:
                d["score"] *= (1 - iou)
            if d["score"] >= score_thresh:
                remaining.append(d)
        absorbed.append(cur_abs)
        dets = sorted(remaining, key=lambda x: x["score"], reverse=True)
    for ki, k in enumerate(kept):
        if len(absorbed[ki]) > 1:
            cat_scores = {}
            for ab in absorbed[ki]: cat_scores[ab["category_id"]] = cat_scores.get(ab["category_id"], 0) + ab["score"]
            kept[ki]["category_id"] = max(cat_scores, key=cat_scores.get)
    return kept


# ── Pipeline ────────────────────────────────────────────────────────────
def run_pipeline(cache, config):
    """Apply post-processing pipeline on cached detections."""
    conf_full = config.get('conf_full', 0.08)
    conf_sahi = config.get('conf_sahi', 0.10)
    wbf_iou = config.get('wbf_iou', 0.50)
    wbf_skip = config.get('wbf_skip', 0.01)
    wbf_conf_type = config.get('wbf_conf_type', 'max')
    nms_type = config.get('nms_type', 'hard')
    nms_iou = config.get('nms_iou', 0.45)
    nms_sigma = config.get('nms_sigma', 0.5)
    max_dets = config.get('max_dets', 200)
    passes = config.get('passes', [1, 2, 3])

    all_preds = []
    for image_id, data in cache.items():
        img_w, img_h = data["img_w"], data["img_h"]

        # Filter by confidence threshold
        pf = [d for d in data["pass_full"] if d["score"] >= conf_full] if 1 in passes else []
        p640 = [d for d in data["pass_640"] if d["score"] >= conf_sahi] if 2 in passes else []
        p960 = [d for d in data["pass_960"] if d["score"] >= conf_sahi] if 3 in passes else []

        pass_list = []
        if pf: pass_list.append(pf)
        if p640: pass_list.append(p640)
        if p960: pass_list.append(p960)

        if not pass_list:
            continue

        # WBF fusion
        try:
            fused = wbf_fuse(pass_list, img_w, img_h,
                           iou_thresh=wbf_iou, skip_box_thresh=wbf_skip,
                           conf_type=wbf_conf_type)
        except Exception:
            fused = []
            for p in pass_list: fused.extend(p)

        # NMS
        if nms_type == 'hard':
            dets = hard_nms_class_agnostic(fused, iou_thresh=nms_iou)
        elif nms_type == 'soft_gaussian':
            dets = soft_nms_class_agnostic(fused, iou_thresh=nms_iou, sigma=nms_sigma, method='gaussian')
        elif nms_type == 'soft_linear':
            dets = soft_nms_class_agnostic(fused, iou_thresh=nms_iou, method='linear')
        else:
            dets = hard_nms_class_agnostic(fused, iou_thresh=nms_iou)

        # Cap
        if len(dets) > max_dets:
            dets.sort(key=lambda x: x["score"], reverse=True)
            dets = dets[:max_dets]

        for d in dets:
            d["image_id"] = image_id
            all_preds.append(d)

    return all_preds


def evaluate_config(cache, gt, config, label=""):
    preds = run_pipeline(cache, config)
    det = evaluate_mAP(preds, gt, iou_threshold=0.5, ignore_category=True)
    cls = evaluate_mAP(preds, gt, iou_threshold=0.5, ignore_category=False)
    combined = 0.7 * det["mAP"] + 0.3 * cls["mAP"]
    return {"label": label, "combined": combined, "det_mAP": det["mAP"],
            "cls_mAP": cls["mAP"], "n_preds": len(preds)}


# ── Sweep configs ───────────────────────────────────────────────────────
def get_all_configs():
    base = {"conf_full": 0.08, "conf_sahi": 0.10, "wbf_iou": 0.50, "wbf_skip": 0.01,
            "wbf_conf_type": "max", "nms_type": "hard", "nms_iou": 0.45,
            "nms_sigma": 0.5, "max_dets": 200, "passes": [1, 2, 3]}

    configs = []

    # Baseline
    configs.append(("BASELINE", {**base}))

    # 1. Confidence thresholds (major axis)
    for f, s in [(0.03, 0.05), (0.05, 0.07), (0.05, 0.10), (0.06, 0.08),
                 (0.08, 0.10), (0.08, 0.12), (0.10, 0.10), (0.10, 0.12),
                 (0.10, 0.15), (0.12, 0.15), (0.15, 0.18), (0.15, 0.20)]:
        configs.append((f"conf={f:.2f}/{s:.2f}", {**base, "conf_full": f, "conf_sahi": s}))

    # 2. WBF IoU threshold
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        configs.append((f"wbf_iou={t:.2f}", {**base, "wbf_iou": t}))

    # 3. WBF conf_type
    for ct in ['max', 'avg', 'box_and_model_avg', 'absent_model_aware_avg']:
        configs.append((f"wbf_conf={ct}", {**base, "wbf_conf_type": ct}))

    # 4. WBF skip threshold
    for sk in [0.001, 0.005, 0.01, 0.02, 0.05, 0.08]:
        configs.append((f"wbf_skip={sk}", {**base, "wbf_skip": sk}))

    # 5. Soft-NMS variants
    for sigma in [0.2, 0.3, 0.5, 0.7, 1.0]:
        configs.append((f"soft_nms_g_s={sigma}", {**base, "nms_type": "soft_gaussian", "nms_sigma": sigma}))
    configs.append(("soft_nms_linear", {**base, "nms_type": "soft_linear"}))

    # 6. NMS IoU threshold
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        configs.append((f"nms_iou={t:.2f}", {**base, "nms_iou": t}))

    # 7. Max detections
    for m in [100, 150, 200, 250, 300, 400, 500]:
        configs.append((f"max_dets={m}", {**base, "max_dets": m}))

    # 8. Pass combinations
    for passes, label in [([1], "full_only"), ([1,2], "full+640"),
                          ([1,3], "full+960"), ([2,3], "sahi_only"),
                          ([1,2,3], "all_passes")]:
        configs.append((f"passes={label}", {**base, "passes": passes}))

    # 9. Combined improvements: Soft-NMS + best WBF + optimal conf
    for sigma in [0.3, 0.5]:
        for wbf_iou in [0.45, 0.50, 0.55]:
            for f, s in [(0.08, 0.10), (0.10, 0.12), (0.06, 0.08)]:
                configs.append((f"combo_s={sigma}_w={wbf_iou}_c={f}/{s}",
                    {**base, "nms_type": "soft_gaussian", "nms_sigma": sigma,
                     "wbf_iou": wbf_iou, "conf_full": f, "conf_sahi": s}))

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=5)
    args = parser.parse_args()

    # Load validation images
    valid_ext = {".jpg", ".jpeg", ".png"}
    img_paths = sorted(p for p in YOLO_VAL_DIR.iterdir() if p.suffix.lower() in valid_ext)
    if args.n_images < len(img_paths):
        step = max(1, len(img_paths) // args.n_images)
        img_paths = [img_paths[i * step] for i in range(args.n_images) if i * step < len(img_paths)]

    img_ids = set(int(p.stem.split("_")[-1]) for p in img_paths)
    gt = load_coco_ground_truth(ANN_PATH, img_ids)
    print(f"Images: {len(img_paths)} | GT annotations: {len(gt)}")

    # Load model
    import torch
    _ol = torch.load
    def _pl(*a, **k):
        if 'weights_only' not in k: k['weights_only'] = False
        return _ol(*a, **k)
    torch.load = _pl
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(WEIGHTS), task="detect")
    # Warmup
    model(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
    print(f"Model loaded on {device}")

    # Cache ALL raw detections with very low conf
    print("\nCaching raw detections (conf=0.01, one-time)...")
    t0 = time.time()
    cache = cache_all_passes(img_paths, model, device)
    print(f"Caching done in {time.time()-t0:.0f}s")

    # Run all configs (FAST — pure post-processing on cached data)
    configs = get_all_configs()
    print(f"\nSweeping {len(configs)} configurations...")
    print(f"{'='*90}")

    results = []
    for label, config in configs:
        t0 = time.time()
        r = evaluate_config(cache, gt, config, label)
        elapsed = time.time() - t0
        results.append(r)
        print(f"  {label:45s} | Score={r['combined']:.4f} | Det={r['det_mAP']:.4f} | "
              f"Cls={r['cls_mAP']:.4f} | N={r['n_preds']:4d} | {elapsed:.2f}s")

    # Ranking
    results.sort(key=lambda r: r["combined"], reverse=True)
    baseline = next((r for r in results if r["label"] == "BASELINE"), results[0])

    print(f"\n{'='*90}")
    print("TOP 20 CONFIGURATIONS")
    print(f"{'='*90}")
    for i, r in enumerate(results[:20]):
        delta = r["combined"] - baseline["combined"]
        arrow = "+" if delta > 0 else "" if delta == 0 else ""
        print(f"  #{i+1:2d} {r['label']:45s} | {r['combined']:.4f} ({arrow}{delta:+.4f}) | "
              f"Det={r['det_mAP']:.4f} | Cls={r['cls_mAP']:.4f}")

    print(f"\n{'='*90}")
    print("BOTTOM 5 CONFIGURATIONS")
    print(f"{'='*90}")
    for i, r in enumerate(results[-5:]):
        delta = r["combined"] - baseline["combined"]
        print(f"  #{len(results)-4+i:2d} {r['label']:45s} | {r['combined']:.4f} ({delta:+.4f})")

    # Save
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to sweep_results.json")

    # Winner
    w = results[0]
    print(f"\nWINNER: {w['label']} → {w['combined']:.4f} (baseline={baseline['combined']:.4f})")


if __name__ == "__main__":
    main()
