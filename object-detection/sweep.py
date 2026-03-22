"""Systematic parameter & strategy sweep for object detection competition.

Tests SOTA post-processing strategies against local validation data:
1. Baseline (current run.py settings)
2. Soft-NMS instead of hard NMS
3. WBF IoU threshold sweep
4. Confidence threshold sweep
5. SAHI overlap sweep
6. Detection cap sweep
7. Score recalibration
8. Multi-scale ensemble weights
9. WBF conf_type comparison (max vs avg vs box_and_model_avg)

Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

Usage:
    python sweep.py                     # Full sweep
    python sweep.py --quick             # Quick (fewer configs)
    python sweep.py --strategy soft_nms # Single strategy
"""
import argparse
import json
import time
import tempfile
import sys
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np

# Import our evaluation
from evaluate_local import evaluate_mAP, load_coco_ground_truth, compute_iou

# ── Configuration ───────────────────────────────────────────────────────
IMG_DIR = Path("data/coco/train/images")
ANN_PATH = Path("data/coco/train/annotations.json")
YOLO_VAL_DIR = Path("data/yolo/images/val")
WEIGHTS = Path("best.onnx")

# ── Model loading ───────────────────────────────────────────────────────
_model = None
_device = None

def get_model():
    global _model, _device
    if _model is None:
        import torch
        # Patch torch.load
        _original_load = torch.load
        def _patched_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)
        torch.load = _patched_load
        
        from ultralytics import YOLO
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = YOLO(str(WEIGHTS), task="detect")
        # Warmup
        imgs = sorted(p for p in YOLO_VAL_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if imgs:
            _model(str(imgs[0]), device=_device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
        print(f"Model loaded on {_device}")
    return _model, _device


# ── Detection extraction ────────────────────────────────────────────────
def extract_dets(results, offset_x=0, offset_y=0, clip_w=None, clip_h=None):
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            x1 += offset_x; y1 += offset_y
            x2 += offset_x; y2 += offset_y
            if clip_w: x1, x2 = max(0, min(x1, clip_w)), max(0, min(x2, clip_w))
            if clip_h: y1, y2 = max(0, min(y1, clip_h)), max(0, min(y2, clip_h))
            w, h = x2 - x1, y2 - y1
            if w < 3 or h < 3:
                continue
            dets.append({
                "category_id": int(r.boxes.cls[i].item()),
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "score": round(float(r.boxes.conf[i].item()), 3),
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


# ── Raw inference (cached per image) ────────────────────────────────────
_raw_cache = {}

def get_raw_passes(img_path, conf_full=0.08, conf_sahi=0.10, sahi_overlap=0.25):
    """Get raw detections from each inference pass (cached)."""
    from PIL import Image
    
    cache_key = (str(img_path), conf_full, conf_sahi, sahi_overlap)
    if cache_key in _raw_cache:
        return _raw_cache[cache_key]
    
    model, device = get_model()
    img = Image.open(img_path)
    img_w, img_h = img.size
    
    # Pass 1: Full image
    results = model(str(img_path), device=device, verbose=False,
                    conf=conf_full, iou=0.5, max_det=400, imgsz=1280, augment=False)
    pass_full = extract_dets(results)
    
    # Pass 2: SAHI 640
    pass_640 = []
    for cx, cy, cw, ch in get_slices(img_w, img_h, 640, sahi_overlap):
        crop = img.crop((cx, cy, cx + cw, cy + ch))
        results = model(np.array(crop), device=device, verbose=False,
                       conf=conf_sahi, iou=0.5, max_det=300, imgsz=640, augment=False)
        pass_640.extend(extract_dets(results, offset_x=cx, offset_y=cy, clip_w=img_w, clip_h=img_h))
    
    # Pass 3: SAHI 960
    pass_960 = []
    for cx, cy, cw, ch in get_slices(img_w, img_h, 960, sahi_overlap):
        crop = img.crop((cx, cy, cx + cw, cy + ch))
        results = model(np.array(crop), device=device, verbose=False,
                       conf=conf_sahi, iou=0.5, max_det=300, imgsz=960, augment=False)
        pass_960.extend(extract_dets(results, offset_x=cx, offset_y=cy, clip_w=img_w, clip_h=img_h))
    
    img.close()
    result = (pass_full, pass_640, pass_960, img_w, img_h)
    _raw_cache[cache_key] = result
    return result


# ── Fusion strategies ───────────────────────────────────────────────────
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
            if x2 <= x1 or y2 <= y1:
                continue
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
            "score": round(float(score), 3),
        })
    return result


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


def class_agnostic_nms(dets, iou_thresh=0.45):
    """Hard NMS with category voting."""
    if not dets:
        return dets
    dets_sorted = sorted(dets, key=lambda x: x["score"], reverse=True)
    kept = []
    absorbed = []
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
    # Category voting
    for ki, k in enumerate(kept):
        if len(absorbed[ki]) > 1:
            cat_scores = {}
            for ab in absorbed[ki]:
                cat = ab["category_id"]
                cat_scores[cat] = cat_scores.get(cat, 0) + ab["score"]
            kept[ki]["category_id"] = max(cat_scores, key=cat_scores.get)
    return kept


def soft_nms(dets, iou_thresh=0.45, sigma=0.5, score_thresh=0.01, method='gaussian'):
    """Soft-NMS: instead of removing overlapping boxes, decrease their scores.
    This is a SOTA improvement over hard NMS - widely used in modern detectors.
    
    Methods:
        'gaussian': score *= exp(-(iou^2) / sigma)  [smoother]
        'linear': score *= (1 - iou) if iou > threshold  [simpler]
    """
    if not dets:
        return dets
    
    dets = [d.copy() for d in dets]
    dets.sort(key=lambda x: x["score"], reverse=True)
    
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        
        remaining = []
        for d in dets:
            iou = _iou(best["bbox"], d["bbox"])
            if method == 'gaussian':
                d["score"] *= np.exp(-(iou ** 2) / sigma)
            elif method == 'linear':
                if iou > iou_thresh:
                    d["score"] *= (1 - iou)
            
            if d["score"] >= score_thresh:
                remaining.append(d)
        
        dets = sorted(remaining, key=lambda x: x["score"], reverse=True)
    
    return kept


def soft_nms_class_agnostic(dets, iou_thresh=0.45, sigma=0.5, score_thresh=0.01, method='gaussian'):
    """Soft-NMS with category voting (class-agnostic version)."""
    if not dets:
        return dets
    
    dets = [d.copy() for d in dets]
    dets.sort(key=lambda x: x["score"], reverse=True)
    
    kept = []
    absorbed = []
    
    while dets:
        best = dets.pop(0)
        kept.append(best)
        cur_absorbed = [best.copy()]
        
        remaining = []
        for d in dets:
            iou = _iou(best["bbox"], d["bbox"])
            if iou >= iou_thresh:
                cur_absorbed.append(d.copy())
                if method == 'gaussian':
                    d["score"] *= np.exp(-(iou ** 2) / sigma)
                elif method == 'linear':
                    d["score"] *= (1 - iou)
            
            if d["score"] >= score_thresh:
                remaining.append(d)
        
        absorbed.append(cur_absorbed)
        dets = sorted(remaining, key=lambda x: x["score"], reverse=True)
    
    # Category voting
    for ki, k in enumerate(kept):
        if len(absorbed[ki]) > 1:
            cat_scores = {}
            for ab in absorbed[ki]:
                cat = ab["category_id"]
                cat_scores[cat] = cat_scores.get(cat, 0) + ab["score"]
            kept[ki]["category_id"] = max(cat_scores, key=cat_scores.get)
    
    return kept


# ── Pipeline configs ────────────────────────────────────────────────────
def run_pipeline(img_paths, config):
    """Run a complete detection pipeline with a given config dict."""
    conf_full = config.get('conf_full', 0.08)
    conf_sahi = config.get('conf_sahi', 0.10)
    wbf_iou = config.get('wbf_iou', 0.50)
    wbf_skip = config.get('wbf_skip', 0.01)
    wbf_conf_type = config.get('wbf_conf_type', 'max')
    nms_type = config.get('nms_type', 'hard')  # hard, soft_gaussian, soft_linear
    nms_iou = config.get('nms_iou', 0.45)
    nms_sigma = config.get('nms_sigma', 0.5)
    max_dets = config.get('max_dets', 200)
    sahi_overlap = config.get('sahi_overlap', 0.25)
    score_recal = config.get('score_recal', None)  # None, 'sqrt', 'log', 'power'
    passes = config.get('passes', [1, 2, 3])  # which passes to use: 1=full, 2=640, 3=960
    
    all_preds = []
    for img_path in img_paths:
        pass_full, pass_640, pass_960, img_w, img_h = get_raw_passes(
            img_path, conf_full, conf_sahi, sahi_overlap)
        
        # Select passes
        pass_list = []
        if 1 in passes: pass_list.append(pass_full)
        if 2 in passes: pass_list.append(pass_640)
        if 3 in passes: pass_list.append(pass_960)
        
        # WBF fusion
        try:
            fused = wbf_fuse(pass_list, img_w, img_h,
                           iou_thresh=wbf_iou, skip_box_thresh=wbf_skip,
                           conf_type=wbf_conf_type)
        except Exception:
            fused = []
            for p in pass_list:
                fused.extend(p)
        
        # NMS
        if nms_type == 'hard':
            dets = class_agnostic_nms(fused, iou_thresh=nms_iou)
        elif nms_type == 'soft_gaussian':
            dets = soft_nms_class_agnostic(fused, iou_thresh=nms_iou, sigma=nms_sigma)
        elif nms_type == 'soft_linear':
            dets = soft_nms_class_agnostic(fused, iou_thresh=nms_iou, method='linear')
        else:
            dets = class_agnostic_nms(fused, iou_thresh=nms_iou)
        
        # Score recalibration
        if score_recal == 'sqrt':
            for d in dets: d["score"] = round(float(np.sqrt(d["score"])), 3)
        elif score_recal == 'power':
            for d in dets: d["score"] = round(float(d["score"] ** 0.7), 3)
        elif score_recal == 'log':
            for d in dets: d["score"] = round(float(-np.log(1 - min(d["score"], 0.999))), 3)
        
        # Cap
        if len(dets) > max_dets:
            dets.sort(key=lambda x: x["score"], reverse=True)
            dets = dets[:max_dets]
        
        image_id = int(img_path.stem.split("_")[-1])
        for d in dets:
            d["image_id"] = image_id
            all_preds.append(d)
    
    return all_preds


def evaluate_config(img_paths, gt, config, label=""):
    """Run pipeline and evaluate."""
    preds = run_pipeline(img_paths, config)
    
    det_result = evaluate_mAP(preds, gt, iou_threshold=0.5, ignore_category=True)
    cls_result = evaluate_mAP(preds, gt, iou_threshold=0.5, ignore_category=False)
    combined = 0.7 * det_result["mAP"] + 0.3 * cls_result["mAP"]
    
    return {
        "label": label,
        "combined": combined,
        "det_mAP": det_result["mAP"],
        "cls_mAP": cls_result["mAP"],
        "n_preds": len(preds),
        "config": config,
    }


# ── Strategy definitions ────────────────────────────────────────────────
def get_strategies(quick=False):
    strategies = {}
    
    # Baseline (current 0.8157 proven settings)
    strategies["baseline"] = [
        {"label": "BASELINE (0.8157 proven)", "conf_full": 0.08, "conf_sahi": 0.10,
         "wbf_iou": 0.50, "nms_type": "hard", "nms_iou": 0.45, "max_dets": 200,
         "wbf_conf_type": "max", "sahi_overlap": 0.25}
    ]
    
    # 1. Soft-NMS (SOTA: used in ATSS, FCOS, modern detectors)
    strategies["soft_nms"] = [
        {"label": "Soft-NMS gaussian s=0.3", "nms_type": "soft_gaussian", "nms_sigma": 0.3, "nms_iou": 0.45},
        {"label": "Soft-NMS gaussian s=0.5", "nms_type": "soft_gaussian", "nms_sigma": 0.5, "nms_iou": 0.45},
        {"label": "Soft-NMS gaussian s=0.7", "nms_type": "soft_gaussian", "nms_sigma": 0.7, "nms_iou": 0.45},
        {"label": "Soft-NMS linear", "nms_type": "soft_linear", "nms_iou": 0.45},
    ]
    
    # 2. WBF IoU threshold
    if quick:
        strategies["wbf_iou"] = [
            {"label": "WBF IoU=0.45", "wbf_iou": 0.45},
            {"label": "WBF IoU=0.55", "wbf_iou": 0.55},
            {"label": "WBF IoU=0.60", "wbf_iou": 0.60},
        ]
    else:
        strategies["wbf_iou"] = [
            {"label": f"WBF IoU={t:.2f}", "wbf_iou": t}
            for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
        ]
    
    # 3. Confidence thresholds
    if quick:
        strategies["conf"] = [
            {"label": "Conf 0.05/0.07", "conf_full": 0.05, "conf_sahi": 0.07},
            {"label": "Conf 0.10/0.12", "conf_full": 0.10, "conf_sahi": 0.12},
            {"label": "Conf 0.12/0.15", "conf_full": 0.12, "conf_sahi": 0.15},
        ]
    else:
        strategies["conf"] = [
            {"label": f"Conf {f:.2f}/{s:.2f}", "conf_full": f, "conf_sahi": s}
            for f, s in [(0.03, 0.05), (0.05, 0.07), (0.06, 0.08), (0.08, 0.10),
                         (0.10, 0.12), (0.12, 0.15), (0.15, 0.18), (0.05, 0.10),
                         (0.08, 0.15), (0.10, 0.10), (0.08, 0.08)]
        ]
    
    # 4. WBF conf_type (comparison of fusion score methods)
    strategies["wbf_conf_type"] = [
        {"label": "WBF conf=max", "wbf_conf_type": "max"},
        {"label": "WBF conf=avg", "wbf_conf_type": "avg"},
        {"label": "WBF conf=box_and_model_avg", "wbf_conf_type": "box_and_model_avg"},
        {"label": "WBF conf=absent_model_aware_avg", "wbf_conf_type": "absent_model_aware_avg"},
    ]
    
    # 5. SAHI overlap
    strategies["sahi_overlap"] = [
        {"label": "SAHI overlap=0.15", "sahi_overlap": 0.15},
        {"label": "SAHI overlap=0.20", "sahi_overlap": 0.20},
        {"label": "SAHI overlap=0.25", "sahi_overlap": 0.25},
        {"label": "SAHI overlap=0.30", "sahi_overlap": 0.30},
        {"label": "SAHI overlap=0.35", "sahi_overlap": 0.35},
    ]
    
    # 6. Detection cap
    strategies["max_dets"] = [
        {"label": "MaxDets=100", "max_dets": 100},
        {"label": "MaxDets=150", "max_dets": 150},
        {"label": "MaxDets=200", "max_dets": 200},
        {"label": "MaxDets=300", "max_dets": 300},
        {"label": "MaxDets=500", "max_dets": 500},
    ]
    
    # 7. Score recalibration
    strategies["score_recal"] = [
        {"label": "No recal (baseline)", "score_recal": None},
        {"label": "Score sqrt", "score_recal": "sqrt"},
        {"label": "Score power^0.7", "score_recal": "power"},
    ]
    
    # 8. NMS IoU threshold
    strategies["nms_iou"] = [
        {"label": f"NMS IoU={t:.2f}", "nms_iou": t}
        for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    ]

    # 9. Pass combinations (which inference passes to use)
    strategies["passes"] = [
        {"label": "Full only", "passes": [1]},
        {"label": "Full + SAHI640", "passes": [1, 2]},
        {"label": "Full + SAHI960", "passes": [1, 3]},
        {"label": "Full + SAHI640 + SAHI960", "passes": [1, 2, 3]},
        {"label": "SAHI640 + SAHI960 only", "passes": [2, 3]},
    ]

    # 10. WBF skip_box_thresh
    strategies["wbf_skip"] = [
        {"label": "WBF skip=0.001", "wbf_skip": 0.001},
        {"label": "WBF skip=0.005", "wbf_skip": 0.005},
        {"label": "WBF skip=0.01", "wbf_skip": 0.01},
        {"label": "WBF skip=0.02", "wbf_skip": 0.02},
        {"label": "WBF skip=0.05", "wbf_skip": 0.05},
    ]
    
    return strategies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer configs per strategy")
    parser.add_argument("--strategy", type=str, default=None, help="Run single strategy")
    parser.add_argument("--n-images", type=int, default=None, help="Number of images (default: all val)")
    args = parser.parse_args()
    
    # Load validation images
    valid_ext = {".jpg", ".jpeg", ".png"}
    img_paths = sorted(p for p in YOLO_VAL_DIR.iterdir() if p.suffix.lower() in valid_ext)
    
    if args.n_images and args.n_images < len(img_paths):
        step = max(1, len(img_paths) // args.n_images)
        img_paths = [img_paths[i * step] for i in range(args.n_images) if i * step < len(img_paths)]
    
    print(f"Using {len(img_paths)} validation images")
    
    # Load ground truth
    img_ids = set(int(p.stem.split("_")[-1]) for p in img_paths)
    gt = load_coco_ground_truth(ANN_PATH, img_ids)
    print(f"Ground truth: {len(gt)} annotations")
    
    # Warmup model
    get_model()
    
    # Pre-cache raw inference for baseline settings
    print("\nPre-caching raw inference passes...")
    t0 = time.time()
    for i, img_path in enumerate(img_paths):
        get_raw_passes(img_path, conf_full=0.08, conf_sahi=0.10, sahi_overlap=0.25)
        if (i + 1) % 10 == 0:
            print(f"  Cached {i+1}/{len(img_paths)}")
    cache_time = time.time() - t0
    print(f"Caching done in {cache_time:.1f}s")
    
    # Default config (baseline)
    base_config = {
        "conf_full": 0.08, "conf_sahi": 0.10,
        "wbf_iou": 0.50, "wbf_skip": 0.01, "wbf_conf_type": "max",
        "nms_type": "hard", "nms_iou": 0.45, "nms_sigma": 0.5,
        "max_dets": 200, "sahi_overlap": 0.25,
        "score_recal": None, "passes": [1, 2, 3],
    }
    
    strategies = get_strategies(quick=args.quick)
    
    if args.strategy:
        if args.strategy not in strategies:
            print(f"Unknown strategy: {args.strategy}")
            print(f"Available: {', '.join(strategies.keys())}")
            return
        strategies = {args.strategy: strategies[args.strategy]}
    
    all_results = []
    
    for strat_name, configs in strategies.items():
        print(f"\n{'='*70}")
        print(f"  STRATEGY: {strat_name}")
        print(f"{'='*70}")
        
        strat_results = []
        for cfg_override in configs:
            label = cfg_override.pop("label", strat_name)
            config = {**base_config, **cfg_override}
            
            t0 = time.time()
            result = evaluate_config(img_paths, gt, config, label=label)
            elapsed = time.time() - t0
            
            strat_results.append(result)
            all_results.append(result)
            
            marker = " ★" if result["combined"] > 0 else ""
            print(f"  {label:40s} | Combined={result['combined']:.4f} | "
                  f"Det={result['det_mAP']:.4f} | Cls={result['cls_mAP']:.4f} | "
                  f"N={result['n_preds']:4d} | {elapsed:.1f}s{marker}")
        
        # Best in this strategy
        best = max(strat_results, key=lambda r: r["combined"])
        baseline_score = None
        for r in all_results:
            if "BASELINE" in r.get("label", ""):
                baseline_score = r["combined"]
                break
        
        if baseline_score:
            delta = best["combined"] - baseline_score
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"  {'BEST':40s} → {best['label']} ({best['combined']:.4f}, {arrow}{abs(delta):.4f} vs baseline)")
    
    # Final summary
    print(f"\n{'='*70}")
    print("  FINAL RANKING (all configs)")
    print(f"{'='*70}")
    all_results.sort(key=lambda r: r["combined"], reverse=True)
    
    baseline_score = None
    for r in all_results:
        if "BASELINE" in r.get("label", ""):
            baseline_score = r["combined"]
            break
    
    for i, r in enumerate(all_results[:20]):
        delta = r["combined"] - baseline_score if baseline_score else 0
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  #{i+1:2d} {r['label']:40s} | {r['combined']:.4f} ({arrow}{abs(delta):.4f}) | "
              f"Det={r['det_mAP']:.4f} | Cls={r['cls_mAP']:.4f}")
    
    # Save results
    out_path = Path("sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Print the winning config
    if all_results:
        winner = all_results[0]
        print(f"\n{'='*70}")
        print(f"  RECOMMENDED CONFIG")
        print(f"{'='*70}")
        print(f"  Score: {winner['combined']:.4f}")
        for k, v in winner['config'].items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
