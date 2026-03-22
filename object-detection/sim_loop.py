"""Continuous improvement simulation loop.
Uses cached detections (v3, v4, v6) to rapidly test parameter combinations.
Logs every experiment to overnight_log.md.
"""
import pickle, json, time, sys, os
from itertools import product
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Load data
# ============================================================================
print("Loading cached detections...")
c3 = pickle.load(open('cache_v3_29.pkl', 'rb'))
c4 = pickle.load(open('cache_v4_29.pkl', 'rb'))
c6 = pickle.load(open('cache_v6_29.pkl', 'rb'))
image_ids = sorted(set(c3.keys()) & set(c4.keys()) & set(c6.keys()))
gt_data = json.load(open('data/coco/train/annotations.json'))
gt_by_image = {}
for ann in gt_data["annotations"]:
    gt_by_image.setdefault(ann["image_id"], []).append(ann)
img_info = {img["id"]: img for img in gt_data["images"]}
print(f"Loaded {len(image_ids)} images with 3-model caches")

# Check available scales per cache
sample_id = image_ids[0]
v3_scales = sorted(c3[sample_id].keys())
v4_scales = sorted(c4[sample_id].keys())
v6_scales = sorted(c6[sample_id].keys())
print(f"v3 scales: {v3_scales}")
print(f"v4 scales: {v4_scales}")
print(f"v6 scales: {v6_scales}")

# ============================================================================
# Core functions
# ============================================================================
def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy; union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0

def soft_nms(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001):
    if not dets: return dets
    dets = [d.copy() for d in dets]
    dets.sort(key=lambda x: x["score"], reverse=True)
    kept, absorbed_groups = [], []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        group = [best.copy()]
        remaining = []
        for d in dets:
            iou = _iou(best["bbox"], d["bbox"])
            if iou >= iou_thresh:
                group.append(d.copy())
            d["score"] *= np.exp(-(iou**2)/sigma)
            if d["score"] >= score_thresh:
                remaining.append(d)
        absorbed_groups.append(group)
        dets = sorted(remaining, key=lambda x: x["score"], reverse=True)
    return kept, absorbed_groups

def soft_nms_with_voting(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001, vote_mode='score'):
    """Soft-NMS with configurable category voting strategy."""
    if not dets: return []
    kept, absorbed = soft_nms(dets, iou_thresh, sigma, score_thresh)
    if vote_mode == 'none':
        return kept
    for ki, k in enumerate(kept):
        group = absorbed[ki]
        if len(group) > 1:
            cat_scores = {}
            for ab in group:
                w = ab["score"] if vote_mode == 'score' else 1.0
                cat_scores[ab["category_id"]] = cat_scores.get(ab["category_id"], 0) + w
            kept[ki]["category_id"] = max(cat_scores, key=cat_scores.get)
    return kept

def wbf(passes, img_w, img_h, iou_thresh=0.50, skip=0.005, ct='box_and_model_avg', weights=None):
    bl, sl, ll = [], [], []
    for p in passes:
        b, s, l = [], [], []
        for d in p:
            x, y, w, h = d["bbox"]
            x1, y1 = max(0, x/img_w), max(0, y/img_h)
            x2, y2 = min(1, (x+w)/img_w), min(1, (y+h)/img_h)
            if x2 <= x1 or y2 <= y1: continue
            b.append([x1,y1,x2,y2]); s.append(d["score"]); l.append(d["category_id"])
        if b:
            bl.append(np.array(b, np.float32))
            sl.append(np.array(s, np.float32))
            ll.append(np.array(l, np.int32))
    if not bl: return []
    kwargs = dict(iou_thr=iou_thresh, skip_box_thr=skip, conf_type=ct)
    if weights: kwargs['weights'] = weights
    fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, **kwargs)
    return [{"category_id": int(la), "bbox": [round(b[0]*img_w,1), round(b[1]*img_h,1),
             round((b[2]-b[0])*img_w,1), round((b[3]-b[1])*img_h,1)], "score": round(float(sc),3)}
            for b, sc, la in zip(fb, fs, fl)]

def compute_ap(preds, gts, iou_thresh=0.5, check_class=False):
    if not gts: return 1.0 if not preds else 0.0
    if not preds: return 0.0
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    matched = [False]*len(gts)
    tp, fp = [], []
    for p in preds:
        best_iou, best_j = 0, -1
        for j, g in enumerate(gts):
            if check_class and p["category_id"] != g["category_id"]: continue
            iou = _iou(p["bbox"], g["bbox"])
            if iou > best_iou: best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0 and not matched[best_j]:
            tp.append(1); fp.append(0); matched[best_j] = True
        else:
            tp.append(0); fp.append(1)
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    rec = tp_c / len(gts); prec = tp_c / (tp_c + fp_c)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre)-2, -1, -1): mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])

def evaluate_config(passes_fn, weights=None, wbf_iou=0.50, snms_sigma=5.0,
                    snms_iou=0.45, snms_score=0.001, max_dets=300, 
                    skip_box=0.005, conf_type='box_and_model_avg', vote_mode='score'):
    """Full evaluation pipeline with all configurable params."""
    det_aps, cls_aps = [], []
    for iid in image_ids:
        info = img_info.get(iid)
        if not info: continue
        img_w, img_h = info["width"], info["height"]
        passes = passes_fn(iid)
        if len(passes) > 1:
            fused = wbf(passes, img_w, img_h, iou_thresh=wbf_iou, skip=skip_box, ct=conf_type, weights=weights)
        else:
            fused = passes[0] if passes else []
        dets = soft_nms_with_voting(fused, iou_thresh=snms_iou, sigma=snms_sigma,
                                     score_thresh=snms_score, vote_mode=vote_mode)
        if len(dets) > max_dets:
            dets.sort(key=lambda x: x["score"], reverse=True)
            dets = dets[:max_dets]
        gts = [{"bbox": g["bbox"], "category_id": g["category_id"]} for g in gt_by_image.get(iid, [])]
        det_p = [{"bbox": d["bbox"], "score": d["score"], "category_id": 0} for d in dets]
        det_g = [{"bbox": g["bbox"], "category_id": 0} for g in gts]
        det_aps.append(compute_ap(det_p, det_g))
        cls_aps.append(compute_ap(dets, gts, check_class=True))
    det = np.mean(det_aps); cls = np.mean(cls_aps)
    return 0.7*det + 0.3*cls, det, cls

# ============================================================================
# Logging
# ============================================================================
LOG_FILE = 'overnight_log.md'

def log_experiment(name, combined, det, cls, params, is_best=False):
    marker = " **NEW BEST**" if is_best else ""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"| {ts} | {name} | {combined:.4f} | {det:.4f} | {cls:.4f} | {params}{marker} |\n"
    with open(LOG_FILE, 'a') as f:
        f.write(line)
    return line

BEST = {'combined': 0.0, 'det': 0.0, 'cls': 0.0, 'name': '', 'params': {}}

def check_best(name, combined, det, cls, params):
    global BEST
    is_best = combined > BEST['combined']
    if is_best:
        BEST = {'combined': combined, 'det': det, 'cls': cls, 'name': name, 'params': params}
    log_experiment(name, combined, det, cls, params, is_best)
    return is_best

# ============================================================================
# Pass generators
# ============================================================================
def passes_8_all(iid):
    return [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
            c4[iid]['full_1280'], c4[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']]

def passes_5_medium(iid):
    return [c3[iid]['full_1280'],
            c4[iid]['full_1280'], c4[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1536']]

# ============================================================================
# ROUND 1: Establish baseline & fine-tune WBF iou threshold
# ============================================================================
def round1_wbf_iou():
    """Fine-grain WBF IoU threshold search around sweet spot."""
    with open(LOG_FILE, 'a') as f:
        f.write("\n### Round 1: WBF IoU threshold fine-tuning (8-pass)\n")
        f.write("| Time | Experiment | Combined | Det | Cls | Params |\n")
        f.write("|------|-----------|----------|-----|-----|--------|\n")
    
    w = [1,1,1,2,2,2,2,2]
    for iou in [0.42, 0.44, 0.46, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60]:
        c, d, cl = evaluate_config(passes_8_all, weights=w, wbf_iou=iou)
        check_best(f"wbf_iou={iou:.2f}", c, d, cl, f"wbf_iou={iou}")
        print(f"  wbf_iou={iou:.2f}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# ROUND 2: Soft-NMS sigma sweep
# ============================================================================
def round2_snms_sigma():
    """Fine-tune soft-NMS Gaussian sigma."""
    with open(LOG_FILE, 'a') as f:
        f.write("\n### Round 2: Soft-NMS sigma sweep\n")
        f.write("| Time | Experiment | Combined | Det | Cls | Params |\n")
        f.write("|------|-----------|----------|-----|-----|--------|\n")
    
    w = [1,1,1,2,2,2,2,2]
    best_iou = BEST['params'].split('=')[1] if 'wbf_iou' in BEST['params'] else '0.50'
    wbf_iou = float(best_iou) if BEST['params'].startswith('wbf_iou') else 0.50
    
    for sigma in [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 50.0]:
        c, d, cl = evaluate_config(passes_8_all, weights=w, wbf_iou=wbf_iou, snms_sigma=sigma)
        check_best(f"sigma={sigma}", c, d, cl, f"sigma={sigma}")
        print(f"  sigma={sigma}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# ROUND 3: Soft-NMS IoU threshold sweep
# ============================================================================
def round3_snms_iou():
    """Fine-tune soft-NMS IoU threshold."""
    with open(LOG_FILE, 'a') as f:
        f.write("\n### Round 3: Soft-NMS IoU threshold sweep\n")
        f.write("| Time | Experiment | Combined | Det | Cls | Params |\n")
        f.write("|------|-----------|----------|-----|-----|--------|\n")
    
    w = [1,1,1,2,2,2,2,2]
    for snms_iou in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        c, d, cl = evaluate_config(passes_8_all, weights=w, snms_iou=snms_iou)
        check_best(f"snms_iou={snms_iou:.2f}", c, d, cl, f"snms_iou={snms_iou}")
        print(f"  snms_iou={snms_iou:.2f}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# ROUND 4: Score threshold sweep 
# ============================================================================
def round4_score_thresh():
    """Fine-tune score threshold."""
    with open(LOG_FILE, 'a') as f:
        f.write("\n### Round 4: Score threshold sweep\n")
        f.write("| Time | Experiment | Combined | Det | Cls | Params |\n")
        f.write("|------|-----------|----------|-----|-----|--------|\n")
    
    w = [1,1,1,2,2,2,2,2]
    for st in [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
        c, d, cl = evaluate_config(passes_8_all, weights=w, snms_score=st)
        check_best(f"score_thresh={st}", c, d, cl, f"score_thresh={st}")
        print(f"  score_thresh={st}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# ROUND 5: skip_box_thresh sweep  
# ============================================================================
def round5_skip_box():
    """Fine-tune WBF skip_box_thresh."""
    with open(LOG_FILE, 'a') as f:
        f.write("\n### Round 5: WBF skip_box_thresh sweep\n")
        f.write("| Time | Experiment | Combined | Det | Cls | Params |\n")
        f.write("|------|-----------|----------|-----|-----|--------|\n")
    
    w = [1,1,1,2,2,2,2,2]
    for sb in [0.0001, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05]:
        c, d, cl = evaluate_config(passes_8_all, weights=w, skip_box=sb)
        check_best(f"skip_box={sb}", c, d, cl, f"skip_box={sb}")
        print(f"  skip_box={sb}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# ROUND 6: WBF conf_type
# ============================================================================
def round6_conf_type():
    """Test different WBF confidence calculation methods."""
    with open(LOG_FILE, 'a') as f:
        f.write("\n### Round 6: WBF conf_type\n")
        f.write("| Time | Experiment | Combined | Det | Cls | Params |\n")
        f.write("|------|-----------|----------|-----|-----|--------|\n")
    
    w = [1,1,1,2,2,2,2,2]
    for ct in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        c, d, cl = evaluate_config(passes_8_all, weights=w, conf_type=ct)
        check_best(f"conf_type={ct}", c, d, cl, f"conf_type={ct}")
        print(f"  conf_type={ct}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# ROUND 7: Fine-grained weight ratios
# ============================================================================
def round7_weights():
    """Exhaustive weight search."""
    with open(LOG_FILE, 'a') as f:
        f.write("\n### Round 7: Fine-grained weight ratios\n")
        f.write("| Time | Experiment | Combined | Det | Cls | Params |\n")
        f.write("|------|-----------|----------|-----|-----|--------|\n")
    
    # Test (v3_w, v4_w, v6_w) combinations as per-model weights expanded to 8 passes
    for v3w, v4w, v6w in [(1,2,2), (1,2,3), (1,3,2), (1,3,3), (2,3,2), (2,3,3),
                           (1,1,2), (2,2,3), (1,2,1), (2,3,1), (1,4,2), (1,4,3),
                           (2,4,3), (1,1,1), (1,2,4), (1,3,4), (2,2,1), (3,2,2),
                           (1,5,3), (2,5,3), (1,1,3)]:
        w = [v3w]*3 + [v4w]*2 + [v6w]*3
        c, d, cl = evaluate_config(passes_8_all, weights=w)
        check_best(f"w=v3:{v3w},v4:{v4w},v6:{v6w}", c, d, cl, f"w=[{','.join(map(str,w))}]")
        print(f"  v3={v3w} v4={v4w} v6={v6w}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# ROUND 8: Category voting strategy
# ============================================================================
def round8_voting():
    """Test voting strategies in soft-NMS."""
    with open(LOG_FILE, 'a') as f:
        f.write("\n### Round 8: Category voting strategy\n")
        f.write("| Time | Experiment | Combined | Det | Cls | Params |\n")
        f.write("|------|-----------|----------|-----|-----|--------|\n")
    
    w = [1,1,1,2,2,2,2,2]
    for vm in ['score', 'count', 'none']:
        c, d, cl = evaluate_config(passes_8_all, weights=w, vote_mode=vm)
        check_best(f"vote={vm}", c, d, cl, f"vote_mode={vm}")
        print(f"  vote_mode={vm}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# ROUND 9: Max detections per image
# ============================================================================
def round9_max_dets():
    """Test max detections cap."""
    with open(LOG_FILE, 'a') as f:
        f.write("\n### Round 9: Max detections per image\n")
        f.write("| Time | Experiment | Combined | Det | Cls | Params |\n")
        f.write("|------|-----------|----------|-----|-----|--------|\n")
    
    w = [1,1,1,2,2,2,2,2]
    for md in [100, 150, 200, 250, 300, 400, 500, 750, 1000]:
        c, d, cl = evaluate_config(passes_8_all, weights=w, max_dets=md)
        check_best(f"max_dets={md}", c, d, cl, f"max_dets={md}")
        print(f"  max_dets={md}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# ROUND 10: Combined best params confirmation 
# ============================================================================
def round10_combined_best():
    """Run the current accumulated best params as combo."""
    with open(LOG_FILE, 'a') as f:
        f.write("\n### Round 10: Combined best params verification\n")
        f.write("| Time | Experiment | Combined | Det | Cls | Params |\n")
        f.write("|------|-----------|----------|-----|-----|--------|\n")
    
    # We'll read best from each round and combine. For now test defaults vs tuned.
    # These will be filled after previous rounds.
    print("  (Will be filled after rounds 1-9 complete)")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    t0 = time.time()
    
    # Start with baseline 
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n## Simulation Run — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    print("="*80)
    print("BASELINE (current v8 config)")
    w = [1,1,1,2,2,2,2,2]
    c, d, cl = evaluate_config(passes_8_all, weights=w)
    BEST = {'combined': c, 'det': d, 'cls': cl, 'name': 'baseline', 'params': 'default'}
    print(f"  BASELINE: {c:.4f} (det={d:.4f} cls={cl:.4f})")
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n**Baseline**: {c:.4f} (det={d:.4f} cls={cl:.4f})\n")
    
    print("\n" + "="*80)
    print("ROUND 1: WBF IoU threshold")
    round1_wbf_iou()
    print(f"\n  >>> Best so far: {BEST['combined']:.4f} ({BEST['name']})")
    
    print("\n" + "="*80)
    print("ROUND 2: Soft-NMS sigma")
    round2_snms_sigma()
    print(f"\n  >>> Best so far: {BEST['combined']:.4f} ({BEST['name']})")
    
    print("\n" + "="*80)
    print("ROUND 3: Soft-NMS IoU threshold")
    round3_snms_iou()
    print(f"\n  >>> Best so far: {BEST['combined']:.4f} ({BEST['name']})")
    
    print("\n" + "="*80)
    print("ROUND 4: Score threshold")
    round4_score_thresh()
    print(f"\n  >>> Best so far: {BEST['combined']:.4f} ({BEST['name']})")
    
    print("\n" + "="*80)
    print("ROUND 5: Skip box threshold")
    round5_skip_box()
    print(f"\n  >>> Best so far: {BEST['combined']:.4f} ({BEST['name']})")
    
    print("\n" + "="*80)
    print("ROUND 6: WBF conf_type")
    round6_conf_type()
    print(f"\n  >>> Best so far: {BEST['combined']:.4f} ({BEST['name']})")
    
    print("\n" + "="*80)
    print("ROUND 7: Weight ratios")
    round7_weights()
    print(f"\n  >>> Best so far: {BEST['combined']:.4f} ({BEST['name']})")
    
    print("\n" + "="*80)
    print("ROUND 8: Category voting")
    round8_voting()
    print(f"\n  >>> Best so far: {BEST['combined']:.4f} ({BEST['name']})")
    
    print("\n" + "="*80)
    print("ROUND 9: Max detections")
    round9_max_dets()
    print(f"\n  >>> Best so far: {BEST['combined']:.4f} ({BEST['name']})")
    
    elapsed = time.time() - t0
    
    print("\n" + "="*80)
    print(f"FINAL BEST: {BEST['combined']:.4f} (det={BEST['det']:.4f} cls={BEST['cls']:.4f})")
    print(f"  Config: {BEST['name']} — {BEST['params']}")
    print(f"  Total time: {elapsed:.1f}s")
    
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n**FINAL BEST**: {BEST['combined']:.4f} (det={BEST['det']:.4f} cls={BEST['cls']:.4f})\n")
        f.write(f"Config: {BEST['name']} — {BEST['params']}\n")
        f.write(f"Total simulation time: {elapsed:.1f}s\n")
    
    print("\nDone. Results logged to overnight_log.md")
