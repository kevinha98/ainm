"""Focused experiment suite: fix vote_mode, max_dets, and explore cls_mAP improvements.
Uses cached 3-model detections (v3+v4+v6) on 29 val images.
"""
import pickle, json, time, sys, os
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from datetime import datetime
from copy import deepcopy

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading cached detections...", flush=True)
c3 = pickle.load(open('cache_v3_29.pkl', 'rb'))
c4 = pickle.load(open('cache_v4_29.pkl', 'rb'))
c6 = pickle.load(open('cache_v6_29.pkl', 'rb'))
image_ids = sorted(set(c3.keys()) & set(c4.keys()) & set(c6.keys()))
gt_data = json.load(open('data/coco/train/annotations.json'))
gt_by_image = {}
for ann in gt_data["annotations"]:
    gt_by_image.setdefault(ann["image_id"], []).append(ann)
img_info = {img["id"]: img for img in gt_data["images"]}
print(f"Loaded {len(image_ids)} val images", flush=True)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy; union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0

def soft_nms(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001, vote_mode='none'):
    if not dets: return []
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
    if vote_mode == 'score':
        for ki, k in enumerate(kept):
            if len(absorbed_groups[ki]) > 1:
                cat_scores = {}
                for ab in absorbed_groups[ki]:
                    cat_scores[ab["category_id"]] = cat_scores.get(ab["category_id"], 0) + ab["score"]
                kept[ki]["category_id"] = max(cat_scores, key=cat_scores.get)
    elif vote_mode == 'count':
        for ki, k in enumerate(kept):
            if len(absorbed_groups[ki]) > 1:
                cat_counts = {}
                for ab in absorbed_groups[ki]:
                    cat_counts[ab["category_id"]] = cat_counts.get(ab["category_id"], 0) + 1
                kept[ki]["category_id"] = max(cat_counts, key=cat_counts.get)
    elif vote_mode == 'weighted':
        # Score-weighted but with decay for lower confidence
        for ki, k in enumerate(kept):
            if len(absorbed_groups[ki]) > 1:
                cat_scores = {}
                for ab in absorbed_groups[ki]:
                    w = ab["score"] ** 2  # Quadratic weighting
                    cat_scores[ab["category_id"]] = cat_scores.get(ab["category_id"], 0) + w
                kept[ki]["category_id"] = max(cat_scores, key=cat_scores.get)
    return kept

def wbf_fuse(passes, img_w, img_h, iou_thresh=0.50, skip=0.005, ct='box_and_model_avg', weights=None):
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

def get_all_passes(iid):
    return [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
            c4[iid]['full_1280'], c4[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']]

def evaluate(weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=1.06,
             snms_iou=0.309, snms_score=9e-6, max_dets=400,
             skip_box=0.005, conf_type='box_and_model_avg', vote_mode='none',
             conf_filter=0.0, cls_boost=1.0):
    """Evaluate with configurable params. Returns (combined, det, cls, n_preds)"""
    det_aps, cls_aps = [], []
    total_preds = 0
    for iid in image_ids:
        info = img_info.get(iid)
        if not info: continue
        img_w, img_h = info["width"], info["height"]
        passes = get_all_passes(iid)
        
        # Optional: filter low-conf detections before WBF
        if conf_filter > 0:
            passes = [[d for d in p if d['score'] >= conf_filter] for p in passes]
        
        fused = wbf_fuse(passes, img_w, img_h, iou_thresh=wbf_iou, skip=skip_box, ct=conf_type, weights=weights)
        dets = soft_nms(fused, iou_thresh=snms_iou, sigma=snms_sigma,
                        score_thresh=snms_score, vote_mode=vote_mode)
        if len(dets) > max_dets:
            dets.sort(key=lambda x: x["score"], reverse=True)
            dets = dets[:max_dets]
        
        total_preds += len(dets)
        gts = [{"bbox": g["bbox"], "category_id": g["category_id"]} for g in gt_by_image.get(iid, [])]
        det_p = [{"bbox": d["bbox"], "score": d["score"], "category_id": 0} for d in dets]
        det_g = [{"bbox": g["bbox"], "category_id": 0} for g in gts]
        det_aps.append(compute_ap(det_p, det_g))
        cls_aps.append(compute_ap(dets, gts, check_class=True))
    det = np.mean(det_aps); cls = np.mean(cls_aps)
    return 0.7*det + 0.3*cls, det, cls, total_preds

# ============================================================================
# EXPERIMENTS
# ============================================================================
LOG = 'overnight_log.md'
BEST = {'combined': 0.0, 'params': {}}
EXP_N = 0

def log_result(name, combined, det, cls, n_preds, params=None, is_best=False):
    global EXP_N
    EXP_N += 1
    ts = datetime.now().strftime("%H:%M:%S")
    marker = " **NEW BEST**" if is_best else ""
    line = f"| {ts} | {name} | {combined:.5f} | {det:.5f} | {cls:.5f} | E{EXP_N}{marker} |"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def test(name, **kwargs):
    global BEST
    c, d, cl, n = evaluate(**kwargs)
    is_best = c > BEST['combined']
    if is_best:
        BEST = {'combined': c, 'det': d, 'cls': cl, 'name': name, 'params': kwargs}
    log_result(name, c, d, cl, n, kwargs, is_best)
    return c, d, cl

def run_all():
    global BEST
    
    # ---------------------------------------------------------------
    # BLOCK 0: Reproduce elite optimizer's best as our baseline
    # ---------------------------------------------------------------
    print("\n=== BLOCK 0: BASELINE (Elite Optimizer Best) ===", flush=True)
    c, d, cl = test("baseline_elite",
        weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=1.06,
        snms_iou=0.309, snms_score=9e-6, max_dets=400, vote_mode='none')
    print(f"  Baseline: combined={c:.5f} det={d:.5f} cls={cl:.5f}", flush=True)

    # ---------------------------------------------------------------
    # BLOCK 1: Test vote_mode (mismatch: run.py does 'score' voting, elite says 'none')
    # ---------------------------------------------------------------
    print("\n=== BLOCK 1: VOTE MODE COMPARISON ===", flush=True)
    for vm in ['none', 'score', 'count', 'weighted']:
        test(f"vote={vm}",
            weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=1.06,
            snms_iou=0.309, snms_score=9e-6, max_dets=400, vote_mode=vm)

    # ---------------------------------------------------------------
    # BLOCK 2: Max detections (run.py uses 450, elite says 400)
    # ---------------------------------------------------------------
    print("\n=== BLOCK 2: MAX DETECTIONS ===", flush=True)
    for md in [300, 350, 400, 450, 500, 600, 800]:
        test(f"maxdets={md}",
            weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=1.06,
            snms_iou=0.309, snms_score=9e-6, max_dets=md, vote_mode='none')

    # ---------------------------------------------------------------
    # BLOCK 3: WBF conf_type alternatives
    # ---------------------------------------------------------------
    print("\n=== BLOCK 3: WBF CONF TYPE ===", flush=True)
    for ct in ['box_and_model_avg', 'avg', 'max', 'absent_model_aware_avg']:
        test(f"conftype={ct}",
            weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=1.06,
            snms_iou=0.309, snms_score=9e-6, max_dets=400, vote_mode='none',
            conf_type=ct)

    # ---------------------------------------------------------------
    # BLOCK 4: Weight redistribution (boost v6 which is freshest model)
    # ---------------------------------------------------------------
    print("\n=== BLOCK 4: WEIGHT COMBOS ===", flush=True)
    weight_configs = [
        # [v3@1280, v3@1408, v3@1536, v4@1280, v4@1536, v6@1280, v6@1408, v6@1536]
        [1, 2, 1, 2, 3, 4, 1, 2],   # current best
        [1, 2, 1, 2, 3, 5, 2, 3],   # boost v6 more
        [1, 2, 1, 2, 3, 6, 2, 3],   # heavy v6
        [1, 2, 1, 3, 4, 4, 1, 2],   # boost v4
        [2, 3, 2, 2, 3, 4, 1, 2],   # boost v3
        [1, 1, 1, 1, 1, 1, 1, 1],   # uniform
        [1, 2, 1, 2, 3, 3, 2, 2],   # balanced
        [1, 3, 1, 2, 4, 5, 2, 3],   # aggressive multi-scale
        [1, 2, 1, 1, 2, 5, 3, 4],   # v6-dominant
        [2, 2, 2, 3, 3, 3, 3, 3],   # equal v4+v6 bias
    ]
    for wc in weight_configs:
        test(f"w={wc}",
            weights=wc, wbf_iou=0.4989, snms_sigma=1.06,
            snms_iou=0.309, snms_score=9e-6, max_dets=400, vote_mode='none')

    # ---------------------------------------------------------------
    # BLOCK 5: Fine-tune around best WBF IoU (0.001 steps near 0.4989)
    # ---------------------------------------------------------------
    print("\n=== BLOCK 5: FINE WBF IoU SWEEP ===", flush=True)
    for wbf in np.arange(0.470, 0.530, 0.003):
        test(f"wbf={wbf:.3f}",
            weights=[1,2,1,2,3,4,1,2], wbf_iou=round(wbf, 4), snms_sigma=1.06,
            snms_iou=0.309, snms_score=9e-6, max_dets=400, vote_mode='none')

    # ---------------------------------------------------------------
    # BLOCK 6: Sigma fine-sweep near 1.06
    # ---------------------------------------------------------------
    print("\n=== BLOCK 6: FINE SIGMA SWEEP ===", flush=True)
    for sig in np.arange(0.80, 1.50, 0.03):
        test(f"sigma={sig:.2f}",
            weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=round(sig, 3),
            snms_iou=0.309, snms_score=9e-6, max_dets=400, vote_mode='none')

    # ---------------------------------------------------------------
    # BLOCK 7: NMS IoU sweep (affects overlap behavior)
    # ---------------------------------------------------------------
    print("\n=== BLOCK 7: NMS IoU SWEEP ===", flush=True)
    for niou in np.arange(0.20, 0.50, 0.02):
        test(f"nmsiou={niou:.2f}",
            weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=1.06,
            snms_iou=round(niou, 3), snms_score=9e-6, max_dets=400, vote_mode='none')

    # ---------------------------------------------------------------
    # BLOCK 8: Score threshold sweep
    # ---------------------------------------------------------------
    print("\n=== BLOCK 8: SCORE THRESHOLD ===", flush=True)
    for st in [1e-8, 5e-7, 1e-6, 5e-6, 9e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        test(f"scorethr={st}",
            weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=1.06,
            snms_iou=0.309, snms_score=st, max_dets=400, vote_mode='none')

    # ---------------------------------------------------------------
    # BLOCK 9: Skip box threshold
    # ---------------------------------------------------------------
    print("\n=== BLOCK 9: SKIP BOX THRESHOLD ===", flush=True)
    for sbt in [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.015, 0.02]:
        test(f"skipbox={sbt}",
            weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=1.06,
            snms_iou=0.309, snms_score=9e-6, max_dets=400, vote_mode='none',
            skip_box=sbt)

    # ---------------------------------------------------------------
    # BLOCK 10: Pre-WBF confidence filter
    # ---------------------------------------------------------------
    print("\n=== BLOCK 10: PRE-WBF CONF FILTER ===", flush=True)
    for cf in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10]:
        test(f"preconf={cf}",
            weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=1.06,
            snms_iou=0.309, snms_score=9e-6, max_dets=400, vote_mode='none',
            conf_filter=cf)

    # ---------------------------------------------------------------
    # BLOCK 11: Joint optimization around best params
    # ---------------------------------------------------------------
    print("\n=== BLOCK 11: JOINT OPTIMIZATION ===", flush=True)
    bp = BEST['params']
    best_wbf = bp.get('wbf_iou', 0.4989)
    best_sigma = bp.get('snms_sigma', 1.06)
    best_niou = bp.get('snms_iou', 0.309)
    best_md = bp.get('max_dets', 400)
    best_vm = bp.get('vote_mode', 'none')
    best_w = bp.get('weights', [1,2,1,2,3,4,1,2])
    
    # Grid around best params
    for wbf_d in [-0.005, 0, 0.005]:
        for sig_d in [-0.05, 0, 0.05]:
            for niou_d in [-0.02, 0, 0.02]:
                wbf = round(best_wbf + wbf_d, 4)
                sig = round(best_sigma + sig_d, 3)
                niou = round(best_niou + niou_d, 3)
                if wbf <= 0 or sig <= 0 or niou <= 0: continue
                test(f"joint:wbf={wbf},s={sig},n={niou}",
                    weights=best_w, wbf_iou=wbf, snms_sigma=sig,
                    snms_iou=niou, snms_score=9e-6, max_dets=best_md,
                    vote_mode=best_vm)

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"BEST FOUND: {BEST['name']}")
    print(f"  Combined: {BEST['combined']:.5f}")
    print(f"  Detection: {BEST['det']:.5f}")
    print(f"  Classification: {BEST['cls']:.5f}")
    print(f"  Params: {BEST['params']}")
    print(f"  Total experiments: {EXP_N}")
    
    # Save best
    with open('experiments_v9_best.json', 'w') as f:
        json.dump(BEST, f, indent=2, default=str)
    
    return BEST

if __name__ == '__main__':
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f"\n## Experiments V9 — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("| Time | Experiment | Combined | Det | Cls | Params |\n")
        f.write("|------|-----------|----------|-----|-----|--------|\n")
    run_all()
