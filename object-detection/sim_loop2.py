"""Simulation 2: Combine best findings from round 1 and explore deeper.

Round 1 winners:
- WBF IoU: 0.48 (0.9491)
- Sigma: 1.0 (0.9499) <<<< biggest
- SNMS IoU: 0.60+ (0.9490)
- Score thresh: 0.0001-0.0005 (0.9488)
- Skip box: doesn't matter <=0.01
- Conf type: box_and_model_avg (0.9487)
- Weights: v3=1,v4=1,v6=2 (0.9492) or v3=1,v4=2,v6=2 (0.9487)
- Vote: none (0.9490) > score (0.9487) > count (0.9479)
- Max dets: 400 (0.9489)

Strategy: Combine winners, then do fine-grained sweeps around the combined best.
"""
import pickle, json, time, sys, os
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
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
print(f"Loaded {len(image_ids)} images")

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

def evaluate(passes_fn, weights=None, wbf_iou=0.50, snms_sigma=5.0,
             snms_iou=0.45, snms_score=0.001, max_dets=300, 
             skip_box=0.005, conf_type='box_and_model_avg', vote_mode='none'):
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
        dets = soft_nms(fused, iou_thresh=snms_iou, sigma=snms_sigma,
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

# Pass generators
def passes_8_all(iid):
    return [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
            c4[iid]['full_1280'], c4[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']]

LOG_FILE = 'overnight_log.md'
BEST = {'combined': 0.0}

def log(msg):
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def test(name, **kwargs):
    global BEST
    c, d, cl = evaluate(passes_8_all, **kwargs)
    is_best = c > BEST.get('combined', 0)
    if is_best:
        BEST = {'combined': c, 'det': d, 'cls': cl, 'name': name, 'kwargs': kwargs}
    ts = datetime.now().strftime("%H:%M:%S")
    marker = " **NEW BEST**" if is_best else ""
    log(f"| {ts} | {name} | {c:.4f} | {d:.4f} | {cl:.4f} | {kwargs}{marker} |")
    return c, d, cl

# ============================================================================
# START SIMULATION 2
# ============================================================================
if __name__ == '__main__':
    t0 = time.time()
    log(f"\n## Simulation 2 - Combined Best + Deep Sweep -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ---- SECTION 1: Combine R1 winners ----
    log("\n### Section 1: Combining round 1 winners")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    # Baseline (current run.py defaults)
    test("baseline_default", weights=[1,1,1,2,2,2,2,2], wbf_iou=0.50, snms_sigma=5.0, snms_iou=0.45, snms_score=0.001, max_dets=300, vote_mode='score')
    
    # Just sigma=1.0 (the biggest single winner)
    test("sigma_1.0_only", weights=[1,1,1,2,2,2,2,2], wbf_iou=0.50, snms_sigma=1.0, snms_iou=0.45, snms_score=0.001, max_dets=300, vote_mode='score')
    
    # sigma=1.0 + wbf_iou=0.48
    test("sigma1+wbf48", weights=[1,1,1,2,2,2,2,2], wbf_iou=0.48, snms_sigma=1.0, snms_iou=0.45, snms_score=0.001, max_dets=300, vote_mode='score')
    
    # sigma=1.0 + wbf=0.48 + vote=none
    test("sigma1+wbf48+noVote", weights=[1,1,1,2,2,2,2,2], wbf_iou=0.48, snms_sigma=1.0, snms_iou=0.45, snms_score=0.001, max_dets=300, vote_mode='none')
    
    # sigma=1.0 + wbf=0.48 + vote=none + max_dets=400
    test("above+maxDets400", weights=[1,1,1,2,2,2,2,2], wbf_iou=0.48, snms_sigma=1.0, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')
    
    # sigma=1.0 + wbf=0.48 + vote=none + score=0.0005
    test("above+score0005", weights=[1,1,1,2,2,2,2,2], wbf_iou=0.48, snms_sigma=1.0, snms_iou=0.45, snms_score=0.0005, max_dets=400, vote_mode='none')
    
    # Try with v3=1,v4=1,v6=2 weights (R7 runner-up)
    test("sigma1+wbf48+v6heavy", weights=[1,1,1,1,1,2,2,2], wbf_iou=0.48, snms_sigma=1.0, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')
    
    # sigma=1.0 + snms_iou=0.60
    test("sigma1+snms60", weights=[1,1,1,2,2,2,2,2], wbf_iou=0.48, snms_sigma=1.0, snms_iou=0.60, snms_score=0.001, max_dets=400, vote_mode='none')

    print(f"\n>>> Section 1 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- SECTION 2: Fine grid around sigma=1.0 ----
    log("\n### Section 2: Fine sigma grid (with best WBF/vote)")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    for sigma in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]:
        test(f"sigma={sigma:.1f}", weights=[1,1,1,2,2,2,2,2], wbf_iou=0.48, 
             snms_sigma=sigma, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')

    print(f"\n>>> Section 2 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- SECTION 3: Fine WBF IoU grid ----
    log("\n### Section 3: Fine WBF IoU grid (with best sigma)")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    # Use the best sigma found so far
    best_sigma = float(BEST.get('kwargs', {}).get('snms_sigma', 1.0))
    for wbf_iou in [0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.52, 0.54, 0.56]:
        test(f"wbf={wbf_iou:.2f}", weights=[1,1,1,2,2,2,2,2], wbf_iou=wbf_iou,
             snms_sigma=best_sigma, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')

    print(f"\n>>> Section 3 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- SECTION 4: Weight search with best params ----
    log("\n### Section 4: Weight search with combined best params")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    best_wbf = float(BEST.get('kwargs', {}).get('wbf_iou', 0.48))
    best_sigma = float(BEST.get('kwargs', {}).get('snms_sigma', 1.0))
    
    for v3w, v4w, v6w in [(1,2,2), (1,1,2), (1,1,1), (2,2,3), (2,3,2), (2,3,3),
                           (1,2,3), (1,3,2), (1,2,1), (2,1,2), (2,1,1), (1,1,3),
                           (3,2,2), (2,2,2), (3,3,2), (1,2,4), (1,3,3)]:
        w = [v3w]*3 + [v4w]*2 + [v6w]*3
        test(f"w={v3w},{v4w},{v6w}", weights=w, wbf_iou=best_wbf,
             snms_sigma=best_sigma, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')

    print(f"\n>>> Section 4 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- SECTION 5: Per-scale asymmetric weights ----
    log("\n### Section 5: Per-scale asymmetric weights")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    # Try giving higher weight to specific scales
    for w_desc, w in [
        ("v3@1280=2,rest=1", [2,1,1,1,1,1,1,1]),
        ("v3@1536=2,rest=1", [1,1,2,1,1,1,1,1]),
        ("v4@1280=3,v4@1536=2,rest=1", [1,1,1,3,2,1,1,1]),
        ("v4@1280=2,v6@1280=2,rest=1", [1,1,1,2,1,2,1,1]),
        ("v3@12=1,v3@14=2,v3@15=2,v4=2,v6=2", [1,2,2,2,2,2,2,2]),
        ("v3@12=2,v3@14=1,v3@15=1,v4=2,v6=2", [2,1,1,2,2,2,2,2]),
        ("all1280=2,all1408=1,all1536=1", [2,1,1,2,1,2,1,1]),
        ("all1536=2,rest=1", [1,1,2,1,2,1,1,2]),
        ("v3=1,v4@12=2,v4@15=3,v6=2", [1,1,1,2,3,2,2,2]),
        ("v3=1,v4@12=3,v4@15=2,v6=2", [1,1,1,3,2,2,2,2]),
    ]:
        test(f"asym:{w_desc[:30]}", weights=w, wbf_iou=best_wbf,
             snms_sigma=best_sigma, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')

    print(f"\n>>> Section 5 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- SECTION 6: SNMS IoU fine-tune with combined best ----
    log("\n### Section 6: SNMS IoU fine-tune")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    best_w = BEST.get('kwargs', {}).get('weights', [1,1,1,2,2,2,2,2])
    for snms_iou in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        test(f"snms_iou={snms_iou:.2f}", weights=best_w, wbf_iou=best_wbf,
             snms_sigma=best_sigma, snms_iou=snms_iou, snms_score=0.001, max_dets=400, vote_mode='none')

    print(f"\n>>> Section 6 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- FINAL ----
    elapsed = time.time() - t0
    log(f"\n**SIM 2 FINAL BEST**: {BEST['combined']:.4f} (det={BEST['det']:.4f} cls={BEST['cls']:.4f})")
    log(f"Config: {BEST['name']}")
    log(f"Params: {BEST.get('kwargs', {})}")
    log(f"Total time: {elapsed:.1f}s")
    
    print(f"\n{'='*80}")
    print(f"FINAL BEST: {BEST['combined']:.4f} (det={BEST['det']:.4f} cls={BEST['cls']:.4f})")
    print(f"Config: {BEST['name']}")
    print(f"Params: {BEST.get('kwargs', {})}")
