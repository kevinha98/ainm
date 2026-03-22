"""Simulation 3: Deep exploration around the winning asymmetric weights.

Sim 2 Best: 0.9506 (det=0.9694, cls=0.9068)
Config: weights=[1,1,1,2,1,2,1,1], wbf_iou=0.48, snms_sigma=1.0,
        snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none'

Pass order: [v3@1280, v3@1408, v3@1536, v4@1280, v4@1536, v6@1280, v6@1408, v6@1536]
Winner: Only v4@1280 and v6@1280 get weight=2, everything else=1

Strategy:
  S1: Perturb each weight position ±1 around [1,1,1,2,1,2,1,1]
  S2: Two-position perturbation combos (promising pairs)
  S3: Drop scales (weight=0) to see which are dispensable
  S4: 2D grid (wbf_iou × sigma) with winning weights
  S5: Score threshold + conf_type with winning weights
  S6: Multi-resolution pass subsets (fewer or more passes)
  S7: Random neighborhood search around overall best
"""
import pickle, json, time, sys, os, itertools, random
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

SCALE_NAMES = ["v3@1280", "v3@1408", "v3@1536", "v4@1280", "v4@1536", "v6@1280", "v6@1408", "v6@1536"]

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
    if weights:
        # Only include weights for non-empty passes
        kwargs['weights'] = weights
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
    """Return all 8 pass detections for an image."""
    return [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
            c4[iid]['full_1280'], c4[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']]

def evaluate(weights=[1,1,1,2,1,2,1,1], wbf_iou=0.48, snms_sigma=1.0,
             snms_iou=0.45, snms_score=0.001, max_dets=400,
             skip_box=0.005, conf_type='box_and_model_avg', vote_mode='none',
             pass_mask=None):
    """Evaluate with given params. pass_mask: list of 8 bools to include/exclude passes."""
    det_aps, cls_aps = [], []
    for iid in image_ids:
        info = img_info.get(iid)
        if not info: continue
        img_w, img_h = info["width"], info["height"]
        all_passes = get_all_passes(iid)
        
        if pass_mask is not None:
            passes = [p for p, m in zip(all_passes, pass_mask) if m]
            w = [wt for wt, m in zip(weights, pass_mask) if m]
        else:
            passes = all_passes
            w = list(weights)
        
        if len(passes) > 1:
            fused = wbf(passes, img_w, img_h, iou_thresh=wbf_iou, skip=skip_box, ct=conf_type, weights=w)
        elif len(passes) == 1:
            fused = passes[0]
        else:
            fused = []
            
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

LOG_FILE = 'overnight_log.md'
BEST = {'combined': 0.0}

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def test(name, **kwargs):
    global BEST
    c, d, cl = evaluate(**kwargs)
    is_best = c > BEST.get('combined', 0)
    if is_best:
        BEST = {'combined': c, 'det': d, 'cls': cl, 'name': name, 'kwargs': kwargs}
    ts = datetime.now().strftime("%H:%M:%S")
    marker = " **NEW BEST**" if is_best else ""
    log(f"| {ts} | {name} | {c:.4f} | {d:.4f} | {cl:.4f} | {kwargs}{marker} |")
    return c, d, cl

# ============================================================================
if __name__ == '__main__':
    t0 = time.time()
    log(f"\n## Simulation 3 - Asymmetric Weight Deep Dive -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    BASE_W = [1,1,1,2,1,2,1,1]
    BASE_KW = dict(wbf_iou=0.48, snms_sigma=1.0, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')

    # ---- SECTION 1: Single-position perturbations ----
    log("\n### Section 1: Single position perturbation +/-1 from [1,1,1,2,1,2,1,1]")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    # Baseline
    test("baseline_best", weights=list(BASE_W), **BASE_KW)

    for i in range(8):
        for delta in [-1, +1]:
            w = list(BASE_W)
            new_val = w[i] + delta
            if new_val < 0: continue
            if w[i] + delta == w[i]: continue  # skip if already tried
            w[i] = new_val
            if w == BASE_W: continue  # skip if same as baseline
            desc = f"{SCALE_NAMES[i]}={new_val}"
            test(f"perturb:{desc}", weights=w, **BASE_KW)

    print(f"\n>>> Section 1 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- SECTION 2: Promising two-position combos ----
    log("\n### Section 2: Two-position perturbation combos")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    # Based on S1 results, try combining promising deltas
    two_pos_combos = [
        # Boost both 1280 scales to 3
        ("v4@1280=3,v6@1280=3", [1,1,1,3,1,3,1,1]),
        # Boost v4@1280=3, keep v6@1280=2
        ("v4@1280=3,v6@1280=2", [1,1,1,3,1,2,1,1]),
        # Keep v4@1280=2, boost v6@1280=3
        ("v4@1280=2,v6@1280=3", [1,1,1,2,1,3,1,1]),
        # Add v6@1408
        ("v4@1280=2,v6@1280=2,v6@1408=2", [1,1,1,2,1,2,2,1]),
        # Add v4@1536
        ("v4@1280=2,v4@1536=2,v6@1280=2", [1,1,1,2,2,2,1,1]),
        # Remove v3@1536 (weight=0)
        ("no_v3@1536,v4@1280=2,v6@1280=2", [1,1,0,2,1,2,1,1]),
        # Remove v3@1408 (weight=0)
        ("no_v3@1408,v4@1280=2,v6@1280=2", [1,0,1,2,1,2,1,1]),
        # Remove both v3@1408+v3@1536
        ("no_v3@14+15,v4@1280=2,v6@1280=2", [1,0,0,2,1,2,1,1]),
        # Boost all 1280s to 2
        ("all1280=2,v4@1280=2,v6@1280=2", [2,1,1,2,1,2,1,1]),
        # v3@1280=2 + v4@1280=2 + v6@1280=2
        ("v3@12=2,v4@12=3,v6@12=2", [2,1,1,3,1,2,1,1]),
        # v3@1280=0
        ("no_v3@1280,v4@12=2,v6@12=2", [0,1,1,2,1,2,1,1]),
        # Higher v4@1280 with v6 boost
        ("v4@12=3,v6@12=3,v6@14=2", [1,1,1,3,1,3,2,1]),
        # v4@1280=2,v4@1536=2,v6@1280=2,v6@1408=2
        ("v4=2,v6@12=2,v6@14=2", [1,1,1,2,2,2,2,1]),
        # Reduce v3 aggressively
        ("v3=0,v4@12=2,v6@12=2", [0,0,0,2,1,2,1,1]),
        # v4@1280=2, v6=2 (all v6 scales)
        ("v4@12=2,v6_all=2", [1,1,1,2,1,2,2,2]),
    ]
    
    for desc, w in two_pos_combos:
        test(f"combo:{desc[:35]}", weights=w, **BASE_KW)

    print(f"\n>>> Section 2 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- SECTION 3: 2D grid (wbf_iou × sigma) with best weights ----
    log("\n### Section 3: 2D grid (wbf_iou x sigma) with current best weights")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    best_w = list(BEST.get('kwargs', {}).get('weights', BASE_W))
    for wbf_iou in [0.45, 0.46, 0.47, 0.48, 0.49, 0.50]:
        for sigma in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            test(f"grid:wbf={wbf_iou:.2f},s={sigma:.1f}", 
                 weights=best_w, wbf_iou=wbf_iou, snms_sigma=sigma, 
                 snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')

    print(f"\n>>> Section 3 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- SECTION 4: Score threshold + conf_type ----
    log("\n### Section 4: Score threshold + conf_type with best")  
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    best_w = list(BEST.get('kwargs', {}).get('weights', BASE_W))
    best_wbf = float(BEST.get('kwargs', {}).get('wbf_iou', 0.48))
    best_sigma = float(BEST.get('kwargs', {}).get('snms_sigma', 1.0))

    # Score thresholds
    for st in [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002]:
        test(f"score={st}", weights=best_w, wbf_iou=best_wbf, snms_sigma=best_sigma,
             snms_iou=0.45, snms_score=st, max_dets=400, vote_mode='none')
    
    # Conf types
    for ct in ['avg', 'box_and_model_avg', 'absent_model_aware_avg']:
        test(f"conf={ct[:15]}", weights=best_w, wbf_iou=best_wbf, snms_sigma=best_sigma,
             snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none', conf_type=ct)

    print(f"\n>>> Section 4 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- SECTION 5: Pass subset exploration ----
    log("\n### Section 5: Pass subset tests (fewer scales)")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    best_w = list(BEST.get('kwargs', {}).get('weights', BASE_W))
    best_kw = dict(
        wbf_iou=float(BEST.get('kwargs', {}).get('wbf_iou', 0.48)),
        snms_sigma=float(BEST.get('kwargs', {}).get('snms_sigma', 1.0)),
        snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none'
    )

    # Drop one scale at a time
    for drop_idx in range(8):
        mask = [True]*8
        mask[drop_idx] = False
        test(f"drop:{SCALE_NAMES[drop_idx]}", weights=best_w, pass_mask=mask, **best_kw)

    # Drop v3 entirely (only v4+v6)
    test("only_v4+v6", weights=best_w, pass_mask=[False,False,False,True,True,True,True,True], **best_kw)
    
    # Just the 1280 scales (3 passes)
    test("only_1280s", weights=best_w, pass_mask=[True,False,False,True,False,True,False,False], **best_kw)
    
    # 1280+1408 (no 1536)
    test("no_1536s", weights=best_w, pass_mask=[True,True,False,True,False,True,True,False], **best_kw)

    print(f"\n>>> Section 5 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- SECTION 6: Random neighborhood search ----
    log("\n### Section 6: Random neighborhood search (50 trials)")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    best_w = list(BEST.get('kwargs', {}).get('weights', BASE_W))
    best_kw = dict(
        wbf_iou=float(BEST.get('kwargs', {}).get('wbf_iou', 0.48)),
        snms_sigma=float(BEST.get('kwargs', {}).get('snms_sigma', 1.0)),
        snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none'
    )

    random.seed(42)
    tried = set()
    tried.add(tuple(best_w))

    for trial in range(50):
        # Perturb 1-3 random positions by -1, 0, or +1
        w = list(best_w)
        n_changes = random.randint(1, 3)
        positions = random.sample(range(8), n_changes)
        for p in positions:
            delta = random.choice([-1, 0, 1])
            w[p] = max(0, w[p] + delta)
        
        wt = tuple(w)
        if wt in tried: continue
        tried.add(wt)
        
        # Also perturb wbf_iou and sigma slightly
        wbf_jitter = random.uniform(-0.02, 0.02)
        sig_jitter = random.uniform(-0.2, 0.2)
        wbf_v = round(best_kw['wbf_iou'] + wbf_jitter, 3)
        sig_v = round(best_kw['snms_sigma'] + sig_jitter, 2)
        wbf_v = max(0.40, min(0.60, wbf_v))
        sig_v = max(0.5, min(2.0, sig_v))

        test(f"rnd{trial:02d}:w={w}", weights=w, wbf_iou=wbf_v, snms_sigma=sig_v,
             snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')

    print(f"\n>>> Section 6 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- SECTION 7: Final fine-tune around absolute best ----
    log("\n### Section 7: Final fine-tune around absolute best")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")

    best_w = list(BEST.get('kwargs', {}).get('weights', BASE_W))
    best_wbf = float(BEST.get('kwargs', {}).get('wbf_iou', 0.48))
    best_sigma = float(BEST.get('kwargs', {}).get('snms_sigma', 1.0))

    # Very fine WBF grid
    for wbf_iou in [best_wbf - 0.02, best_wbf - 0.01, best_wbf - 0.005, best_wbf, 
                     best_wbf + 0.005, best_wbf + 0.01, best_wbf + 0.02]:
        wbf_iou = round(wbf_iou, 3)
        test(f"finetune:wbf={wbf_iou}", weights=best_w, wbf_iou=wbf_iou, 
             snms_sigma=best_sigma, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')

    # Very fine sigma grid
    for sigma in [best_sigma - 0.2, best_sigma - 0.1, best_sigma - 0.05, best_sigma,
                   best_sigma + 0.05, best_sigma + 0.1, best_sigma + 0.2]:
        sigma = round(sigma, 2)
        if sigma <= 0: continue
        test(f"finetune:sigma={sigma}", weights=best_w, wbf_iou=best_wbf,
             snms_sigma=sigma, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')

    # Max dets fine
    for md in [350, 400, 450, 500, 600]:
        test(f"finetune:maxd={md}", weights=best_w, wbf_iou=best_wbf,
             snms_sigma=best_sigma, snms_iou=0.45, snms_score=0.001, max_dets=md, vote_mode='none')

    print(f"\n>>> Section 7 best: {BEST['combined']:.4f} ({BEST['name']})")

    # ---- FINAL ----
    elapsed = time.time() - t0
    log(f"\n**SIM 3 FINAL BEST**: {BEST['combined']:.4f} (det={BEST['det']:.4f} cls={BEST['cls']:.4f})")
    log(f"Config: {BEST['name']}")
    log(f"Params: {BEST.get('kwargs', {})}")
    log(f"Total time: {elapsed:.1f}s")

    print(f"\n{'='*80}")
    print(f"SIM 3 FINAL BEST: {BEST['combined']:.4f} (det={BEST['det']:.4f} cls={BEST['cls']:.4f})")
    print(f"Config: {BEST['name']}")
    print(f"Params: {BEST.get('kwargs', {})}")
