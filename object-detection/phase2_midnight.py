"""PHASE 2 MIDNIGHT OPTIMIZER - Runs until midnight (March 22 2026).

Starting from ELITE BEST: 0.95145
Config: weights=[1,2,1,2,3,4,1,2], wbf=0.4989, sigma=1.06, snms_iou=0.309, snms_score=9e-6

Strategy:
  Phase A: Systematic 1D sweeps around elite best (hyper-fine)
  Phase B: Genetic algorithm with tournament selection
  Phase C: Latin hypercube sampling of unexplored regions
  Phase D: Focused exploitation around any new bests
  Phase E: Score-threshold / conf-type / skip-box interaction searches
  Phases repeat until midnight.
"""
import pickle, json, time, sys, os, random, math, copy
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# DATA LOADING
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
print(f"Loaded {len(image_ids)} images", flush=True)

SCALE_NAMES = ["v3@1280", "v3@1408", "v3@1536", "v4@1280", "v4@1536", "v6@1280", "v6@1408", "v6@1536"]

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

def get_passes(iid):
    return [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
            c4[iid]['full_1280'], c4[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']]

def evaluate(weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=1.06,
             snms_iou=0.309, snms_score=9e-6, max_dets=400,
             skip_box=0.005, conf_type='box_and_model_avg', vote_mode='none'):
    det_aps, cls_aps = [], []
    for iid in image_ids:
        info = img_info.get(iid)
        if not info: continue
        img_w, img_h = info["width"], info["height"]
        passes = get_passes(iid)
        w = list(weights)
        if len(passes) > 1:
            fused = wbf(passes, img_w, img_h, iou_thresh=wbf_iou, skip=skip_box, ct=conf_type, weights=w)
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

# ============================================================================
# LOGGING
# ============================================================================
LOG_FILE = 'overnight_log.md'
BEST = {
    'combined': 0.95145,
    'det': 0.97013,
    'cls': 0.90785,
    'name': 'elite_seed',
    'kwargs': dict(weights=[1,2,1,2,3,4,1,2], wbf_iou=0.4989, snms_sigma=1.06,
                   snms_iou=0.309, snms_score=9e-6, max_dets=400, vote_mode='none')
}
EVAL_COUNT = 0
PHASE_COUNT = 0

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def test(name, **kwargs):
    global BEST, EVAL_COUNT
    EVAL_COUNT += 1
    c, d, cl = evaluate(**kwargs)
    is_best = c > BEST.get('combined', 0)
    if is_best:
        BEST = {'combined': c, 'det': d, 'cls': cl, 'name': name, 'kwargs': kwargs}
        # Save best immediately
        save_best()
    ts = datetime.now().strftime("%H:%M:%S")
    marker = " **NEW BEST**" if is_best else ""
    log(f"| {ts} | {name[:40]} | {c:.5f} | {d:.5f} | {cl:.5f} | E{EVAL_COUNT}{marker} |")
    return c, d, cl

def save_best():
    with open('sims/PHASE2_BEST.json', 'w') as f:
        json.dump({
            'combined': BEST['combined'],
            'det': BEST['det'],
            'cls': BEST['cls'],
            'name': BEST['name'],
            'kwargs': BEST['kwargs'],
            'eval_count': EVAL_COUNT,
            'phase_count': PHASE_COUNT,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

def get_best_kw():
    return dict(BEST['kwargs'])

# ============================================================================
# PHASE A: Ultra-fine 1D sweeps
# ============================================================================
def phase_a():
    global PHASE_COUNT
    PHASE_COUNT += 1
    log(f"\n### Phase A-{PHASE_COUNT}: Ultra-fine 1D sweeps around best")
    log("| Time | Experiment | Combined | Det | Cls | Info |")
    log("|------|-----------|----------|-----|-----|------|")
    
    kw = get_best_kw()
    
    # WBF IoU: fine sweep ±0.03
    base_wbf = kw['wbf_iou']
    for delta in np.arange(-0.03, 0.031, 0.003):
        v = round(base_wbf + delta, 4)
        if 0.35 <= v <= 0.65:
            kw2 = dict(kw); kw2['wbf_iou'] = v
            test(f"wbf={v:.4f}", **kw2)
    
    # Sigma: fine sweep ±0.3
    kw = get_best_kw()
    base_sig = kw['snms_sigma']
    for delta in np.arange(-0.3, 0.31, 0.03):
        v = round(base_sig + delta, 3)
        if 0.3 <= v <= 3.0:
            kw2 = dict(kw); kw2['snms_sigma'] = v
            test(f"sig={v:.3f}", **kw2)
    
    # SNMS IoU: fine sweep
    kw = get_best_kw()
    base_siou = kw['snms_iou']
    for delta in np.arange(-0.1, 0.11, 0.01):
        v = round(base_siou + delta, 3)
        if 0.15 <= v <= 0.65:
            kw2 = dict(kw); kw2['snms_iou'] = v
            test(f"siou={v:.3f}", **kw2)
    
    # Score threshold sweep
    kw = get_best_kw()
    for st in [1e-7, 5e-7, 1e-6, 5e-6, 9e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        kw2 = dict(kw); kw2['snms_score'] = st
        test(f"score={st:.1e}", **kw2)
    
    # Skip box threshold
    kw = get_best_kw()
    for sb in [0.0001, 0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02]:
        kw2 = dict(kw); kw2['skip_box'] = sb
        test(f"skip={sb}", **kw2)
    
    # Max dets
    kw = get_best_kw()
    for md in [300, 350, 400, 450, 500, 600, 800]:
        kw2 = dict(kw); kw2['max_dets'] = md
        test(f"maxd={md}", **kw2)

# ============================================================================
# PHASE B: Genetic Algorithm
# ============================================================================
def phase_b(pop_size=20, generations=10):
    global PHASE_COUNT
    PHASE_COUNT += 1
    log(f"\n### Phase B-{PHASE_COUNT}: Genetic Algorithm ({generations} gens, pop={pop_size})")
    log("| Time | Experiment | Combined | Det | Cls | Info |")
    log("|------|-----------|----------|-----|-----|------|")
    
    def random_individual():
        kw = get_best_kw()
        # Mutate weights
        w = list(kw['weights'])
        for i in range(8):
            if random.random() < 0.4:
                w[i] = max(0, w[i] + random.choice([-2, -1, 0, 1, 2]))
        kw['weights'] = w
        # Mutate continuous params
        kw['wbf_iou'] = round(max(0.35, min(0.65, kw['wbf_iou'] + random.gauss(0, 0.03))), 4)
        kw['snms_sigma'] = round(max(0.3, min(3.0, kw['snms_sigma'] + random.gauss(0, 0.2))), 3)
        kw['snms_iou'] = round(max(0.15, min(0.65, kw['snms_iou'] + random.gauss(0, 0.05))), 3)
        kw['snms_score'] = max(1e-7, min(0.01, kw['snms_score'] * (10 ** random.gauss(0, 0.5))))
        return kw
    
    def crossover(p1, p2):
        child = {}
        for k in p1:
            if k == 'weights':
                child[k] = [p1[k][i] if random.random() < 0.5 else p2[k][i] for i in range(8)]
            elif isinstance(p1[k], (int, float)):
                alpha = random.random()
                child[k] = round(alpha * p1[k] + (1-alpha) * p2[k], 6)
            else:
                child[k] = p1[k] if random.random() < 0.5 else p2[k]
        return child
    
    def mutate(ind, strength=0.3):
        kw = dict(ind)
        if random.random() < strength:
            w = list(kw['weights'])
            idx = random.randint(0, 7)
            w[idx] = max(0, w[idx] + random.choice([-1, 1]))
            kw['weights'] = w
        if random.random() < strength:
            kw['wbf_iou'] = round(max(0.35, min(0.65, kw['wbf_iou'] + random.gauss(0, 0.01))), 4)
        if random.random() < strength:
            kw['snms_sigma'] = round(max(0.3, min(3.0, kw['snms_sigma'] + random.gauss(0, 0.1))), 3)
        if random.random() < strength:
            kw['snms_iou'] = round(max(0.15, min(0.65, kw['snms_iou'] + random.gauss(0, 0.02))), 3)
        return kw
    
    # Initialize population: 50% from best, 50% random
    population = []
    for _ in range(pop_size // 2):
        population.append(get_best_kw())
    for _ in range(pop_size - len(population)):
        population.append(random_individual())
    
    # Evaluate initial population
    scored = []
    for i, ind in enumerate(population):
        c, d, cl = test(f"gen0_ind{i}", **ind)
        scored.append((c, ind))
    
    for gen in range(1, generations + 1):
        scored.sort(key=lambda x: -x[0])
        
        # Elitism: keep top 25%
        elite_n = max(2, pop_size // 4)
        new_pop = [s[1] for s in scored[:elite_n]]
        
        # Fill rest with crossover + mutation
        while len(new_pop) < pop_size:
            # Tournament selection
            t1 = max(random.sample(scored, min(3, len(scored))), key=lambda x: x[0])
            t2 = max(random.sample(scored, min(3, len(scored))), key=lambda x: x[0])
            child = crossover(t1[1], t2[1])
            child = mutate(child, strength=0.3 * (1 - gen/generations))  # Reduce mutation over time
            new_pop.append(child)
        
        # Evaluate
        scored = []
        for i, ind in enumerate(new_pop):
            c, d, cl = test(f"gen{gen}_ind{i}", **ind)
            scored.append((c, ind))

# ============================================================================
# PHASE C: Latin Hypercube Sampling
# ============================================================================
def phase_c(n_samples=40):
    global PHASE_COUNT
    PHASE_COUNT += 1
    log(f"\n### Phase C-{PHASE_COUNT}: Latin Hypercube Exploration ({n_samples} samples)")
    log("| Time | Experiment | Combined | Det | Cls | Info |")
    log("|------|-----------|----------|-----|-----|------|")
    
    # Define parameter ranges
    def lhs_sample(n):
        """Simple LHS for 5 continuous params + 8 weight positions."""
        samples = []
        for _ in range(n):
            kw = get_best_kw()
            # Continuous params with wider exploration
            kw['wbf_iou'] = round(random.uniform(0.40, 0.58), 4)
            kw['snms_sigma'] = round(random.uniform(0.5, 2.0), 3)
            kw['snms_iou'] = round(random.uniform(0.20, 0.55), 3)
            kw['snms_score'] = 10 ** random.uniform(-6, -2)
            kw['max_dets'] = random.choice([300, 350, 400, 450, 500])
            # Weights: explore diverse combinations
            w = [random.randint(0, 4) for _ in range(8)]
            # Ensure at least base scale has weight > 0
            w[0] = max(1, w[0])  # v3@1280 always present
            w[3] = max(1, w[3])  # v4@1280 always present
            w[5] = max(1, w[5])  # v6@1280 always present
            kw['weights'] = w
            samples.append(kw)
        return samples
    
    for i, kw in enumerate(lhs_sample(n_samples)):
        test(f"lhs_{i:03d}", **kw)

# ============================================================================
# PHASE D: Focused exploitation around current best
# ============================================================================
def phase_d(n_trials=50):
    global PHASE_COUNT
    PHASE_COUNT += 1
    log(f"\n### Phase D-{PHASE_COUNT}: Focused exploitation ({n_trials} trials)")
    log("| Time | Experiment | Combined | Det | Cls | Info |")
    log("|------|-----------|----------|-----|-----|------|")
    
    for i in range(n_trials):
        kw = get_best_kw()
        # Small perturbations
        n_changes = random.randint(1, 3)
        changes = random.sample(['w', 'wbf', 'sig', 'siou', 'score', 'skip'], min(n_changes, 6))
        
        for ch in changes:
            if ch == 'w':
                w = list(kw['weights'])
                idx = random.randint(0, 7)
                w[idx] = max(0, w[idx] + random.choice([-1, 0, 1]))
                kw['weights'] = w
            elif ch == 'wbf':
                kw['wbf_iou'] = round(max(0.35, min(0.65, kw['wbf_iou'] + random.gauss(0, 0.008))), 4)
            elif ch == 'sig':
                kw['snms_sigma'] = round(max(0.3, min(3.0, kw['snms_sigma'] + random.gauss(0, 0.05))), 3)
            elif ch == 'siou':
                kw['snms_iou'] = round(max(0.15, min(0.65, kw['snms_iou'] + random.gauss(0, 0.015))), 3)
            elif ch == 'score':
                kw['snms_score'] = max(1e-7, min(0.01, kw['snms_score'] * (10 ** random.gauss(0, 0.3))))
            elif ch == 'skip':
                kw['skip_box'] = max(0.0001, min(0.05, kw.get('skip_box', 0.005) + random.gauss(0, 0.002)))
        
        test(f"focus_{i:03d}", **kw)

# ============================================================================
# PHASE E: Weight-focused deep exploration
# ============================================================================
def phase_e():
    global PHASE_COUNT
    PHASE_COUNT += 1
    log(f"\n### Phase E-{PHASE_COUNT}: Weight interaction deep search")
    log("| Time | Experiment | Combined | Det | Cls | Info |")
    log("|------|-----------|----------|-----|-----|------|")
    
    kw = get_best_kw()
    base_w = list(kw['weights'])
    
    # Try all 2-position changes
    changes_2 = []
    for i in range(8):
        for j in range(i+1, 8):
            for di in [-1, 1]:
                for dj in [-1, 1]:
                    w = list(base_w)
                    w[i] = max(0, w[i] + di)
                    w[j] = max(0, w[j] + dj)
                    if w != base_w:
                        changes_2.append((f"w[{i}]+={di},w[{j}]+={dj}", w))
    
    random.shuffle(changes_2)
    for desc, w in changes_2[:60]:
        kw2 = get_best_kw()
        kw2['weights'] = w
        test(f"w2:{desc[:30]}", **kw2)

# ============================================================================
# PHASE F: 2D grid around best (WBF x sigma)
# ============================================================================
def phase_f():
    global PHASE_COUNT
    PHASE_COUNT += 1
    log(f"\n### Phase F-{PHASE_COUNT}: 2D fine grid (wbf x sigma)")
    log("| Time | Experiment | Combined | Det | Cls | Info |")
    log("|------|-----------|----------|-----|-----|------|")
    
    kw = get_best_kw()
    base_wbf = kw['wbf_iou']
    base_sig = kw['snms_sigma']
    
    for dw in np.arange(-0.015, 0.016, 0.003):
        for ds in np.arange(-0.15, 0.16, 0.03):
            wbf_v = round(base_wbf + dw, 4)
            sig_v = round(base_sig + ds, 3)
            if 0.35 <= wbf_v <= 0.60 and 0.3 <= sig_v <= 3.0:
                kw2 = get_best_kw()
                kw2['wbf_iou'] = wbf_v
                kw2['snms_sigma'] = sig_v
                test(f"grid:wbf={wbf_v:.4f},s={sig_v:.3f}", **kw2)

# ============================================================================
# PHASE G: Conf type comparison with best
# ============================================================================
def phase_g():
    global PHASE_COUNT
    PHASE_COUNT += 1
    log(f"\n### Phase G-{PHASE_COUNT}: Conf type + vote mode interaction")
    log("| Time | Experiment | Combined | Det | Cls | Info |")
    log("|------|-----------|----------|-----|-----|------|")
    
    kw = get_best_kw()
    for ct in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        for vm in ['none', 'score']:
            kw2 = dict(kw)
            kw2['conf_type'] = ct
            kw2['vote_mode'] = vm
            test(f"ct={ct[:15]},vm={vm}", **kw2)

# ============================================================================
# MAIN LOOP - RUNS UNTIL MIDNIGHT
# ============================================================================
if __name__ == '__main__':
    t0 = time.time()
    midnight = datetime.now().replace(hour=23, minute=55, second=0)
    
    log(f"\n## Phase 2 Midnight Optimizer -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"Starting best: {BEST['combined']:.5f} (det={BEST['det']:.5f} cls={BEST['cls']:.5f})")
    log(f"Target: midnight ({midnight.strftime('%H:%M')})")
    
    cycle = 0
    while datetime.now() < midnight:
        cycle += 1
        remaining = midnight - datetime.now()
        log(f"\n--- CYCLE {cycle} | Best: {BEST['combined']:.5f} | Evals: {EVAL_COUNT} | Remaining: {remaining} ---")
        
        try:
            # Phase A: 1D sweeps (fast)
            if datetime.now() < midnight:
                phase_a()
            
            # Phase B: Genetic algorithm (medium)
            if datetime.now() < midnight:
                phase_b(pop_size=15, generations=8)
            
            # Phase C: Latin hypercube (exploratory)
            if datetime.now() < midnight:
                phase_c(n_samples=30)
            
            # Phase D: Focused exploitation (narrow)
            if datetime.now() < midnight:
                phase_d(n_trials=40)
            
            # Phase E: Weight interactions
            if datetime.now() < midnight:
                phase_e()
            
            # Phase F: 2D grid refinement
            if datetime.now() < midnight:
                phase_f()
            
            # Phase G: Conf type + vote mode
            if datetime.now() < midnight:
                phase_g()
            
        except Exception as e:
            log(f"ERROR in cycle {cycle}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)
        
        # Save checkpoint
        save_best()
        log(f"\n--- CYCLE {cycle} COMPLETE | Best: {BEST['combined']:.5f} | Evals: {EVAL_COUNT} ---")
    
    # Final save
    elapsed = time.time() - t0
    log(f"\n**PHASE 2 FINAL BEST**: {BEST['combined']:.5f} (det={BEST['det']:.5f} cls={BEST['cls']:.5f})")
    log(f"Config: {BEST['name']}")
    log(f"Params: {BEST['kwargs']}")
    log(f"Total evals: {EVAL_COUNT}, Total cycles: {cycle}, Total time: {elapsed:.1f}s")
    
    # Save final
    with open('sims/PHASE2_FINAL.json', 'w') as f:
        json.dump({
            'global_best': BEST,
            'total_evals': EVAL_COUNT,
            'total_cycles': cycle,
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"PHASE 2 FINAL: {BEST['combined']:.5f}")
    print(f"Config: {BEST['kwargs']}")
