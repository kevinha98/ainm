"""Phase 2: Elite Midnight Optimizer
Runs until midnight (2026-03-21 23:59:59), continuously improving.

Strategy:
1. Load all sim results from sims/ folder
2. Extract top-N configs
3. Run increasingly fine searches around the best
4. Adaptive: track which dimensions give most improvement
5. Save every improvement to overnight_log.md
6. Never stop until midnight
"""
import pickle, json, time, sys, os, random, glob, math
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from datetime import datetime, timedelta
from copy import deepcopy

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MIDNIGHT = datetime(2026, 3, 22, 0, 0, 0)  # midnight tonight
LOG_FILE = 'overnight_log.md'

# ============================================================================
# CORE EVALUATION ENGINE (copied from sim framework)
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

def evaluate(weights=[1,1,1,2,1,2,1,1], wbf_iou=0.48, snms_sigma=1.0,
             snms_iou=0.45, snms_score=0.001, max_dets=400,
             skip_box=0.005, conf_type='box_and_model_avg', vote_mode='none'):
    det_aps, cls_aps = [], []
    for iid in image_ids:
        info = img_info.get(iid)
        if not info: continue
        img_w, img_h = info["width"], info["height"]
        passes = get_all_passes(iid)
        if len(passes) > 1:
            fused = wbf_fuse(passes, img_w, img_h, iou_thresh=wbf_iou, skip=skip_box, ct=conf_type, weights=weights)
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
# LOGGING & TRACKING
# ============================================================================
GLOBAL_BEST = {'combined': 0.0}
EVAL_COUNT = 0
PHASE_COUNT = 0

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def test(name, **kwargs):
    global GLOBAL_BEST, EVAL_COUNT
    EVAL_COUNT += 1
    c, d, cl = evaluate(**kwargs)
    is_best = c > GLOBAL_BEST.get('combined', 0)
    if is_best:
        GLOBAL_BEST = {'combined': c, 'det': d, 'cls': cl, 'name': name, 'kwargs': kwargs}
        # Save best config to file
        with open('sims/ELITE_BEST.json', 'w') as f:
            json.dump({'combined': c, 'det': d, 'cls': cl, 'name': name, 'kwargs': kwargs,
                       'eval_count': EVAL_COUNT, 'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)
    ts = datetime.now().strftime("%H:%M:%S")
    marker = " **NEW BEST**" if is_best else ""
    log(f"| {ts} | {name} | {c:.5f} | {d:.5f} | {cl:.5f} | {kwargs}{marker} |")
    return c, d, cl

def time_remaining():
    return (MIDNIGHT - datetime.now()).total_seconds()

# ============================================================================
# PHASE 1: Load top configs from all sim results
# ============================================================================
def load_top_configs(n=10):
    results = []
    for f in glob.glob('sims/sim_*_result.json'):
        with open(f) as fh:
            d = json.load(fh)
            kw = d['best'].get('kwargs', {})
            results.append((d['best']['combined'], kw))
    results.sort(key=lambda x: x[0], reverse=True)
    # Deduplicate by rounding params
    seen = set()
    unique = []
    for score, kw in results:
        key = (tuple(kw.get('weights',[])), round(kw.get('wbf_iou',0),3), round(kw.get('snms_sigma',0),2))
        if key not in seen:
            seen.add(key)
            unique.append((score, kw))
    return unique[:n]

# ============================================================================
# OPTIMIZATION STRATEGIES
# ============================================================================
def strategy_fine_grid(base_kw, grid_step=0.005, sigma_step=0.05):
    """Ultra-fine 2D grid around a config."""
    kw = deepcopy(base_kw)
    wbf_c = kw.get('wbf_iou', 0.47)
    sig_c = kw.get('snms_sigma', 1.0)
    
    for wbf_d in np.arange(-3*grid_step, 3.01*grid_step, grid_step):
        for sig_d in np.arange(-3*sigma_step, 3.01*sigma_step, sigma_step):
            if time_remaining() < 30: return
            wbf_v = round(wbf_c + wbf_d, 4)
            sig_v = round(sig_c + sig_d, 3)
            if wbf_v <= 0.3 or wbf_v >= 0.7 or sig_v <= 0.1: continue
            params = deepcopy(kw)
            params['wbf_iou'] = wbf_v
            params['snms_sigma'] = sig_v
            test(f"fine:wbf={wbf_v},s={sig_v}", **params)

def strategy_weight_perturbation(base_kw, n_trials=50):
    """Random weight perturbations around the best."""
    kw = deepcopy(base_kw)
    base_w = list(kw.get('weights', [1,1,1,2,1,2,1,1]))
    
    for trial in range(n_trials):
        if time_remaining() < 30: return
        w = list(base_w)
        # Perturb 1-3 positions
        n_changes = random.randint(1, 3)
        positions = random.sample(range(8), n_changes)
        for p in positions:
            delta = random.choice([-1, 0, 1])
            w[p] = max(0, min(5, w[p] + delta))
        params = deepcopy(kw)
        params['weights'] = w
        test(f"wpert:{w}", **params)

def strategy_snms_sweep(base_kw):
    """Fine sweep of SNMS parameters."""
    kw = deepcopy(base_kw)
    
    # SNMS IoU sweep
    for snms_iou in [0.30, 0.35, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.55, 0.60]:
        if time_remaining() < 30: return
        params = deepcopy(kw)
        params['snms_iou'] = snms_iou
        test(f"snms_iou={snms_iou}", **params)
    
    # Score threshold
    for st in [0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015]:
        if time_remaining() < 30: return
        params = deepcopy(kw)
        params['snms_score'] = st
        test(f"snms_st={st}", **params)
    
    # Max dets
    for md in [300, 350, 400, 450, 500, 600, 800]:
        if time_remaining() < 30: return
        params = deepcopy(kw)
        params['max_dets'] = md
        test(f"maxd={md}", **params)

def strategy_skip_box(base_kw):
    """Skip box threshold sweep."""
    kw = deepcopy(base_kw)
    for skip in [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]:
        if time_remaining() < 30: return
        params = deepcopy(kw)
        params['skip_box'] = skip
        test(f"skip={skip}", **params)

def strategy_conf_type(base_kw):
    """Try different confidence aggregation types."""
    for ct in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        if time_remaining() < 30: return
        params = deepcopy(base_kw)
        params['conf_type'] = ct
        test(f"ct={ct[:15]}", **params)

def strategy_genetic_crossover(configs, n_offspring=30):
    """Genetic algorithm: crossover top configs."""
    if len(configs) < 2: return
    
    for _ in range(n_offspring):
        if time_remaining() < 30: return
        # Pick two parents
        p1_kw = deepcopy(random.choice(configs[:5])[1])
        p2_kw = deepcopy(random.choice(configs[:5])[1])
        
        # Crossover: mix params
        child = {}
        for key in ['wbf_iou', 'snms_sigma', 'snms_iou', 'snms_score', 'max_dets', 'vote_mode', 'skip_box', 'conf_type']:
            if key in p1_kw or key in p2_kw:
                child[key] = p1_kw.get(key, p2_kw.get(key)) if random.random() < 0.5 else p2_kw.get(key, p1_kw.get(key))
        
        # Crossover weights
        w1 = p1_kw.get('weights', [1]*8)
        w2 = p2_kw.get('weights', [1]*8)
        child_w = [w1[i] if random.random() < 0.5 else w2[i] for i in range(8)]
        child['weights'] = child_w
        
        # Mutation: slight jitter
        if random.random() < 0.3:
            child['wbf_iou'] = round(child.get('wbf_iou', 0.47) + random.uniform(-0.01, 0.01), 3)
        if random.random() < 0.3:
            child['snms_sigma'] = round(child.get('snms_sigma', 1.0) + random.uniform(-0.1, 0.1), 2)
        if random.random() < 0.2:
            idx = random.randint(0, 7)
            child_w[idx] = max(0, min(5, child_w[idx] + random.choice([-1, 0, 1])))
            child['weights'] = child_w
        
        # Clamp
        child['wbf_iou'] = max(0.35, min(0.60, child.get('wbf_iou', 0.47)))
        child['snms_sigma'] = max(0.3, min(3.0, child.get('snms_sigma', 1.0)))
        
        test(f"gen:w={child_w},wbf={child.get('wbf_iou'):.3f}", **child)

def strategy_simulated_annealing(base_kw, n_steps=100, temp_start=0.002, temp_end=0.0001):
    """Simulated annealing around top config."""
    current_kw = deepcopy(base_kw)
    current_score, _, _ = evaluate(**current_kw)
    best_score = current_score
    best_kw = deepcopy(current_kw)
    
    for step in range(n_steps):
        if time_remaining() < 30: break
        temp = temp_start * (temp_end / temp_start) ** (step / max(1, n_steps - 1))
        
        # Perturb
        candidate = deepcopy(current_kw)
        
        # Random dimension to perturb
        dim = random.choice(['wbf_iou', 'snms_sigma', 'weight', 'snms_iou', 'snms_score'])
        
        if dim == 'wbf_iou':
            candidate['wbf_iou'] = round(max(0.35, min(0.60, candidate.get('wbf_iou', 0.47) + random.gauss(0, 0.008))), 4)
        elif dim == 'snms_sigma':
            candidate['snms_sigma'] = round(max(0.3, min(3.0, candidate.get('snms_sigma', 1.0) + random.gauss(0, 0.08))), 3)
        elif dim == 'weight':
            w = list(candidate.get('weights', [1]*8))
            idx = random.randint(0, 7)
            w[idx] = max(0, min(5, w[idx] + random.choice([-1, 0, 1])))
            candidate['weights'] = w
        elif dim == 'snms_iou':
            candidate['snms_iou'] = round(max(0.2, min(0.7, candidate.get('snms_iou', 0.45) + random.gauss(0, 0.03))), 3)
        elif dim == 'snms_score':
            log_st = math.log10(max(1e-6, candidate.get('snms_score', 0.001)))
            log_st += random.gauss(0, 0.2)
            candidate['snms_score'] = round(10**max(-6, min(-1, log_st)), 6)
        
        cand_score, d, cl = evaluate(**candidate)
        
        # Accept or reject
        delta = cand_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
            current_kw = candidate
            current_score = cand_score
            if cand_score > best_score:
                best_score = cand_score
                best_kw = deepcopy(candidate)
                # Log to global
                ts = datetime.now().strftime("%H:%M:%S")
                log(f"| {ts} | SA_step{step} | {cand_score:.5f} | {d:.5f} | {cl:.5f} | {candidate} **NEW SA BEST** |")
                # Update global
                global GLOBAL_BEST, EVAL_COUNT
                EVAL_COUNT += 1
                if cand_score > GLOBAL_BEST.get('combined', 0):
                    GLOBAL_BEST = {'combined': cand_score, 'det': d, 'cls': cl, 'name': f'SA_step{step}', 'kwargs': candidate}
                    with open('sims/ELITE_BEST.json', 'w') as f:
                        json.dump(GLOBAL_BEST | {'eval_count': EVAL_COUNT, 'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)
        
        EVAL_COUNT += 1
    
    return best_kw, best_score

def strategy_ultra_fine(base_kw):
    """Ultra-fine 1D sweeps on each dimension."""
    kw = deepcopy(base_kw)
    
    # Ultra-fine WBF
    wbf_c = kw.get('wbf_iou', 0.47)
    for d in np.arange(-0.015, 0.016, 0.001):
        if time_remaining() < 30: return
        v = round(wbf_c + d, 4)
        if v <= 0.3: continue
        params = deepcopy(kw)
        params['wbf_iou'] = v
        test(f"uf_wbf={v}", **params)
    
    # Ultra-fine sigma
    sig_c = kw.get('snms_sigma', 1.0)
    for d in np.arange(-0.3, 0.31, 0.02):
        if time_remaining() < 30: return
        v = round(sig_c + d, 3)
        if v <= 0.1: continue
        params = deepcopy(kw)
        params['snms_sigma'] = v
        test(f"uf_sig={v}", **params)

# ============================================================================
# MAIN LOOP: Run until midnight
# ============================================================================
if __name__ == '__main__':
    t0 = time.time()
    random.seed(42)
    
    log(f"\n\n## ========== PHASE 2: ELITE MIDNIGHT OPTIMIZER ==========")
    log(f"## Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"## Target: Run until {MIDNIGHT.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"## Time remaining: {time_remaining()/3600:.1f} hours")
    
    # Load top configs from Phase 1
    top_configs = load_top_configs(15)
    log(f"\n### Loaded {len(top_configs)} unique top configs from sims/")
    log(f"### Top-3 starting scores: {[f'{s:.5f}' for s,_ in top_configs[:3]]}")
    
    # Initialize global best from top configs
    if top_configs:
        GLOBAL_BEST = {'combined': top_configs[0][0], 'kwargs': top_configs[0][1], 'name': 'loaded_best'}
    
    phase_num = 0
    
    while time_remaining() > 60:  # Stop 1 minute before midnight
        phase_num += 1
        remaining_h = time_remaining() / 3600
        
        log(f"\n### Phase 2.{phase_num} | {datetime.now().strftime('%H:%M:%S')} | {remaining_h:.1f}h remaining | Best: {GLOBAL_BEST.get('combined',0):.5f} | Evals: {EVAL_COUNT}")
        log("| Time | Experiment | Combined | Det | Cls | Params |")
        log("|------|-----------|----------|-----|-----|--------|")
        
        # Get current best config
        best_kw = deepcopy(GLOBAL_BEST.get('kwargs', top_configs[0][1] if top_configs else {}))
        
        # Cycle through strategies, spending proportional time
        if phase_num == 1:
            # First phase: comprehensive sweeps
            log(f"#### Strategy: Ultra-fine 1D sweeps on best config")
            strategy_ultra_fine(best_kw)
        
        elif phase_num == 2:
            log(f"#### Strategy: Fine 2D grid (wbf x sigma)")
            strategy_fine_grid(best_kw, grid_step=0.003, sigma_step=0.03)
        
        elif phase_num == 3:
            log(f"#### Strategy: SNMS parameter sweep")
            strategy_snms_sweep(best_kw)
        
        elif phase_num == 4:
            log(f"#### Strategy: Weight perturbations (50 trials)")
            strategy_weight_perturbation(best_kw, n_trials=50)
        
        elif phase_num == 5:
            log(f"#### Strategy: Skip box + conf type")
            strategy_skip_box(best_kw)
            strategy_conf_type(best_kw)
        
        elif phase_num == 6:
            log(f"#### Strategy: Genetic crossover from top configs")
            strategy_genetic_crossover(top_configs, n_offspring=40)
        
        elif phase_num == 7:
            log(f"#### Strategy: Simulated annealing (200 steps)")
            sa_kw, sa_score = strategy_simulated_annealing(best_kw, n_steps=200)
        
        else:
            # After initial strategies, keep cycling advanced ones
            cycle = (phase_num - 8) % 5
            
            if cycle == 0:
                log(f"#### Strategy: Ultra-fine around current best")
                strategy_ultra_fine(deepcopy(GLOBAL_BEST.get('kwargs', best_kw)))
            elif cycle == 1:
                log(f"#### Strategy: Simulated annealing (300 steps, refined)")
                strategy_simulated_annealing(deepcopy(GLOBAL_BEST.get('kwargs', best_kw)), n_steps=300, temp_start=0.001, temp_end=0.00005)
            elif cycle == 2:
                log(f"#### Strategy: Genetic crossover (refreshed)")
                # Refresh top configs with new best
                updated_configs = [(GLOBAL_BEST.get('combined', 0), GLOBAL_BEST.get('kwargs', {}))] + top_configs[:9]
                strategy_genetic_crossover(updated_configs, n_offspring=50)
            elif cycle == 3:
                log(f"#### Strategy: Weight mutations (aggressive)")
                strategy_weight_perturbation(deepcopy(GLOBAL_BEST.get('kwargs', best_kw)), n_trials=80)
            elif cycle == 4:
                log(f"#### Strategy: Fine 2D grid (tighter)")
                strategy_fine_grid(deepcopy(GLOBAL_BEST.get('kwargs', best_kw)), grid_step=0.002, sigma_step=0.02)
        
        # Status update
        elapsed = time.time() - t0
        log(f"\n**Phase 2.{phase_num} complete** | Evals: {EVAL_COUNT} | Best: {GLOBAL_BEST.get('combined',0):.5f} | Elapsed: {elapsed:.0f}s")
    
    # ── FINAL ──
    elapsed = time.time() - t0
    log(f"\n## ========== PHASE 2 COMPLETE ==========")
    log(f"## Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"## Total evaluations: {EVAL_COUNT}")
    log(f"## Total phases: {phase_num}")
    log(f"## Elapsed: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    log(f"## **GLOBAL BEST: {GLOBAL_BEST.get('combined',0):.5f}** (det={GLOBAL_BEST.get('det',0):.5f} cls={GLOBAL_BEST.get('cls',0):.5f})")
    log(f"## Config: {GLOBAL_BEST.get('name','')}")
    log(f"## Params: {GLOBAL_BEST.get('kwargs',{})}")
    
    # Save final state
    with open('sims/ELITE_FINAL.json', 'w') as f:
        json.dump({
            'global_best': GLOBAL_BEST,
            'total_evals': EVAL_COUNT,
            'total_phases': phase_num,
            'elapsed_seconds': elapsed,
            'finished': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"PHASE 2 DONE: {EVAL_COUNT} evals, {phase_num} phases")
    print(f"GLOBAL BEST: {GLOBAL_BEST.get('combined',0):.5f}")
    print(f"Config: {GLOBAL_BEST.get('kwargs',{})}")
