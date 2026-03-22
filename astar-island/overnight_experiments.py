"""
Overnight experiment suite for Astar Island.
Tests multiple improvements using correct entropy-weighted KL scoring.
Baseline: a=0.75, clip=1e-6, zero-mountain -> LOO-CV 68.40
"""
import json, sys, time, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import uniform_filter, distance_transform_edt
from scipy.optimize import differential_evolution
from datetime import datetime

sys.path.insert(0, "src")
DATA_DIR = Path("data")
LOG_FILE = Path("overnight_log.md")
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}

# -- Load all GT data --
all_entries = []
for gf in sorted(DATA_DIR.glob('ground_truth_*.json')):
    rid = gf.stem.replace('ground_truth_', '')
    with open(gf) as f:
        data = json.load(f)
    for sk, entry in data.items():
        ig = np.array(entry['initial_grid'])
        gt = np.array(entry['ground_truth'])
        if ig.shape == (40,40) and gt.shape == (40,40,6):
            all_entries.append({'rid': rid, 'ig': ig, 'gt': gt, 'seed': sk})
print(f"Loaded {len(all_entries)} seeds from {len(set(e['rid'] for e in all_entries))} rounds")

# -- Competition scoring function --
def comp_score(pred, gt):
    eps = 1e-15
    gt_s = np.clip(gt, eps, None)
    pred_s = np.clip(pred, eps, None)
    entropy = -np.sum(gt * np.log(gt_s), axis=-1)
    kl = np.sum(gt * np.log(gt_s / pred_s), axis=-1)
    te = entropy.sum()
    if te < eps: return 100.0
    wkl = np.sum(entropy * kl) / te
    return max(0, min(100, 100 * np.exp(-3 * wkl)))

# -- Spatial features (auto_runner_v2 style with distance_transform) --
def compute_features_edt(ig):
    """Compute spatial features using EDT (same as auto_runner_v2.py)."""
    mapped = np.vectorize(GRID_TO_CLASS.get)(ig)
    H, W = mapped.shape
    settlement = (mapped == 1)
    dist_s = distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20.0)
    forest = (mapped == 4)
    dist_f = distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20.0)
    ocean = (ig == 10)
    dist_o = distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 40.0)
    port = (mapped == 2)
    dist_p = distance_transform_edt(~port) if port.any() else np.full((H,W), 40.0)
    settle_bin = np.full((H,W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    coastal = (dist_o <= 1.5).astype(int)
    near_port = (dist_p <= 2.0).astype(int)
    return mapped, settle_bin, near_forest, coastal, near_port, dist_s, dist_f, dist_o, dist_p

# -- Precompute features --
print("Precomputing features...")
for e in all_entries:
    ig = e['ig']
    mapped, sb, nf, co, np_ , ds, df, do, dp = compute_features_edt(ig)
    e['mapped'] = mapped
    e['feats'] = (sb, nf, co, np_)
    e['dist_s'] = ds
    e['dist_f'] = df
    e['dist_o'] = do
    e['dist_p'] = dp
    e['mtn'] = (mapped == 5)

# -- Cell model --
from simulator.cell_model import CellParams, predict_cell_distributions, params_from_vector
cell_vec = np.load(DATA_DIR / 'cell_model_params.npy')
cell_p = params_from_vector(cell_vec)

print("Precomputing cell model predictions...")
for e in all_entries:
    cd = predict_cell_distributions(e['ig'], cell_p)
    mtn = e['mtn']
    cd[~mtn, 5] = 0.0
    s = cd.sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1, s)
    cd /= s
    e['cell_zm'] = cd

# -- Round grouping --
rounds_map = defaultdict(list)
for i, e in enumerate(all_entries):
    rounds_map[e['rid']].append(i)

# -- Global tally (key: 5-feature tuple -> sum of GT distributions) --
def build_tallies(entries_idx, all_e):
    """Build feature-keyed tallies from a list of entry indices."""
    tally = defaultdict(lambda: np.zeros(6))
    for idx in entries_idx:
        e = all_e[idx]
        mapped = e['mapped']
        sb, nf, co, np_ = e['feats']
        gt = e['gt']
        for r in range(40):
            for c in range(40):
                key = (int(mapped[r,c]), int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]))
                tally[key] += gt[r,c]
    return dict(tally)

print("Building global tallies...")
global_tally = defaultdict(lambda: np.zeros(6))
round_tally = {}
for rid in rounds_map:
    rt = defaultdict(lambda: np.zeros(6))
    for idx in rounds_map[rid]:
        e = all_entries[idx]
        mapped = e['mapped']
        sb, nf, co, np_ = e['feats']
        gt = e['gt']
        for r in range(40):
            for c in range(40):
                key = (int(mapped[r,c]), int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]))
                rt[key] += gt[r,c]
                global_tally[key] += gt[r,c]
    round_tally[rid] = dict(rt)

# -- LOO-CV engine --
def build_loo_lut(hold_rid, min_n=20):
    """Build LUT excluding one round, with 5->4->3->class fallback."""
    train_tally = {}
    for k, v in global_tally.items():
        sub = round_tally[hold_rid].get(k, np.zeros(6))
        remainder = v - sub
        if remainder.sum() > 0:
            train_tally[k] = remainder
    
    fb4 = defaultdict(lambda: np.zeros(6))
    fb3 = defaultdict(lambda: np.zeros(6))
    class_totals = np.zeros((6, 6))
    for k, v in train_tally.items():
        fb4[(k[0], k[1], k[2], k[3])] += v
        fb3[(k[0], k[1], k[2])] += v
        class_totals[k[0]] += v
    
    lut = {k: v/v.sum() for k,v in train_tally.items() if v.sum() >= min_n}
    fb4_lut = {k: v/v.sum() for k,v in fb4.items() if v.sum() >= min_n}
    fb3_lut = {k: v/v.sum() for k,v in fb3.items() if v.sum() >= min_n}
    ca = {}
    for ci in range(6):
        s = class_totals[ci].sum()
        ca[ci] = class_totals[ci] / s if s > 0 else np.ones(6)/6
    return lut, fb4_lut, fb3_lut, ca

def predict_lut(idx, lut, fb4_lut, fb3_lut, ca, zero_mtn=True):
    """Build LUT prediction for one seed."""
    e = all_entries[idx]
    mapped = e['mapped']
    sb, nf, co, np_ = e['feats']
    pred = np.zeros((40,40,6))
    for r in range(40):
        for c in range(40):
            ic = int(mapped[r,c])
            key5 = (ic, int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]))
            if key5 in lut:
                pred[r,c] = lut[key5]
            elif key5[:4] in fb4_lut:
                pred[r,c] = fb4_lut[key5[:4]]
            elif key5[:3] in fb3_lut:
                pred[r,c] = fb3_lut[key5[:3]]
            else:
                pred[r,c] = ca.get(ic, np.ones(6)/6)
    if zero_mtn:
        mtn = e['mtn']
        pred[~mtn, 5] = 0.0
        s = pred.sum(axis=-1, keepdims=True)
        s = np.where(s == 0, 1, s)
        pred /= s
        pred[mtn] = np.array([0,0,0,0,0,1.0])
    return pred

def loo_cv_score(alpha=0.75, clip=1e-6, temperature=1.0, per_class_temp=None, smooth_sigma=0.0):
    """Full LOO-CV with given parameters. Returns (mean, std, per_round_scores)."""
    scores = []
    per_round = {}
    for hold_rid in rounds_map:
        lut, fb4, fb3, ca = build_loo_lut(hold_rid)
        round_scores = []
        for idx in rounds_map[hold_rid]:
            e = all_entries[idx]
            pred_lut = predict_lut(idx, lut, fb4, fb3, ca)
            
            if alpha > 0:
                cell_zm = e['cell_zm']
                lut_log = np.log(np.clip(pred_lut, clip, None))
                cell_log = np.log(np.clip(cell_zm, clip, None))
                mixed = (1-alpha) * lut_log + alpha * cell_log
                mixed -= mixed.max(axis=-1, keepdims=True)
                p = np.exp(mixed)
                p /= p.sum(axis=-1, keepdims=True)
            else:
                p = pred_lut.copy()
            
            # Temperature scaling
            if per_class_temp is not None:
                # Per-class temperature: different T for each output class
                mtn = e['mtn']
                non_mtn = ~mtn
                if non_mtn.any():
                    p_nm = np.clip(p[non_mtn], 1e-10, None)
                    log_p = np.log(p_nm)
                    for ci in range(6):
                        if per_class_temp[ci] != 1.0:
                            log_p[:, ci] /= per_class_temp[ci]
                    log_p -= log_p.max(axis=-1, keepdims=True)
                    p_nm = np.exp(log_p)
                    p_nm /= p_nm.sum(axis=-1, keepdims=True)
                    p[non_mtn] = p_nm
            elif temperature != 1.0:
                mtn = e['mtn']
                non_mtn = ~mtn
                if non_mtn.any():
                    p_nm = np.clip(p[non_mtn], 1e-10, None)
                    p_nm = np.exp(np.log(p_nm) / temperature)
                    p_nm /= p_nm.sum(axis=-1, keepdims=True)
                    p[non_mtn] = p_nm
            
            # Spatial smoothing
            if smooth_sigma > 0:
                mtn = e['mtn']
                for ci in range(6):
                    p[:,:,ci] = uniform_filter(p[:,:,ci], size=smooth_sigma, mode='constant')
                p /= p.sum(axis=-1, keepdims=True)
                if mtn.any():
                    p[mtn] = np.array([0,0,0,0,0,1.0])
            
            # Final clip
            p = np.clip(p, 1e-8, None)
            p /= p.sum(axis=-1, keepdims=True)
            
            s = comp_score(p, e['gt'])
            round_scores.append(s)
            scores.append(s)
        per_round[hold_rid] = np.mean(round_scores)
    return np.mean(scores), np.std(scores), per_round

def log_result(experiment, params_str, score_mean, score_std, improved, baseline=68.40):
    """Log experiment result to overnight_log.md"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    delta = score_mean - baseline
    status = "IMPROVED" if improved else "no improvement"
    line = f"| {ts} | {experiment} | {params_str} | {score_mean:.2f} +/- {score_std:.2f} | {delta:+.2f} | {status} |\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"  [{status}] {experiment}: {score_mean:.2f} +/- {score_std:.2f} (d={delta:+.2f})")

# Initialize log
if not LOG_FILE.exists():
    with open(LOG_FILE, "w") as f:
        f.write("# Overnight Experiment Log\n\n")
        f.write("| Timestamp | Experiment | Params | Score | Delta | Status |\n")
        f.write("|-----------|------------|--------|-------|-------|--------|\n")

# ==========================================================
# EXPERIMENT 1: Verify baseline
# ==========================================================
print("\n" + "="*60)
print("EXPERIMENT 1: Verify baseline (a=0.75, clip=1e-6, T=1.0)")
print("="*60)
mean, std, pr = loo_cv_score(alpha=0.75, clip=1e-6, temperature=1.0)
BASELINE = mean
print(f"  Baseline: {mean:.2f} +/- {std:.2f}")
print(f"  Per-round: {', '.join(f'{v:.1f}' for v in sorted(pr.values()))}")
log_result("Baseline", "a=0.75 clip=1e-6 T=1.0", mean, std, False, mean)

# ==========================================================
# EXPERIMENT 2: Temperature sweep
# ==========================================================
print("\n" + "="*60)
print("EXPERIMENT 2: Temperature sweep")
print("="*60)
best_t = 1.0
best_t_score = BASELINE
for T in [0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]:
    mean, std, _ = loo_cv_score(alpha=0.75, clip=1e-6, temperature=T)
    improved = mean > best_t_score + 0.01
    log_result("Temperature", f"T={T:.2f}", mean, std, improved, BASELINE)
    if mean > best_t_score:
        best_t = T
        best_t_score = mean
print(f"  Best temperature: T={best_t} -> {best_t_score:.2f}")

# ==========================================================
# EXPERIMENT 3: Alpha fine-tuning around 0.75
# ==========================================================
print("\n" + "="*60)
print("EXPERIMENT 3: Alpha fine-tuning")
print("="*60)
best_alpha = 0.75
best_alpha_score = BASELINE
for a in [0.65, 0.68, 0.70, 0.72, 0.73, 0.74, 0.76, 0.77, 0.78, 0.80, 0.82, 0.85]:
    mean, std, _ = loo_cv_score(alpha=a, clip=1e-6, temperature=best_t)
    improved = mean > best_alpha_score + 0.01
    log_result("Alpha fine-tune", f"a={a:.2f} T={best_t}", mean, std, improved, BASELINE)
    if mean > best_alpha_score:
        best_alpha = a
        best_alpha_score = mean
print(f"  Best alpha: a={best_alpha} -> {best_alpha_score:.2f}")

# ==========================================================
# EXPERIMENT 4: Spatial smoothing
# ==========================================================
print("\n" + "="*60)
print("EXPERIMENT 4: Spatial smoothing")
print("="*60)
best_smooth = 0.0
best_smooth_score = best_alpha_score
for sigma in [3, 5, 7]:
    mean, std, _ = loo_cv_score(alpha=best_alpha, clip=1e-6, temperature=best_t, smooth_sigma=sigma)
    improved = mean > best_smooth_score + 0.01
    log_result("Spatial smooth", f"s={sigma} a={best_alpha} T={best_t}", mean, std, improved, BASELINE)
    if mean > best_smooth_score:
        best_smooth = sigma
        best_smooth_score = mean
print(f"  Best smooth: s={best_smooth} -> {best_smooth_score:.2f}")

# ==========================================================
# EXPERIMENT 5: Per-class temperature
# ==========================================================
print("\n" + "="*60)
print("EXPERIMENT 5: Per-class temperature")
print("="*60)
# Classes: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
# Try boosting confidence on majority class (empty) and reducing on minority
base_temp = [1.0]*6
best_pct = None
best_pct_score = best_alpha_score
configs = [
    ("Sharpen empty", [0.8, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ("Sharpen settle", [1.0, 0.8, 1.0, 1.0, 1.0, 1.0]),
    ("Sharpen forest", [1.0, 1.0, 1.0, 1.0, 0.8, 1.0]),
    ("Widen settle", [1.0, 1.2, 1.0, 1.0, 1.0, 1.0]),
    ("Widen forest", [1.0, 1.0, 1.0, 1.0, 1.2, 1.0]),
    ("Widen minor", [1.0, 1.0, 1.3, 1.3, 1.0, 1.0]),
    ("Overall sharpen", [0.9, 0.9, 0.9, 0.9, 0.9, 1.0]),
    ("Overall widen", [1.1, 1.1, 1.1, 1.1, 1.1, 1.0]),
]
for name, pct in configs:
    mean, std, _ = loo_cv_score(alpha=best_alpha, clip=1e-6, per_class_temp=pct)
    improved = mean > best_pct_score + 0.01
    log_result("Per-class T", f"{name} {pct[:5]}", mean, std, improved, BASELINE)
    if mean > best_pct_score:
        best_pct = pct
        best_pct_score = mean
        print(f"  NEW BEST: {name} -> {mean:.2f}")
if best_pct:
    print(f"  Best per-class T: {best_pct} -> {best_pct_score:.2f}")
else:
    print(f"  No improvement from per-class T")

# ==========================================================
# EXPERIMENT 6: Min-N threshold sweep
# ==========================================================
print("\n" + "="*60)
print("EXPERIMENT 6: Min-N threshold for LUT buckets")
print("="*60)
# This requires modifying build_loo_lut, so let's do it inline
best_min_n = 20
best_min_n_score = BASELINE
for min_n in [5, 10, 15, 25, 30, 40, 50]:
    scores = []
    for hold_rid in rounds_map:
        # Build LUT with custom min_n
        train_tally = {}
        for k, v in global_tally.items():
            sub = round_tally[hold_rid].get(k, np.zeros(6))
            remainder = v - sub
            if remainder.sum() > 0:
                train_tally[k] = remainder
        fb4 = defaultdict(lambda: np.zeros(6))
        fb3 = defaultdict(lambda: np.zeros(6))
        ct = np.zeros((6,6))
        for k, v in train_tally.items():
            fb4[k[:4]] += v
            fb3[k[:3]] += v
            ct[k[0]] += v
        lut = {k: v/v.sum() for k,v in train_tally.items() if v.sum() >= min_n}
        fb4l = {k: v/v.sum() for k,v in fb4.items() if v.sum() >= min_n}
        fb3l = {k: v/v.sum() for k,v in fb3.items() if v.sum() >= min_n}
        ca = {}
        for ci in range(6):
            s = ct[ci].sum()
            ca[ci] = ct[ci]/s if s > 0 else np.ones(6)/6
        
        for idx in rounds_map[hold_rid]:
            e = all_entries[idx]
            pred_lut = predict_lut(idx, lut, fb4l, fb3l, ca)
            cell_zm = e['cell_zm']
            a = best_alpha
            lut_log = np.log(np.clip(pred_lut, 1e-6, None))
            cell_log = np.log(np.clip(cell_zm, 1e-6, None))
            mixed = (1-a)*lut_log + a*cell_log
            mixed -= mixed.max(axis=-1, keepdims=True)
            p = np.exp(mixed)
            p /= p.sum(axis=-1, keepdims=True)
            p = np.clip(p, 1e-8, None)
            p /= p.sum(axis=-1, keepdims=True)
            scores.append(comp_score(p, e['gt']))
    mean_s, std_s = np.mean(scores), np.std(scores)
    improved = mean_s > best_min_n_score + 0.01
    log_result("Min-N", f"min_n={min_n} a={best_alpha}", mean_s, std_s, improved, BASELINE)
    if mean_s > best_min_n_score:
        best_min_n = min_n  
        best_min_n_score = mean_s
print(f"  Best min_n: {best_min_n} -> {best_min_n_score:.2f}")

# ==========================================================
# EXPERIMENT 7: Re-optimize cell model with correct metric
# ==========================================================
print("\n" + "="*60)
print("EXPERIMENT 7: Re-optimize cell model (differential_evolution)")
print("="*60)

def cell_objective(param_vec):
    """Evaluate cell model parameters using LOO-CV with correct scoring."""
    try:
        params = params_from_vector(param_vec)
    except Exception:
        return 100.0  # penalty
    
    scores = []
    for hold_rid in rounds_map:
        lut, fb4, fb3, ca = build_loo_lut(hold_rid)
        for idx in rounds_map[hold_rid]:
            e = all_entries[idx]
            # Predict with new cell params
            cd = predict_cell_distributions(e['ig'], params)
            mtn = e['mtn']
            cd[~mtn, 5] = 0.0
            s = cd.sum(axis=-1, keepdims=True)
            s = np.where(s == 0, 1, s)
            cd /= s
            
            # Build LUT pred
            pred_lut = predict_lut(idx, lut, fb4, fb3, ca)
            
            # Blend
            lut_log = np.log(np.clip(pred_lut, 1e-6, None))
            cell_log = np.log(np.clip(cd, 1e-6, None))
            mixed = (1-best_alpha)*lut_log + best_alpha*cell_log
            mixed -= mixed.max(axis=-1, keepdims=True)
            p = np.exp(mixed)
            p /= p.sum(axis=-1, keepdims=True)
            p = np.clip(p, 1e-8, None)
            p /= p.sum(axis=-1, keepdims=True)
            scores.append(comp_score(p, e['gt']))
    return -np.mean(scores)  # minimize negative score

# Current cell model score
current_cell_score = -cell_objective(cell_vec)
print(f"  Current cell params score: {current_cell_score:.2f}")

# Bounds for 15 parameters
bounds = [
    (0.01, 0.8),   # settle_base_empty
    (0.5, 15.0),   # settle_scale_empty
    (0.01, 0.8),   # settle_base_forest
    (0.5, 20.0),   # settle_scale_forest (current=13.0)
    (0.005, 0.3),  # port_base
    (0.5, 15.0),   # port_scale
    (0.001, 0.15), # ruin_from_settle
    (0.001, 0.1),  # ruin_persistence
    (0.1, 0.9),    # settle_survival_base
    (0.0, 1.0),    # settle_survival_density_bonus
    (0.3, 0.99),   # forest_persist_base
    (0.05, 5.0),   # forest_settle_penalty
    (0.001, 0.15), # forest_from_empty
    (0.05, 0.5),   # forest_from_settle
    (0.05, 0.5),   # port_survival
]

print("  Starting differential_evolution (popsize=15, maxiter=50)...")
t0 = time.time()
iter_count = [0]
def de_callback(xk, convergence):
    iter_count[0] += 1
    if iter_count[0] % 10 == 0:
        print(f"    DE iteration {iter_count[0]}, convergence={convergence:.4f}")
    return False

result = differential_evolution(
    cell_objective, bounds,
    seed=42, popsize=15, maxiter=50,
    mutation=(0.5, 1.5), recombination=0.9,
    x0=cell_vec, polish=True,
    callback=de_callback
)
elapsed = time.time() - t0
new_cell_score = -result.fun
print(f"  Optimization done in {elapsed:.0f}s")
print(f"  Old cell score: {current_cell_score:.2f}")
print(f"  New cell score: {new_cell_score:.2f}")

if new_cell_score > current_cell_score + 0.05:
    print(f"  OK IMPROVED by {new_cell_score - current_cell_score:.2f}")
    # Save new params
    np.save(DATA_DIR / 'cell_model_params_new.npy', result.x)
    log_result("Cell model reopt", f"score {current_cell_score:.2f}->{new_cell_score:.2f}", 
               new_cell_score, 0, True, BASELINE)
    
    # Verify with full LOO-CV using new params
    print("  Verifying with full LOO-CV...")
    # Update cell predictions
    new_params = params_from_vector(result.x)
    for e in all_entries:
        cd = predict_cell_distributions(e['ig'], new_params)
        mtn = e['mtn']
        cd[~mtn, 5] = 0.0
        s = cd.sum(axis=-1, keepdims=True)
        s = np.where(s == 0, 1, s)
        cd /= s
        e['cell_zm'] = cd
    
    verify_mean, verify_std, _ = loo_cv_score(alpha=best_alpha, clip=1e-6, temperature=best_t)
    print(f"  Verified LOO-CV: {verify_mean:.2f} +/- {verify_std:.2f}")
    
    if verify_mean > BASELINE:
        # Deploy: overwrite cell_model_params.npy
        np.save(DATA_DIR / 'cell_model_params.npy', result.x)
        print(f"  OK DEPLOYED new cell params (backup at cell_model_params_new.npy)")
        BASELINE = verify_mean
        log_result("Cell deploy", f"new baseline {verify_mean:.2f}", verify_mean, verify_std, True, 68.40)
    else:
        # Revert cell predictions
        for e in all_entries:
            cd = predict_cell_distributions(e['ig'], cell_p)
            mtn = e['mtn']
            cd[~mtn, 5] = 0.0
            s = cd.sum(axis=-1, keepdims=True)
            s = np.where(s == 0, 1, s)
            cd /= s
            e['cell_zm'] = cd
        print(f"  FAIL Full LOO-CV check failed, reverted")
else:
    print(f"  FAIL No meaningful improvement ({new_cell_score:.2f} vs {current_cell_score:.2f})")
    log_result("Cell model reopt", f"no improve {new_cell_score:.2f}", new_cell_score, 0, False, BASELINE)


# ==========================================================
# EXPERIMENT 8: Additional spatial features
# ==========================================================
print("\n" + "="*60)
print("EXPERIMENT 8: Additional spatial features")
print("="*60)

# Try adding terrain roughness (variance of class types in neighborhood)
# Try adding settlement cluster size
# Try quadrant position

def compute_extended_features(ig, mapped):
    """Compute additional spatial features."""
    H, W = mapped.shape
    
    # Terrain diversity: count unique classes in 5x5 neighborhood
    diversity = np.zeros((H, W))
    for r in range(H):
        for c in range(W):
            r0, r1 = max(0, r-2), min(H, r+3)
            c0, c1 = max(0, c-2), min(W, c+3)
            patch = mapped[r0:r1, c0:c1]
            diversity[r, c] = len(np.unique(patch))
    diversity_bin = (diversity >= 3).astype(int)  # "diverse" = 3+ classes nearby
    
    # Quadrant (0=NW, 1=NE, 2=SW, 3=SE)
    quadrant = np.zeros((H, W), dtype=int)
    quadrant[H//2:, :] += 2
    quadrant[:, W//2:] += 1
    
    # Edge cells (adjacent to ocean/nothing)
    is_ocean = (ig == 10)
    is_land = ~is_ocean
    from scipy.ndimage import binary_dilation
    land_edge = binary_dilation(~is_land) & is_land
    edge_bin = land_edge.astype(int)
    
    return diversity_bin, quadrant, edge_bin

print("  Computing extended features...")
for e in all_entries:
    div_bin, quad, edge = compute_extended_features(e['ig'], e['mapped'])
    e['diversity'] = div_bin
    e['quadrant'] = quad
    e['edge'] = edge

# Test 6-feature LUT: add diversity_bin
print("  Testing 6-feature LUT (+ diversity)...")
# Build 6-feat tallies
global_tally6 = defaultdict(lambda: np.zeros(6))
round_tally6 = {}
for rid in rounds_map:
    rt6 = defaultdict(lambda: np.zeros(6))
    for idx in rounds_map[rid]:
        e = all_entries[idx]
        mapped = e['mapped']
        sb, nf, co, np_ = e['feats']
        div_bin = e['diversity']
        gt = e['gt']
        for r in range(40):
            for c in range(40):
                key = (int(mapped[r,c]), int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]), int(div_bin[r,c]))
                rt6[key] += gt[r,c]
                global_tally6[key] += gt[r,c]
    round_tally6[rid] = dict(rt6)

# LOO-CV with 6 features
scores_6f = []
for hold_rid in rounds_map:
    train6 = {}
    for k, v in global_tally6.items():
        sub = round_tally6[hold_rid].get(k, np.zeros(6))
        rem = v - sub
        if rem.sum() > 0:
            train6[k] = rem
    
    # Build fallbacks
    fb5 = defaultdict(lambda: np.zeros(6))
    for k, v in train6.items():
        fb5[k[:5]] += v
    
    min_n = 20
    lut6 = {k: v/v.sum() for k,v in train6.items() if v.sum() >= min_n}
    fb5_lut = {k: v/v.sum() for k,v in fb5.items() if v.sum() >= min_n}
    
    # Also need 4/3/class fallbacks from 5-feat
    lut5, fb4, fb3, ca = build_loo_lut(hold_rid)
    
    for idx in rounds_map[hold_rid]:
        e = all_entries[idx]
        mapped = e['mapped']
        sb, nf, co, np_ = e['feats']
        div_bin = e['diversity']
        mtn = e['mtn']
        
        pred = np.zeros((40,40,6))
        for r in range(40):
            for c in range(40):
                ic = int(mapped[r,c])
                key6 = (ic, int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]), int(div_bin[r,c]))
                key5 = key6[:5]
                if key6 in lut6:
                    pred[r,c] = lut6[key6]
                elif key5 in fb5_lut:
                    pred[r,c] = fb5_lut[key5]
                elif key5 in lut5:
                    pred[r,c] = lut5[key5]
                elif key5[:4] in fb4:
                    pred[r,c] = fb4[key5[:4]]
                elif key5[:3] in fb3:
                    pred[r,c] = fb3[key5[:3]]
                else:
                    pred[r,c] = ca.get(ic, np.ones(6)/6)
        
        # Zero mountain
        pred[~mtn, 5] = 0.0
        s = pred.sum(axis=-1, keepdims=True)
        s = np.where(s == 0, 1, s)
        pred /= s
        pred[mtn] = np.array([0,0,0,0,0,1.0])
        
        # Blend with cell model
        cell_zm = e['cell_zm']
        lut_log = np.log(np.clip(pred, 1e-6, None))
        cell_log = np.log(np.clip(cell_zm, 1e-6, None))
        mixed = (1-best_alpha)*lut_log + best_alpha*cell_log
        mixed -= mixed.max(axis=-1, keepdims=True)
        p = np.exp(mixed)
        p /= p.sum(axis=-1, keepdims=True)
        p = np.clip(p, 1e-8, None)
        p /= p.sum(axis=-1, keepdims=True)
        scores_6f.append(comp_score(p, e['gt']))

mean_6f = np.mean(scores_6f)
std_6f = np.std(scores_6f)
improved = mean_6f > BASELINE + 0.01
log_result("6-feat LUT (+diversity)", f"6-feat + a={best_alpha}", mean_6f, std_6f, improved, BASELINE)
print(f"  6-feat LUT: {mean_6f:.2f} +/- {std_6f:.2f} (baseline: {BASELINE:.2f})")

# ==========================================================
# EXPERIMENT 9: Obs-alpha simulation
# ==========================================================
print("\n" + "="*60)
print("EXPERIMENT 9: Simulated obs-alpha tuning")
print("="*60)
# Simulate having observations by using within-round GT as "observations"
# Then blend (obs-LUT) with cell model at different alphas
# This simulates the with-observations scenario

best_obs_alpha = 0.10
best_obs_score = 0
for obs_alpha in [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.50]:
    scores_obs = []
    for hold_rid in rounds_map:
        # "Observations": use GT from current round directly as the LUT
        # This simulates perfect within-round observations (upper bound)
        obs_tally = round_tally[hold_rid]
        
        # Build LUT from observations (within-round GT)
        fb4 = defaultdict(lambda: np.zeros(6))
        fb3 = defaultdict(lambda: np.zeros(6))
        ct = np.zeros((6,6))
        for k, v in obs_tally.items():
            fb4[k[:4]] += v
            fb3[k[:3]] += v
            ct[k[0]] += v
        
        obs_lut = {k: v/v.sum() for k,v in obs_tally.items() if v.sum() >= 20}
        fb4l = {k: v/v.sum() for k,v in fb4.items() if v.sum() >= 20}
        fb3l = {k: v/v.sum() for k,v in fb3.items() if v.sum() >= 20}
        ca = {}
        for ci in range(6):
            s = ct[ci].sum()
            ca[ci] = ct[ci]/s if s > 0 else np.ones(6)/6
        
        for idx in rounds_map[hold_rid]:
            e = all_entries[idx]
            pred_lut = predict_lut(idx, obs_lut, fb4l, fb3l, ca)
            
            if obs_alpha > 0:
                cell_zm = e['cell_zm']
                lut_log = np.log(np.clip(pred_lut, 1e-6, None))
                cell_log = np.log(np.clip(cell_zm, 1e-6, None))
                mixed = (1-obs_alpha)*lut_log + obs_alpha*cell_log
                mixed -= mixed.max(axis=-1, keepdims=True)
                p = np.exp(mixed)
                p /= p.sum(axis=-1, keepdims=True)
            else:
                p = pred_lut.copy()
            p = np.clip(p, 1e-8, None)
            p /= p.sum(axis=-1, keepdims=True)
            scores_obs.append(comp_score(p, e['gt']))
    
    mean_o = np.mean(scores_obs)
    std_o = np.std(scores_obs)
    improved = mean_o > best_obs_score + 0.01
    log_result("Obs-alpha sim", f"obs_a={obs_alpha:.2f}", mean_o, std_o, False, BASELINE)
    if mean_o > best_obs_score:
        best_obs_alpha = obs_alpha
        best_obs_score = mean_o

print(f"  Best obs-alpha (simulated): {best_obs_alpha} -> {best_obs_score:.2f}")
print(f"  NOTE: This uses within-round GT as 'observations' (upper bound)")

# ==========================================================
# SUMMARY
# ==========================================================
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"  Original baseline:    68.40")
print(f"  Best temperature:     T={best_t} -> {best_t_score:.2f}")
print(f"  Best alpha:           a={best_alpha} -> {best_alpha_score:.2f}")
print(f"  Best min_n:           {best_min_n}")
print(f"  Best smoothing:       s={best_smooth}")
if best_pct:
    print(f"  Best per-class T:     {best_pct}")
print(f"  Best obs-alpha (sim): {best_obs_alpha}")
print(f"  Current BASELINE:     {BASELINE:.2f}")
print(f"\nExperiments complete. Check overnight_log.md for details.")
