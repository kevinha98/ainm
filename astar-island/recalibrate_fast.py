"""
Fast vectorized recalibration using ACTUAL competition scoring.
Score = 100 * exp(-3 * weighted_kl)
"""
import json, sys, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import uniform_filter

sys.path.insert(0, "src")
DATA_DIR = Path("data")
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}

# ── Load all GT
gt_files = sorted(DATA_DIR.glob('ground_truth_*.json'))
print(f"Loading {len(gt_files)} GT files...")

all_entries = []
for gf in gt_files:
    rid = gf.stem.replace('ground_truth_', '')
    with open(gf) as f:
        data = json.load(f)
    for seed_key, entry in data.items():
        ig = np.array(entry['initial_grid'])
        gt = np.array(entry['ground_truth'])
        if ig.shape == (40, 40) and gt.shape == (40, 40, 6):
            all_entries.append({'rid': rid, 'seed': seed_key, 'ig': ig, 'gt': gt})

print(f"Total seeds: {len(all_entries)}")

# ── Spatial features
def compute_spatial_features(ig):
    mapped = np.vectorize(GRID_TO_CLASS.get)(ig)
    is_settle = (mapped == 1).astype(float)
    settle_density = uniform_filter(is_settle, size=5, mode='constant')
    settle_bin = np.zeros_like(mapped, dtype=int)
    settle_bin[settle_density > 0.3] = 2
    settle_bin[(settle_density > 0.05) & (settle_density <= 0.3)] = 1
    
    is_forest = (mapped == 4).astype(float)
    near_forest = (uniform_filter(is_forest, size=3, mode='constant') > 0.1).astype(int)
    
    is_ocean = (ig == 10).astype(float)
    ocean_nearby = uniform_filter(is_ocean, size=3, mode='constant')
    coastal = ((ocean_nearby > 0) & (ocean_nearby < 1.0)).astype(int)
    
    is_port = (mapped == 2).astype(float)
    near_port = (uniform_filter(is_port, size=5, mode='constant') > 0).astype(int)
    
    return settle_bin, near_forest, coastal, near_port

# ── Cell model
from simulator.cell_model import CellParams, predict_cell_distributions
cell_params = np.load(DATA_DIR / 'cell_model_params.npy')
cell_p = CellParams(*cell_params)

# ── Actual competition score (vectorized)
def competition_score(pred, gt):
    eps = 1e-15
    gt_safe = np.clip(gt, eps, None)
    pred_safe = np.clip(pred, eps, None)
    entropy = -np.sum(gt * np.log(gt_safe), axis=-1)  # (40,40)
    kl = np.sum(gt * np.log(gt_safe / pred_safe), axis=-1)  # (40,40)
    total_entropy = entropy.sum()
    if total_entropy < eps:
        return 100.0
    weighted_kl = np.sum(entropy * kl) / total_entropy
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))

# ── Precompute features for all entries
print("Precomputing features...")
for e in all_entries:
    ig = e['ig']
    e['mapped'] = np.vectorize(GRID_TO_CLASS.get)(ig)
    sb, nf, co, np_ = compute_spatial_features(ig)
    e['sb'] = sb; e['nf'] = nf; e['co'] = co; e['np'] = np_
    # Compute 5-feature key as flat array for fast lookup
    e['keys'] = e['mapped'] * 1000 + sb * 100 + nf * 10 + co * 2 + np_
    # Cell model predictions
    e['cell_dist'] = predict_cell_distributions(ig, cell_p)

# Group by round
rounds_map = defaultdict(list)
for i, e in enumerate(all_entries):
    rounds_map[e['rid']].append(i)

round_ids = sorted(rounds_map.keys())
print(f"Rounds: {len(round_ids)}")

# ── Build tally per round (precompute for LOO)
# Global tally, then subtract per-round
print("Building global tally...")
global_tally = defaultdict(lambda: np.zeros(6))
round_tally = {}  # round_tally[rid][(key)] = contribution

for rid in round_ids:
    rt = defaultdict(lambda: np.zeros(6))
    for idx in rounds_map[rid]:
        e = all_entries[idx]
        mapped, sb, nf, co, np_, gt = e['mapped'], e['sb'], e['nf'], e['co'], e['np'], e['gt']
        for r in range(40):
            for c in range(40):
                key = (int(mapped[r,c]), int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]))
                rt[key] += gt[r,c]
                global_tally[key] += gt[r,c]
    round_tally[rid] = dict(rt)

print(f"Global LUT buckets: {len(global_tally)}")

# ── Fast LOO-CV
def run_cv(alpha, clip_floor, submission_floor, min_n=20):
    scores = []
    
    for hold_rid in round_ids:
        # Subtract held-out round from global tally
        train_tally = {}
        for k, v in global_tally.items():
            sub = round_tally[hold_rid].get(k, np.zeros(6))
            remainder = v - sub
            if remainder.sum() > 0:
                train_tally[k] = remainder
        
        # Build fallback LUTs
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
        class_avgs = {}
        for ci in range(6):
            s = class_totals[ci].sum()
            if s > 0:
                class_avgs[ci] = class_totals[ci] / s
        
        # Score each held-out seed
        for hold_idx in rounds_map[hold_rid]:
            e = all_entries[hold_idx]
            mapped = e['mapped']
            sb, nf, co, np_ = e['sb'], e['nf'], e['co'], e['np']
            gt = e['gt']
            cell_dist = e['cell_dist']
            
            # Vectorized LUT lookup
            pred = np.zeros((40, 40, 6))
            for r in range(40):
                for c in range(40):
                    ic = int(mapped[r,c])
                    key5 = (ic, int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]))
                    key4 = (ic, int(sb[r,c]), int(nf[r,c]), int(co[r,c]))
                    key3 = (ic, int(sb[r,c]), int(nf[r,c]))
                    
                    if key5 in lut:
                        lut_p = lut[key5]
                    elif key4 in fb4_lut:
                        lut_p = fb4_lut[key4]
                    elif key3 in fb3_lut:
                        lut_p = fb3_lut[key3]
                    elif ic in class_avgs:
                        lut_p = class_avgs[ic]
                    else:
                        lut_p = np.ones(6) / 6
                    
                    pred[r, c] = lut_p
            
            # Ensemble with cell model in log space (vectorized)
            lut_log = np.log(np.clip(pred, clip_floor, None))
            cell_log = np.log(np.clip(cell_dist, clip_floor, None))
            mixed = (1 - alpha) * lut_log + alpha * cell_log
            mixed -= mixed.max(axis=-1, keepdims=True)
            pred = np.exp(mixed)
            pred /= pred.sum(axis=-1, keepdims=True)
            
            # Apply submission floor
            pred = np.maximum(pred, submission_floor)
            pred /= pred.sum(axis=-1, keepdims=True)
            
            score = competition_score(pred, gt)
            scores.append(score)
    
    return np.mean(scores), np.std(scores)

# ── Test submission floors
print("\n=== Submission floor sweep (α=0.35, clip=0.0001) ===")
for sf in [0.0, 1e-6, 0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
    mean, std = run_cv(0.35, 0.0001, sf)
    print(f"  floor={sf:.4f}: {mean:.2f} ± {std:.2f}")

print("\n=== Alpha sweep (clip=0.0001, floor=best from above) ===")
# Run with 0.01 floor first, then try best
for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    mean, std = run_cv(alpha, 0.0001, 0.01)
    print(f"  α={alpha:.2f}: {mean:.2f} ± {std:.2f}")

# LUT-only (no cell model)
print("\n=== LUT only (α=0) with different floors ===")
for sf in [0.005, 0.01, 0.015, 0.02, 0.03]:
    mean, std = run_cv(0.0, 0.0001, sf)
    print(f"  floor={sf:.3f}: {mean:.2f} ± {std:.2f}")

print("\n=== Grid search: best combo ===")
best_score = 0
best_cfg = None
for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    for sf in [0.005, 0.008, 0.01, 0.012, 0.015, 0.02]:
        mean, _ = run_cv(alpha, 0.0001, sf)
        if mean > best_score:
            best_score = mean
            best_cfg = (alpha, sf)

print(f"\nBEST CONFIG: α={best_cfg[0]:.2f}, submission_floor={best_cfg[1]:.3f} → score={best_score:.2f}")
