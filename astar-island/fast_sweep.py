"""Fast alpha + floor sweep with zero-mountain fix, using actual competition scoring."""
import json, sys, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import uniform_filter

sys.path.insert(0, "src")
DATA_DIR = Path("data")
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}

# Load GT
all_entries = []
for gf in sorted(DATA_DIR.glob('ground_truth_*.json')):
    rid = gf.stem.replace('ground_truth_', '')
    with open(gf) as f:
        data = json.load(f)
    for sk, entry in data.items():
        ig = np.array(entry['initial_grid'])
        gt = np.array(entry['ground_truth'])
        if ig.shape == (40,40) and gt.shape == (40,40,6):
            all_entries.append({'rid': rid, 'ig': ig, 'gt': gt})
print(f"Seeds: {len(all_entries)}")

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

from simulator.cell_model import CellParams, predict_cell_distributions
cell_params = np.load(DATA_DIR / 'cell_model_params.npy')
cell_p = CellParams(*cell_params)

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

# Precompute everything
print("Precomputing...")
for e in all_entries:
    ig = e['ig']
    e['mapped'] = np.vectorize(GRID_TO_CLASS.get)(ig)
    e['feats'] = compute_spatial_features(ig)
    cd = predict_cell_distributions(ig, cell_p)
    # Zero mountain for non-mountain cells
    mtn = (e['mapped'] == 5)
    cd[~mtn, 5] = 0.0
    s = cd.sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1, s)
    cd /= s
    e['cell_zm'] = cd
    e['mtn'] = mtn

# Precompute tallies by round
rounds_map = defaultdict(list)
for i, e in enumerate(all_entries):
    rounds_map[e['rid']].append(i)

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

print(f"Buckets: {len(global_tally)}")

# Precompute LUT predictions for each held-out round
print("Precomputing LOO LUT predictions...")
loo_preds = {}  # (rid, idx) -> pred_lut_zm (40,40,6)
for hold_rid in rounds_map:
    # Build LUT excluding this round
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
    
    MIN_N = 20
    lut = {k: v/v.sum() for k,v in train_tally.items() if v.sum() >= MIN_N}
    fb4_lut = {k: v/v.sum() for k,v in fb4.items() if v.sum() >= MIN_N}
    fb3_lut = {k: v/v.sum() for k,v in fb3.items() if v.sum() >= MIN_N}
    ca = {}
    for ci in range(6):
        s = class_totals[ci].sum()
        if s > 0:
            ca[ci] = class_totals[ci] / s
    
    for idx in rounds_map[hold_rid]:
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
        
        # Zero mountain for non-mountain
        pred[~e['mtn'], 5] = 0.0
        s = pred.sum(axis=-1, keepdims=True)
        s = np.where(s == 0, 1, s)
        pred /= s
        
        loo_preds[(hold_rid, idx)] = pred

print("LUT predictions precomputed!")

# Now fast sweep: only need to blend + score
def sweep_score(alpha, clip_floor=0.0001):
    scores = []
    for rid in rounds_map:
        for idx in rounds_map[rid]:
            e = all_entries[idx]
            pred_lut = loo_preds[(rid, idx)]
            
            if alpha > 0:
                cell_zm = e['cell_zm']
                lut_log = np.log(np.clip(pred_lut, clip_floor, None))
                cell_log = np.log(np.clip(cell_zm, clip_floor, None))
                mixed = (1-alpha) * lut_log + alpha * cell_log
                mixed -= mixed.max(axis=-1, keepdims=True)
                p = np.exp(mixed)
                p /= p.sum(axis=-1, keepdims=True)
            else:
                p = pred_lut.copy()
            
            scores.append(comp_score(p, e['gt']))
    return np.mean(scores), np.std(scores)

# Alpha sweep with zero mountain
print("\n=== Alpha sweep (zero-mountain, clip=0.0001) ===")
for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
    mean, std = sweep_score(alpha)
    print(f"  α={alpha:.2f}: {mean:.2f} ± {std:.2f}")

# Best alpha with different clip floors
print("\n=== Clip floor sweep (α=0.35 with zero-mountain) ===")
for cf in [1e-6, 1e-5, 0.0001, 0.001, 0.005, 0.01]:
    mean, std = sweep_score(0.35, cf)
    print(f"  clip={cf:.6f}: {mean:.2f} ± {std:.2f}")

# LUT-only (no cell model)
print(f"\n  LUT-only (α=0): {sweep_score(0.0)[0]:.2f}")
