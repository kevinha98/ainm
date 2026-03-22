"""Compare: class avg vs LUT vs LUT+cell on R15, using actual scoring."""
import json, sys, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import uniform_filter

sys.path.insert(0, "src")
DATA_DIR = Path("data")
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}

all_entries = []
for gf in sorted(DATA_DIR.glob('ground_truth_*.json')):
    rid = gf.stem.replace('ground_truth_', '')
    with open(gf) as f:
        data = json.load(f)
    for sk, entry in data.items():
        ig = np.array(entry['initial_grid'])
        gt = np.array(entry['ground_truth'])
        if ig.shape == (40,40) and gt.shape == (40,40,6):
            all_entries.append({'rid': rid, 'seed': sk, 'ig': ig, 'gt': gt})

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

# Build LUT from all rounds EXCEPT test round
def build_luts_excluding(exclude_rid, min_n=20):
    train = [e for e in all_entries if e['rid'] != exclude_rid]
    
    tally5 = defaultdict(lambda: np.zeros(6))
    class_counts = np.zeros((6, 6))
    
    for e in train:
        mapped = np.vectorize(GRID_TO_CLASS.get)(e['ig'])
        sb, nf, co, np_ = compute_spatial_features(e['ig'])
        for r in range(40):
            for c in range(40):
                key = (int(mapped[r,c]), int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]))
                tally5[key] += e['gt'][r,c]
                class_counts[int(mapped[r,c])] += e['gt'][r,c]
    
    # Build fallbacks
    fb4 = defaultdict(lambda: np.zeros(6))
    fb3 = defaultdict(lambda: np.zeros(6))
    for k, v in tally5.items():
        fb4[(k[0], k[1], k[2], k[3])] += v
        fb3[(k[0], k[1], k[2])] += v
    
    lut = {k: v/v.sum() for k,v in tally5.items() if v.sum() >= min_n}
    fb4_lut = {k: v/v.sum() for k,v in fb4.items() if v.sum() >= min_n}
    fb3_lut = {k: v/v.sum() for k,v in fb3.items() if v.sum() >= min_n}
    class_avgs = {}
    for ci in range(6):
        s = class_counts[ci].sum()
        if s > 0:
            class_avgs[ci] = class_counts[ci] / s
    
    return lut, fb4_lut, fb3_lut, class_avgs

# Test on each round (LOO)
print("=== LOO-CV: Class avg vs LUT vs LUT+Cell ===")
print(f"{'Round':<12} {'ClassAvg':>8} {'LUT':>8} {'LUT+Cell35':>10} {'LUT+Cell15':>10}")
print("-" * 55)

rounds = defaultdict(list)
for i, e in enumerate(all_entries):
    rounds[e['rid']].append(i)

all_ca = []; all_lut = []; all_lc35 = []; all_lc15 = []

for rid in sorted(rounds.keys()):
    lut, fb4, fb3, ca = build_luts_excluding(rid)
    
    ca_scores = []; lut_scores = []; lc35_scores = []; lc15_scores = []
    
    for idx in rounds[rid]:
        e = all_entries[idx]
        ig, gt = e['ig'], e['gt']
        mapped = np.vectorize(GRID_TO_CLASS.get)(ig)
        sb, nf, co, np_ = compute_spatial_features(ig)
        cell_dist = predict_cell_distributions(ig, cell_p)
        
        pred_ca = np.zeros((40,40,6))
        pred_lut = np.zeros((40,40,6))
        
        for r in range(40):
            for c in range(40):
                ic = int(mapped[r,c])
                pred_ca[r,c] = ca.get(ic, np.ones(6)/6)
                
                key5 = (ic, int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]))
                key4 = (ic, int(sb[r,c]), int(nf[r,c]), int(co[r,c]))
                key3 = (ic, int(sb[r,c]), int(nf[r,c]))
                
                if key5 in lut:
                    pred_lut[r,c] = lut[key5]
                elif key4 in fb4:
                    pred_lut[r,c] = fb4[key4]
                elif key3 in fb3:
                    pred_lut[r,c] = fb3[key3]
                else:
                    pred_lut[r,c] = ca.get(ic, np.ones(6)/6)
        
        # LUT + cell model
        clip_f = 0.0001
        lut_log = np.log(np.clip(pred_lut, clip_f, None))
        cell_log = np.log(np.clip(cell_dist, clip_f, None))
        
        mixed35 = 0.65 * lut_log + 0.35 * cell_log
        mixed35 -= mixed35.max(axis=-1, keepdims=True)
        p35 = np.exp(mixed35)
        p35 /= p35.sum(axis=-1, keepdims=True)
        
        mixed15 = 0.85 * lut_log + 0.15 * cell_log
        mixed15 -= mixed15.max(axis=-1, keepdims=True)
        p15 = np.exp(mixed15)
        p15 /= p15.sum(axis=-1, keepdims=True)
        
        ca_scores.append(comp_score(pred_ca, gt))
        lut_scores.append(comp_score(pred_lut, gt))
        lc35_scores.append(comp_score(p35, gt))
        lc15_scores.append(comp_score(p15, gt))
    
    avg_ca = np.mean(ca_scores)
    avg_lut = np.mean(lut_scores)
    avg_lc35 = np.mean(lc35_scores)
    avg_lc15 = np.mean(lc15_scores)
    all_ca.extend(ca_scores); all_lut.extend(lut_scores)
    all_lc35.extend(lc35_scores); all_lc15.extend(lc15_scores)
    
    marker = " ← R15" if rid == 'cc5442dd' else ""
    print(f"  {rid:<10} {avg_ca:8.1f} {avg_lut:8.1f} {avg_lc35:10.1f} {avg_lc15:10.1f}{marker}")

print("-" * 55)
print(f"  {'MEAN':<10} {np.mean(all_ca):8.1f} {np.mean(all_lut):8.1f} {np.mean(all_lc35):10.1f} {np.mean(all_lc15):10.1f}")
print(f"  {'STD':<10} {np.std(all_ca):8.1f} {np.std(all_lut):8.1f} {np.std(all_lc35):10.1f} {np.std(all_lc15):10.1f}")

# ── Now test with FIXES: zero mountain for non-mountain cells
print("\n\n=== WITH FIXES: zero mountain + retune ===")
print(f"{'Round':<12} {'LUT':>8} {'LUT+C35':>8} {'LUT+C35+ZM':>10} {'LUT+ZM':>8}")
print("-" * 55)

all_lut2 = []; all_lc35_zm = []; all_lut_zm = []

for rid in sorted(rounds.keys()):
    lut, fb4, fb3, ca = build_luts_excluding(rid)
    
    lut_scores = []; lc35_scores = []; lc35_zm_scores = []; lut_zm_scores = []
    
    for idx in rounds[rid]:
        e = all_entries[idx]
        ig, gt = e['ig'], e['gt']
        mapped = np.vectorize(GRID_TO_CLASS.get)(ig)
        sb, nf, co, np_ = compute_spatial_features(ig)
        cell_dist = predict_cell_distributions(ig, cell_p)
        
        # Zero out mountain in cell model for non-mountain cells
        mtn_mask = (mapped == 5)
        cell_dist_zm = cell_dist.copy()
        cell_dist_zm[~mtn_mask, 5] = 0.0
        # Renormalize
        sums = cell_dist_zm.sum(axis=-1, keepdims=True)
        sums = np.where(sums == 0, 1, sums)
        cell_dist_zm = cell_dist_zm / sums
        
        pred_lut = np.zeros((40,40,6))
        for r in range(40):
            for c in range(40):
                ic = int(mapped[r,c])
                key5 = (ic, int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]))
                key4 = (ic, int(sb[r,c]), int(nf[r,c]), int(co[r,c]))
                key3 = (ic, int(sb[r,c]), int(nf[r,c]))
                if key5 in lut:
                    pred_lut[r,c] = lut[key5]
                elif key4 in fb4:
                    pred_lut[r,c] = fb4[key4]
                elif key3 in fb3:
                    pred_lut[r,c] = fb3[key3]
                else:
                    pred_lut[r,c] = ca.get(ic, np.ones(6)/6)
        
        # Also zero mountain in LUT for non-mountain
        pred_lut_zm = pred_lut.copy()
        pred_lut_zm[~mtn_mask, 5] = 0.0
        sums = pred_lut_zm.sum(axis=-1, keepdims=True)
        sums = np.where(sums == 0, 1, sums)
        pred_lut_zm = pred_lut_zm / sums
        
        # LUT only + zero mountain
        lut_zm_scores.append(comp_score(pred_lut_zm, gt))
        
        # LUT + cell (α=0.35) + zero mountain
        clip_f = 0.0001
        lut_log = np.log(np.clip(pred_lut_zm, clip_f, None))
        cell_log = np.log(np.clip(cell_dist_zm, clip_f, None))
        mixed = 0.65 * lut_log + 0.35 * cell_log
        mixed -= mixed.max(axis=-1, keepdims=True)
        p = np.exp(mixed)
        p /= p.sum(axis=-1, keepdims=True)
        lc35_zm_scores.append(comp_score(p, gt))
        
        lut_scores.append(comp_score(pred_lut, gt))
    
    all_lut2.extend(lut_scores)
    all_lc35_zm.extend(lc35_zm_scores)
    all_lut_zm.extend(lut_zm_scores)
    
    marker = " ← R15" if rid == 'cc5442dd' else ""
    print(f"  {rid:<10} {np.mean(lut_scores):8.1f} {'-':>8} {np.mean(lc35_zm_scores):10.1f} {np.mean(lut_zm_scores):8.1f}{marker}")

print("-" * 55)
print(f"  {'MEAN':<10} {np.mean(all_lut2):8.1f} {'':>8} {np.mean(all_lc35_zm):10.1f} {np.mean(all_lut_zm):8.1f}")

# Alpha sweep with zero mountain fix
print("\n=== Alpha sweep with zero-mountain fix ===")
for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    scores = []
    for rid in sorted(rounds.keys()):
        lut, fb4, fb3, ca = build_luts_excluding(rid)
        for idx in rounds[rid]:
            e = all_entries[idx]
            ig, gt = e['ig'], e['gt']
            mapped = np.vectorize(GRID_TO_CLASS.get)(ig)
            sb, nf, co, np_ = compute_spatial_features(ig)
            cell_dist = predict_cell_distributions(ig, cell_p)
            
            mtn_mask = (mapped == 5)
            cell_dist_zm = cell_dist.copy()
            cell_dist_zm[~mtn_mask, 5] = 0.0
            s = cell_dist_zm.sum(axis=-1, keepdims=True)
            s = np.where(s == 0, 1, s)
            cell_dist_zm /= s
            
            pred_lut = np.zeros((40,40,6))
            for r in range(40):
                for c in range(40):
                    ic = int(mapped[r,c])
                    key5 = (ic, int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]))
                    key4 = (ic, int(sb[r,c]), int(nf[r,c]), int(co[r,c]))
                    key3 = (ic, int(sb[r,c]), int(nf[r,c]))
                    if key5 in lut:
                        pred_lut[r,c] = lut[key5]
                    elif key4 in fb4:
                        pred_lut[r,c] = fb4[key4]
                    elif key3 in fb3:
                        pred_lut[r,c] = fb3[key3]
                    else:
                        pred_lut[r,c] = ca.get(ic, np.ones(6)/6)
            
            pred_lut[~mtn_mask, 5] = 0.0
            s = pred_lut.sum(axis=-1, keepdims=True)
            s = np.where(s == 0, 1, s)
            pred_lut /= s
            
            if alpha > 0:
                clip_f = 0.0001
                lut_log = np.log(np.clip(pred_lut, clip_f, None))
                cell_log = np.log(np.clip(cell_dist_zm, clip_f, None))
                mixed = (1-alpha) * lut_log + alpha * cell_log
                mixed -= mixed.max(axis=-1, keepdims=True)
                p = np.exp(mixed)
                p /= p.sum(axis=-1, keepdims=True)
            else:
                p = pred_lut
            
            scores.append(comp_score(p, gt))
    
    print(f"  α={alpha:.2f}: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
