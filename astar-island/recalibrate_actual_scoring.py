"""
Recalibrate using the ACTUAL competition scoring metric:
  score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
  weighted_kl = sum(entropy_i * KL_i) / sum(entropy_i)
  KL(p||q) = sum(p_i * log(p_i / q_i))
  entropy = -sum(p_i * log(p_i))
  
Key insight: we need a SUBMISSION floor, not just an internal clip floor.
The hackathon docs recommend 0.01.
"""
import json, sys, time, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import uniform_filter

sys.path.insert(0, "src")
DATA_DIR = Path("data")
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}

# ── Load all GT data
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

print(f"Total seeds with GT: {len(all_entries)}")

# ── Spatial features (same as auto_runner_v2)
def compute_spatial_features(ig):
    mapped = np.vectorize(GRID_TO_CLASS.get)(ig)
    is_settle = (mapped == 1).astype(float)
    settle_density = uniform_filter(is_settle, size=5, mode='constant')
    settle_bin = np.zeros_like(mapped, dtype=int)
    settle_bin[settle_density > 0.3] = 2
    settle_bin[(settle_density > 0.05) & (settle_density <= 0.3)] = 1
    
    is_forest = (mapped == 4).astype(float)
    forest_nearby = uniform_filter(is_forest, size=3, mode='constant')
    near_forest = (forest_nearby > 0.1).astype(int)
    
    is_ocean = (ig == 10).astype(float)
    ocean_nearby = uniform_filter(is_ocean, size=3, mode='constant')
    coastal = ((ocean_nearby > 0) & (ocean_nearby < 1.0)).astype(int)
    
    is_port = (mapped == 2).astype(float)
    port_nearby = uniform_filter(is_port, size=5, mode='constant')
    near_port = (port_nearby > 0).astype(int)
    
    return settle_bin, near_forest, coastal, near_port

# ── Cell model
from simulator.cell_model import CellParams, predict_cell_distributions
cell_params = np.load(DATA_DIR / 'cell_model_params.npy')
cell_p = CellParams(*cell_params)

# ── ACTUAL competition scoring function
def competition_score(pred, gt):
    """
    Compute the actual competition score for a 40x40 grid.
    pred: (40, 40, 6) predicted probabilities
    gt: (40, 40, 6) ground truth probabilities
    Returns: score in [0, 100]
    """
    eps = 1e-15
    
    # Compute entropy for each cell
    gt_safe = np.clip(gt, eps, None)
    entropy = -np.sum(gt * np.log(gt_safe), axis=-1)  # (40, 40)
    
    # KL divergence per cell: KL(p||q) = sum(p * log(p/q))
    pred_safe = np.clip(pred, eps, None)
    kl = np.sum(gt * np.log(gt_safe / pred_safe), axis=-1)  # (40, 40)
    
    # Weighted KL (only cells with entropy > 0)
    total_entropy = entropy.sum()
    if total_entropy < eps:
        return 100.0  # All static cells = perfect
    
    weighted_kl = np.sum(entropy * kl) / total_entropy
    
    score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))
    return score

# ── LOO-CV with actual scoring metric
def run_cv(alpha, clip_floor, submission_floor, min_n=20):
    """Run LOO-CV with the actual competition scoring metric."""
    scores = []
    
    # Group entries by round
    rounds = defaultdict(list)
    for i, e in enumerate(all_entries):
        rounds[e['rid']].append(i)
    
    for hold_rid, hold_indices in rounds.items():
        # Build LUT from all other rounds
        train_indices = [i for i in range(len(all_entries)) if all_entries[i]['rid'] != hold_rid]
        
        train_tally = defaultdict(lambda: np.zeros(6))
        for i in train_indices:
            e = all_entries[i]
            ig, gt = e['ig'], e['gt']
            mapped = np.vectorize(GRID_TO_CLASS.get)(ig)
            sb, nf, co, np_ = compute_spatial_features(ig)
            for r in range(40):
                for c in range(40):
                    key = (mapped[r,c], sb[r,c], nf[r,c], co[r,c], np_[r,c])
                    train_tally[key] += gt[r,c]
        
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
        for hold_idx in hold_indices:
            e = all_entries[hold_idx]
            ig, gt = e['ig'], e['gt']
            mapped = np.vectorize(GRID_TO_CLASS.get)(ig)
            sb, nf, co, np_ = compute_spatial_features(ig)
            
            # Cell model
            cell_dist = predict_cell_distributions(ig, cell_p)
            
            pred = np.zeros((40, 40, 6))
            for r in range(40):
                for c in range(40):
                    ic = mapped[r,c]
                    key5 = (ic, sb[r,c], nf[r,c], co[r,c], np_[r,c])
                    key4 = (ic, sb[r,c], nf[r,c], co[r,c])
                    key3 = (ic, sb[r,c], nf[r,c])
                    
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
                    
                    cell_p_dist = cell_dist[r, c]
                    
                    # Ensemble in log space
                    lut_log = np.log(np.clip(lut_p, clip_floor, None))
                    cell_log = np.log(np.clip(cell_p_dist, clip_floor, None))
                    mixed = (1 - alpha) * lut_log + alpha * cell_log
                    mixed -= mixed.max()
                    probs = np.exp(mixed)
                    probs /= probs.sum()
                    
                    pred[r, c] = probs
            
            # Apply submission floor (THIS IS THE KEY DIFFERENCE)
            pred = np.maximum(pred, submission_floor)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            
            # Score with competition metric
            score = competition_score(pred, gt)
            scores.append(score)
    
    return np.mean(scores), np.std(scores), scores

# ── Test different configurations
print("\n=== Testing submission floors (α=0.35, clip=0.0001) ===")
for sub_floor in [0.0, 1e-6, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05]:
    mean, std, _ = run_cv(alpha=0.35, clip_floor=0.0001, submission_floor=sub_floor)
    print(f"  floor={sub_floor:.4f}: {mean:.2f} ± {std:.2f}")

print("\n=== Testing alphas with floor=0.01 ===")
for alpha in [0.0, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50]:
    mean, std, _ = run_cv(alpha=alpha, clip_floor=0.0001, submission_floor=0.01)
    print(f"  α={alpha:.2f}: {mean:.2f} ± {std:.2f}")

print("\n=== Testing clip_floor with sub_floor=0.01 ===")
for clip in [0.0001, 0.001, 0.005, 0.01]:
    mean, std, _ = run_cv(alpha=0.35, clip_floor=clip, submission_floor=0.01)
    print(f"  clip={clip:.4f}: {mean:.2f} ± {std:.2f}")

# Find best overall config
print("\n=== Grid search: best config ===")
best_score = 0
best_config = None
for alpha in [0.0, 0.10, 0.20, 0.25, 0.30, 0.35]:
    for sub_floor in [0.005, 0.01, 0.015, 0.02]:
        mean, std, _ = run_cv(alpha=alpha, clip_floor=0.0001, submission_floor=sub_floor)
        if mean > best_score:
            best_score = mean
            best_config = (alpha, sub_floor)
        print(f"  α={alpha:.2f}, floor={sub_floor:.3f}: {mean:.2f}")

print(f"\n=== BEST: α={best_config[0]:.2f}, floor={best_config[1]:.3f} → {best_score:.2f} ===")
