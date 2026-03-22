"""Debug R15: compare our actual submission vs alternatives."""
import json, numpy as np, sys
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import uniform_filter

sys.path.insert(0, "src")
DATA_DIR = Path("data")
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}

# Load R15 GT
gt_path = DATA_DIR / "ground_truth_cc5442dd.json"
with open(gt_path) as f:
    r15_data = json.load(f)

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

# Load ALL GT for building a robust predictor
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

# ── Compute class averages (excluding R15)
non_r15 = [e for e in all_entries if e['rid'] != 'cc5442dd']
class_counts = np.zeros((6, 6))
for e in non_r15:
    mapped = np.vectorize(GRID_TO_CLASS.get)(e['ig'])
    for r in range(40):
        for c in range(40):
            class_counts[mapped[r,c]] += e['gt'][r,c]

class_avgs = {}
for ci in range(6):
    s = class_counts[ci].sum()
    if s > 0:
        class_avgs[ci] = class_counts[ci] / s

# Score R15 with different strategies
print("=== R15 Scoring Analysis ===\n")

for seed_key in sorted(r15_data.keys()):
    entry = r15_data[seed_key]
    ig = np.array(entry['initial_grid'])
    gt = np.array(entry['ground_truth'])
    mapped = np.vectorize(GRID_TO_CLASS.get)(ig)
    
    # Strategy 1: Class averages
    pred_ca = np.zeros((40,40,6))
    for r in range(40):
        for c in range(40):
            pred_ca[r,c] = class_avgs[mapped[r,c]]
    s1 = comp_score(pred_ca, gt)
    
    # Strategy 2: Uniform
    pred_uni = np.ones((40,40,6)) / 6
    s2 = comp_score(pred_uni, gt)
    
    # Strategy 3: Class avg with floor 0.01
    pred_ca_f = np.maximum(pred_ca, 0.01)
    pred_ca_f /= pred_ca_f.sum(axis=-1, keepdims=True)
    s3 = comp_score(pred_ca_f, gt)
    
    # Strategy 4: Class avg with floor 0.001
    pred_ca_f2 = np.maximum(pred_ca, 0.001)
    pred_ca_f2 /= pred_ca_f2.sum(axis=-1, keepdims=True)
    s4 = comp_score(pred_ca_f2, gt)
    
    print(f"Seed {seed_key}:")
    print(f"  Class avg:           {s1:.1f}")
    print(f"  Class avg + 0.01:    {s3:.1f}")
    print(f"  Class avg + 0.001:   {s4:.1f}")
    print(f"  Uniform:             {s2:.1f}")

# Now examine WHERE the KL is high
print("\n=== Cell-level KL analysis (seed 0) ===")
entry = r15_data['0']
ig = np.array(entry['initial_grid'])
gt = np.array(entry['ground_truth'])
mapped = np.vectorize(GRID_TO_CLASS.get)(ig)

# Use class avg prediction
pred = np.zeros((40,40,6))
for r in range(40):
    for c in range(40):
        pred[r,c] = class_avgs[mapped[r,c]]

eps = 1e-15
gt_s = np.clip(gt, eps, None)
pred_s = np.clip(pred, eps, None)
entropy = -np.sum(gt * np.log(gt_s), axis=-1)
kl = np.sum(gt * np.log(gt_s / pred_s), axis=-1)

# Top 20 worst cells (highest entropy * kl)
damage = entropy * kl
flat_idx = np.argsort(damage.ravel())[::-1][:20]
print(f"Top 20 most damaging cells:")
print(f"  {'Row':>3} {'Col':>3} {'Class':>5} {'Entropy':>8} {'KL':>8} {'Damage':>8} {'GT dist':<40} {'Pred dist':<40}")
for fi in flat_idx:
    r, c = divmod(fi, 40)
    ic = mapped[r,c]
    gt_dist = gt[r,c]
    pred_dist = pred[r,c]
    print(f"  {r:3d} {c:3d} {ic:5d} {entropy[r,c]:8.3f} {kl[r,c]:8.3f} {damage[r,c]:8.3f} "
          f"[{', '.join(f'{x:.3f}' for x in gt_dist)}] [{', '.join(f'{x:.3f}' for x in pred_dist)}]")

# Summary: which CLASS causes most damage?
print("\n=== Damage by initial class ===")
for ci in range(6):
    mask = (mapped == ci)
    if mask.any():
        total_dmg = damage[mask].sum()
        n_cells = mask.sum()
        n_dynamic = (entropy[mask] > 0.01).sum()
        print(f"  Class {ci}: {n_cells} cells, {n_dynamic} dynamic, total_damage={total_dmg:.1f}, avg_kl={kl[mask].mean():.4f}")
