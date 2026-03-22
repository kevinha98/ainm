"""Quick analysis: per-round difficulty & baseline scores."""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('data')
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

print(f"Total seeds: {len(all_entries)}")

def comp_score(pred, gt):
    eps = 1e-15
    gt_s = np.clip(gt, eps, None)
    pred_s = np.clip(pred, eps, None)
    entropy = -np.sum(gt * np.log(gt_s), axis=-1)
    kl = np.sum(gt * np.log(gt_s / pred_s), axis=-1)
    te = entropy.sum()
    if te < eps: return 100.0, 0.0, 0
    wkl = np.sum(entropy * kl) / te
    dynamic = (entropy > 0.01).sum()
    return max(0, min(100, 100 * np.exp(-3 * wkl))), entropy[entropy > 0.01].mean(), dynamic

# Class averages (non-LOO, just baseline)
class_counts = np.zeros((6, 6))
for e in all_entries:
    mapped = np.vectorize(GRID_TO_CLASS.get)(e['ig'])
    for r in range(40):
        for c in range(40):
            class_counts[mapped[r,c]] += e['gt'][r,c]
class_avgs = {}
for ci in range(6):
    s = class_counts[ci].sum()
    if s > 0:
        class_avgs[ci] = class_counts[ci] / s

print("\nClass averages:")
for ci in range(6):
    if ci in class_avgs:
        dist = class_avgs[ci]
        print(f"  Class {ci}: [{', '.join(f'{d:.3f}' for d in dist)}]")

# Per-round scores
print("\n=== Per-round baseline (class avg) ===")
scores_by_round = defaultdict(list)
for e in all_entries:
    mapped = np.vectorize(GRID_TO_CLASS.get)(e['ig'])
    pred = np.zeros((40,40,6))
    for r in range(40):
        for c in range(40):
            pred[r,c] = class_avgs[mapped[r,c]]
    score, avg_ent, n_dyn = comp_score(pred, e['gt'])
    scores_by_round[e['rid']].append((score, avg_ent, n_dyn))

for rid in sorted(scores_by_round.keys()):
    entries = scores_by_round[rid]
    avg_s = np.mean([x[0] for x in entries])
    avg_e = np.mean([x[1] for x in entries])
    avg_d = np.mean([x[2] for x in entries])
    print(f"  {rid}: score={avg_s:.1f}, avg_entropy={avg_e:.3f}, dynamic_cells={avg_d:.0f}")

all_s = [x[0] for ss in scores_by_round.values() for x in ss]
print(f"  OVERALL: {np.mean(all_s):.1f} ± {np.std(all_s):.1f}")

# Uniform prediction score
print("\n=== Uniform prediction (1/6 each) ===")
uniform_scores = []
for e in all_entries:
    pred = np.ones((40,40,6)) / 6
    score, _, _ = comp_score(pred, e['gt'])
    uniform_scores.append(score)
print(f"  Uniform: {np.mean(uniform_scores):.1f} ± {np.std(uniform_scores):.1f}")

# Perfect initial class prediction (predict everything stays as initial)
print("\n=== Perfect-class prediction (100% stays as initial) ===")
pclass_scores = []
for e in all_entries:
    mapped = np.vectorize(GRID_TO_CLASS.get)(e['ig'])
    pred = np.full((40,40,6), 0.001)
    for r in range(40):
        for c in range(40):
            pred[r,c,mapped[r,c]] = 1.0
    pred /= pred.sum(axis=-1, keepdims=True)
    score, _, _ = comp_score(pred, e['gt'])
    pclass_scores.append(score)
print(f"  Stay-as-initial: {np.mean(pclass_scores):.1f} ± {np.std(pclass_scores):.1f}")
