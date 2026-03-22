"""Cross-round validation: train on N-1 rounds, test on held-out round.
This tests whether per-class averages transfer across different maps/rounds.
"""
import json
import numpy as np
from pathlib import Path
from src.settings import DATA_DIR, NUM_CLASSES, CLASS_NAMES
from src.models import build_class_grid

# Load all ground truth files
gt_files = {
    "R1": DATA_DIR / "ground_truth_71451d74.json",
    "R2": DATA_DIR / "ground_truth_76909e29.json",
    "R3": DATA_DIR / "ground_truth_f1dac9a9.json",
    "R4": DATA_DIR / "ground_truth_8e839974.json",
}

all_gt = {}
for name, path in gt_files.items():
    if path.exists():
        with open(path) as f:
            all_gt[name] = json.load(f)
        print(f"Loaded {name}: {path.name}")

def score_pred(pred, gt):
    """Score: 100 * exp(-KL(gt || pred))"""
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    kl = np.sum(gt_safe * np.log(gt_safe / (pred + 1e-15)), axis=-1).mean()
    return 100 * np.exp(-kl)

def build_per_class_avg(gt_data_list):
    """Build per-class averages from list of GT data dicts."""
    by_class = {c: [] for c in range(NUM_CLASSES)}
    for gt_data in gt_data_list:
        for si in range(5):
            si_str = str(si)
            if si_str not in gt_data:
                continue
            gt = np.array(gt_data[si_str].get('ground_truth', []))
            ig = np.array(gt_data[si_str].get('initial_grid', []))
            if gt.size == 0 or ig.size == 0:
                continue
            ic = build_class_grid(ig)
            H, W = ig.shape[:2]
            for y in range(H):
                for x in range(W):
                    by_class[ic[y, x]].append(gt[y, x])
    
    avg = {}
    for c in range(NUM_CLASSES):
        if by_class[c]:
            avg[c] = np.mean(by_class[c], axis=0)
        else:
            avg[c] = np.ones(6) / 6
    return avg

print("\n" + "=" * 70)
print("  CROSS-ROUND VALIDATION (Leave-One-Round-Out)")
print("=" * 70)

round_names = sorted(all_gt.keys())
results = []

for test_round in round_names:
    train_rounds = [r for r in round_names if r != test_round]
    
    # Build averages from training rounds
    train_data = [all_gt[r] for r in train_rounds]
    class_avg = build_per_class_avg(train_data)
    
    # Also test with ALL rounds (including test — in-sample)
    all_data = [all_gt[r] for r in round_names]
    class_avg_all = build_per_class_avg(all_data)
    
    # Also test with just the test round (in-sample per-seed)
    class_avg_self = build_per_class_avg([all_gt[test_round]])
    
    # Global average from training rounds only
    all_train_gt = []
    for r in train_rounds:
        for si in range(5):
            si_str = str(si)
            if si_str in all_gt[r]:
                gt = np.array(all_gt[r][si_str].get('ground_truth', []))
                if gt.size > 0:
                    all_train_gt.append(gt.reshape(-1, 6))
    global_avg = np.vstack(all_train_gt).mean(axis=0) if all_train_gt else np.ones(6) / 6
    
    # Score on test round
    scores_loo = []
    scores_all = []
    scores_self = []
    scores_global = []
    scores_uniform = []
    
    for si in range(5):
        si_str = str(si)
        if si_str not in all_gt[test_round]:
            continue
        gt = np.array(all_gt[test_round][si_str]['ground_truth'])
        ig = np.array(all_gt[test_round][si_str]['initial_grid'])
        ic = build_class_grid(ig)
        H, W = ig.shape[:2]
        
        pred_loo = np.zeros((H, W, 6))
        pred_all = np.zeros((H, W, 6))
        pred_self = np.zeros((H, W, 6))
        pred_global = np.tile(global_avg, (H, W, 1))
        pred_uniform = np.full((H, W, 6), 1/6)
        
        for y in range(H):
            for x in range(W):
                c = ic[y, x]
                pred_loo[y, x] = class_avg.get(c, class_avg[0])
                pred_all[y, x] = class_avg_all.get(c, class_avg_all[0])
                pred_self[y, x] = class_avg_self.get(c, class_avg_self[0])
        
        scores_loo.append(score_pred(pred_loo, gt))
        scores_all.append(score_pred(pred_all, gt))
        scores_self.append(score_pred(pred_self, gt))
        scores_global.append(score_pred(pred_global, gt))
        scores_uniform.append(score_pred(pred_uniform, gt))
    
    avg_loo = np.mean(scores_loo)
    avg_all = np.mean(scores_all)
    avg_self = np.mean(scores_self)
    avg_global = np.mean(scores_global)
    avg_uniform = np.mean(scores_uniform)
    
    print(f"\nTest={test_round} (train={','.join(train_rounds)}):")
    print(f"  LOO per-class:  {avg_loo:.2f}  (seeds: {[f'{s:.1f}' for s in scores_loo]})")
    print(f"  All per-class:  {avg_all:.2f}  (in-sample)")
    print(f"  Self per-class: {avg_self:.2f}  (same-round only)")
    print(f"  Global avg:     {avg_global:.2f}")
    print(f"  Uniform:        {avg_uniform:.2f}")
    
    results.append({
        'round': test_round,
        'loo': avg_loo,
        'all': avg_all,
        'self': avg_self,
        'global': avg_global,
        'uniform': avg_uniform,
    })

print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"{'Strategy':<25s} {'Avg':<8s} " + " ".join(f"{r['round']:<8s}" for r in results))
for strategy in ['loo', 'all', 'self', 'global', 'uniform']:
    label = {'loo': 'LOO per-class', 'all': 'All per-class', 'self': 'Self per-class', 'global': 'Global avg', 'uniform': 'Uniform'}[strategy]
    vals = [r[strategy] for r in results]
    avg = np.mean(vals)
    print(f"{label:<25s} {avg:<8.2f} " + " ".join(f"{v:<8.2f}" for v in vals))
