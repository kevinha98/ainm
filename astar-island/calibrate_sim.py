"""
Calibrate simulator parameters against GT data using Bayesian optimization.
Uses optuna to find parameters that minimize KL divergence against real GT.
"""
import json
import numpy as np
from pathlib import Path
import time
from dataclasses import asdict

from simulator.engine import SimParams, simulate_distribution, GRID_TO_CLASS

DATA_DIR = Path("data")


def kl_score(pred, gt):
    """KL-divergence score (higher = better, 100 = perfect)."""
    pred = np.clip(pred, 1e-8, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    kl = np.where(gt > 0, gt * np.log(np.clip(gt, 1e-15, None) / pred), 0).sum(axis=-1)
    kl = np.where(np.isfinite(kl), kl, 0)
    return 100 - kl.mean() * 100


def load_all_gt():
    """Load all GT files. Returns list of (initial_grid, ground_truth) per seed."""
    entries = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        with open(gf) as f:
            data = json.load(f)
        for si_str in sorted(data.keys()):
            entry = data[si_str]
            ig = np.array(entry['initial_grid'])
            gt = np.array(entry['ground_truth'])
            entries.append((ig, gt))
    return entries


def eval_params(params, entries, n_sims=30, max_seeds=10):
    """Evaluate param set against GT entries. Returns avg KL score."""
    scores = []
    for i, (ig, gt) in enumerate(entries[:max_seeds]):
        pred = simulate_distribution(ig, params, n_sims=n_sims, base_seed=i * 1000)
        s = kl_score(pred, gt)
        scores.append(s)
    return np.mean(scores)


def eval_transitions(params, entries, n_sims=50, max_seeds=5):
    """Compare per-class transition distributions between sim and GT."""
    cls_names = ['Empty', 'Settle', 'Port', 'Ruin', 'Forest', 'Mountain']
    sim_trans = {ic: [] for ic in range(6)}
    gt_trans = {ic: [] for ic in range(6)}

    for i, (ig, gt) in enumerate(entries[:max_seeds]):
        pred = simulate_distribution(ig, params, n_sims=n_sims, base_seed=i * 1000)
        cls = np.zeros_like(ig)
        for gv, c in GRID_TO_CLASS.items():
            cls[ig == gv] = c
        for ic in range(6):
            mask = cls == ic
            if mask.sum() > 0:
                sim_trans[ic].append(pred[mask].mean(axis=0))
                gt_trans[ic].append(gt[mask].mean(axis=0))

    print("\nTransition comparison (sim vs GT):")
    for ic in range(6):
        if sim_trans[ic]:
            s = np.mean(sim_trans[ic], axis=0)
            g = np.mean(gt_trans[ic], axis=0)
            diff = s - g
            print(f"  {cls_names[ic]:>8s} sim:  {np.round(s, 3).tolist()}")
            print(f"  {cls_names[ic]:>8s} GT:   {np.round(g, 3).tolist()}")
            print(f"  {cls_names[ic]:>8s} diff: {np.round(diff, 3).tolist()}")


def grid_search():
    """Manual grid search over key parameters."""
    entries = load_all_gt()
    print(f"Loaded {len(entries)} GT entries ({len(entries)//5} rounds × 5 seeds)")
    
    # Use subset for speed
    test_entries = entries[::7][:10]  # ~10 diverse seeds
    print(f"Testing on {len(test_entries)} seeds")

    # Baseline
    base = SimParams()
    t0 = time.time()
    base_score = eval_params(base, test_entries, n_sims=20)
    t1 = time.time()
    print(f"\nBaseline: {base_score:.2f} ({t1-t0:.1f}s)")
    eval_transitions(base, test_entries, n_sims=20, max_seeds=3)

    # Key insight from GT: 11% of empty cells become settlements
    # This requires very aggressive expansion
    best_score = base_score
    best_params = base

    param_grid = {
        'expand_pop': [60, 80, 100],
        'expand_prob': [0.3, 0.5, 0.7],
        'expand_radius': [3, 5, 7],
        'food_per_plains': [2.0, 3.0, 5.0],
        'raid_prob': [0.1, 0.25, 0.4],
        'winter_mean': [0.5, 1.0, 1.5],
        'forest_clear_prob': [0.05, 0.15, 0.3],
    }

    # Test each parameter one at a time
    print("\n--- Single parameter sweeps ---")
    for param_name, values in param_grid.items():
        print(f"\n{param_name}:")
        for val in values:
            p = SimParams(**{param_name: val})
            t0 = time.time()
            score = eval_params(p, test_entries, n_sims=20)
            t1 = time.time()
            marker = " ***" if score > best_score else ""
            print(f"  {val:>6} → {score:.2f} ({t1-t0:.1f}s){marker}")
            if score > best_score:
                best_score = score
                best_params = p

    print(f"\n{'='*50}")
    print(f"Best score: {best_score:.2f}")
    print(f"Best params: {asdict(best_params)}")

    # Show transitions for best params
    eval_transitions(best_params, test_entries, n_sims=30, max_seeds=3)


if __name__ == "__main__":
    grid_search()
