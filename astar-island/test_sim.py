"""
Quick test: run simulator on one GT round and compare distributions.
"""
import json
import numpy as np
from pathlib import Path
import time

from simulator.engine import SimParams, simulate_distribution, simulate_once, GRID_TO_CLASS

DATA_DIR = Path("data")


def kl_score(pred, gt):
    """KL-divergence based score (higher is better, 100 = perfect)."""
    pred = np.clip(pred, 1e-8, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    # Only count KL where GT has nonzero mass
    kl = np.where(gt > 0, gt * np.log(gt_safe / pred), 0).sum(axis=-1)
    kl = np.where(np.isfinite(kl), kl, 0)
    return 100 - kl.mean() * 100


def load_gt(gt_path):
    """Load all seeds from a GT file. Returns list of (initial_grid, ground_truth)."""
    with open(gt_path) as f:
        data = json.load(f)
    entries = []
    for si_str in sorted(data.keys()):
        entry = data[si_str]
        ig = np.array(entry['initial_grid'])
        gt = np.array(entry['ground_truth'])
        entries.append((ig, gt))
    return entries


def main():
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    print(f"Found {len(gt_files)} GT files")
    
    # Test on first GT file
    gt_path = gt_files[0]
    print(f"\nTesting on {gt_path.name}")
    entries = load_gt(gt_path)
    print(f"  Seeds: {len(entries)}")
    
    ig, gt = entries[0]
    print(f"  Grid shape: {ig.shape}")
    print(f"  Initial terrain: {dict(zip(*np.unique(ig, return_counts=True)))}")
    
    # Test single simulation
    print("\n--- Single simulation ---")
    t0 = time.time()
    params = SimParams()
    final = simulate_once(ig, params, seed=42)
    t1 = time.time()
    print(f"  Time: {t1-t0:.3f}s")
    unique, counts = np.unique(final, return_counts=True)
    for u, c in zip(unique, counts):
        pct = 100 * c / final.size
        cls_names = ['Empty', 'Settle', 'Port', 'Ruin', 'Forest', 'Mountain']
        print(f"  Class {u} ({cls_names[u]}): {c} cells ({pct:.1f}%)")
    
    # Test distribution (small N first)
    print("\n--- Distribution (N=50) ---")
    t0 = time.time()
    pred = simulate_distribution(ig, params, n_sims=50, base_seed=0)
    t1 = time.time()
    print(f"  Time: {t1-t0:.3f}s ({(t1-t0)/50:.3f}s per sim)")
    
    score = kl_score(pred, gt)
    print(f"  KL Score: {score:.2f}")
    
    # Per-class breakdown
    cls_grid = np.zeros_like(ig)
    for gv, cls in GRID_TO_CLASS.items():
        cls_grid[ig == gv] = cls
    
    cls_names = ['Empty', 'Settle', 'Port', 'Ruin', 'Forest', 'Mountain']
    for ic in range(6):
        mask = cls_grid == ic
        if mask.sum() == 0:
            continue
        pred_avg = pred[mask].mean(axis=0)
        gt_avg = gt[mask].mean(axis=0)
        print(f"  {cls_names[ic]:>8s} pred: {np.round(pred_avg, 3).tolist()}")
        print(f"  {cls_names[ic]:>8s} GT:   {np.round(gt_avg, 3).tolist()}")
    
    # Test across all seeds of this round
    print("\n--- All seeds (N=50 each) ---")
    scores = []
    for i, (ig_i, gt_i) in enumerate(entries):
        pred_i = simulate_distribution(ig_i, params, n_sims=50, base_seed=i*1000)
        s = kl_score(pred_i, gt_i)
        scores.append(s)
        print(f"  Seed {i}: {s:.2f}")
    print(f"  Mean: {np.mean(scores):.2f}")
    
    # Test across first 3 GT files
    print("\n--- Multiple rounds (N=30 each, first seed only) ---")
    for gf in gt_files[:3]:
        entries_r = load_gt(gf)
        ig_r, gt_r = entries_r[0]
        pred_r = simulate_distribution(ig_r, params, n_sims=30, base_seed=0)
        s = kl_score(pred_r, gt_r)
        print(f"  {gf.name}: {s:.2f}")


if __name__ == "__main__":
    main()
