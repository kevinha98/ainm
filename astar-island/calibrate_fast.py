"""
Fast simulator using multiprocessing for parallel simulations.
Also includes a quick-calibrate mode that uses transition statistics matching.
"""
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, fields
import time

from simulator.engine import SimParams, simulate_once, GRID_TO_CLASS

DATA_DIR = Path("data")


def kl_score(pred, gt):
    pred = np.clip(pred, 1e-8, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    kl = np.where(gt > 0, gt * np.log(np.clip(gt, 1e-15, None) / pred), 0).sum(axis=-1)
    kl = np.where(np.isfinite(kl), kl, 0)
    return 100 - kl.mean() * 100


def _run_one_sim(args):
    """Worker function for multiprocessing."""
    initial_grid, params_dict, seed = args
    params = SimParams(**params_dict)
    cls = simulate_once(initial_grid, params, seed=seed)
    return cls


def simulate_distribution_parallel(initial_grid, params, n_sims=200, base_seed=42, max_workers=6):
    """Run N simulations in parallel and return H×W×6 probability distribution."""
    H, W = initial_grid.shape
    counts = np.zeros((H, W, 6), dtype=np.float64)
    params_dict = asdict(params)

    args_list = [(initial_grid, params_dict, base_seed + i) for i in range(n_sims)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_one_sim, args) for args in args_list]
        for f in as_completed(futures):
            cls = f.result()
            for c in range(6):
                counts[:, :, c] += (cls == c)

    probs = counts / n_sims
    probs = np.clip(probs, 1e-6, None)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


def compute_transition_stats(initial_grid, params, n_sims=50, base_seed=42):
    """
    Compute per-class transition statistics from simulations.
    Returns dict: initial_class → avg_distribution[6]
    """
    H, W = initial_grid.shape
    cls_grid = np.zeros_like(initial_grid)
    for gv, c in GRID_TO_CLASS.items():
        cls_grid[initial_grid == gv] = c

    counts_per_class = {ic: np.zeros(6) for ic in range(6)}
    n_per_class = {ic: 0 for ic in range(6)}

    for i in range(n_sims):
        final_cls = simulate_once(initial_grid, params, seed=base_seed + i)
        for ic in range(6):
            mask = cls_grid == ic
            if mask.sum() == 0:
                continue
            for oc in range(6):
                counts_per_class[ic][oc] += (final_cls[mask] == oc).sum()
            n_per_class[ic] += mask.sum()

    transitions = {}
    for ic in range(6):
        if n_per_class[ic] > 0:
            transitions[ic] = counts_per_class[ic] / n_per_class[ic]
        else:
            transitions[ic] = np.zeros(6)
    return transitions


def transition_error(sim_trans, gt_trans):
    """MSE between simulated and GT transition probabilities."""
    err = 0.0
    n = 0
    for ic in range(6):
        if ic in sim_trans and ic in gt_trans:
            diff = sim_trans[ic] - gt_trans[ic]
            err += (diff ** 2).sum()
            n += 1
    return err / max(n, 1)


def compute_gt_transitions(entries):
    """Compute average transition stats from GT entries."""
    gt_trans = {ic: [] for ic in range(6)}
    for ig, gt in entries:
        cls = np.zeros_like(ig)
        for gv, c in GRID_TO_CLASS.items():
            cls[ig == gv] = c
        for ic in range(6):
            mask = cls == ic
            if mask.sum() > 0:
                gt_trans[ic].append(gt[mask].mean(axis=0))
    return {ic: np.mean(v, axis=0) if v else np.zeros(6) for ic, v in gt_trans.items()}


def load_all_gt():
    entries = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        with open(gf) as f:
            data = json.load(f)
        for si_str in sorted(data.keys()):
            entry = data[si_str]
            entries.append((np.array(entry['initial_grid']), np.array(entry['ground_truth'])))
    return entries


def quick_calibrate():
    """
    Fast calibration: match per-class-level transition stats.
    Instead of full KL on every cell, match the 5×6=30 transition probabilities.
    """
    entries = load_all_gt()
    print(f"Loaded {len(entries)} GT entries")

    # GT transitions
    gt_trans = compute_gt_transitions(entries)
    print("\nGT transition matrix:")
    cls_names = ['Empty', 'Settle', 'Port', 'Ruin', 'Forest', 'Mountain']
    for ic in range(6):
        if gt_trans[ic].sum() > 0:
            print(f"  {cls_names[ic]:>8s} → {np.round(gt_trans[ic], 3).tolist()}")

    # Use a single representative seed for calibration
    test_ig = entries[0][0]

    # Parameters to sweep
    configs = []

    # The key parameters that control transition rates:
    # expand_pop/prob → controls how many settlements appear on empty/forest cells
    # winter_mean/std → controls settlement survival rate
    # forest_clear_prob → controls forest→empty transition
    # ruin_nature_rate → controls ruin cleanup speed

    for expand_pop in [50, 80, 120]:
        for expand_prob in [0.3, 0.5, 0.8]:
            for expand_radius in [3, 5, 7]:
                for winter_mean in [0.5, 0.8, 1.2]:
                    for forest_clear in [0.05, 0.15, 0.3]:
                        configs.append(SimParams(
                            expand_pop=expand_pop,
                            expand_prob=expand_prob,
                            expand_radius=expand_radius,
                            winter_mean=winter_mean,
                            forest_clear_prob=forest_clear,
                        ))

    print(f"\nTesting {len(configs)} parameter configurations...")
    best_err = float('inf')
    best_params = None
    best_trans = None

    for i, params in enumerate(configs):
        sim_trans = compute_transition_stats(test_ig, params, n_sims=15, base_seed=42)
        err = transition_error(sim_trans, gt_trans)
        if err < best_err:
            best_err = err
            best_params = params
            best_trans = sim_trans
            print(f"  [{i}/{len(configs)}] MSE={err:.6f} ← NEW BEST")
            for ic in [0, 1, 4]:
                print(f"    {cls_names[ic]:>8s} sim: {np.round(sim_trans[ic], 3).tolist()}")
                print(f"    {cls_names[ic]:>8s} GT:  {np.round(gt_trans[ic], 3).tolist()}")

    print(f"\n{'='*60}")
    print(f"Best MSE: {best_err:.6f}")
    print(f"Best params:")
    for f in fields(SimParams):
        val = getattr(best_params, f.name)
        default = f.default
        if val != default:
            print(f"  {f.name} = {val} (default: {default})")

    print("\nBest transition match:")
    for ic in range(6):
        if best_trans[ic].sum() > 0:
            print(f"  {cls_names[ic]:>8s} sim: {np.round(best_trans[ic], 3).tolist()}")
            print(f"  {cls_names[ic]:>8s}  GT: {np.round(gt_trans[ic], 3).tolist()}")

    # Now evaluate KL score with best params
    print("\nEvaluating KL score with best params (N=50, 5 seeds)...")
    scores = []
    for i, (ig, gt) in enumerate(entries[:5]):
        t0 = time.time()
        from simulator.engine import simulate_distribution
        pred = simulate_distribution(ig, best_params, n_sims=50, base_seed=i*1000)
        s = kl_score(pred, gt)
        scores.append(s)
        print(f"  Seed {i}: {s:.2f} ({time.time()-t0:.1f}s)")
    print(f"  Mean: {np.mean(scores):.2f}")

    return best_params


if __name__ == "__main__":
    quick_calibrate()
