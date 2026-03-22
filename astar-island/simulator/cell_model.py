"""
Cell-level stochastic simulator — directly calibrated to GT data.

Instead of an agent-based model with settlements/food/conflict,
this models each cell's transition independently based on:
  - Initial class
  - Distance to nearest settlement
  - Distance to nearest forest
  - Coastal flag
  - Local settlement density

The transition probabilities are parameterized functions
calibrated against the 14 GT rounds.

This is MUCH faster than the agent-based model and directly
targets the distributions we need to predict.
"""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.optimize import minimize
import time
from dataclasses import dataclass

DATA_DIR = Path("data")
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


def build_class_grid(grid_np):
    cg = np.zeros_like(grid_np)
    for gv, cls in GRID_TO_CLASS.items():
        cg[grid_np == gv] = cls
    return cg


def compute_features(ig):
    """Compute per-cell features from initial grid."""
    H, W = ig.shape
    cls = build_class_grid(ig)

    # Distance to nearest settlement
    settle = (cls == 1) | (cls == 2)
    dist_s = ndimage.distance_transform_edt(~settle) if settle.any() else np.full((H, W), 40.0)

    # Distance to nearest forest
    forest = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 40.0)

    # Coastal flag
    ocean = (ig == 10)
    dist_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    coastal = (dist_o <= 1.5).astype(float)

    # Local settlement density (radius 5)
    settle_density = ndimage.uniform_filter(settle.astype(float), size=7, mode='constant')

    # Is it ocean?
    is_ocean = ocean.astype(float)

    return cls, dist_s, dist_f, coastal, settle_density, is_ocean


@dataclass
class CellParams:
    """Parameters for the cell-level transition model."""
    # Settlement spread: prob of empty→settlement as function of settlement distance
    # P(settle | empty, dist_s) = settle_base * exp(-dist_s / settle_scale)
    settle_base_empty: float = 0.35
    settle_scale_empty: float = 3.0
    settle_base_forest: float = 0.30
    settle_scale_forest: float = 2.5

    # Port formation: P(port | near coastal settlement)
    port_base: float = 0.05
    port_scale: float = 2.0

    # Ruin formation: P(ruin | was settlement)
    ruin_from_settle: float = 0.025
    ruin_persistence: float = 0.01  # P(ruin | was empty/forest)

    # Settlement survival: P(stay settlement | was settlement)
    settle_survival_base: float = 0.35
    settle_survival_density_bonus: float = 0.5  # bonus for high density

    # Forest dynamics
    forest_persist_base: float = 0.80   # P(forest | was forest, far from settle)
    forest_settle_penalty: float = 0.25  # penalty per unit of settle density
    forest_from_empty: float = 0.03     # P(forest | was empty)
    forest_from_settle: float = 0.20    # P(forest | was settlement that collapsed)

    # Port survival
    port_survival: float = 0.20

    # Ocean: always stays empty
    ocean_persist: float = 1.0


def predict_cell_distributions(ig, params: CellParams):
    """
    Given initial grid, predict H×W×6 probability distributions.
    Fully vectorized numpy implementation — no per-cell loops.
    """
    H, W = ig.shape
    cls, dist_s, dist_f, coastal, settle_density, is_ocean = compute_features(ig)

    pred = np.zeros((H, W, 6))

    # Special cells first
    m5 = (cls == 5)  # Mountain
    pred[m5, 5] = 1.0

    deep_ocean = (is_ocean > 0.5) & (dist_s > 5) & ~m5
    pred[deep_ocean, 0] = 1.0

    # Masks for each initial class (excluding special cells)
    special = m5 | deep_ocean
    m0 = (cls == 0) & ~special  # Empty/Plains/Ocean (non-deep-ocean)
    m1 = (cls == 1) & ~special  # Settlement
    m2 = (cls == 2) & ~special  # Port
    m3 = (cls == 3) & ~special  # Ruin
    m4 = (cls == 4) & ~special  # Forest

    # --- Class 0: Empty/Plains/Ocean ---
    if m0.any():
        ds = dist_s[m0]
        df = dist_f[m0]
        co = coastal[m0]
        p_settle = params.settle_base_empty * np.exp(-ds / params.settle_scale_empty)
        p_port = params.port_base * co * np.exp(-ds / params.port_scale)
        p_ruin = np.full_like(ds, params.ruin_persistence)
        p_forest = params.forest_from_empty * (1 + 1.0 / (1 + df))
        p_empty = 1.0 - p_settle - p_port - p_ruin - p_forest
        p = np.stack([np.maximum(0, p_empty), p_settle, p_port, p_ruin, np.maximum(0, p_forest), np.zeros_like(ds)], axis=-1)
        pred[m0] = p

    # --- Class 1: Settlement ---
    if m1.any():
        sd = settle_density[m1]
        co = coastal[m1]
        p_survive = np.minimum(params.settle_survival_base + params.settle_survival_density_bonus * sd, 0.9)
        p_port = params.port_base * co * 2
        p_ruin = np.full_like(sd, params.ruin_from_settle)
        p_forest = params.forest_from_settle * (1 - p_survive * 0.5)
        p_empty = 1.0 - p_survive - p_port - p_ruin - p_forest
        p = np.stack([np.maximum(0, p_empty), p_survive, p_port, p_ruin, np.maximum(0, p_forest), np.zeros_like(sd)], axis=-1)
        pred[m1] = p

    # --- Class 2: Port ---
    if m2.any():
        n2 = m2.sum()
        p_survive_port = np.full(n2, params.port_survival)
        p_survive_settle = np.full(n2, 0.10)
        p_ruin = np.full(n2, params.ruin_from_settle)
        p_forest = np.full(n2, params.forest_from_settle * 0.8)
        p_empty = 1.0 - p_survive_port - p_survive_settle - p_ruin - p_forest
        p = np.stack([np.maximum(0, p_empty), p_survive_settle, p_survive_port, p_ruin, np.maximum(0, p_forest), np.zeros(n2)], axis=-1)
        pred[m2] = p

    # --- Class 3: Ruin ---
    if m3.any():
        ds = dist_s[m3]
        p_settle = params.settle_base_empty * np.exp(-ds / params.settle_scale_empty) * 1.2
        p_forest = np.full_like(ds, 0.3)
        p_ruin = np.full_like(ds, 0.05)
        p_empty = 1.0 - p_settle - p_forest - p_ruin
        p = np.stack([np.maximum(0, p_empty), p_settle, np.zeros_like(ds), p_ruin, np.maximum(0, p_forest), np.zeros_like(ds)], axis=-1)
        pred[m3] = p

    # --- Class 4: Forest ---
    if m4.any():
        ds = dist_s[m4]
        co = coastal[m4]
        sd = settle_density[m4]
        forest_survive = np.clip(params.forest_persist_base - params.forest_settle_penalty * sd, 0.3, 0.99)
        p_settle = params.settle_base_forest * np.exp(-ds / params.settle_scale_forest)
        p_port = params.port_base * co * np.exp(-ds / params.port_scale) * 0.5
        p_ruin = np.full_like(ds, params.ruin_persistence)
        p_empty = 1.0 - forest_survive - p_settle - p_port - p_ruin
        p = np.stack([np.maximum(0, p_empty), p_settle, p_port, p_ruin, forest_survive, np.zeros_like(ds)], axis=-1)
        pred[m4] = p

    # Normalize all non-special cells
    active = m0 | m1 | m2 | m3 | m4
    if active.any():
        pred[active] = np.clip(pred[active], 1e-6, None)
        pred[active] = pred[active] / pred[active].sum(axis=-1, keepdims=True)

    return pred


def kl_score(pred, gt):
    pred = np.clip(pred, 1e-8, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    kl = np.where(gt > 0, gt * np.log(np.clip(gt, 1e-15, None) / pred), 0).sum(axis=-1)
    kl = np.where(np.isfinite(kl), kl, 0)
    return 100 - kl.mean() * 100


def load_all_gt():
    entries = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        with open(gf) as f:
            data = json.load(f)
        for si_str in sorted(data.keys()):
            entry = data[si_str]
            entries.append((np.array(entry['initial_grid']), np.array(entry['ground_truth'])))
    return entries


def params_from_vector(x):
    """Convert optimization vector to CellParams."""
    return CellParams(
        settle_base_empty=x[0],
        settle_scale_empty=x[1],
        settle_base_forest=x[2],
        settle_scale_forest=x[3],
        port_base=x[4],
        port_scale=x[5],
        ruin_from_settle=x[6],
        ruin_persistence=x[7],
        settle_survival_base=x[8],
        settle_survival_density_bonus=x[9],
        forest_persist_base=x[10],
        forest_settle_penalty=x[11],
        forest_from_empty=x[12],
        forest_from_settle=x[13],
        port_survival=x[14],
    )


def vector_from_params(p: CellParams):
    return np.array([
        p.settle_base_empty, p.settle_scale_empty,
        p.settle_base_forest, p.settle_scale_forest,
        p.port_base, p.port_scale,
        p.ruin_from_settle, p.ruin_persistence,
        p.settle_survival_base, p.settle_survival_density_bonus,
        p.forest_persist_base, p.forest_settle_penalty,
        p.forest_from_empty, p.forest_from_settle,
        p.port_survival,
    ])


def objective(x, entries):
    """Negative KL score to minimize."""
    params = params_from_vector(x)
    scores = []
    for ig, gt in entries:
        pred = predict_cell_distributions(ig, params)
        s = kl_score(pred, gt)
        scores.append(s)
    return -np.mean(scores)


def calibrate():
    entries = load_all_gt()
    print(f"Loaded {len(entries)} GT entries")

    # Use subset for speed during calibration
    cal_entries = entries[::5][:14]  # 1 per round
    print(f"Calibrating on {len(cal_entries)} seeds")

    # Test default params
    params = CellParams()
    t0 = time.time()
    scores = []
    for ig, gt in cal_entries[:3]:
        pred = predict_cell_distributions(ig, params)
        s = kl_score(pred, gt)
        scores.append(s)
    t1 = time.time()
    print(f"Default params: KL={np.mean(scores):.2f} ({t1-t0:.2f}s)")

    # Show per-class comparison
    ig, gt = cal_entries[0]
    pred = predict_cell_distributions(ig, params)
    cls = build_class_grid(ig)
    cls_names = ['Empty', 'Settle', 'Port', 'Ruin', 'Forest', 'Mountain']
    print("\nDefault per-class comparison:")
    for ic in range(6):
        mask = cls == ic
        if mask.sum() > 0:
            print(f"  {cls_names[ic]:>8s} pred: {np.round(pred[mask].mean(0), 3).tolist()}")
            print(f"  {cls_names[ic]:>8s}   GT: {np.round(gt[mask].mean(0), 3).tolist()}")

    # Optimize
    print("\nOptimizing parameters...")
    x0 = vector_from_params(params)

    # Bounds
    bounds = [
        (0.01, 0.8),  # settle_base_empty
        (1.0, 10.0),  # settle_scale_empty
        (0.01, 0.8),  # settle_base_forest
        (1.0, 10.0),  # settle_scale_forest
        (0.001, 0.2), # port_base
        (1.0, 10.0),  # port_scale
        (0.001, 0.1), # ruin_from_settle
        (0.001, 0.05),# ruin_persistence
        (0.05, 0.8),  # settle_survival_base
        (0.0, 2.0),   # settle_survival_density_bonus
        (0.5, 0.99),  # forest_persist_base
        (0.0, 1.0),   # forest_settle_penalty
        (0.001, 0.1), # forest_from_empty
        (0.05, 0.5),  # forest_from_settle
        (0.05, 0.5),  # port_survival
    ]

    best_score = -float('inf')
    best_x = x0

    # Multiple random restarts
    for restart in range(5):
        if restart == 0:
            x_init = x0
        else:
            x_init = np.array([b[0] + np.random.random() * (b[1] - b[0]) for b in bounds])

        result = minimize(
            objective, x_init, args=(cal_entries,),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 0.001, 'fatol': 0.01},
        )
        score = -result.fun
        print(f"  Restart {restart}: score={score:.2f} (iterations={result.nit})")
        if score > best_score:
            best_score = score
            best_x = result.x

    best_params = params_from_vector(best_x)
    print(f"\nBest score: {best_score:.2f}")

    # Evaluate on full dataset
    print("\nFull evaluation...")
    all_scores = []
    for i, (ig, gt) in enumerate(entries[:20]):
        pred = predict_cell_distributions(ig, best_params)
        s = kl_score(pred, gt)
        all_scores.append(s)
    print(f"  Mean KL (20 seeds): {np.mean(all_scores):.2f}")

    # Per-class comparison
    ig, gt = entries[0]
    pred = predict_cell_distributions(ig, best_params)
    cls = build_class_grid(ig)
    print("\nOptimized per-class comparison:")
    for ic in range(6):
        mask = cls == ic
        if mask.sum() > 0:
            print(f"  {cls_names[ic]:>8s} pred: {np.round(pred[mask].mean(0), 3).tolist()}")
            print(f"  {cls_names[ic]:>8s}   GT: {np.round(gt[mask].mean(0), 3).tolist()}")

    # Print best params
    print(f"\nBest CellParams:")
    for name, val in zip(
        ['settle_base_empty', 'settle_scale_empty', 'settle_base_forest', 'settle_scale_forest',
         'port_base', 'port_scale', 'ruin_from_settle', 'ruin_persistence',
         'settle_survival_base', 'settle_survival_density_bonus',
         'forest_persist_base', 'forest_settle_penalty', 'forest_from_empty', 'forest_from_settle',
         'port_survival'],
        best_x
    ):
        print(f"  {name} = {val:.4f}")

    return best_params


if __name__ == "__main__":
    calibrate()
