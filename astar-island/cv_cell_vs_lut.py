"""
LOO CV comparison: Cell model vs Bucket LUT.

Tests whether the cell-level parametric model can beat the bucket LUT
(our current best at CV 94.02).

The cell model produces CONTINUOUS predictions based on distance features,
while the LUT uses discrete buckets. The cell model should capture
spatial gradients better.
"""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.optimize import minimize
import time
from dataclasses import asdict

DATA_DIR = Path("data")
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


def build_class_grid(ig):
    cg = np.zeros_like(ig)
    for gv, cls in GRID_TO_CLASS.items():
        cg[ig == gv] = cls
    return cg


def kl_score(pred, gt):
    pred = np.clip(pred, 1e-8, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    kl = np.where(gt > 0, gt * np.log(np.clip(gt, 1e-15, None) / pred), 0).sum(axis=-1)
    kl = np.where(np.isfinite(kl), kl, 0)
    return 100 - kl.mean() * 100


def load_all_gt():
    entries = []
    round_entries = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        with open(gf) as f:
            data = json.load(f)
        round_seeds = []
        for si_str in sorted(data.keys()):
            entry = data[si_str]
            pair = (np.array(entry['initial_grid']), np.array(entry['ground_truth']))
            entries.append(pair)
            round_seeds.append(pair)
        round_entries.append(round_seeds)
    return entries, round_entries


# ─── Bucket LUT (current best) ────────────────────────────

def compute_spatial_features(cls):
    H, W = cls.shape
    settlement = (cls == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    forest = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20.0)
    settle_bin = np.full((H, W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    return settle_bin, near_forest


def lut_predict_loo(round_entries, test_round_idx, clip_floor=0.0005, min_n=50):
    """LOO: train LUT on all rounds except test_round_idx, predict test round."""
    # Build LUT from training rounds
    lut_counts = {}
    lut_total = {}
    class_counts = {}
    class_total = {}

    for ri, seeds in enumerate(round_entries):
        if ri == test_round_idx:
            continue
        for ig, gt in seeds:
            cls = build_class_grid(ig)
            sb, nf = compute_spatial_features(cls)
            H, W = ig.shape
            for y in range(H):
                for x in range(W):
                    key = (int(cls[y, x]), int(sb[y, x]), int(nf[y, x]))
                    if key not in lut_counts:
                        lut_counts[key] = np.zeros(6)
                        lut_total[key] = 0
                    lut_counts[key] += gt[y, x]
                    lut_total[key] += 1

                    ic = int(cls[y, x])
                    if ic not in class_counts:
                        class_counts[ic] = np.zeros(6)
                        class_total[ic] = 0
                    class_counts[ic] += gt[y, x]
                    class_total[ic] += 1

    # Build LUT
    class_avgs = {}
    for ic in range(6):
        if ic in class_total and class_total[ic] > 0:
            class_avgs[ic] = class_counts[ic] / class_total[ic]
        else:
            class_avgs[ic] = np.ones(6) / 6

    lut = {}
    for key, total_prob in lut_counts.items():
        n = lut_total[key]
        if n >= min_n:
            avg = total_prob / n
            avg = np.clip(avg, clip_floor, None)
            avg /= avg.sum()
            lut[key] = avg
        else:
            lut[key] = class_avgs[key[0]]

    # Predict test round
    scores = []
    for ig, gt in round_entries[test_round_idx]:
        cls = build_class_grid(ig)
        sb, nf = compute_spatial_features(cls)
        H, W = ig.shape
        pred = np.ones((H, W, 6)) / 6
        for y in range(H):
            for x in range(W):
                ic = int(cls[y, x])
                if ic == 5:  # mountain
                    pred[y, x] = [0, 0, 0, 0, 0, 1]
                    continue
                key = (ic, int(sb[y, x]), int(nf[y, x]))
                if key in lut:
                    pred[y, x] = lut[key]
                else:
                    pred[y, x] = class_avgs.get(ic, np.ones(6) / 6)

        pred = np.clip(pred, clip_floor, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        s = kl_score(pred, gt)
        scores.append(s)

    return np.mean(scores)


# ─── Cell Model ────────────────────────────────────────────

def cell_model_predict(ig, params_vec, features=None):
    """Cell model prediction using optimized parameters."""
    from simulator.cell_model import params_from_vector, predict_cell_distributions
    params = params_from_vector(params_vec)
    return predict_cell_distributions(ig, params)


def cell_model_loo(round_entries, test_round_idx, params_vec):
    """LOO with cell model. params_vec should be pre-optimized on all data."""
    # For a proper LOO, we'd re-optimize params excluding the test round.
    # For speed, we use the same params (slight optimistic bias).
    scores = []
    for ig, gt in round_entries[test_round_idx]:
        pred = cell_model_predict(ig, params_vec)
        s = kl_score(pred, gt)
        scores.append(s)
    return np.mean(scores)


# ─── Hybrid: Cell Model + LUT blend ───────────────────────

def hybrid_predict(ig, gt, lut_pred, cell_pred, alpha=0.5):
    """Blend LUT and cell model predictions."""
    pred = alpha * cell_pred + (1 - alpha) * lut_pred
    pred = np.clip(pred, 1e-8, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    return kl_score(pred, gt)


def main():
    entries, round_entries = load_all_gt()
    n_rounds = len(round_entries)
    print(f"Loaded {n_rounds} rounds, {len(entries)} total seeds")

    # LOO CV for LUT
    print("\n=== Bucket LUT (current best) ===")
    lut_scores = []
    for ri in range(n_rounds):
        s = lut_predict_loo(round_entries, ri)
        lut_scores.append(s)
        print(f"  Round {ri}: {s:.2f}")
    print(f"  LOO Mean: {np.mean(lut_scores):.2f}")

    # Cell model (use default params for now)
    print("\n=== Cell Model (default params) ===")
    from simulator.cell_model import CellParams, vector_from_params
    params_vec = vector_from_params(CellParams())
    cell_scores = []
    for ri in range(n_rounds):
        scores = []
        for ig, gt in round_entries[ri]:
            pred = cell_model_predict(ig, params_vec)
            scores.append(kl_score(pred, gt))
        avg = np.mean(scores)
        cell_scores.append(avg)
        print(f"  Round {ri}: {avg:.2f}")
    print(f"  Mean: {np.mean(cell_scores):.2f}")


if __name__ == "__main__":
    main()
