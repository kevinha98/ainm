"""Tune ensemble parameters: alpha, temperature, clip_floor.
Tests all combinations with LOO-CV on 5-feat LUT + cell model."""
import numpy as np
import json
from pathlib import Path
from scipy import ndimage
from itertools import product

DATA_DIR = Path("data")
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6


def build_class_grid(grid_np):
    cg = np.zeros_like(grid_np)
    for gv, cls in GRID_TO_CLASS.items():
        cg[grid_np == gv] = cls
    return cg


def compute_spatial_features(cls, ig):
    H, W = cls.shape
    settlement = (cls == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    forest = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20.0)
    ocean = (ig == 10)
    dist_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    port = (cls == 2)
    dist_p = ndimage.distance_transform_edt(~port) if port.any() else np.full((H, W), 40.0)
    settle_bin = np.full((H, W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    coastal = (dist_o <= 1.5).astype(int)
    near_port = (dist_p <= 2.0).astype(int)
    return settle_bin, near_forest, coastal, near_port


def kl_score(pred, gt):
    pred = np.clip(pred, 1e-8, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    kl = np.where(gt > 0, gt * np.log(np.clip(gt, 1e-15, None) / pred), 0).sum(axis=-1)
    return 100 - np.mean(np.where(np.isfinite(kl), kl, 0)) * 100


# Load all GT
gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
all_entries = []
for gf in gt_files:
    with open(gf) as f:
        data = json.load(f)
    for si_str in sorted(data.keys()):
        entry = data[si_str]
        all_entries.append((np.array(entry['initial_grid']), np.array(entry['ground_truth'])))

n_rounds = len(all_entries) // 5
print(f"Loaded {len(all_entries)} entries from {n_rounds} rounds")

# Load cell model
from simulator.cell_model import predict_cell_distributions, params_from_vector
opt_vec = np.load('data/cell_model_params.npy')
cell_params = params_from_vector(opt_vec)

# Pre-compute all cell model predictions (fast now!)
cell_preds = []
for ig, gt in all_entries:
    cell_preds.append(predict_cell_distributions(ig, cell_params))
print("Pre-computed all cell model predictions")


def build_lut_loo(leave_out_round, clip_floor, min_n=50):
    """Build 5-feat LUT leaving out one round."""
    cross_counts = {}
    cross_total = {}
    fb4_counts = {}
    fb4_total = {}
    fb3_counts = {}
    fb3_total = {}
    
    for r in range(n_rounds):
        if r == leave_out_round:
            continue
        for s in range(5):
            idx = r * 5 + s
            ig, gt = all_entries[idx]
            cls = build_class_grid(ig)
            sb, nf, co, np_ = compute_spatial_features(cls, ig)
            H, W = ig.shape
            for y in range(H):
                for x in range(W):
                    ic = int(cls[y, x])
                    key5 = (ic, int(sb[y, x]), int(nf[y, x]), int(co[y, x]), int(np_[y, x]))
                    key4 = key5[:4]
                    key3 = key5[:3]
                    cross_counts.setdefault(key5, np.zeros(6))
                    cross_total.setdefault(key5, 0)
                    cross_counts[key5] += gt[y, x]
                    cross_total[key5] += 1
                    fb4_counts.setdefault(key4, np.zeros(6))
                    fb4_total.setdefault(key4, 0)
                    fb4_counts[key4] += gt[y, x]
                    fb4_total[key4] += 1
                    fb3_counts.setdefault(key3, np.zeros(6))
                    fb3_total.setdefault(key3, 0)
                    fb3_counts[key3] += gt[y, x]
                    fb3_total[key3] += 1

    class_avgs = {}
    for ic in range(6):
        tc, tn = np.zeros(6), 0
        for k, v in cross_counts.items():
            if k[0] == ic:
                tc += v
                tn += cross_total[k]
        class_avgs[ic] = tc / max(tn, 1) if tn > 0 else np.ones(6) / 6

    fb3_lut = {}
    for k3, tp in fb3_counts.items():
        n = fb3_total[k3]
        if n >= 10:
            avg = np.clip(tp / n, clip_floor, None)
            fb3_lut[k3] = avg / avg.sum()

    fb4_lut = {}
    for k4, tp in fb4_counts.items():
        n = fb4_total[k4]
        if n >= 10:
            avg = np.clip(tp / n, clip_floor, None)
            fb4_lut[k4] = avg / avg.sum()

    lut = {}
    for k5, tp in cross_counts.items():
        n = cross_total[k5]
        if n >= min_n:
            avg = np.clip(tp / n, clip_floor, None)
            lut[k5] = avg / avg.sum()
        else:
            k4, k3 = k5[:4], k5[:3]
            lut[k5] = fb4_lut.get(k4, fb3_lut.get(k3, class_avgs[k5[0]]))

    return lut, fb4_lut, fb3_lut, class_avgs


def evaluate_ensemble(alpha, temperature, clip_floor, min_n=50):
    """LOO-CV score for given ensemble parameters."""
    round_scores = []
    for leave_out in range(n_rounds):
        lut, fb4, fb3, class_avgs = build_lut_loo(leave_out, clip_floor, min_n)
        
        seeds = list(range(leave_out*5, (leave_out+1)*5))
        scores = []
        for idx in seeds:
            ig, gt = all_entries[idx]
            cls = build_class_grid(ig)
            sb, nf, co, np_ = compute_spatial_features(cls, ig)
            H, W = ig.shape
            
            # LUT prediction
            pred_lut = np.ones((H, W, 6)) / 6
            for y in range(H):
                for x in range(W):
                    ic = int(cls[y, x])
                    key5 = (ic, int(sb[y, x]), int(nf[y, x]), int(co[y, x]), int(np_[y, x]))
                    pred_lut[y, x] = lut.get(key5, fb4.get(key5[:4], fb3.get(key5[:3], class_avgs.get(ic, np.ones(6)/6))))
            
            mtn = (cls == 5)
            if mtn.any():
                pred_lut[mtn] = [0, 0, 0, 0, 0, 1]

            # Cell model prediction (pre-computed)
            pred_cell = cell_preds[idx]
            
            # Ensemble blend
            if alpha > 0:
                pred_lut_safe = np.clip(pred_lut, 1e-10, None)
                pred_cell_safe = np.clip(pred_cell, 1e-10, None)
                log_blend = (1 - alpha) * np.log(pred_lut_safe) + alpha * np.log(pred_cell_safe)
                pred = np.exp(log_blend)
            else:
                pred = pred_lut

            # Temperature
            if temperature != 1.0:
                non_mtn = ~mtn
                if non_mtn.any():
                    p = np.clip(pred[non_mtn], 1e-10, None)
                    pred[non_mtn] = np.exp(np.log(p) / temperature)

            # Final clip + normalize
            pred = np.clip(pred, clip_floor, None)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            scores.append(kl_score(pred, gt))
        
        round_scores.append(np.mean(scores))
    
    return np.mean(round_scores), round_scores


# Grid search ensemble parameters
print("\n=== Ensemble Parameter Tuning (LOO-CV) ===\n")

alphas = [0.0, 0.1, 0.2, 0.3, 0.35, 0.40, 0.45, 0.50, 0.6]
temperatures = [0.90, 0.95, 1.0, 1.05, 1.10]
clip_floors = [0.0001, 0.0005, 0.001, 0.002]

# First: find best alpha at T=1.0, clip=0.0005
print("--- Alpha sweep (T=1.0, clip=0.0005) ---")
best_alpha = 0.4
best_score = 0
for a in alphas:
    score, rs = evaluate_ensemble(a, 1.0, 0.0005)
    flag = " *" if score > best_score else ""
    print(f"  α={a:.2f}: {score:.4f} (std={np.std(rs):.2f}){flag}")
    if score > best_score:
        best_score = score
        best_alpha = a

print(f"\nBest alpha: {best_alpha} ({best_score:.4f})")

# Second: sweep T at best alpha
print(f"\n--- Temperature sweep (α={best_alpha}, clip=0.0005) ---")
best_temp = 1.0
best_score_t = 0
for t in temperatures:
    score, rs = evaluate_ensemble(best_alpha, t, 0.0005)
    flag = " *" if score > best_score_t else ""
    print(f"  T={t:.2f}: {score:.4f}{flag}")
    if score > best_score_t:
        best_score_t = score
        best_temp = t

print(f"\nBest temperature: {best_temp} ({best_score_t:.4f})")

# Third: sweep clip at best alpha + T
print(f"\n--- Clip floor sweep (α={best_alpha}, T={best_temp}) ---")
best_clip = 0.0005
best_score_c = 0
for c in clip_floors:
    score, rs = evaluate_ensemble(best_alpha, best_temp, c)
    flag = " *" if score > best_score_c else ""
    print(f"  clip={c:.4f}: {score:.4f}{flag}")
    if score > best_score_c:
        best_score_c = score
        best_clip = c

print(f"\nBest clip: {best_clip} ({best_score_c:.4f})")

# Final: best combo per-round detail
print(f"\n=== Best Ensemble: α={best_alpha}, T={best_temp}, clip={best_clip} ===")
final_score, final_rounds = evaluate_ensemble(best_alpha, best_temp, best_clip)
for i, rs in enumerate(final_rounds):
    print(f"  Round {i+1}: {rs:.4f}")
print(f"  Mean: {final_score:.4f}")

# Also test min_n values
print(f"\n--- MIN_N sweep (α={best_alpha}, T={best_temp}, clip={best_clip}) ---")
for mn in [20, 30, 50, 70, 100]:
    score, _ = evaluate_ensemble(best_alpha, best_temp, best_clip, min_n=mn)
    print(f"  min_n={mn}: {score:.4f}")
