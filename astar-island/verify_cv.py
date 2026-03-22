"""End-to-end LOO-CV verification of the EXACT auto_runner_v2.py model.
Tests the deployed build_fallback_lut + cell model ensemble logic."""
import numpy as np
import json
from pathlib import Path
from scipy import ndimage

DATA_DIR = Path("data")
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6

# Import the exact functions from auto_runner_v2 
import importlib.util
spec = importlib.util.spec_from_file_location("ar2", "auto_runner_v2.py")
ar2 = importlib.util.module_from_spec(spec)

# We need to import the module but avoid running main and lock logic
import sys
sys.modules['ar2'] = ar2
spec.loader.exec_module(ar2)

# Use tuned params
CLIP_FLOOR = 0.0001
TEMPERATURE = 1.0
MIN_N = 20
ENSEMBLE_ALPHA = 0.35


def build_class_grid(grid_np):
    cg = np.zeros_like(grid_np)
    for gv, cls in GRID_TO_CLASS.items():
        cg[grid_np == gv] = cls
    return cg

compute_spatial_features = ar2.compute_spatial_features


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
opt_vec = np.load(DATA_DIR / "cell_model_params.npy")
cell_params = params_from_vector(opt_vec)

# LOO-CV matching exact auto_runner_v2 logic
print(f"\n=== LOO-CV (α={ENSEMBLE_ALPHA}, T={TEMPERATURE}, clip={CLIP_FLOOR}, min_n={MIN_N}) ===\n")

round_scores = []
for leave_out in range(n_rounds):
    # Build LUT from all rounds except leave_out (same as build_fallback_lut but LOO)
    cross_counts = {}
    cross_total = {}
    fb4_counts = {}
    fb4_total = {}
    fb3_counts = {}
    fb3_total = {}
    
    for r in range(n_rounds):
        if r == leave_out:
            continue
        for s in range(5):
            ig, gt = all_entries[r * 5 + s]
            cls = build_class_grid(ig)
            sb, nf, co, np_ = compute_spatial_features(cls, ig)
            H, W = ig.shape
            for y in range(H):
                for x in range(W):
                    ic = int(cls[y, x])
                    key5 = (ic, int(sb[y, x]), int(nf[y, x]), int(co[y, x]), int(np_[y, x]))
                    key4 = key5[:4]
                    key3 = key5[:3]
                    for counts, total, key in [(cross_counts, cross_total, key5), (fb4_counts, fb4_total, key4), (fb3_counts, fb3_total, key3)]:
                        counts.setdefault(key, np.zeros(6))
                        total.setdefault(key, 0)
                        counts[key] += gt[y, x]
                        total[key] += 1
    
    # Build LUTs
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
            avg = np.clip(tp / n, CLIP_FLOOR, None)
            fb3_lut[k3] = avg / avg.sum()
    
    fb4_lut = {}
    for k4, tp in fb4_counts.items():
        n = fb4_total[k4]
        if n >= 10:
            avg = np.clip(tp / n, CLIP_FLOOR, None)
            fb4_lut[k4] = avg / avg.sum()
    
    lut = {}
    for k5, tp in cross_counts.items():
        n = cross_total[k5]
        if n >= MIN_N:
            avg = np.clip(tp / n, CLIP_FLOOR, None)
            lut[k5] = avg / avg.sum()
        else:
            k4, k3 = k5[:4], k5[:3]
            lut[k5] = fb4_lut.get(k4, fb3_lut.get(k3, class_avgs[k5[0]]))
    
    # Predict left-out round
    seed_scores = []
    for s in range(5):
        idx = leave_out * 5 + s
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
                if key5 in lut:
                    pred_lut[y, x] = lut[key5]
                else:
                    k4, k3 = key5[:4], key5[:3]
                    pred_lut[y, x] = fb4_lut.get(k4, fb3_lut.get(k3, class_avgs.get(ic, np.ones(6)/6)))
        
        mtn = (cls == 5)
        if mtn.any():
            pred_lut[mtn] = [0, 0, 0, 0, 0, 1]
        
        # Cell model
        pred_cell = predict_cell_distributions(ig, cell_params)
        pred_cell = np.clip(pred_cell, CLIP_FLOOR, None)
        pred_cell = pred_cell / pred_cell.sum(axis=-1, keepdims=True)
        if mtn.any():
            pred_cell[mtn] = [0, 0, 0, 0, 0, 1]
        
        # Log-space ensemble
        pred_lut_safe = np.clip(pred_lut, 1e-10, None)
        pred_cell_safe = np.clip(pred_cell, 1e-10, None)
        log_blend = (1 - ENSEMBLE_ALPHA) * np.log(pred_lut_safe) + ENSEMBLE_ALPHA * np.log(pred_cell_safe)
        pred = np.exp(log_blend)
        
        # Final clip + normalize
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        
        seed_scores.append(kl_score(pred, gt))
    
    rm = np.mean(seed_scores)
    round_scores.append(rm)
    print(f"  Round {leave_out+1}: {rm:.4f} ({' '.join(f'{s:.1f}' for s in seed_scores)})")

print(f"\n  Mean: {np.mean(round_scores):.4f}")
print(f"  Std:  {np.std(round_scores):.4f}")
print(f"  Min:  {np.min(round_scores):.4f} (Round {np.argmin(round_scores)+1})")
print(f"  Max:  {np.max(round_scores):.4f} (Round {np.argmax(round_scores)+1})")
