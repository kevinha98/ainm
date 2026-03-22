"""Test observation-aware ensemble performance.
Simulates what happens when we observe 45 viewports per round."""
import numpy as np
import json
from pathlib import Path
from scipy import ndimage

DATA_DIR = Path("data")
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6
CLIP_FLOOR = 0.0001
MIN_N = 20
ENSEMBLE_ALPHA = 0.35


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
opt_vec = np.load(DATA_DIR / "cell_model_params.npy")
cell_params = params_from_vector(opt_vec)

# Viewport positions for 40x40 grid
viewport_positions = [(r, c) for r in [0, 12, 25] for c in [0, 12, 25]]


def simulate_observations(round_entries, gt_entries, n_obs=45):
    """Simulate observing viewports and building observation counts.
    We sample from the GT distributions to simulate observations."""
    obs_counts = {}
    obs_total = {}
    obs4_counts = {}
    obs4_total = {}
    obs3_counts = {}
    obs3_total = {}
    
    obs_used = 0
    for si in range(len(round_entries)):
        ig, gt = round_entries[si]
        cls = build_class_grid(ig)
        sb, nf, co, np_ = compute_spatial_features(cls, ig)
        H, W = ig.shape
        
        for row, col in viewport_positions:
            if obs_used >= n_obs:
                break
            obs_used += 1
            
            for vy in range(15):
                for vx in range(15):
                    gy, gx = row + vy, col + vx
                    if gy >= H or gx >= W:
                        continue
                    ic = int(cls[gy, gx])
                    sb_val = int(sb[gy, gx])
                    nf_val = int(nf[gy, gx])
                    co_val = int(co[gy, gx])
                    np_val = int(np_[gy, gx])
                    
                    # Sample an outcome from the GT distribution
                    oc = np.random.choice(6, p=gt[gy, gx])
                    
                    key5 = (ic, sb_val, nf_val, co_val, np_val)
                    key4 = key5[:4]
                    key3 = key5[:3]
                    obs_counts.setdefault(key5, np.zeros(6))
                    obs_total.setdefault(key5, 0)
                    obs_counts[key5][oc] += 1
                    obs_total[key5] += 1
                    obs4_counts.setdefault(key4, np.zeros(6))
                    obs4_total.setdefault(key4, 0)
                    obs4_counts[key4][oc] += 1
                    obs4_total[key4] += 1
                    obs3_counts.setdefault(key3, np.zeros(6))
                    obs3_total.setdefault(key3, 0)
                    obs3_counts[key3][oc] += 1
                    obs3_total[key3] += 1
        if obs_used >= n_obs:
            break
    
    return obs_counts, obs_total, obs4_counts, obs4_total, obs3_counts, obs3_total


# LOO-CV with observations
print(f"\n=== Obs-aware LOO-CV (α={ENSEMBLE_ALPHA}, clip={CLIP_FLOOR}) ===")
print(f"Simulating 45 observations per round from GT distributions\n")

n_trials = 5  # multiple trials to average out randomness
all_trial_scores = []

for trial in range(n_trials):
    np.random.seed(42 + trial)
    round_scores = []
    
    for leave_out in range(n_rounds):
        # Build cross-round GT LUT (excluding leave_out)
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
        
        # Build fallback LUTs
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
        
        fallback_lut = {}
        for k5, tp in cross_counts.items():
            n = cross_total[k5]
            if n >= MIN_N:
                avg = np.clip(tp / n, CLIP_FLOOR, None)
                fallback_lut[k5] = avg / avg.sum()
            else:
                k4, k3 = k5[:4], k5[:3]
                fallback_lut[k5] = fb4_lut.get(k4, fb3_lut.get(k3, class_avgs[k5[0]]))
        
        # Simulate observations from this round
        lo_seeds = all_entries[leave_out*5:(leave_out+1)*5]
        obs_c, obs_t, obs4_c, obs4_t, obs3_c, obs3_t = simulate_observations(lo_seeds, lo_seeds)
        
        # Build obs LUTs
        obs4_lut = {}
        for k4, counts in obs4_c.items():
            n = obs4_t[k4]
            if n >= MIN_N:
                obs4_lut[k4] = counts / n
        obs3_lut = {}
        for k3, counts in obs3_c.items():
            n = obs3_t[k3]
            if n >= MIN_N:
                obs3_lut[k3] = counts / n
        
        # Merge: obs > fallback
        lut = dict(fallback_lut)
        for k5, counts in obs_c.items():
            n = obs_t[k5]
            if n >= MIN_N:
                lut[k5] = counts / n
            else:
                k4, k3 = k5[:4], k5[:3]
                if k4 in obs4_lut:
                    lut[k5] = obs4_lut[k4]
                elif k3 in obs3_lut:
                    lut[k5] = obs3_lut[k3]
        
        # Predict
        seed_scores = []
        for s in range(5):
            ig, gt = lo_seeds[s]
            cls = build_class_grid(ig)
            sb, nf, co, np_ = compute_spatial_features(cls, ig)
            H, W = ig.shape
            
            # LUT prediction
            pred_lut = np.ones((H, W, 6)) / 6
            for y in range(H):
                for x in range(W):
                    ic = int(cls[y, x])
                    key5 = (ic, int(sb[y, x]), int(nf[y, x]), int(co[y, x]), int(np_[y, x]))
                    pred_lut[y, x] = lut.get(key5, fb4_lut.get(key5[:4], fb3_lut.get(key5[:3], class_avgs.get(ic, np.ones(6)/6))))
            
            mtn = (cls == 5)
            if mtn.any():
                pred_lut[mtn] = [0, 0, 0, 0, 0, 1]
            
            # Cell model + ensemble
            pred_cell = predict_cell_distributions(ig, cell_params)
            pred_cell = np.clip(pred_cell, CLIP_FLOOR, None)
            pred_cell = pred_cell / pred_cell.sum(axis=-1, keepdims=True)
            if mtn.any():
                pred_cell[mtn] = [0, 0, 0, 0, 0, 1]
            
            pred_lut_safe = np.clip(pred_lut, 1e-10, None)
            pred_cell_safe = np.clip(pred_cell, 1e-10, None)
            log_blend = (1 - ENSEMBLE_ALPHA) * np.log(pred_lut_safe) + ENSEMBLE_ALPHA * np.log(pred_cell_safe)
            pred = np.exp(log_blend)
            pred = np.clip(pred, CLIP_FLOOR, None)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            
            seed_scores.append(kl_score(pred, gt))
        
        round_scores.append(np.mean(seed_scores))
    
    trial_mean = np.mean(round_scores)
    all_trial_scores.append(trial_mean)
    print(f"  Trial {trial+1}: {trial_mean:.4f}")

print(f"\n  Mean across {n_trials} trials: {np.mean(all_trial_scores):.4f} (±{np.std(all_trial_scores):.4f})")

# Compare to no-obs baseline
print(f"\n  No-obs baseline: 89.32")
print(f"  Obs improvement: +{np.mean(all_trial_scores) - 89.32:.2f}")
