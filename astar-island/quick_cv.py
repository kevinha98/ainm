"""Fast LOO CV: observation-frequency baseline with parameter sweep (vectorized)."""
import json, numpy as np, time
from pathlib import Path
from scipy import ndimage
from src.settings import DATA_DIR, NUM_CLASSES
from src.models import build_class_grid

SETTLE_DIST_THRESH = 2.0

ROUND_MAP = {
    '71451d74': 'R1', '76909e29': 'R2', 'f1dac9a9': 'R3', '8e839974': 'R4',
    'fd3c92ff': 'R5', 'ae78003a': 'R6', '36e581f1': 'R7', 'c5cdf100': 'R8',
    '2a341ace': 'R9', '75e625c3': 'R10', '324fde07': 'R11', '795bfb1f': 'R12'
}

def score_pred(pred, gt):
    pred = np.clip(pred, 1e-12, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    kl = np.sum(gt_safe * np.log(gt_safe / pred), axis=-1).mean()
    return 100 * np.exp(-kl)

def simulate_observations(test_data, seed_offset=42):
    """Simulate observations from GT distributions (vectorized per viewport)."""
    obs_counts = {}
    obs_total = {}
    
    for si_str in sorted(test_data.keys()):
        entry = test_data[si_str]
        if not isinstance(entry, dict):
            continue
        gt = np.array(entry['ground_truth'])
        ig = np.array(entry['initial_grid'])
        H, W = ig.shape
        cls = build_class_grid(ig)
        settlement = (cls == 1)
        dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
        near_settle = (dist_s <= SETTLE_DIST_THRESH)
        
        rng = np.random.RandomState(seed_offset + int(si_str))
        
        for row in [0, 12, 25]:
            for col in [0, 12, 25]:
                r_end = min(row + 15, H)
                c_end = min(col + 15, W)
                sub_cls = cls[row:r_end, col:c_end].ravel()
                sub_ns = near_settle[row:r_end, col:c_end].ravel()
                sub_gt = gt[row:r_end, col:c_end].reshape(-1, 6)
                
                cumprob = np.cumsum(np.clip(sub_gt, 0, None), axis=1)
                cumprob /= cumprob[:, -1:] + 1e-15
                u = rng.random(len(sub_cls))
                obs_class = (u[:, None] > cumprob).sum(axis=1).clip(0, 5)
                
                for i in range(len(sub_cls)):
                    key = (int(sub_cls[i]), bool(sub_ns[i]))
                    if key not in obs_counts:
                        obs_counts[key] = np.zeros(6)
                        obs_total[key] = 0
                    obs_counts[key][obs_class[i]] += 1
                    obs_total[key] += 1
    
    return obs_counts, obs_total

def build_lut(obs_counts, obs_total):
    """Build a lookup table [6 classes, 2 near_settle, 6 output classes]."""
    lut = np.ones((6, 2, 6)) / 6  # default uniform
    # First: compute merged (class-only) frequencies
    merged = {}
    for ic in range(6):
        tc, tn = np.zeros(6), 0
        for ns in [True, False]:
            k = (ic, ns)
            if k in obs_counts:
                tc += obs_counts[k]
                tn += obs_total[k]
        merged[ic] = (tc / max(tn, 1), tn)
    
    for ic in range(6):
        for ns_idx in [0, 1]:
            key = (ic, bool(ns_idx))
            if key in obs_counts and obs_total[key] >= 10:
                lut[ic, ns_idx] = obs_counts[key] / obs_total[key]
            else:
                freq, tn = merged[ic]
                if tn > 0:
                    lut[ic, ns_idx] = freq
    return lut

def predict(test_data, obs_counts, obs_total, temperature=1.0, clip_floor=0.0001):
    """Predict using observation frequencies, fully vectorized."""
    lut = build_lut(obs_counts, obs_total)
    
    scores = []
    for si_str in sorted(test_data.keys()):
        entry = test_data[si_str]
        if not isinstance(entry, dict):
            continue
        gt = np.array(entry['ground_truth'])
        ig = np.array(entry['initial_grid'])
        H, W = ig.shape
        cls = build_class_grid(ig)
        settlement = (cls == 1)
        dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
        near_settle = (dist_s <= SETTLE_DIST_THRESH).astype(int)
        
        pred = lut[cls, near_settle]  # (H, W, 6)
        
        # Mountain override
        mtn = (cls == 5)
        if mtn.any():
            pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
        
        # Temperature scaling
        if temperature != 1.0:
            pred = np.clip(pred, 1e-10, None)
            log_pred = np.log(pred)
            pred = np.exp(log_pred / temperature)
        
        pred = np.clip(pred, clip_floor, None)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        
        scores.append(score_pred(pred, gt))
    
    return scores

def main():
    t0 = time.time()
    gt_files = sorted(DATA_DIR.glob('ground_truth_*.json'))
    all_gt = {}
    for gf in gt_files:
        rid = gf.stem.replace('ground_truth_', '')
        with open(gf) as f:
            all_gt[rid] = json.load(f)
    rids = sorted(all_gt.keys())
    print(f"Loaded {len(all_gt)} GT files in {time.time()-t0:.1f}s\n")
    
    # Pre-compute observations for each round
    obs_cache = {}
    for rid in rids:
        obs_cache[rid] = simulate_observations(all_gt[rid])
    print(f"Observations computed in {time.time()-t0:.1f}s\n")
    
    # ===== Test 1: Pure obs baseline =====
    print("=" * 70)
    print("TEST 1: Pure obs freq (T=1.0, mountain override, near_settle split)")
    print("=" * 70)
    all_scores = []
    for rid in rids:
        rname = ROUND_MAP.get(rid[:8], rid[:8])
        obs_c, obs_t = obs_cache[rid]
        scores = predict(all_gt[rid], obs_c, obs_t, temperature=1.0)
        avg = np.mean(scores)
        all_scores.append(avg)
        print(f"  {rname}: {avg:.3f}")
    print(f"  MEAN: {np.mean(all_scores):.3f}  ({time.time()-t0:.1f}s)\n")
    
    # ===== Test 2: Temperature sweep =====
    print("=" * 70)
    print("TEST 2: Temperature sweep")
    print("=" * 70)
    temps = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.08, 1.10, 1.12, 1.15, 1.20, 1.30, 1.50, 2.00]
    best_temp = 1.0
    best_mean = 0
    for temp in temps:
        round_scores = []
        for rid in rids:
            obs_c, obs_t = obs_cache[rid]
            scores = predict(all_gt[rid], obs_c, obs_t, temperature=temp)
            round_scores.append(np.mean(scores))
        mean_s = np.mean(round_scores)
        mark = " <-- BEST" if mean_s > best_mean else ""
        if mean_s > best_mean:
            best_mean = mean_s
            best_temp = temp
        print(f"  T={temp:.2f}: mean={mean_s:.3f}{mark}")
    print(f"\n  Best temperature: {best_temp:.2f} (mean={best_mean:.3f})")
    print(f"  ({time.time()-t0:.1f}s elapsed)\n")
    
    # ===== Test 3: Multiple random seeds =====
    print("=" * 70)
    print(f"TEST 3: Robustness (T={best_temp:.2f}) across 10 random seeds")
    print("=" * 70)
    seed_means = []
    for seed in [0, 13, 42, 77, 100, 256, 500, 777, 1234, 9999]:
        round_scores = []
        for rid in rids:
            obs_c, obs_t = simulate_observations(all_gt[rid], seed_offset=seed)
            scores = predict(all_gt[rid], obs_c, obs_t, temperature=best_temp)
            round_scores.append(np.mean(scores))
        mean_s = np.mean(round_scores)
        seed_means.append(mean_s)
        print(f"  seed={seed:5d}: mean={mean_s:.3f}")
    print(f"  Mean +/- std: {np.mean(seed_means):.3f} +/- {np.std(seed_means):.3f}\n")
    
    # ===== Test 4: Clip floor sweep =====
    print("=" * 70)
    print(f"TEST 4: Clip floor sweep (T={best_temp:.2f})")
    print("=" * 70)
    floors = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05]
    best_floor = 1e-4
    best_floor_mean = 0
    for floor in floors:
        round_scores = []
        for rid in rids:
            obs_c, obs_t = obs_cache[rid]
            scores = predict(all_gt[rid], obs_c, obs_t, temperature=best_temp, clip_floor=floor)
            round_scores.append(np.mean(scores))
        mean_s = np.mean(round_scores)
        mark = " <-- BEST" if mean_s > best_floor_mean else ""
        if mean_s > best_floor_mean:
            best_floor_mean = mean_s
            best_floor = floor
        print(f"  floor={floor:.1e}: mean={mean_s:.3f}{mark}")
    print(f"\n  Best floor: {best_floor:.1e} (mean={best_floor_mean:.3f})\n")
    
    # ===== Test 5: Per-round detail =====
    print("=" * 70)
    print(f"TEST 5: Per-round detail (T={best_temp:.2f}, floor={best_floor:.1e})")
    print("=" * 70)
    all_final = []
    for rid in rids:
        rname = ROUND_MAP.get(rid[:8], rid[:8])
        obs_c, obs_t = obs_cache[rid]
        scores = predict(all_gt[rid], obs_c, obs_t, temperature=best_temp, clip_floor=best_floor)
        avg = np.mean(scores)
        all_final.append(avg)
        print(f"  {rname}: {avg:.3f}  seeds={[f'{s:.1f}' for s in scores]}")
    print(f"\n  FINAL MEAN: {np.mean(all_final):.3f}")
    print(f"  Total time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
