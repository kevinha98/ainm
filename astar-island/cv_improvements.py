"""
Comprehensive CV: Test cell-level Bayesian model targeting rank #1.

Tests:
1. Bucket-only LUT (baseline settle_bins+forest) vs cell-level Bayesian
2. Alpha sweep for cell-level Bayesian prior strength
3. Spatial smoothing
4. Combined best settings
"""
import json, numpy as np, time
from pathlib import Path
from scipy import ndimage
from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from src.models import build_class_grid

ROUND_MAP = {
    '71451d74': 'R1', '76909e29': 'R2', 'f1dac9a9': 'R3', '8e839974': 'R4',
    'fd3c92ff': 'R5', 'ae78003a': 'R6', '36e581f1': 'R7', 'c5cdf100': 'R8',
    '2a341ace': 'R9', '75e625c3': 'R10', '324fde07': 'R11', '795bfb1f': 'R12',
    '7b4bda99': 'R13',
}

def score_pred(pred, gt):
    pred = np.clip(pred, 1e-12, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    kl = np.sum(gt_safe * np.log(gt_safe / pred), axis=-1).mean()
    return 100 * np.exp(-kl)


def compute_features(ig):
    cls = build_class_grid(ig)
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
    return cls, settle_bin, near_forest


def sim_obs_with_cells(test_data, seed_offset=42):
    """Simulate observations returning bucket counts AND per-cell observations."""
    obs_counts, obs_total = {}, {}
    cell_obs = {}  # (si_str, y, x) → list of observed classes
    seed_feats = {}

    for si_str in sorted(test_data.keys()):
        entry = test_data[si_str]
        if not isinstance(entry, dict): continue
        gt = np.array(entry['ground_truth'])
        ig = np.array(entry['initial_grid'])
        H, W = ig.shape
        cls, sb, nf = compute_features(ig)
        seed_feats[si_str] = (cls, sb, nf)

        rng = np.random.RandomState(seed_offset + int(si_str))
        for row in [0, 12, 25]:
            for col in [0, 12, 25]:
                r_end, c_end = min(row+15, H), min(col+15, W)
                sub_gt = gt[row:r_end, col:c_end].reshape(-1, 6)
                cumprob = np.cumsum(np.clip(sub_gt, 0, None), axis=1)
                cumprob /= cumprob[:, -1:] + 1e-15
                u = rng.random(len(sub_gt))
                obs_class = (u[:, None] > cumprob).sum(axis=1).clip(0, 5)

                idx = 0
                for vy in range(r_end - row):
                    for vx in range(c_end - col):
                        gy, gx = row + vy, col + vx
                        ic = int(cls[gy, gx])
                        sb_val = int(sb[gy, gx])
                        nf_val = int(nf[gy, gx])
                        oc = int(obs_class[idx])

                        key = (ic, sb_val, nf_val)
                        if key not in obs_counts:
                            obs_counts[key] = np.zeros(6)
                            obs_total[key] = 0
                        obs_counts[key][oc] += 1
                        obs_total[key] += 1

                        cell_key = (si_str, gy, gx)
                        if cell_key not in cell_obs:
                            cell_obs[cell_key] = []
                        cell_obs[cell_key].append(oc)
                        idx += 1

    return obs_counts, obs_total, cell_obs, seed_feats


def build_lut(obs_counts, obs_total, min_n=10):
    """Build bucket LUT with class-level fallback."""
    class_avgs = {}
    for ic in range(6):
        tc, tn = np.zeros(6), 0
        for k, c in obs_counts.items():
            if k[0] == ic: tc += c; tn += obs_total[k]
        class_avgs[ic] = tc / max(tn, 1) if tn > 0 else np.ones(6)/6
    lut = {}
    for k, counts in obs_counts.items():
        n = obs_total[k]
        lut[k] = counts / n if n >= min_n else class_avgs[k[0]]
    return lut, class_avgs


def predict_bucket_only(test_data, obs_counts, obs_total, clip_floor=0.0001):
    """Predict using bucket LUT only (no cell-level updates)."""
    lut, class_avgs = build_lut(obs_counts, obs_total)
    scores = []
    for si_str in sorted(test_data.keys()):
        entry = test_data[si_str]
        if not isinstance(entry, dict): continue
        gt = np.array(entry['ground_truth'])
        ig = np.array(entry['initial_grid'])
        H, W = ig.shape
        cls, sb, nf = compute_features(ig)
        pred = np.ones((H, W, 6)) / 6
        for y in range(H):
            for x in range(W):
                key = (int(cls[y,x]), int(sb[y,x]), int(nf[y,x]))
                pred[y,x] = lut.get(key, class_avgs.get(int(cls[y,x]), np.ones(6)/6))
        mtn = (cls == 5)
        if mtn.any(): pred[mtn] = [0,0,0,0,0,1.0]
        pred = np.clip(pred, clip_floor, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        scores.append(score_pred(pred, gt))
    return scores


def predict_cell_bayesian(test_data, obs_counts, obs_total, cell_obs, seed_feats,
                          alpha_prior=10.0, clip_floor=0.0001):
    """Predict using bucket LUT + per-cell Bayesian updates."""
    lut, class_avgs = build_lut(obs_counts, obs_total)
    n_seeds = len([k for k in test_data.keys() if isinstance(test_data[k], dict)])
    seed_strs = sorted(k for k in test_data.keys() if isinstance(test_data[k], dict))

    scores = []
    for si_str in seed_strs:
        entry = test_data[si_str]
        gt = np.array(entry['ground_truth'])
        ig = np.array(entry['initial_grid'])
        H, W = ig.shape
        cls, sb, nf = seed_feats[si_str]

        pred = np.ones((H, W, 6)) / 6
        for y in range(H):
            for x in range(W):
                key = (int(cls[y,x]), int(sb[y,x]), int(nf[y,x]))
                bucket_prior = lut.get(key, class_avgs.get(int(cls[y,x]), np.ones(6)/6)).copy()

                # Collect cell obs from ALL seeds
                cell_samples = []
                for other_si in seed_strs:
                    cell_key = (other_si, y, x)
                    if cell_key in cell_obs:
                        cell_samples.extend(cell_obs[cell_key])

                if cell_samples:
                    cell_counts = np.zeros(6)
                    for s in cell_samples:
                        cell_counts[s] += 1
                    posterior = bucket_prior * alpha_prior + cell_counts
                    posterior = np.clip(posterior, 1e-10, None)
                    posterior /= posterior.sum()
                    pred[y,x] = posterior
                else:
                    pred[y,x] = bucket_prior

        mtn = (cls == 5)
        if mtn.any(): pred[mtn] = [0,0,0,0,0,1.0]
        pred = np.clip(pred, clip_floor, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        scores.append(score_pred(pred, gt))
    return scores


def predict_cell_bayesian_smooth(test_data, obs_counts, obs_total, cell_obs, seed_feats,
                                  alpha_prior=10.0, smooth_sigma=0.5, clip_floor=0.0001):
    """Bucket LUT + cell Bayesian + spatial Gaussian smoothing."""
    lut, class_avgs = build_lut(obs_counts, obs_total)
    seed_strs = sorted(k for k in test_data.keys() if isinstance(test_data[k], dict))

    scores = []
    for si_str in seed_strs:
        entry = test_data[si_str]
        gt = np.array(entry['ground_truth'])
        ig = np.array(entry['initial_grid'])
        H, W = ig.shape
        cls, sb, nf = seed_feats[si_str]

        pred = np.ones((H, W, 6)) / 6
        for y in range(H):
            for x in range(W):
                key = (int(cls[y,x]), int(sb[y,x]), int(nf[y,x]))
                bucket_prior = lut.get(key, class_avgs.get(int(cls[y,x]), np.ones(6)/6)).copy()
                cell_samples = []
                for other_si in seed_strs:
                    cell_key = (other_si, y, x)
                    if cell_key in cell_obs:
                        cell_samples.extend(cell_obs[cell_key])
                if cell_samples:
                    cell_counts = np.zeros(6)
                    for s in cell_samples:
                        cell_counts[s] += 1
                    posterior = bucket_prior * alpha_prior + cell_counts
                    posterior = np.clip(posterior, 1e-10, None)
                    posterior /= posterior.sum()
                    pred[y,x] = posterior
                else:
                    pred[y,x] = bucket_prior

        mtn = (cls == 5)
        if mtn.any(): pred[mtn] = [0,0,0,0,0,1.0]

        # Spatial smoothing per class channel (skip mountain)
        if smooth_sigma > 0:
            for c in range(6):
                channel = pred[:,:,c].copy()
                smoothed = ndimage.gaussian_filter(channel, sigma=smooth_sigma)
                non_mtn = ~mtn
                pred[:,:,c] = np.where(non_mtn, smoothed, channel)

        pred = np.clip(pred, clip_floor, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        scores.append(score_pred(pred, gt))
    return scores


def main():
    t0 = time.time()
    gt_files = sorted(DATA_DIR.glob('ground_truth_*.json'))
    all_gt = {}
    for gf in gt_files:
        rid = gf.stem.replace('ground_truth_', '')
        with open(gf) as f: all_gt[rid] = json.load(f)
    rids = sorted(all_gt.keys())
    print(f"Loaded {len(all_gt)} GT files ({time.time()-t0:.1f}s)\n")

    # Pre-compute observations
    all_obs = {}
    for rid in rids:
        all_obs[rid] = sim_obs_with_cells(all_gt[rid])
    print(f"Observations precomputed ({time.time()-t0:.1f}s)\n")

    # === EXPERIMENT 1: Bucket-only baseline (settle_bins+forest) ===
    print("=" * 70)
    print("EXP 1: Bucket-only LUT (settle_bins + near_forest)")
    print("=" * 70)
    bucket_scores = []
    for rid in rids:
        obs_c, obs_t, _, _ = all_obs[rid]
        scores = predict_bucket_only(all_gt[rid], obs_c, obs_t)
        avg = np.mean(scores)
        bucket_scores.append(avg)
        rname = ROUND_MAP.get(rid[:8], rid[:8])
        print(f"  {rname}: {avg:.3f}")
    print(f"  MEAN: {np.mean(bucket_scores):.3f}\n")

    # === EXPERIMENT 2: Cell-level Bayesian, alpha sweep ===
    print("=" * 70)
    print("EXP 2: Cell-level Bayesian — alpha sweep")
    print("=" * 70)
    best_alpha, best_mean = 10, 0
    for alpha in [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]:
        round_scores = []
        for rid in rids:
            obs_c, obs_t, cell_obs, seed_feats = all_obs[rid]
            scores = predict_cell_bayesian(all_gt[rid], obs_c, obs_t, cell_obs, seed_feats, alpha)
            round_scores.append(np.mean(scores))
        mean = np.mean(round_scores)
        mark = " ***" if mean > best_mean else ""
        if mean > best_mean:
            best_mean, best_alpha = mean, alpha
        print(f"  alpha={alpha:6.1f}: mean={mean:.3f}{mark}")
    print(f"  Best alpha: {best_alpha} (mean={best_mean:.3f})\n")

    # === EXPERIMENT 3: Spatial smoothing sigma sweep ===
    print("=" * 70)
    print(f"EXP 3: Spatial smoothing — sigma sweep (alpha={best_alpha})")
    print("=" * 70)
    best_sigma, best_smooth_mean = 0, best_mean
    for sigma in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        round_scores = []
        for rid in rids:
            obs_c, obs_t, cell_obs, seed_feats = all_obs[rid]
            scores = predict_cell_bayesian_smooth(
                all_gt[rid], obs_c, obs_t, cell_obs, seed_feats,
                alpha_prior=best_alpha, smooth_sigma=sigma)
            round_scores.append(np.mean(scores))
        mean = np.mean(round_scores)
        mark = " ***" if mean > best_smooth_mean else ""
        if mean > best_smooth_mean:
            best_smooth_mean, best_sigma = mean, sigma
        print(f"  sigma={sigma:.1f}: mean={mean:.3f}{mark}")
    print(f"  Best sigma: {best_sigma} (mean={best_smooth_mean:.3f})\n")

    # === EXPERIMENT 4: Robustness across seeds ===
    print("=" * 70)
    print(f"EXP 4: Robustness (alpha={best_alpha}, sigma={best_sigma}) x5 seeds")
    print("=" * 70)
    seed_means = []
    for seed in [0, 42, 100, 777, 9999]:
        round_scores = []
        for rid in rids:
            obs_c, obs_t, cell_obs, seed_feats = sim_obs_with_cells(all_gt[rid], seed_offset=seed)
            if best_sigma > 0:
                scores = predict_cell_bayesian_smooth(
                    all_gt[rid], obs_c, obs_t, cell_obs, seed_feats,
                    alpha_prior=best_alpha, smooth_sigma=best_sigma)
            else:
                scores = predict_cell_bayesian(
                    all_gt[rid], obs_c, obs_t, cell_obs, seed_feats, best_alpha)
            round_scores.append(np.mean(scores))
        mean = np.mean(round_scores)
        seed_means.append(mean)
        print(f"  seed={seed:5d}: mean={mean:.3f}")
    print(f"  Mean ± std: {np.mean(seed_means):.3f} ± {np.std(seed_means):.3f}\n")

    # === EXPERIMENT 5: Per-round detail for best config ===
    print("=" * 70)
    print(f"EXP 5: Per-round detail (alpha={best_alpha}, sigma={best_sigma})")
    print("=" * 70)
    for rid in rids:
        obs_c, obs_t, cell_obs, seed_feats = all_obs[rid]
        if best_sigma > 0:
            scores = predict_cell_bayesian_smooth(
                all_gt[rid], obs_c, obs_t, cell_obs, seed_feats,
                alpha_prior=best_alpha, smooth_sigma=best_sigma)
        else:
            scores = predict_cell_bayesian(
                all_gt[rid], obs_c, obs_t, cell_obs, seed_feats, best_alpha)
        rname = ROUND_MAP.get(rid[:8], rid[:8])
        print(f"  {rname}: {np.mean(scores):.3f}  seeds={[f'{s:.1f}' for s in scores]}")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Bucket-only: {np.mean(bucket_scores):.3f}")
    print(f"  Cell Bayesian (alpha={best_alpha}): {best_mean:.3f}")
    print(f"  + Smoothing (sigma={best_sigma}): {best_smooth_mean:.3f}")
    print(f"  Improvement: +{best_smooth_mean - np.mean(bucket_scores):.3f}")
    print(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
