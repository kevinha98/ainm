"""Fast feature comparison: within-round obs + weather-prediction concepts."""
import json, numpy as np, time
from pathlib import Path
from scipy import ndimage
from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS
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
    ocean = (cls == 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 20.0)
    forest = (cls == 4)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20.0)
    mountain = (cls == 5)
    dist_mtn = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H, W), 20.0)

    settle_bin = np.full((H, W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0

    return {
        "cls": cls, "near_settle": (dist_s <= 2.0).astype(int),
        "settle_bin": settle_bin,
        "near_ocean": (dist_ocean <= 2.0).astype(int),
        "near_forest": (dist_forest <= 2.0).astype(int),
        "near_mountain": (dist_mtn <= 2.0).astype(int),
    }

# Key functions
def key_baseline(f, y, x):
    return (int(f["cls"][y,x]), int(f["near_settle"][y,x]))
def key_settle_bins(f, y, x):
    return (int(f["cls"][y,x]), int(f["settle_bin"][y,x]))
def key_settle_ocean(f, y, x):
    return (int(f["cls"][y,x]), int(f["near_settle"][y,x]), int(f["near_ocean"][y,x]))
def key_bins_ocean(f, y, x):
    return (int(f["cls"][y,x]), int(f["settle_bin"][y,x]), int(f["near_ocean"][y,x]))
def key_bins_ocean_forest(f, y, x):
    return (int(f["cls"][y,x]), int(f["settle_bin"][y,x]),
            int(f["near_ocean"][y,x]), int(f["near_forest"][y,x]))
def key_all_dist(f, y, x):
    return (int(f["cls"][y,x]), int(f["settle_bin"][y,x]),
            int(f["near_ocean"][y,x]), int(f["near_forest"][y,x]),
            int(f["near_mountain"][y,x]))

def simulate_and_predict(test_data, key_fn, method="empirical", min_n=10,
                         prior_strength=5.0, clip_floor=0.0001, seed_offset=42):
    obs_counts, obs_total = {}, {}
    seed_feats = {}
    for si_str in sorted(test_data.keys()):
        entry = test_data[si_str]
        if not isinstance(entry, dict): continue
        gt = np.array(entry['ground_truth'])
        ig = np.array(entry['initial_grid'])
        H, W = ig.shape
        feats = compute_features(ig)
        seed_feats[si_str] = feats
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
                        key = key_fn(feats, gy, gx)
                        if key not in obs_counts:
                            obs_counts[key] = np.zeros(6)
                            obs_total[key] = 0
                        obs_counts[key][obs_class[idx]] += 1
                        obs_total[key] += 1
                        idx += 1

    # Class-level averages for fallback
    class_avgs = {}
    for ic in range(6):
        tc, tn = np.zeros(6), 0
        for k, c in obs_counts.items():
            if k[0] == ic: tc += c; tn += obs_total[k]
        class_avgs[ic] = tc / max(tn, 1) if tn > 0 else np.ones(6)/6

    lut = {}
    if method == "bayesian":
        for k, counts in obs_counts.items():
            n = obs_total[k]
            prior = class_avgs[k[0]]
            posterior = (counts + prior_strength * prior) / (n + prior_strength)
            lut[k] = posterior / posterior.sum()
    else:
        for k, counts in obs_counts.items():
            n = obs_total[k]
            lut[k] = counts / n if n >= min_n else class_avgs[k[0]]

    scores = []
    for si_str in sorted(test_data.keys()):
        entry = test_data[si_str]
        if not isinstance(entry, dict): continue
        gt = np.array(entry['ground_truth'])
        ig = np.array(entry['initial_grid'])
        H, W = ig.shape
        feats = seed_feats[si_str]
        pred = np.ones((H, W, 6)) / 6
        for y in range(H):
            for x in range(W):
                key = key_fn(feats, y, x)
                pred[y, x] = lut.get(key, class_avgs.get(int(feats["cls"][y,x]), np.ones(6)/6))
        mtn = (feats["cls"] == 5)
        if mtn.any(): pred[mtn] = np.array([0,0,0,0,0,1.0])
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
        with open(gf) as f: all_gt[rid] = json.load(f)
    rids = sorted(all_gt.keys())
    print(f"Loaded {len(all_gt)} GT files ({time.time()-t0:.1f}s)\n")

    configs = [
        ("baseline (cls,ns)",                key_baseline,           "empirical", {}),
        ("settle_bins",                      key_settle_bins,        "empirical", {}),
        ("settle+ocean",                     key_settle_ocean,       "empirical", {}),
        ("bins+ocean",                       key_bins_ocean,         "empirical", {}),
        ("bins+ocean+forest",                key_bins_ocean_forest,  "empirical", {}),
        ("all_dist",                         key_all_dist,           "empirical", {}),
        ("baseline/bayes(5)",                key_baseline,           "bayesian",  {"prior_strength": 5}),
        ("settle_bins/bayes(5)",             key_settle_bins,        "bayesian",  {"prior_strength": 5}),
        ("bins+ocean/bayes(5)",              key_bins_ocean,         "bayesian",  {"prior_strength": 5}),
        ("settle_bins(min_n=1)",             key_settle_bins,        "empirical", {"min_n": 1}),
        ("settle_bins(min_n=5)",             key_settle_bins,        "empirical", {"min_n": 5}),
        ("settle_bins(min_n=20)",            key_settle_bins,        "empirical", {"min_n": 20}),
    ]

    print(f"{'Config':<35s} {'Mean':>7s} {'Worst':>7s} {'Best':>7s}")
    print("-" * 60)
    best_mean, best_label = 0, ""
    for label, key_fn, method, kwargs in configs:
        round_scores = []
        for rid in rids:
            scores = simulate_and_predict(all_gt[rid], key_fn, method, **kwargs)
            round_scores.append(np.mean(scores))
        mean, worst, best = np.mean(round_scores), min(round_scores), max(round_scores)
        mark = " ***" if mean > best_mean else ""
        if mean > best_mean: best_mean, best_label = mean, label
        print(f"  {label:<33s} {mean:7.3f} {worst:7.1f} {best:7.1f}{mark}")

    print(f"\nBest: {best_label} (mean={best_mean:.3f})")

    # Per-round detail for top 3
    print(f"\n{'='*60}")
    top3 = sorted(configs, key=lambda c: -np.mean([np.mean(simulate_and_predict(all_gt[rid], c[1], c[2], **c[3])) for rid in rids]))[:3]
    for label, key_fn, method, kwargs in top3:
        print(f"\n{label}:")
        for rid in rids:
            rname = ROUND_MAP.get(rid[:8], rid[:8])
            s = simulate_and_predict(all_gt[rid], key_fn, method, **kwargs)
            print(f"  {rname}: {np.mean(s):.3f}")

    print(f"\nTotal: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
