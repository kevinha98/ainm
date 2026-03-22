"""
LOO CV comparison: old HGB+calibration vs new observation-first approach.
Simulates observations from GT data to test both approaches.

For each held-out round:
  1. Train HGB on remaining rounds
  2. Simulate "observations" by sampling from the held-out GT
  3. Score both approaches against the held-out GT
"""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from src.models import build_class_grid

CLIP_FLOOR = 0.0001
TEMPERATURE_OLD = 1.15   # Old model temperature
TEMPERATURE_NEW = 1.10   # New model temperature
HGB_PRIOR_WEIGHT = 0.15  # Weight for HGB in new model
SETTLE_DIST_THRESH = 2.0

# Round ID to name mapping
ROUND_MAP = {
    '71451d74': 'R1', '76909e29': 'R2', 'f1dac9a9': 'R3', '8e839974': 'R4',
    'fd3c92ff': 'R5', 'ae78003a': 'R6', '36e581f1': 'R7', 'c5cdf100': 'R8',
    '2a341ace': 'R9', '75e625c3': 'R10', '324fde07': 'R11', '795bfb1f': 'R12'
}

def score_pred(pred, gt):
    """Score: 100 * exp(-KL(gt || pred))"""
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    kl = np.sum(gt_safe * np.log(gt_safe / (pred + 1e-15)), axis=-1).mean()
    return 100 * np.exp(-kl)


def extract_features(ig):
    cls = build_class_grid(ig)
    H, W = ig.shape
    ocean = (ig == 10); mountain = (ig == 5)
    settlement = (cls == 1); forest = (cls == 4); empty = (cls == 0)

    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H,W), 20)

    k3, k7, k11 = np.ones((3,3)), np.ones((7,7)), np.ones((11,11))
    n_s3 = ndimage.convolve(settlement.astype(float), k3, mode='constant')
    n_s7 = ndimage.convolve(settlement.astype(float), k7, mode='constant')
    n_f7 = ndimage.convolve(forest.astype(float), k7, mode='constant')
    n_o7 = ndimage.convolve(ocean.astype(float), k7, mode='constant')
    n_e7 = ndimage.convolve(empty.astype(float), k7, mode='constant')
    n_s11 = ndimage.convolve(settlement.astype(float), k11, mode='constant')

    cls_oh = np.zeros((H, W, NUM_CLASSES))
    for c in range(NUM_CLASSES): cls_oh[:,:,c] = (cls==c).astype(float)

    features = np.concatenate([
        cls_oh,
        dist_ocean[:,:,None], dist_settle[:,:,None], dist_forest[:,:,None], dist_mountain[:,:,None],
        n_s3[:,:,None], n_s7[:,:,None], n_f7[:,:,None], n_o7[:,:,None], n_e7[:,:,None], n_s11[:,:,None],
        is_coast[:,:,None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1])


def train_hgb(gt_data_list):
    """Train HGB on a list of GT data dicts."""
    X_all, Y_all = [], []
    for gt_data in gt_data_list:
        for si_str in sorted(gt_data.keys()):
            entry = gt_data[si_str]
            if not isinstance(entry, dict): continue
            gt = np.array(entry.get('ground_truth', []))
            ig = np.array(entry.get('initial_grid', []))
            if gt.size == 0 or ig.size == 0: continue
            X_all.append(extract_features(ig))
            Y_all.append(gt.reshape(-1, 6))

    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)

    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, learning_rate=0.05,
            min_samples_leaf=50, random_state=42
        )
        m.fit(X_all, Y_all[:, c])
        models.append(m)
    return models


def simulate_observations_from_gt(gt_data, n_obs_per_seed=9):
    """
    Simulate observations by sampling from the GT probability distributions.
    For each cell, sample a class from the GT distribution (simulating a single sim run).
    Returns observation counts per (class, spatial_bucket).
    """
    obs_counts = {}   # (ic, near_settle) → counts
    obs_total = {}
    obs_fine = {}     # (ic, spatial_type) → counts
    obs_fine_n = {}

    vp_size = 15

    for si_str in sorted(gt_data.keys()):
        entry = gt_data[si_str]
        if not isinstance(entry, dict): continue
        gt = np.array(entry['ground_truth'])
        ig = np.array(entry['initial_grid'])
        H, W = ig.shape

        cls = build_class_grid(ig)
        settlement = (cls == 1)
        ocean = (ig == 10)
        dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
        near_settle = dist_s <= SETTLE_DIST_THRESH
        is_coast = ~(ig == 10) & (ndimage.maximum_filter((ig == 10).astype(float), size=3) > 0)

        # Viewport positions
        def grid_positions(dim):
            n = max(1, -(-dim // vp_size))
            if n == 1: return [0]
            step = (dim - vp_size) / (n - 1)
            return [round(i * step) for i in range(n)]

        for row in grid_positions(H):
            for col in grid_positions(W):
                for vy in range(min(vp_size, H - row)):
                    for vx in range(min(vp_size, W - col)):
                        gy, gx = row + vy, col + vx
                        ic = cls[gy, gx]
                        ns = bool(near_settle[gy, gx])
                        coast = bool(is_coast[gy, gx])

                        # Sample observed class from GT distribution
                        gt_dist = gt[gy, gx]
                        oc = np.random.choice(6, p=np.clip(gt_dist, 0, None) / np.clip(gt_dist, 0, None).sum())

                        key = (ic, ns)
                        if key not in obs_counts:
                            obs_counts[key] = np.zeros(6)
                            obs_total[key] = 0
                        obs_counts[key][oc] += 1
                        obs_total[key] += 1

                        if ns: stype = 0
                        elif coast: stype = 1
                        else: stype = 2
                        fkey = (ic, stype)
                        if fkey not in obs_fine:
                            obs_fine[fkey] = np.zeros(6)
                            obs_fine_n[fkey] = 0
                        obs_fine[fkey][oc] += 1
                        obs_fine_n[fkey] += 1

    return obs_counts, obs_total, obs_fine, obs_fine_n


def predict_old_approach(models, test_gt_data, obs_counts, obs_total):
    """Old approach: HGB + multiplicative calibration with temperature."""
    scores = []
    for si_str in sorted(test_gt_data.keys()):
        entry = test_gt_data[si_str]
        if not isinstance(entry, dict): continue
        gt = np.array(entry['ground_truth'])
        ig = np.array(entry['initial_grid'])
        H, W = ig.shape
        cls = build_class_grid(ig)

        # HGB prediction
        X = extract_features(ig)
        pred = np.column_stack([m.predict(X) for m in models])
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)

        # Temperature scaling
        log_p = np.log(pred)
        scaled = log_p / TEMPERATURE_OLD
        scaled -= scaled.max(axis=-1, keepdims=True)
        pred = np.exp(scaled)
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        pred = pred.reshape(H, W, 6)

        # Multiplicative calibration
        settlement = (cls == 1)
        dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
        near_settle = dist_s <= SETTLE_DIST_THRESH

        # Compute predicted averages per bucket
        pred_avg = {}
        pred_n = {}
        for y in range(H):
            for x in range(W):
                ic = cls[y, x]
                ns = bool(near_settle[y, x])
                key = (ic, ns)
                if key not in pred_avg:
                    pred_avg[key] = np.zeros(6)
                    pred_n[key] = 0
                pred_avg[key] += pred[y, x]
                pred_n[key] += 1

        # Apply multiplicative calibration
        for y in range(H):
            for x in range(W):
                ic = cls[y, x]
                ns = bool(near_settle[y, x])
                key = (ic, ns)
                if key in obs_counts and obs_total[key] >= 10 and pred_n.get(key, 0) >= 10:
                    obs_freq = obs_counts[key] / obs_total[key]
                    p_avg = pred_avg[key] / pred_n[key]
                    ratio = np.ones(6)
                    for k in range(6):
                        if p_avg[k] > 0.01:
                            ratio[k] = np.clip(obs_freq[k] / p_avg[k], 0.01, 100.0)
                    pred[y, x] *= ratio

        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)

        scores.append(score_pred(pred, gt))
    return np.mean(scores)


def predict_new_approach(models, test_gt_data, obs_counts, obs_total, obs_fine, obs_fine_n, hgb_weight=HGB_PRIOR_WEIGHT, temperature=TEMPERATURE_NEW):
    """New approach: observation-first with optional HGB prior blend."""
    scores = []
    for si_str in sorted(test_gt_data.keys()):
        entry = test_gt_data[si_str]
        if not isinstance(entry, dict): continue
        gt = np.array(entry['ground_truth'])
        ig = np.array(entry['initial_grid'])
        H, W = ig.shape
        cls = build_class_grid(ig)

        settlement = (cls == 1)
        ocean = (ig == 10)
        mountain = (ig == 5)
        dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
        near_settle = dist_s <= SETTLE_DIST_THRESH
        is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)

        # HGB prior (if available)
        hgb_pred = None
        if models is not None and hgb_weight > 0:
            X = extract_features(ig)
            hp = np.column_stack([m.predict(X) for m in models])
            hp = np.clip(hp, CLIP_FLOOR, None)
            hp /= hp.sum(axis=-1, keepdims=True)
            hgb_pred = hp.reshape(H, W, 6)

        pred = np.zeros((H, W, 6))
        for y in range(H):
            for x in range(W):
                ic = cls[y, x]

                # Mountain override
                if ic == 5:
                    pred[y, x] = [0, 0, 0, 0, 0, 1.0]
                    continue

                ns = bool(near_settle[y, x])
                coast = bool(is_coast[y, x])

                if ns: stype = 0
                elif coast: stype = 1
                else: stype = 2

                fkey = (ic, stype)
                key = (ic, ns)

                obs_freq = None
                if fkey in obs_fine and obs_fine_n[fkey] >= 30:
                    obs_freq = obs_fine[fkey] / obs_fine_n[fkey]
                elif key in obs_counts and obs_total[key] >= 10:
                    obs_freq = obs_counts[key] / obs_total[key]
                else:
                    total_c = np.zeros(6)
                    total_n = 0
                    for ns_v in [True, False]:
                        k = (ic, ns_v)
                        if k in obs_counts:
                            total_c += obs_counts[k]
                            total_n += obs_total[k]
                    if total_n >= 5:
                        obs_freq = total_c / total_n

                if obs_freq is not None:
                    obs_freq = np.clip(obs_freq, CLIP_FLOOR, None)
                    obs_freq /= obs_freq.sum()
                    if hgb_pred is not None:
                        hgb_cell = np.clip(hgb_pred[y, x], CLIP_FLOOR, None)
                        hgb_cell /= hgb_cell.sum()
                        blended = (1 - hgb_weight) * obs_freq + hgb_weight * hgb_cell
                    else:
                        blended = obs_freq
                else:
                    if hgb_pred is not None:
                        blended = np.clip(hgb_pred[y, x], CLIP_FLOOR, None)
                        blended /= blended.sum()
                    else:
                        blended = np.ones(6) / 6

                pred[y, x] = blended

        # Temperature scaling (skip mountains)
        mountain_mask = (cls == 5)
        flat = pred.reshape(-1, 6)
        for i in range(len(flat)):
            yy, xx = divmod(i, W)
            if mountain_mask[yy, xx]: continue
            log_p = np.log(np.clip(flat[i], CLIP_FLOOR, None))
            scaled = log_p / temperature
            scaled -= scaled.max()
            flat[i] = np.exp(scaled)
            flat[i] = np.clip(flat[i], CLIP_FLOOR, None)
            flat[i] /= flat[i].sum()

        pred = flat.reshape(H, W, 6)
        scores.append(score_pred(pred, gt))
    return np.mean(scores)


def main():
    # Load all GT files
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    all_gt = {}
    for gf in gt_files:
        rid = gf.stem.replace("ground_truth_", "")
        with open(gf) as f:
            all_gt[rid] = json.load(f)
    print(f"Loaded {len(all_gt)} GT files")

    # LOO CV
    results_old = []
    results_new_pure = []
    results_new_blend = []
    results_new_notemp = []

    rids = sorted(all_gt.keys())
    for test_rid in rids:
        rname = ROUND_MAP.get(test_rid[:8], test_rid[:8])
        train_data = [all_gt[r] for r in rids if r != test_rid]
        test_data = all_gt[test_rid]

        # Train HGB on other rounds
        models = train_hgb(train_data)

        # Simulate observations from test GT (as if we're observing the round)
        np.random.seed(42)
        obs_counts, obs_total, obs_fine, obs_fine_n = simulate_observations_from_gt(test_data)

        # Old approach: HGB + multiplicative calibration
        score_old = predict_old_approach(models, test_data, obs_counts, obs_total)

        # New approach: pure observations (no HGB)
        score_new_pure = predict_new_approach(None, test_data, obs_counts, obs_total, obs_fine, obs_fine_n,
                                               hgb_weight=0.0, temperature=TEMPERATURE_NEW)

        # New approach: observations + HGB prior blend
        score_new_blend = predict_new_approach(models, test_data, obs_counts, obs_total, obs_fine, obs_fine_n,
                                                hgb_weight=HGB_PRIOR_WEIGHT, temperature=TEMPERATURE_NEW)

        # New approach: pure obs, no temperature
        score_new_notemp = predict_new_approach(None, test_data, obs_counts, obs_total, obs_fine, obs_fine_n,
                                                  hgb_weight=0.0, temperature=1.0)

        results_old.append(score_old)
        results_new_pure.append(score_new_pure)
        results_new_blend.append(score_new_blend)
        results_new_notemp.append(score_new_notemp)

        print(f"{rname:4s}: old={score_old:6.2f}  new_pure={score_new_pure:6.2f}  new_blend={score_new_blend:6.2f}  new_notemp={score_new_notemp:6.2f}")

    print(f"\n{'='*80}")
    print(f"{'AVG':4s}: old={np.mean(results_old):6.2f}  new_pure={np.mean(results_new_pure):6.2f}  new_blend={np.mean(results_new_blend):6.2f}  new_notemp={np.mean(results_new_notemp):6.2f}")
    print(f"{'MIN':4s}: old={np.min(results_old):6.2f}  new_pure={np.min(results_new_pure):6.2f}  new_blend={np.min(results_new_blend):6.2f}  new_notemp={np.min(results_new_notemp):6.2f}")
    print(f"{'MAX':4s}: old={np.max(results_old):6.2f}  new_pure={np.max(results_new_pure):6.2f}  new_blend={np.max(results_new_blend):6.2f}  new_notemp={np.max(results_new_notemp):6.2f}")

    # Also test different HGB weights and temperatures
    print(f"\n{'='*80}")
    print("PARAMETER SWEEP (new approach):")
    print(f"{'Weight':>8s} {'Temp':>6s} {'AVG':>8s} {'MIN':>8s}")
    for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        for t in [1.0, 1.05, 1.10, 1.15, 1.20]:
            scores = []
            for test_rid in rids:
                train_data = [all_gt[r] for r in rids if r != test_rid]
                test_data = all_gt[test_rid]
                models = train_hgb(train_data) if w > 0 else None
                np.random.seed(42)
                obs_c, obs_t, obs_f, obs_fn = simulate_observations_from_gt(test_data)
                s = predict_new_approach(models, test_data, obs_c, obs_t, obs_f, obs_fn,
                                         hgb_weight=w, temperature=t)
                scores.append(s)
            avg = np.mean(scores)
            mn = np.min(scores)
            print(f"{w:8.2f} {t:6.2f} {avg:8.2f} {mn:8.2f}")


if __name__ == "__main__":
    main()
