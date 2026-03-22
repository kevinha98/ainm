"""Test wider clip ranges for per-class multiplicative calibration.
R6 has extreme dynamics (66× ratio for mountains) that get clipped to 3×.
"""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("data")
NC = 6
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
CLIP = 0.0001


def build_class_grid(ig):
    cls = np.zeros_like(ig)
    for raw, c in GRID_TO_CLASS.items():
        cls[ig == raw] = c
    return cls


def extract_features(ig):
    cls = build_class_grid(ig)
    H, W = ig.shape
    ocean = ig == 10; mountain = ig == 5
    settlement = cls == 1; forest = cls == 4; empty = cls == 0
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H, W), 20)
    k3, k7, k11 = np.ones((3, 3)), np.ones((7, 7)), np.ones((11, 11))
    cls_oh = np.zeros((H, W, NC))
    for c in range(NC):
        cls_oh[:, :, c] = (cls == c).astype(float)
    return np.concatenate([
        cls_oh, dist_ocean[:, :, None], dist_settle[:, :, None],
        dist_forest[:, :, None], dist_mountain[:, :, None],
        ndimage.convolve(settlement.astype(float), k3, mode="constant")[:, :, None],
        ndimage.convolve(settlement.astype(float), k7, mode="constant")[:, :, None],
        ndimage.convolve(forest.astype(float), k7, mode="constant")[:, :, None],
        ndimage.convolve(ocean.astype(float), k7, mode="constant")[:, :, None],
        ndimage.convolve(empty.astype(float), k7, mode="constant")[:, :, None],
        ndimage.convolve(settlement.astype(float), k11, mode="constant")[:, :, None],
        is_coast[:, :, None].astype(float),
    ], axis=-1).reshape(-1, 17)


def kl_score(gt, pred):
    g = np.clip(gt.reshape(-1, NC), 1e-10, None)
    p = np.clip(pred.reshape(-1, NC), 1e-10, None)
    p /= p.sum(axis=-1, keepdims=True)
    g /= g.sum(axis=-1, keepdims=True)
    return 100 * np.exp(-np.mean(np.sum(g * np.log(g / p), axis=-1)))


def sample_obs(gt, rng):
    """Sample one observation from GT distribution for each cell."""
    flat = np.clip(gt.reshape(-1, NC), 1e-10, None)
    flat /= flat.sum(axis=-1, keepdims=True)
    cumsum = np.cumsum(flat, axis=-1)
    u = rng.random(len(flat))
    return (u[:, None] < cumsum).argmax(axis=-1).reshape(gt.shape[:2])


def load_all():
    rounds = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        rid = gf.stem.replace("ground_truth_", "")
        with open(gf) as f:
            data = json.load(f)
        seeds = []
        for si in sorted(data.keys()):
            gt = np.array(data[si].get("ground_truth", []))
            ig = np.array(data[si].get("initial_grid", []))
            if gt.size > 0 and ig.size > 0:
                seeds.append((ig, gt))
        rounds.append((rid, seeds))
    return rounds


def test_clip_range(rounds, clip_lo, clip_hi, rng, n_mc=5):
    """LOO CV with per-class mult calibration at given clip range."""
    scores_all = []
    for mc in range(n_mc):
        scores_per_round = []
        for hold in range(len(rounds)):
            _, test_seeds = rounds[hold]
            train = [s for i, (_, ss) in enumerate(rounds) if i != hold for s in ss]

            # Train HGB
            X, Y = [], []
            for ig, gt in train:
                X.append(extract_features(ig))
                Y.append(gt.reshape(-1, NC))
            X, Y = np.vstack(X), np.vstack(Y)
            models = [
                HistGradientBoostingRegressor(
                    max_iter=100, max_depth=4, learning_rate=0.05,
                    min_samples_leaf=50, random_state=42,
                ).fit(X, Y[:, c])
                for c in range(NC)
            ]

            # Predict
            grids = [ig for ig, _ in test_seeds]
            gts = [gt for _, gt in test_seeds]
            n_seeds = len(grids)
            H, W = grids[0].shape

            preds = {}
            for si in range(n_seeds):
                Xt = extract_features(grids[si])
                p = np.column_stack([m.predict(Xt) for m in models])
                p = np.clip(p, CLIP, None)
                p /= p.sum(axis=-1, keepdims=True)
                preds[si] = p.reshape(H, W, NC)

            # Simulate observations (all seeds, 9 viewports each)
            vp_size = 15
            def grid_positions(dim):
                n = max(1, -(-dim // vp_size))
                if n == 1:
                    return [0]
                step = (dim - vp_size) / (n - 1)
                return [round(i * step) for i in range(n)]

            rows = grid_positions(H)
            cols = grid_positions(W)
            vps = [(r, c) for r in rows for c in cols]

            per_cls_obs = np.zeros((NC, NC))
            per_cls_pred = np.zeros((NC, NC))
            per_cls_n = np.zeros(NC)

            obs_used = 0
            for si in range(n_seeds):
                obs_grid = sample_obs(gts[si], rng)
                for row, col in vps:
                    if obs_used >= 45:
                        break
                    cls = build_class_grid(grids[si])
                    for vy in range(min(vp_size, H - row)):
                        for vx in range(min(vp_size, W - col)):
                            gy, gx = row + vy, col + vx
                            if gy >= H or gx >= W:
                                continue
                            ic = cls[gy, gx]
                            oc = obs_grid[gy, gx]
                            per_cls_obs[ic, oc] += 1
                            per_cls_pred[ic] += preds[si][gy, gx]
                            per_cls_n[ic] += 1
                    obs_used += 1
                if obs_used >= 45:
                    break

            # Calibrate with given clip range
            calibrated = {}
            for si in range(n_seeds):
                pred = preds[si].copy().reshape(-1, NC)
                cls = build_class_grid(grids[si]).ravel()
                for ic in range(NC):
                    n = per_cls_n[ic]
                    if n < 10:
                        continue
                    obs_freq = per_cls_obs[ic] / n
                    pred_avg = per_cls_pred[ic] / n
                    ratio = np.where(
                        pred_avg > 0.01,
                        np.clip(obs_freq / pred_avg, clip_lo, clip_hi),
                        1.0,
                    )
                    pred[cls == ic] *= ratio
                pred = np.clip(pred, CLIP, None)
                pred /= pred.sum(axis=-1, keepdims=True)
                calibrated[si] = pred.reshape(H, W, NC)

            round_score = np.mean([kl_score(gts[si], calibrated[si]) for si in range(n_seeds)])
            scores_per_round.append(round_score)
        scores_all.append(np.mean(scores_per_round))
    return np.mean(scores_all), np.std(scores_all)


if __name__ == "__main__":
    print("=== CLIP RANGE SENSITIVITY TEST ===\n")
    rounds = load_all()
    rng = np.random.default_rng(42)

    configs = [
        (0.3, 3.0),    # current
        (0.2, 5.0),    # wider
        (0.1, 10.0),   # much wider
        (0.05, 20.0),  # very wide
        (0.01, 100.0), # extreme
        (0.5, 2.0),    # narrower
    ]

    for lo, hi in configs:
        avg, std = test_clip_range(rounds, lo, hi, rng, n_mc=3)
        print(f"  clip=[{lo}, {hi:>5.1f}]: {avg:.2f} ±{std:.2f}")
