"""Fast clip range test - reuses cached HGB predictions."""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
import time

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


t0 = time.time()
print("=== CLIP RANGE SWEEP (cached HGB) ===\n")
rounds = load_all()
n_rounds = len(rounds)
vp_size = 15


def grid_positions(dim):
    n = max(1, -(-dim // vp_size))
    if n == 1:
        return [0]
    step = (dim - vp_size) / (n - 1)
    return [round(i * step) for i in range(n)]


# Step 1: Pre-train LOO HGB models and cache predictions
print("[1] Pre-training LOO HGB models...", flush=True)
cached = {}
for hold in range(n_rounds):
    rid = rounds[hold][0][:8]
    _, test_seeds = rounds[hold]
    train = [s for i, (_, ss) in enumerate(rounds) if i != hold for s in ss]

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

    grids = [ig for ig, _ in test_seeds]
    gts = [gt for _, gt in test_seeds]
    H, W = grids[0].shape
    rows = grid_positions(H)
    cols = grid_positions(W)
    vps = [(r, c) for r in rows for c in cols]

    preds = {}
    for si in range(len(grids)):
        Xt = extract_features(grids[si])
        p = np.column_stack([m.predict(Xt) for m in models])
        p = np.clip(p, CLIP, None)
        p /= p.sum(axis=-1, keepdims=True)
        preds[si] = p.reshape(H, W, NC)

    cached[hold] = {
        "grids": grids,
        "gts": gts,
        "preds": preds,
        "vps": vps,
        "H": H,
        "W": W,
    }
    score_no_obs = np.mean([kl_score(gts[si], preds[si]) for si in range(len(grids))])
    print(f"  Fold {hold} ({rid}): {score_no_obs:.2f}", flush=True)

print(f"  Done in {time.time()-t0:.0f}s\n")

# Step 2: Test clip ranges with MC sampling
print("[2] Testing clip ranges (5 MC each):\n", flush=True)

configs = [
    (0.5, 2.0),
    (0.3, 3.0),
    (0.2, 5.0),
    (0.1, 10.0),
    (0.05, 20.0),
    (0.01, 100.0),
    (0.0, 1000.0),  # effectively no clip
]

for clip_lo, clip_hi in configs:
    mc_scores = []
    for mc in range(5):
        rng = np.random.default_rng(mc * 1000 + 42)
        fold_scores = []
        for hold in range(n_rounds):
            c = cached[hold]
            grids, gts, preds = c["grids"], c["gts"], c["preds"]
            vps = c["vps"]
            H, W = c["H"], c["W"]
            n_seeds = len(grids)

            # Simulate observations
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

            # Calibrate
            calibrated = {}
            for si in range(n_seeds):
                pred = preds[si].copy().reshape(-1, NC)
                cls = build_class_grid(grids[si]).ravel()
                for ic in range(NC):
                    n = per_cls_n[ic]
                    if n < 10:
                        continue
                    of = per_cls_obs[ic] / n
                    pa = per_cls_pred[ic] / n
                    r = np.where(pa > 0.01, np.clip(of / pa, clip_lo, clip_hi), 1.0)
                    pred[cls == ic] *= r
                pred = np.clip(pred, CLIP, None)
                pred /= pred.sum(axis=-1, keepdims=True)
                calibrated[si] = pred.reshape(H, W, NC)

            fold_score = np.mean([kl_score(gts[si], calibrated[si]) for si in range(n_seeds)])
            fold_scores.append(fold_score)
        mc_scores.append(np.mean(fold_scores))

    avg = np.mean(mc_scores)
    std = np.std(mc_scores)
    per_fold = ""
    # Show per-fold detail for first MC
    if True:
        rng = np.random.default_rng(42)
        for hold in range(n_rounds):
            c = cached[hold]
            grids, gts, preds = c["grids"], c["gts"], c["preds"]
            vps = c["vps"]
            H, W = c["H"], c["W"]
            n_seeds = len(grids)
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
            calibrated = {}
            for si in range(n_seeds):
                pred = preds[si].copy().reshape(-1, NC)
                cls_flat = build_class_grid(grids[si]).ravel()
                for ic in range(NC):
                    n = per_cls_n[ic]
                    if n < 10:
                        continue
                    of_v = per_cls_obs[ic] / n
                    pa_v = per_cls_pred[ic] / n
                    r_v = np.where(pa_v > 0.01, np.clip(of_v / pa_v, clip_lo, clip_hi), 1.0)
                    pred[cls_flat == ic] *= r_v
                pred = np.clip(pred, CLIP, None)
                pred /= pred.sum(axis=-1, keepdims=True)
                calibrated[si] = pred.reshape(H, W, NC)
            fs = np.mean([kl_score(gts[si], calibrated[si]) for si in range(n_seeds)])
            per_fold += f" {fs:.1f}"

    print(f"  clip=[{clip_lo:.2f}, {clip_hi:>6.1f}]: {avg:.2f} ±{std:.2f} | folds:{per_fold}", flush=True)

print(f"\nTotal time: {time.time()-t0:.0f}s")
