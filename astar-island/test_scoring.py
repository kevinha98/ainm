"""Test alternative scoring formulas to match server scores.
Hypothesis: score = mean_cells(100 * exp(-KL_cell)) instead of 
            score = 100 * exp(-mean_cells(KL_cell)).
Due to Jensen's inequality, the first gives lower scores.
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
    for c in range(NC): cls_oh[:, :, c] = (cls == c).astype(float)
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


def per_cell_kl(gt, pred):
    """Return per-cell KL divergences."""
    g = np.clip(gt.reshape(-1, NC), 1e-10, None)
    p = np.clip(pred.reshape(-1, NC), 1e-10, None)
    p /= p.sum(axis=-1, keepdims=True)
    g /= g.sum(axis=-1, keepdims=True)
    return np.sum(g * np.log(g / p), axis=-1)


# Load R5 GT and R1-R4 for training
gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
X_train, Y_train = [], []
for gf in gt_files:
    if "fd3c92ff" in gf.name:
        continue
    with open(gf) as f:
        data = json.load(f)
    for si in sorted(data.keys()):
        ig = np.array(data[si]["initial_grid"])
        gt = np.array(data[si]["ground_truth"])
        X_train.append(extract_features(ig))
        Y_train.append(gt.reshape(-1, 6))
X_train, Y_train = np.vstack(X_train), np.vstack(Y_train)

models = [
    HistGradientBoostingRegressor(
        max_iter=100, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    ).fit(X_train, Y_train[:, c])
    for c in range(NC)
]

with open(DATA_DIR / "ground_truth_fd3c92ff.json") as f:
    r5_data = json.load(f)

print("=== Alternative Scoring Formulas for R5 ===\n")
print("Server scores: [79.47, 77.08, 80.19, 81.42, 82.18], avg=80.07\n")

formulas = {}
for si in sorted(r5_data.keys()):
    ig = np.array(r5_data[si]["initial_grid"])
    gt = np.array(r5_data[si]["ground_truth"])
    X = extract_features(ig)
    pred = np.column_stack([m.predict(X) for m in models])
    pred = np.clip(pred, CLIP, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    
    kl_cells = per_cell_kl(gt, pred.reshape(ig.shape[0], ig.shape[1], NC))

    # Formula 1: 100 * exp(-mean(KL))  [our current]
    s1 = 100 * np.exp(-kl_cells.mean())
    
    # Formula 2: mean(100 * exp(-KL))  [per-cell aggregate]
    s2 = np.mean(100 * np.exp(-kl_cells))
    
    # Formula 3: 100 * exp(-median(KL))
    s3 = 100 * np.exp(-np.median(kl_cells))
    
    # Formula 4: geometric mean of per-cell scores
    s4 = np.exp(np.mean(np.log(np.clip(100 * np.exp(-kl_cells), 1e-10, None))))
    
    # Formula 5: 100 - 100 * mean(KL) (linear)
    s5 = 100 - 100 * kl_cells.mean()
    
    # Formula 6: 100 / (1 + mean(KL))
    s6 = 100 / (1 + kl_cells.mean())
    
    # Formula 7: KL normalized differently
    # Maybe: 100 * exp(-sum_cells_KL * 6 / n_cells)
    s7 = 100 * np.exp(-kl_cells.sum() * NC / len(kl_cells))
    
    # Formula 8: RMSE based
    g = np.clip(gt.reshape(-1, NC), 1e-10, None)
    g /= g.sum(axis=-1, keepdims=True)
    p = pred.reshape(-1, NC)
    rmse = np.sqrt(np.mean((g - p) ** 2))
    s8 = 100 * (1 - rmse)
    
    # Formula 9: Brier score
    brier = np.mean(np.sum((g - p) ** 2, axis=-1))
    s9 = 100 * (1 - brier / 2)  # max brier = 2 for 6 classes
    
    # Formula 10: Cosine similarity
    cos_sim = np.mean(np.sum(g * p, axis=-1) / (np.sqrt(np.sum(g**2, axis=-1)) * np.sqrt(np.sum(p**2, axis=-1))))
    s10 = 100 * cos_sim
    
    for name, val in [("f1:exp(-mean_kl)", s1), ("f2:mean(exp(-kl))", s2),
                       ("f3:exp(-median_kl)", s3), ("f4:geomean(exp(-kl))", s4),
                       ("f5:100-100*kl", s5), ("f6:100/(1+kl)", s6),
                       ("f7:exp(-6*mean_kl)", s7), ("f8:100*(1-rmse)", s8),
                       ("f9:brier", s9), ("f10:cosine", s10)]:
        if name not in formulas:
            formulas[name] = []
        formulas[name].append(val)

# Print results vs server
server_seeds = [79.47, 77.08, 80.19, 81.42, 82.18]
server_avg = np.mean(server_seeds)

print(f"{'Formula':<30s} {'Seeds':>40s} {'Avg':>8s} {'Server':>8s} {'Err':>8s}")
print("-" * 100)
for name, vals in formulas.items():
    avg = np.mean(vals)
    err = avg - server_avg
    seed_str = ", ".join(f"{v:.2f}" for v in vals)
    print(f"{name:<30s} {seed_str:>40s} {avg:>8.2f} {server_avg:>8.2f} {err:>+8.2f}")

# Find best match
print("\n--- Best match to server per-seed scores ---")
for name, vals in sorted(formulas.items(), key=lambda x: sum((a-b)**2 for a,b in zip(x[1], server_seeds))):
    mse = np.mean([(a-b)**2 for a, b in zip(vals, server_seeds)])
    print(f"  {name:<30s} MSE={mse:.4f} seeds={[round(v,2) for v in vals]}")
