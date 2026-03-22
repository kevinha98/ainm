"""Verify R5 score gap: why LOO CV gives 94.23 but actual was 80.07.
Reproduce the exact R5 submission and score against our R5 GT."""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("data")
NUM_CLASSES = 6
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
    ocean = ig == 10
    mountain = ig == 5
    settlement = cls == 1
    forest = cls == 4
    empty = cls == 0
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H, W), 20)
    k3, k7, k11 = np.ones((3, 3)), np.ones((7, 7)), np.ones((11, 11))
    n_s3 = ndimage.convolve(settlement.astype(float), k3, mode="constant")
    n_s7 = ndimage.convolve(settlement.astype(float), k7, mode="constant")
    n_f7 = ndimage.convolve(forest.astype(float), k7, mode="constant")
    n_o7 = ndimage.convolve(ocean.astype(float), k7, mode="constant")
    n_e7 = ndimage.convolve(empty.astype(float), k7, mode="constant")
    n_s11 = ndimage.convolve(settlement.astype(float), k11, mode="constant")
    cls_oh = np.zeros((H, W, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        cls_oh[:, :, c] = (cls == c).astype(float)
    features = np.concatenate(
        [
            cls_oh,
            dist_ocean[:, :, None],
            dist_settle[:, :, None],
            dist_forest[:, :, None],
            dist_mountain[:, :, None],
            n_s3[:, :, None],
            n_s7[:, :, None],
            n_f7[:, :, None],
            n_o7[:, :, None],
            n_e7[:, :, None],
            n_s11[:, :, None],
            is_coast[:, :, None].astype(float),
        ],
        axis=-1,
    )
    return features.reshape(-1, features.shape[-1])


def kl_score(gt, pred):
    gt = np.clip(gt, 1e-10, None)
    pred = np.clip(pred, 1e-10, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt = gt / gt.sum(axis=-1, keepdims=True)
    kl = np.sum(gt * np.log(gt / pred), axis=-1)
    return 100 * np.exp(-kl.mean())


# Load all GT files and identify rounds
gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
print(f"Found {len(gt_files)} GT files:")
for gf in gt_files:
    with open(gf) as f:
        data = json.load(f)
    n_seeds = len(data)
    first_ig = np.array(data["0"]["initial_grid"])
    print(f"  {gf.name}: {n_seeds} seeds, grid={first_ig.shape}")

# Load R5 GT (fd3c92ff)
r5_file = DATA_DIR / "ground_truth_fd3c92ff.json"
with open(r5_file) as f:
    r5_data = json.load(f)

r5_seeds = []
for si_str in sorted(r5_data.keys()):
    gt = np.array(r5_data[si_str]["ground_truth"])
    ig = np.array(r5_data[si_str]["initial_grid"])
    r5_seeds.append((ig, gt))
print(f"\nR5: {len(r5_seeds)} seeds, grid={r5_seeds[0][0].shape}")

# Train HGB on ONLY R1-R4 (exclude R5)
print("\nTraining HGB on R1-R4 only (exclude fd3c92ff)...")
X_all, Y_all = [], []
for gf in gt_files:
    if "fd3c92ff" in gf.name:
        print(f"  Skipping {gf.name} (R5)")
        continue
    print(f"  Loading {gf.name}")
    with open(gf) as f:
        data = json.load(f)
    for si_str in sorted(data.keys()):
        gt = np.array(data[si_str].get("ground_truth", []))
        ig = np.array(data[si_str].get("initial_grid", []))
        if gt.size == 0 or ig.size == 0:
            continue
        X_all.append(extract_features(ig))
        Y_all.append(gt.reshape(-1, 6))

X_all = np.vstack(X_all)
Y_all = np.vstack(Y_all)
print(f"Training data: {len(X_all)} cells")

models = []
for c in range(NUM_CLASSES):
    m = HistGradientBoostingRegressor(
        max_iter=100, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    )
    m.fit(X_all, Y_all[:, c])
    models.append(m)
print("Models trained.")

# Predict R5 and score
print("\n--- R5 prediction vs cached GT ---")
scores = []
for si, (ig, gt) in enumerate(r5_seeds):
    X = extract_features(ig)
    pred = np.column_stack([m.predict(X) for m in models])
    pred = np.clip(pred, CLIP, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    pred_grid = pred.reshape(ig.shape[0], ig.shape[1], 6)
    s = kl_score(gt, pred_grid)
    scores.append(s)
    print(f"  Seed {si}: KL score = {s:.2f}")

avg = np.mean(scores)
print(f"\n  Average R5 score (vs cached GT): {avg:.2f}")
print(f"  Actual R5 score (server): 80.07")
print(f"  Gap: {avg - 80.07:.2f}")

# Also check: what if we apply the extra clip + renormalize that run_v10.py does on submission?
print("\n--- With submission-style clip (1e-6) ---")
scores2 = []
for si, (ig, gt) in enumerate(r5_seeds):
    X = extract_features(ig)
    pred = np.column_stack([m.predict(X) for m in models])
    pred = np.clip(pred, CLIP, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    # Extra submission clip
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    pred_grid = pred.reshape(ig.shape[0], ig.shape[1], 6)
    s = kl_score(gt, pred_grid)
    scores2.append(s)
    print(f"  Seed {si}: KL score = {s:.2f}")

avg2 = np.mean(scores2)
print(f"  Average: {avg2:.2f}")

# Check R5 GT statistics
print("\n--- R5 GT statistics ---")
for si, (ig, gt) in enumerate(r5_seeds):
    cls = build_class_grid(ig)
    print(f"\n  Seed {si}:")
    for c in range(NUM_CLASSES):
        mask = cls == c
        n = mask.sum()
        if n > 0:
            avg_gt = gt[mask].mean(axis=0)
            ent = -np.sum(avg_gt * np.log(np.clip(avg_gt, 1e-10, None)))
            print(f"    Class {c} (n={n:4d}): avg_gt={np.round(avg_gt, 3).tolist()} ent={ent:.3f}")
