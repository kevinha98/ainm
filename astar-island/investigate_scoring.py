"""Investigate scoring formula. 
Compare our local KL scoring against known server scores.
"""
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
    ocean = ig == 10; mountain = ig == 5
    settlement = cls == 1; forest = cls == 4; empty = cls == 0
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
    for c in range(NUM_CLASSES): cls_oh[:, :, c] = (cls == c).astype(float)
    return np.concatenate([
        cls_oh, dist_ocean[:, :, None], dist_settle[:, :, None],
        dist_forest[:, :, None], dist_mountain[:, :, None],
        n_s3[:, :, None], n_s7[:, :, None], n_f7[:, :, None],
        n_o7[:, :, None], n_e7[:, :, None], n_s11[:, :, None],
        is_coast[:, :, None].astype(float),
    ], axis=-1).reshape(-1, 17)


def score_kl(gt, pred):
    """KL divergence scoring: 100 * exp(-mean_KL)."""
    gt = np.clip(gt, 1e-10, None)
    pred = np.clip(pred, 1e-10, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt = gt / gt.sum(axis=-1, keepdims=True)
    kl = np.sum(gt * np.log(gt / pred), axis=-1)
    return 100 * np.exp(-kl.mean())


def score_ce(gt, pred):
    """Cross-entropy scoring: 100 * exp(-mean_CE)."""
    gt = np.clip(gt, 1e-10, None)
    pred = np.clip(pred, 1e-10, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt = gt / gt.sum(axis=-1, keepdims=True)
    ce = -np.sum(gt * np.log(pred), axis=-1)
    return 100 * np.exp(-ce.mean())


def score_kl_sum(gt, pred):
    """KL with sum (not mean) over cells: 100 * exp(-sum_KL / N)."""
    gt = np.clip(gt, 1e-10, None)
    pred = np.clip(pred, 1e-10, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt = gt / gt.sum(axis=-1, keepdims=True)
    kl = np.sum(gt * np.log(gt / pred), axis=-1)
    return 100 * np.exp(-kl.sum() / gt.shape[0])


# Load round data
print("Loading rounds...")
import sys
sys.path.insert(0, ".")
from src.api import AstarAPI

api = AstarAPI()
my_rounds = api.get_my_rounds()

# Map round IDs to file names
round_map = {}
for mr in my_rounds:
    rn = mr.get("round_number")
    rid = mr.get("id")
    score = mr.get("round_score")
    seed_scores = mr.get("seed_scores", [])
    round_map[rn] = {
        "id": rid,
        "server_score": score,
        "seed_scores": seed_scores,
    }

# Now test: load R2 GT, compute per-class avg from R1 only, score it
print("\n=== Reproducing R2 server score ===")
print("R2 was submitted with v7 per-class avg trained on R1 only")
r1_file = DATA_DIR / "ground_truth_71451d74.json"  # R1
r2_file = DATA_DIR / "ground_truth_76909e29.json"  # R2

with open(r1_file) as f:
    r1_data = json.load(f)
with open(r2_file) as f:
    r2_data = json.load(f)

# Compute per-class avg from R1
all_cls, all_gt = [], []
for si_str in sorted(r1_data.keys()):
    ig = np.array(r1_data[si_str]["initial_grid"])
    gt = np.array(r1_data[si_str]["ground_truth"])
    all_cls.append(build_class_grid(ig).ravel())
    all_gt.append(gt.reshape(-1, 6))
all_cls = np.concatenate(all_cls)
all_gt = np.vstack(all_gt)

avgs = {}
for c in range(NUM_CLASSES):
    mask = all_cls == c
    avgs[c] = all_gt[mask].mean(axis=0) if mask.any() else np.ones(6) / 6

# Test on R2
print("\nPer-class avg (from R1 only) on R2:")
r2_scores_kl = []
r2_scores_ce = []
for si_str in sorted(r2_data.keys()):
    ig = np.array(r2_data[si_str]["initial_grid"])
    gt = np.array(r2_data[si_str]["ground_truth"])
    cls = build_class_grid(ig)
    pred = np.zeros_like(gt)
    for c in range(NUM_CLASSES):
        pred[cls == c] = avgs[c]
    pred = np.clip(pred, CLIP, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    # Extra submission clip
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)

    s_kl = score_kl(gt, pred)
    s_ce = score_ce(gt, pred)
    r2_scores_kl.append(s_kl)
    r2_scores_ce.append(s_ce)
    print(f"  Seed {si_str}: KL={s_kl:.2f} CE={s_ce:.2f}")

print(f"\n  Avg KL = {np.mean(r2_scores_kl):.2f}")
print(f"  Avg CE = {np.mean(r2_scores_ce):.2f}")

r2_server = round_map.get(2, {})
print(f"  Server score: {r2_server.get('server_score')}")
if r2_server.get("seed_scores"):
    print(f"  Server seeds: {[round(s, 2) for s in r2_server['seed_scores']]}")

# Test R5 with HGB
print("\n=== Reproducing R5 server score ===")
print("R5 was submitted with HGB trained on R1-R4, no obs")

# Train on R1-R4
gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
X_train, Y_train = [], []
for gf in gt_files:
    if "fd3c92ff" in gf.name:
        continue
    with open(gf) as f:
        data = json.load(f)
    for si_str in sorted(data.keys()):
        gt = np.array(data[si_str]["ground_truth"])
        ig = np.array(data[si_str]["initial_grid"])
        X_train.append(extract_features(ig))
        Y_train.append(gt.reshape(-1, 6))
X_train, Y_train = np.vstack(X_train), np.vstack(Y_train)

models = [
    HistGradientBoostingRegressor(
        max_iter=100, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    ).fit(X_train, Y_train[:, c])
    for c in range(6)
]

r5_file = DATA_DIR / "ground_truth_fd3c92ff.json"
with open(r5_file) as f:
    r5_data = json.load(f)

print("\nHGB (R1-R4 trained) on R5:")
r5_scores_kl = []
r5_scores_ce = []
for si_str in sorted(r5_data.keys()):
    ig = np.array(r5_data[si_str]["initial_grid"])
    gt = np.array(r5_data[si_str]["ground_truth"])
    X = extract_features(ig)
    pred = np.column_stack([m.predict(X) for m in models])
    pred = np.clip(pred, CLIP, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    pred_grid = pred.reshape(ig.shape[0], ig.shape[1], 6)
    # With extra submission clip
    pred_sub = np.clip(pred_grid, 1e-6, None)
    pred_sub /= pred_sub.sum(axis=-1, keepdims=True)

    s_kl = score_kl(gt, pred_sub)
    s_ce = score_ce(gt, pred_sub)
    r5_scores_kl.append(s_kl)
    r5_scores_ce.append(s_ce)
    print(f"  Seed {si_str}: KL={s_kl:.2f} CE={s_ce:.2f}")

print(f"\n  Avg KL = {np.mean(r5_scores_kl):.2f}")
print(f"  Avg CE = {np.mean(r5_scores_ce):.2f}")

r5_server = round_map.get(5, {})
print(f"  Server score: {r5_server.get('server_score')}")
if r5_server.get("seed_scores"):
    print(f"  Server seeds: {[round(s, 2) for s in r5_server['seed_scores']]}")

# Also try: what formula gives server scores?
print("\n=== Formula reverse-engineering ===")
for rn, rdata in sorted(round_map.items()):
    server = rdata.get("server_score")
    if server is not None:
        kl_from_server = -np.log(server / 100)
        print(f"R{rn}: server={server} -> implied_mean_kl={kl_from_server:.4f}")
