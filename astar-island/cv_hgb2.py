"""Cross-round LOO: HGB regressor — optimized version."""
import json, time, sys
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from src.settings import DATA_DIR, NUM_CLASSES
from src.models import build_class_grid

gt_files = {
    "R1": DATA_DIR / "ground_truth_71451d74.json",
    "R2": DATA_DIR / "ground_truth_76909e29.json",
    "R3": DATA_DIR / "ground_truth_f1dac9a9.json",
    "R4": DATA_DIR / "ground_truth_8e839974.json",
}
all_gt = {}
for name, path in gt_files.items():
    with open(path) as f:
        all_gt[name] = json.load(f)

def score_pred(pred, gt):
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    kl = np.sum(gt_safe * np.log(gt_safe / (pred + 1e-15)), axis=-1).mean()
    return 100 * np.exp(-kl)

def extract_features(ig):
    cls = build_class_grid(ig)
    H, W = ig.shape
    ocean = (ig == 10)
    mountain = (ig == 5)
    settlement = (cls == 1)
    forest = (cls == 4)
    
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    
    k7 = np.ones((7, 7))
    k11 = np.ones((11, 11))
    n_settle_3 = ndimage.convolve(settlement.astype(float), k7, mode='constant')
    n_forest_3 = ndimage.convolve(forest.astype(float), k7, mode='constant')
    n_ocean_3 = ndimage.convolve(ocean.astype(float), k7, mode='constant')
    n_settle_5 = ndimage.convolve(settlement.astype(float), k11, mode='constant')
    
    cls_onehot = np.zeros((H, W, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        cls_onehot[:, :, c] = (cls == c).astype(float)
    
    features = np.concatenate([
        cls_onehot,
        dist_ocean[:, :, None],
        dist_settle[:, :, None],
        n_settle_3[:, :, None],
        n_forest_3[:, :, None],
        n_ocean_3[:, :, None],
        n_settle_5[:, :, None],
        is_coast[:, :, None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1]), cls.ravel()

# Precompute all features and targets
print("Precomputing features...", flush=True)
t0 = time.time()
round_data = {}  # round_name -> list of (X, Y, cls)
for rname, rdata in all_gt.items():
    entries = []
    for si_str in sorted(rdata.keys()):
        gt = np.array(rdata[si_str]['ground_truth'])
        ig = np.array(rdata[si_str]['initial_grid'])
        X, cls = extract_features(ig)
        Y = gt.reshape(-1, 6)
        entries.append((X, Y, cls))
    round_data[rname] = entries
print(f"Done in {time.time()-t0:.1f}s", flush=True)

print("\n=== CROSS-ROUND LOO: HGB REGRESSOR ===\n", flush=True)
round_names = sorted(round_data.keys())

for test_round in round_names:
    t1 = time.time()
    train_rounds = [r for r in round_names if r != test_round]
    
    X_train = np.vstack([e[0] for r in train_rounds for e in round_data[r]])
    Y_train = np.vstack([e[1] for r in train_rounds for e in round_data[r]])
    cls_train = np.concatenate([e[2] for r in train_rounds for e in round_data[r]])
    
    # Per-class averages as baseline
    avg_class = {}
    for c in range(NUM_CLASSES):
        mask = cls_train == c
        if mask.any():
            avg_class[c] = Y_train[mask].mean(axis=0)
        else:
            avg_class[c] = np.ones(6) / 6
    
    # Train HGB
    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, learning_rate=0.05,
            min_samples_leaf=50, random_state=42
        )
        m.fit(X_train, Y_train[:, c])
        models.append(m)
    
    # Test
    scores_hgb, scores_basic = [], []
    for X_test, Y_test, cls_test in round_data[test_round]:
        H = W = 40
        gt = Y_test.reshape(H, W, 6)
        
        pred_hgb = np.column_stack([m.predict(X_test) for m in models])
        pred_hgb = np.clip(pred_hgb, 0.002, None)
        pred_hgb /= pred_hgb.sum(axis=-1, keepdims=True)
        pred_hgb = pred_hgb.reshape(H, W, 6)
        
        pred_basic = np.zeros((H, W, 6))
        for c in range(NUM_CLASSES):
            mask = (cls_test.reshape(H, W) == c)
            pred_basic[mask] = avg_class.get(c, avg_class[0])
        pred_basic = np.clip(pred_basic, 0.002, None)
        pred_basic /= pred_basic.sum(axis=-1, keepdims=True)
        
        scores_hgb.append(score_pred(pred_hgb, gt))
        scores_basic.append(score_pred(pred_basic, gt))
    
    avg_h = np.mean(scores_hgb)
    avg_b = np.mean(scores_basic)
    print(f"{test_round}: basic={avg_b:.2f}  HGB={avg_h:.2f}  diff={avg_h-avg_b:+.2f}  ({time.time()-t1:.1f}s)", flush=True)

print(f"\n(binned approach scored: R1=92.9, R2=90.2, R3=81.7, R4=97.0, avg=90.4)", flush=True)
