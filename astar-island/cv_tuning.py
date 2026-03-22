"""Cross-round LOO: Test Dirichlet smoothing and KL-aware clipping.
The KL divergence is asymmetric: GT * log(GT/pred) — being wrong on high-GT cells
hurts more. Dirichlet smoothing may help by avoiding extreme predictions.
"""
import json, time
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

def kl_score(pred, gt):
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
    empty = (cls == 0)
    
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H,W), 20)
    
    k3 = np.ones((3, 3))
    k7 = np.ones((7, 7))
    k11 = np.ones((11, 11))
    
    n_settle_3 = ndimage.convolve(settlement.astype(float), k3, mode='constant')
    n_settle_7 = ndimage.convolve(settlement.astype(float), k7, mode='constant')
    n_forest_7 = ndimage.convolve(forest.astype(float), k7, mode='constant')
    n_ocean_7 = ndimage.convolve(ocean.astype(float), k7, mode='constant')
    n_empty_7 = ndimage.convolve(empty.astype(float), k7, mode='constant')
    n_settle_11 = ndimage.convolve(settlement.astype(float), k11, mode='constant')
    
    cls_onehot = np.zeros((H, W, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        cls_onehot[:, :, c] = (cls == c).astype(float)
    
    features = np.concatenate([
        cls_onehot,
        dist_ocean[:, :, None], dist_settle[:, :, None],
        dist_forest[:, :, None], dist_mountain[:, :, None],
        n_settle_3[:, :, None], n_settle_7[:, :, None],
        n_forest_7[:, :, None], n_ocean_7[:, :, None],
        n_empty_7[:, :, None], n_settle_11[:, :, None],
        is_coast[:, :, None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1]), cls.ravel()

# Precompute
round_data = {}
for rname, rdata in all_gt.items():
    entries = []
    for si_str in sorted(rdata.keys()):
        gt = np.array(rdata[si_str]['ground_truth'])
        ig = np.array(rdata[si_str]['initial_grid'])
        X, cls = extract_features(ig)
        Y = gt.reshape(-1, 6)
        entries.append((X, Y, cls))
    round_data[rname] = entries

round_names = sorted(round_data.keys())

# Sweep min_clip values — this directly affects KL score
print("=== SWEEP: Clip floor for HGB predictions ===\n", flush=True)
clip_values = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

for clip_val in clip_values:
    round_scores = []
    
    for test_round in round_names:
        train_rounds = [r for r in round_names if r != test_round]
        X_train = np.vstack([e[0] for r in train_rounds for e in round_data[r]])
        Y_train = np.vstack([e[1] for r in train_rounds for e in round_data[r]])
        
        models = []
        for c in range(NUM_CLASSES):
            m = HistGradientBoostingRegressor(
                max_iter=100, max_depth=4, learning_rate=0.05,
                min_samples_leaf=50, random_state=42
            )
            m.fit(X_train, Y_train[:, c])
            models.append(m)
        
        scores = []
        for X_test, Y_test, cls_test in round_data[test_round]:
            gt = Y_test.reshape(40, 40, 6)
            pred = np.column_stack([m.predict(X_test) for m in models])
            pred = np.clip(pred, clip_val, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            pred = pred.reshape(40, 40, 6)
            scores.append(kl_score(pred, gt))
        round_scores.append(np.mean(scores))
    
    avg = np.mean(round_scores)
    print(f"  clip={clip_val:<8.4f}  avg={avg:.3f}  {' '.join(f'{s:.1f}' for s in round_scores)}", flush=True)

# Sweep Dirichlet-style smoothing: pred = (1-alpha)*hgb + alpha*uniform
print("\n=== SWEEP: Dirichlet smoothing (blend with uniform) ===\n", flush=True)
alpha_values = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

for alpha in alpha_values:
    round_scores = []
    
    for test_round in round_names:
        train_rounds = [r for r in round_names if r != test_round]
        X_train = np.vstack([e[0] for r in train_rounds for e in round_data[r]])
        Y_train = np.vstack([e[1] for r in train_rounds for e in round_data[r]])
        
        models = []
        for c in range(NUM_CLASSES):
            m = HistGradientBoostingRegressor(
                max_iter=100, max_depth=4, learning_rate=0.05,
                min_samples_leaf=50, random_state=42
            )
            m.fit(X_train, Y_train[:, c])
            models.append(m)
        
        scores = []
        for X_test, Y_test, cls_test in round_data[test_round]:
            gt = Y_test.reshape(40, 40, 6)
            pred = np.column_stack([m.predict(X_test) for m in models])
            pred = np.clip(pred, 0.001, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            # Blend with uniform
            uniform = np.ones_like(pred) / 6
            pred = (1 - alpha) * pred + alpha * uniform
            pred = pred.reshape(40, 40, 6)
            scores.append(kl_score(pred, gt))
        round_scores.append(np.mean(scores))
    
    avg = np.mean(round_scores)
    print(f"  alpha={alpha:<6.2f}  avg={avg:.3f}  {' '.join(f'{s:.1f}' for s in round_scores)}", flush=True)

# Sweep blending with per-class average
print("\n=== SWEEP: Blend HGB with per-class average ===\n", flush=True)
blend_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

for blend in blend_values:
    round_scores = []
    
    for test_round in round_names:
        train_rounds = [r for r in round_names if r != test_round]
        X_train = np.vstack([e[0] for r in train_rounds for e in round_data[r]])
        Y_train = np.vstack([e[1] for r in train_rounds for e in round_data[r]])
        cls_train = np.concatenate([e[2] for r in train_rounds for e in round_data[r]])
        
        avg_class = {}
        for c in range(NUM_CLASSES):
            mask = cls_train == c
            avg_class[c] = Y_train[mask].mean(axis=0) if mask.any() else np.ones(6)/6
        
        models = []
        for c in range(NUM_CLASSES):
            m = HistGradientBoostingRegressor(
                max_iter=100, max_depth=4, learning_rate=0.05,
                min_samples_leaf=50, random_state=42
            )
            m.fit(X_train, Y_train[:, c])
            models.append(m)
        
        scores = []
        for X_test, Y_test, cls_test in round_data[test_round]:
            gt = Y_test.reshape(40, 40, 6)
            pred_hgb = np.column_stack([m.predict(X_test) for m in models])
            pred_avg = np.zeros_like(pred_hgb)
            for c in range(NUM_CLASSES):
                pred_avg[cls_test == c] = avg_class[c]
            pred = (1-blend) * pred_hgb + blend * pred_avg
            pred = np.clip(pred, 0.001, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            pred = pred.reshape(40, 40, 6)
            scores.append(kl_score(pred, gt))
        round_scores.append(np.mean(scores))
    
    avg = np.mean(round_scores)
    print(f"  blend={blend:<6.2f}  avg={avg:.3f}  {' '.join(f'{s:.1f}' for s in round_scores)}", flush=True)
