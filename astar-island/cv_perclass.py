"""Cross-round LOO: Per-class HGB models (train separate HGB per initial class).
This avoids mixing gradients between very different distributions (ocean vs mountain).
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

def score_pred(pred, gt):
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    kl = np.sum(gt_safe * np.log(gt_safe / (pred + 1e-15)), axis=-1).mean()
    return 100 * np.exp(-kl)

def extract_spatial_features(ig):
    """Similar features but NO class one-hot (since we train per-class)."""
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
    k5 = np.ones((5, 5))
    k7 = np.ones((7, 7))
    
    n_settle_3 = ndimage.convolve(settlement.astype(float), k3, mode='constant')
    n_settle_5 = ndimage.convolve(settlement.astype(float), k5, mode='constant')
    n_forest_7 = ndimage.convolve(forest.astype(float), k7, mode='constant')
    n_ocean_7 = ndimage.convolve(ocean.astype(float), k7, mode='constant')
    n_empty_7 = ndimage.convolve(empty.astype(float), k7, mode='constant')
    
    features = np.stack([
        dist_ocean, dist_settle, dist_forest, dist_mountain,
        n_settle_3, n_settle_5, n_forest_7, n_ocean_7, n_empty_7,
        is_coast.astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1]), cls.ravel()

# Precompute
print("Precomputing features...", flush=True)
round_data = {}
for rname, rdata in all_gt.items():
    entries = []
    for si_str in sorted(rdata.keys()):
        gt = np.array(rdata[si_str]['ground_truth'])
        ig = np.array(rdata[si_str]['initial_grid'])
        X, cls = extract_spatial_features(ig)
        Y = gt.reshape(-1, 6)
        entries.append((X, Y, cls))
    round_data[rname] = entries
print("Done\n", flush=True)

CLASS_LABELS = ["Empty", "Settlement", "Port", "Shipwreck", "Forest", "Mountain"]
round_names = sorted(round_data.keys())

# Also test a global HGB for comparison
for mode in ["per-class-hgb", "global-hgb"]:
    t0 = time.time()
    round_scores = []
    
    for test_round in round_names:
        train_rounds = [r for r in round_names if r != test_round]
        
        X_train = np.vstack([e[0] for r in train_rounds for e in round_data[r]])
        Y_train = np.vstack([e[1] for r in train_rounds for e in round_data[r]])
        cls_train = np.concatenate([e[2] for r in train_rounds for e in round_data[r]])
        
        if mode == "per-class-hgb":
            # Train separate HGB per initial class
            class_models = {}
            class_avg = {}
            for ic in range(NUM_CLASSES):
                mask = cls_train == ic
                n = mask.sum()
                class_avg[ic] = Y_train[mask].mean(axis=0) if n > 0 else np.ones(6)/6
                
                if n < 30:
                    class_models[ic] = None
                    continue
                
                X_c = X_train[mask]
                Y_c = Y_train[mask]
                models = []
                for oc in range(NUM_CLASSES):
                    m = HistGradientBoostingRegressor(
                        max_iter=80, max_depth=3, learning_rate=0.05,
                        min_samples_leaf=max(10, n // 20), random_state=42
                    )
                    m.fit(X_c, Y_c[:, oc])
                    models.append(m)
                class_models[ic] = models
        else:
            # Global HGB
            models = []
            class_avg = {}
            for ic in range(NUM_CLASSES):
                mask = cls_train == ic
                class_avg[ic] = Y_train[mask].mean(axis=0) if mask.any() else np.ones(6)/6
            
            # Add class one-hot
            cls_oh = np.zeros((len(cls_train), NUM_CLASSES))
            for c in range(NUM_CLASSES):
                cls_oh[cls_train == c, c] = 1
            X_aug = np.hstack([X_train, cls_oh])
            
            for oc in range(NUM_CLASSES):
                m = HistGradientBoostingRegressor(
                    max_iter=100, max_depth=4, learning_rate=0.05,
                    min_samples_leaf=50, random_state=42
                )
                m.fit(X_aug, Y_train[:, oc])
                models.append(m)
        
        # Test
        scores = []
        for X_test, Y_test, cls_test in round_data[test_round]:
            H = W = 40
            gt = Y_test.reshape(H, W, 6)
            
            if mode == "per-class-hgb":
                pred = np.zeros((X_test.shape[0], 6))
                for ic in range(NUM_CLASSES):
                    mask = cls_test == ic
                    if not mask.any():
                        continue
                    if class_models[ic] is None:
                        pred[mask] = class_avg[ic]
                    else:
                        for oc in range(NUM_CLASSES):
                            pred[mask, oc] = class_models[ic][oc].predict(X_test[mask])
            else:
                cls_oh = np.zeros((len(cls_test), NUM_CLASSES))
                for c in range(NUM_CLASSES):
                    cls_oh[cls_test == c, c] = 1
                X_aug = np.hstack([X_test, cls_oh])
                pred = np.column_stack([m.predict(X_aug) for m in models])
            
            pred = np.clip(pred, 0.002, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            pred = pred.reshape(H, W, 6)
            scores.append(score_pred(pred, gt))
        
        round_scores.append(np.mean(scores))
    
    avg = np.mean(round_scores)
    detail = " ".join(f"{r}={s:.1f}" for r, s in zip(round_names, round_scores))
    print(f"{mode:20s}  avg={avg:.2f}  {detail}  ({time.time()-t0:.1f}s)", flush=True)

print(f"\nBaseline:  binned=90.43, HGB-base=91.0", flush=True)
