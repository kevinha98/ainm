"""Sweep: HGB tuning + blending with binned/per-class avg.
Tests multiple configurations in LOO cross-validation.
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

def get_spatial_bin(ig, cls):
    H, W = ig.shape
    ocean = (ig == 10)
    mountain = (ig == 5)
    settlement = (cls == 1)
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    k7 = np.ones((7, 7))
    n_settle_3 = ndimage.convolve(settlement.astype(float), k7, mode='constant')
    bins = np.full((H, W), 4, dtype=int)  # default: inland_far
    bins[ocean] = 0
    bins[mountain] = 1
    bins[is_coast & ~ocean & ~mountain] = 2
    bins[(n_settle_3 >= 2) & ~ocean & ~mountain & ~is_coast] = 3
    return bins

def extract_features_v2(ig):
    """Extended features."""
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
    k11 = np.ones((11, 11))
    
    n_settle_3 = ndimage.convolve(settlement.astype(float), k3, mode='constant')
    n_settle_5 = ndimage.convolve(settlement.astype(float), k5, mode='constant')
    n_settle_7 = ndimage.convolve(settlement.astype(float), k7, mode='constant')
    n_forest_7 = ndimage.convolve(forest.astype(float), k7, mode='constant')
    n_ocean_7 = ndimage.convolve(ocean.astype(float), k7, mode='constant')
    n_empty_7 = ndimage.convolve(empty.astype(float), k7, mode='constant')
    n_settle_11 = ndimage.convolve(settlement.astype(float), k11, mode='constant')
    
    # Edge features: row/col as fractions
    rows = np.repeat(np.arange(H)[:, None], W, axis=1).astype(float) / H
    cols = np.repeat(np.arange(W)[None, :], H, axis=0).astype(float) / W
    
    cls_onehot = np.zeros((H, W, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        cls_onehot[:, :, c] = (cls == c).astype(float)
    
    # Spatial bin as feature too
    bins = get_spatial_bin(ig, cls).astype(float) / 4.0
    
    features = np.concatenate([
        cls_onehot,                                    # 0-5: class one-hot
        dist_ocean[:, :, None],                        # 6
        dist_settle[:, :, None],                       # 7
        dist_forest[:, :, None],                       # 8
        dist_mountain[:, :, None],                     # 9
        n_settle_3[:, :, None],                        # 10
        n_settle_5[:, :, None],                        # 11
        n_settle_7[:, :, None],                        # 12
        n_forest_7[:, :, None],                        # 13
        n_ocean_7[:, :, None],                         # 14
        n_empty_7[:, :, None],                         # 15
        n_settle_11[:, :, None],                       # 16
        is_coast[:, :, None].astype(float),            # 17
        rows[:, :, None],                              # 18
        cols[:, :, None],                              # 19
        bins[:, :, None],                              # 20
    ], axis=-1)
    return features.reshape(-1, features.shape[-1]), cls.ravel()

# Precompute
print("Precomputing extended features...", flush=True)
t0 = time.time()
round_data = {}
for rname, rdata in all_gt.items():
    entries = []
    for si_str in sorted(rdata.keys()):
        gt = np.array(rdata[si_str]['ground_truth'])
        ig = np.array(rdata[si_str]['initial_grid'])
        X, cls = extract_features_v2(ig)
        Y = gt.reshape(-1, 6)
        bins = get_spatial_bin(ig, build_class_grid(ig))
        entries.append((X, Y, cls, ig, bins.ravel()))
    round_data[rname] = entries
print(f"Done in {time.time()-t0:.1f}s\n", flush=True)

# Configs to test
configs = [
    {"name": "HGB-base", "max_iter": 100, "max_depth": 4, "lr": 0.05, "min_leaf": 50, "blend_binned": 0.0, "blend_avg": 0.0},
    {"name": "HGB-deeper", "max_iter": 150, "max_depth": 5, "lr": 0.05, "min_leaf": 30, "blend_binned": 0.0, "blend_avg": 0.0},
    {"name": "HGB-more", "max_iter": 200, "max_depth": 4, "lr": 0.03, "min_leaf": 50, "blend_binned": 0.0, "blend_avg": 0.0},
    {"name": "HGB+avg20", "max_iter": 100, "max_depth": 4, "lr": 0.05, "min_leaf": 50, "blend_binned": 0.0, "blend_avg": 0.2},
    {"name": "HGB+avg30", "max_iter": 100, "max_depth": 4, "lr": 0.05, "min_leaf": 50, "blend_binned": 0.0, "blend_avg": 0.3},
    {"name": "HGB+bin20", "max_iter": 100, "max_depth": 4, "lr": 0.05, "min_leaf": 50, "blend_binned": 0.2, "blend_avg": 0.0},
    {"name": "HGB+bin30", "max_iter": 100, "max_depth": 4, "lr": 0.05, "min_leaf": 50, "blend_binned": 0.3, "blend_avg": 0.0},
    {"name": "HGB-conservative", "max_iter": 50, "max_depth": 3, "lr": 0.1, "min_leaf": 100, "blend_binned": 0.0, "blend_avg": 0.0},
]

round_names = sorted(round_data.keys())

for cfg in configs:
    t1 = time.time()
    round_scores = []
    
    for test_round in round_names:
        train_rounds = [r for r in round_names if r != test_round]
        
        X_train = np.vstack([e[0] for r in train_rounds for e in round_data[r]])
        Y_train = np.vstack([e[1] for r in train_rounds for e in round_data[r]])
        cls_train = np.concatenate([e[2] for r in train_rounds for e in round_data[r]])
        bins_train = np.concatenate([e[4] for r in train_rounds for e in round_data[r]])
        
        # Per-class averages
        avg_class = {}
        for c in range(NUM_CLASSES):
            mask = cls_train == c
            avg_class[c] = Y_train[mask].mean(axis=0) if mask.any() else np.ones(6)/6
        
        # Per-(class, bin) averages
        avg_binned = {}
        for c in range(NUM_CLASSES):
            for b in range(5):
                mask = (cls_train == c) & (bins_train == b)
                if mask.sum() >= 5:
                    avg_binned[(c, b)] = Y_train[mask].mean(axis=0)
                else:
                    avg_binned[(c, b)] = avg_class[c]
        
        # Train HGB
        models = []
        for c in range(NUM_CLASSES):
            m = HistGradientBoostingRegressor(
                max_iter=cfg["max_iter"], max_depth=cfg["max_depth"],
                learning_rate=cfg["lr"], min_samples_leaf=cfg["min_leaf"],
                random_state=42
            )
            m.fit(X_train, Y_train[:, c])
            models.append(m)
        
        # Test
        scores = []
        for X_test, Y_test, cls_test, ig_test, bins_test in round_data[test_round]:
            H = W = 40
            gt = Y_test.reshape(H, W, 6)
            
            pred_hgb = np.column_stack([m.predict(X_test) for m in models])
            pred_hgb = np.clip(pred_hgb, 0.001, None)
            
            # Blend with per-class avg
            if cfg["blend_avg"] > 0:
                pred_avg = np.zeros_like(pred_hgb)
                for c in range(NUM_CLASSES):
                    mask = cls_test == c
                    pred_avg[mask] = avg_class[c]
                pred_hgb = (1 - cfg["blend_avg"]) * pred_hgb + cfg["blend_avg"] * pred_avg
            
            # Blend with binned avg
            if cfg["blend_binned"] > 0:
                pred_bin = np.zeros_like(pred_hgb)
                for c in range(NUM_CLASSES):
                    for b in range(5):
                        mask = (cls_test == c) & (bins_test == b)
                        pred_bin[mask] = avg_binned[(c, b)]
                pred_hgb = (1 - cfg["blend_binned"]) * pred_hgb + cfg["blend_binned"] * pred_bin
            
            pred_hgb = np.clip(pred_hgb, 0.002, None)
            pred_hgb /= pred_hgb.sum(axis=-1, keepdims=True)
            pred_hgb = pred_hgb.reshape(H, W, 6)
            
            scores.append(score_pred(pred_hgb, gt))
        round_scores.append(np.mean(scores))
    
    avg = np.mean(round_scores)
    detail = " ".join(f"{r}={s:.1f}" for r, s in zip(round_names, round_scores))
    print(f"{cfg['name']:20s}  avg={avg:.2f}  {detail}  ({time.time()-t1:.1f}s)", flush=True)

print(f"\nBaseline:  binned avg=90.43, basic avg=86.92", flush=True)
