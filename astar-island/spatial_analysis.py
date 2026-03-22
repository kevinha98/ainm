"""Explore whether spatial features improve GT distribution prediction.
Goal: find features that reduce KL divergence beyond per-class average.
"""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from src.settings import DATA_DIR, NUM_CLASSES, CLASS_NAMES
from src.models import build_class_grid

# Load all ground truth
gt_files = {
    "R1": DATA_DIR / "ground_truth_71451d74.json",
    "R2": DATA_DIR / "ground_truth_76909e29.json",
    "R3": DATA_DIR / "ground_truth_f1dac9a9.json",
    "R4": DATA_DIR / "ground_truth_8e839974.json",
}

all_gt = {}
for name, path in gt_files.items():
    if path.exists():
        with open(path) as f:
            all_gt[name] = json.load(f)

def score_pred(pred, gt):
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    kl = np.sum(gt_safe * np.log(gt_safe / (pred + 1e-15)), axis=-1).mean()
    return 100 * np.exp(-kl)


def extract_spatial_features(ig):
    """Extract per-cell spatial features from initial grid."""
    H, W = ig.shape
    cls = build_class_grid(ig)
    ocean = (ig == 10)
    mountain = (ig == 5)
    settlement = (cls == 1)
    forest = (cls == 4)
    plains = ((ig == 11) | (ig == 0))
    
    # Distance transforms
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H,W), 20)
    dist_settlement = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20)
    
    # Neighbor counts in radius 3
    kernel3 = np.ones((7, 7))
    n_ocean_3 = ndimage.convolve(ocean.astype(float), kernel3, mode='constant')
    n_settlement_3 = ndimage.convolve(settlement.astype(float), kernel3, mode='constant')
    n_forest_3 = ndimage.convolve(forest.astype(float), kernel3, mode='constant')
    n_mountain_3 = ndimage.convolve(mountain.astype(float), kernel3, mode='constant')
    
    # Edge features
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    is_mountain_edge = ~mountain & (ndimage.maximum_filter(mountain.astype(float), size=3) > 0)
    
    features = np.stack([
        cls.astype(float),             # 0: initial class
        dist_ocean,                     # 1: distance to ocean
        dist_mountain,                  # 2: distance to mountain
        dist_settlement,               # 3: distance to nearest settlement
        dist_forest,                   # 4: distance to nearest forest
        n_ocean_3,                     # 5: ocean neighbors in r=3
        n_settlement_3,                # 6: settlement neighbors in r=3
        n_forest_3,                    # 7: forest neighbors in r=3
        n_mountain_3,                  # 8: mountain neighbors in r=3
        is_coast.astype(float),        # 9: is coastal
        is_mountain_edge.astype(float),# 10: is mountain edge
    ], axis=-1)
    
    return features


# Build training data from all rounds
print("Building spatial feature dataset...")
X_all, Y_all, C_all = [], [], []  # features, GT distributions, initial classes

for rname, gt_data in sorted(all_gt.items()):
    for si in range(5):
        si_str = str(si)
        if si_str not in gt_data:
            continue
        gt = np.array(gt_data[si_str].get('ground_truth', []))
        ig = np.array(gt_data[si_str].get('initial_grid', []))
        if gt.size == 0 or ig.size == 0:
            continue
        
        cls = build_class_grid(ig)
        feats = extract_spatial_features(ig)
        
        X_all.append(feats.reshape(-1, feats.shape[-1]))
        Y_all.append(gt.reshape(-1, 6))
        C_all.append(cls.ravel())

X = np.vstack(X_all)
Y = np.vstack(Y_all)
C = np.concatenate(C_all)
print(f"Dataset: {X.shape[0]} cells, {X.shape[1]} features")

# Analyze: for each initial class, how much variance is explained by spatial features?
print("\n=== VARIANCE ANALYSIS ===")
for c in range(NUM_CLASSES):
    mask = C == c
    if mask.sum() < 10:
        continue
    
    y_c = Y[mask]
    x_c = X[mask]
    
    avg = y_c.mean(axis=0)
    total_var = np.mean((y_c - avg) ** 2)
    
    # Check if features correlate with GT deviation from avg
    deviations = y_c - avg  # How much each cell deviates from class avg
    
    print(f"\n{CLASS_NAMES[c]} (n={mask.sum()}):")
    print(f"  Class avg: {[round(v,3) for v in avg.tolist()]}")
    print(f"  Total var: {total_var:.6f}")
    
    # Correlation of each feature with GT class-0 probability
    for fi in range(x_c.shape[1]):
        feat_vals = x_c[:, fi]
        if feat_vals.std() < 1e-6:
            continue
        # Correlate with each GT class probability
        best_corr = 0
        best_cls = 0
        for gi in range(6):
            corr = np.corrcoef(feat_vals, y_c[:, gi])[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_cls = gi
        if abs(best_corr) > 0.1:
            feat_names = ['class', 'dist_ocean', 'dist_mtn', 'dist_settle', 'dist_forest',
                         'n_ocean_3', 'n_settle_3', 'n_forest_3', 'n_mtn_3', 'is_coast', 'is_mtn_edge']
            print(f"  {feat_names[fi]:15s} → {CLASS_NAMES[best_cls]:10s} r={best_corr:.3f}")

# Quick test: binned spatial predictions
print("\n\n=== BINNED SPATIAL TEST ===")
print("Test: do settlements near ocean have different GT than inland settlements?")

# For settlements, bin by distance to ocean
settle_mask = C == 1
if settle_mask.sum() > 0:
    x_settle = X[settle_mask]
    y_settle = Y[settle_mask]
    dist_ocean = x_settle[:, 1]
    
    for thresh in [2, 5, 10]:
        near = dist_ocean <= thresh
        far = dist_ocean > thresh
        if near.sum() > 5 and far.sum() > 5:
            avg_near = y_settle[near].mean(axis=0)
            avg_far = y_settle[far].mean(axis=0)
            print(f"\n  Settlements dist_ocean <= {thresh} (n={near.sum()}):")
            print(f"    {[round(v,3) for v in avg_near.tolist()]}")
            print(f"  Settlements dist_ocean > {thresh} (n={far.sum()}):")
            print(f"    {[round(v,3) for v in avg_far.tolist()]}")

# For plains, bin by number of nearby settlements  
plains_mask = C == 0
if plains_mask.sum() > 0:
    x_plains = X[plains_mask]
    y_plains = Y[plains_mask]
    n_settle_near = x_plains[:, 6]
    
    print(f"\n  Plains near settlements (n_settle_r3 > 0) vs far:")
    near = n_settle_near > 0
    far = n_settle_near == 0
    if near.sum() > 5 and far.sum() > 5:
        print(f"    Near (n={near.sum()}): {[round(v,3) for v in y_plains[near].mean(axis=0).tolist()]}")
        print(f"    Far  (n={far.sum()}):  {[round(v,3) for v in y_plains[far].mean(axis=0).tolist()]}")

# Try a simple binned approach: per-class + coastal vs inland
print("\n\n=== CROSS-ROUND LOO: BINNED SPATIAL ===")
round_names = sorted(all_gt.keys())

for test_round in round_names:
    train_rounds = [r for r in round_names if r != test_round]
    
    # Standard per-class avg from training
    by_class = {c: [] for c in range(NUM_CLASSES)}
    by_class_coast = {c: [] for c in range(NUM_CLASSES)}
    by_class_inland = {c: [] for c in range(NUM_CLASSES)}
    
    for r in train_rounds:
        for si in range(5):
            si_str = str(si)
            if si_str not in all_gt[r]:
                continue
            gt = np.array(all_gt[r][si_str]['ground_truth'])
            ig = np.array(all_gt[r][si_str]['initial_grid'])
            ic = build_class_grid(ig)
            ocean = (ig == 10)
            is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
            
            for y in range(ig.shape[0]):
                for x in range(ig.shape[1]):
                    c = ic[y, x]
                    by_class[c].append(gt[y, x])
                    if is_coast[y, x]:
                        by_class_coast[c].append(gt[y, x])
                    else:
                        by_class_inland[c].append(gt[y, x])
    
    avg = {c: np.mean(v, axis=0) if v else np.ones(6)/6 for c, v in by_class.items()}
    avg_coast = {c: np.mean(v, axis=0) if v else avg[c] for c, v in by_class_coast.items()}
    avg_inland = {c: np.mean(v, axis=0) if v else avg[c] for c, v in by_class_inland.items()}
    
    # Score on test round
    scores_basic = []
    scores_coastal = []
    
    for si in range(5):
        si_str = str(si)
        if si_str not in all_gt[test_round]:
            continue
        gt = np.array(all_gt[test_round][si_str]['ground_truth'])
        ig = np.array(all_gt[test_round][si_str]['initial_grid'])
        ic = build_class_grid(ig)
        ocean = (ig == 10)
        is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
        H, W = ig.shape
        
        pred_basic = np.zeros((H, W, 6))
        pred_coastal = np.zeros((H, W, 6))
        
        for y in range(H):
            for x in range(W):
                c = ic[y, x]
                pred_basic[y, x] = avg.get(c, avg[0])
                if is_coast[y, x]:
                    pred_coastal[y, x] = avg_coast.get(c, avg[c])
                else:
                    pred_coastal[y, x] = avg_inland.get(c, avg[c])
        
        scores_basic.append(score_pred(pred_basic, gt))
        scores_coastal.append(score_pred(pred_coastal, gt))
    
    print(f"  {test_round}: basic={np.mean(scores_basic):.2f}, coastal_split={np.mean(scores_coastal):.2f}")
