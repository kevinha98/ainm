"""Cross-round LOO validation: HGB regressor trained on spatial features.
Trains on N-1 rounds, tests on held-out round.
"""
import json
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from src.settings import DATA_DIR, NUM_CLASSES, CLASS_NAMES
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
    """Extract per-cell features."""
    cls = build_class_grid(ig)
    H, W = ig.shape
    ocean = (ig == 10)
    mountain = (ig == 5)
    settlement = (cls == 1)
    forest = (cls == 4)
    
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    
    kernel3 = np.ones((7, 7))
    kernel5 = np.ones((11, 11))
    n_settle_3 = ndimage.convolve(settlement.astype(float), kernel3, mode='constant')
    n_forest_3 = ndimage.convolve(forest.astype(float), kernel3, mode='constant')
    n_ocean_3 = ndimage.convolve(ocean.astype(float), kernel3, mode='constant')
    n_settle_5 = ndimage.convolve(settlement.astype(float), kernel5, mode='constant')
    
    # One-hot encode initial class
    cls_onehot = np.zeros((H, W, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        cls_onehot[:, :, c] = (cls == c).astype(float)
    
    features = np.concatenate([
        cls_onehot,                                    # 0-5: class one-hot
        dist_ocean[:, :, None],                        # 6
        dist_settle[:, :, None],                       # 7
        n_settle_3[:, :, None],                        # 8
        n_forest_3[:, :, None],                        # 9
        n_ocean_3[:, :, None],                         # 10
        n_settle_5[:, :, None],                        # 11
        is_coast[:, :, None].astype(float),            # 12
    ], axis=-1)
    
    return features.reshape(-1, features.shape[-1])

print("=== CROSS-ROUND LOO: HGB REGRESSOR ===\n")
round_names = sorted(all_gt.keys())

for test_round in round_names:
    train_rounds = [r for r in round_names if r != test_round]
    
    # Build training data
    X_train, Y_train = [], []
    by_class = {c: [] for c in range(NUM_CLASSES)}
    
    for r in train_rounds:
        for si in range(5):
            si_str = str(si)
            if si_str not in all_gt[r]:
                continue
            gt = np.array(all_gt[r][si_str]['ground_truth'])
            ig = np.array(all_gt[r][si_str]['initial_grid'])
            if gt.size == 0 or ig.size == 0:
                continue
            
            X_train.append(extract_features(ig))
            Y_train.append(gt.reshape(-1, 6))
            ic = build_class_grid(ig)
            for y in range(ig.shape[0]):
                for x in range(ig.shape[1]):
                    by_class[ic[y, x]].append(gt[y, x])
    
    X_train = np.vstack(X_train)
    Y_train = np.vstack(Y_train)
    avg_class = {c: np.mean(v, axis=0) if v else np.ones(6)/6 for c, v in by_class.items()}
    
    # Train regressors
    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, learning_rate=0.05,
            min_samples_leaf=50, random_state=42
        )
        m.fit(X_train, Y_train[:, c])
        models.append(m)
    
    # Test
    scores_hgb = []
    scores_basic = []
    
    for si in range(5):
        si_str = str(si)
        if si_str not in all_gt[test_round]:
            continue
        gt = np.array(all_gt[test_round][si_str]['ground_truth'])
        ig = np.array(all_gt[test_round][si_str]['initial_grid'])
        ic = build_class_grid(ig)
        H, W = ig.shape
        
        X_test = extract_features(ig)
        
        pred_hgb = np.zeros((X_test.shape[0], 6))
        for c in range(NUM_CLASSES):
            pred_hgb[:, c] = models[c].predict(X_test)
        pred_hgb = np.clip(pred_hgb, 0.002, None)
        pred_hgb /= pred_hgb.sum(axis=-1, keepdims=True)
        pred_hgb = pred_hgb.reshape(H, W, 6)
        
        pred_basic = np.zeros((H, W, 6))
        for y in range(H):
            for x in range(W):
                pred_basic[y, x] = avg_class.get(ic[y, x], avg_class[0])
        pred_basic = np.clip(pred_basic, 0.002, None)
        pred_basic /= pred_basic.sum(axis=-1, keepdims=True)
        
        scores_hgb.append(score_pred(pred_hgb, gt))
        scores_basic.append(score_pred(pred_basic, gt))
    
    avg_h = np.mean(scores_hgb)
    avg_b = np.mean(scores_basic)
    print(f"{test_round}: basic={avg_b:.2f}  HGB={avg_h:.2f}  diff={avg_h-avg_b:+.2f}")

print(f"\n(binned approach scored: R1=92.9, R2=90.2, R3=81.7, R4=97.0, avg=90.4)")
