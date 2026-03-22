"""Simulate observation-calibrated HGB.
Idea: observe some cells, compute correction factors per initial class,
apply to all predictions.
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
    ocean = (ig == 10); mountain = (ig == 5)
    settlement = (cls == 1); forest = (cls == 4); empty = (cls == 0)
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H,W), 20)
    k3, k7, k11 = np.ones((3,3)), np.ones((7,7)), np.ones((11,11))
    n_s3 = ndimage.convolve(settlement.astype(float), k3, mode='constant')
    n_s7 = ndimage.convolve(settlement.astype(float), k7, mode='constant')
    n_f7 = ndimage.convolve(forest.astype(float), k7, mode='constant')
    n_o7 = ndimage.convolve(ocean.astype(float), k7, mode='constant')
    n_e7 = ndimage.convolve(empty.astype(float), k7, mode='constant')
    n_s11 = ndimage.convolve(settlement.astype(float), k11, mode='constant')
    cls_oh = np.zeros((H, W, NUM_CLASSES))
    for c in range(NUM_CLASSES): cls_oh[:,:,c] = (cls==c).astype(float)
    features = np.concatenate([cls_oh, dist_ocean[:,:,None], dist_settle[:,:,None], dist_forest[:,:,None], dist_mountain[:,:,None], n_s3[:,:,None], n_s7[:,:,None], n_f7[:,:,None], n_o7[:,:,None], n_e7[:,:,None], n_s11[:,:,None], is_coast[:,:,None].astype(float)], axis=-1)
    return features.reshape(-1, features.shape[-1])

# Precompute
round_data = {}
for rname, rdata in all_gt.items():
    entries = []
    for si_str in sorted(rdata.keys()):
        gt = np.array(rdata[si_str]['ground_truth'])
        ig = np.array(rdata[si_str]['initial_grid'])
        X = extract_features(ig)
        cls = build_class_grid(ig)
        entries.append((X, gt, ig, cls))
    round_data[rname] = entries

round_names = sorted(round_data.keys())
print("=== OBSERVATION CALIBRATION SIMULATION ===\n", flush=True)

# Strategy: for each test round, observe N random cells from seed 0,
# compute per-class correction = observed_avg / predicted_avg
# Then apply correction to all predictions
for n_obs in [0, 5, 10, 20, 50]:
    np.random.seed(42)
    round_scores = []
    
    for test_round in round_names:
        train_rounds = [r for r in round_names if r != test_round]
        X_train = np.vstack([e[0] for r in train_rounds for e in round_data[r]])
        Y_train = np.vstack([e[1].reshape(-1, 6) for r in train_rounds for e in round_data[r]])
        
        models = []
        for c in range(NUM_CLASSES):
            m = HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42)
            m.fit(X_train, Y_train[:, c])
            models.append(m)
        
        scores = []
        for X_test, gt_test, ig_test, cls_test in round_data[test_round]:
            H, W = ig_test.shape
            pred = np.column_stack([m.predict(X_test) for m in models])
            pred = np.clip(pred, 1e-4, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            
            if n_obs > 0:
                # Simulate observations: pick random non-ocean/non-mountain cells
                land_mask = ~(ig_test == 10) & ~(ig_test == 5)
                land_indices = np.argwhere(land_mask)
                obs_indices = land_indices[np.random.choice(len(land_indices), min(n_obs, len(land_indices)), replace=False)]
                
                # Compute per-class correction
                for ic in range(NUM_CLASSES):
                    obs_this_class = [(y, x) for y, x in obs_indices if cls_test[y, x] == ic]
                    if len(obs_this_class) < 2:
                        continue
                    
                    obs_gt_avg = np.mean([gt_test[y, x] for y, x in obs_this_class], axis=0)
                    obs_pred_avg = np.mean([pred[y * W + x] for y, x in obs_this_class], axis=0)
                    
                    # Multiplicative correction: ratio = gt / pred where both are > eps
                    ratio = np.ones(6)
                    for k in range(6):
                        if obs_pred_avg[k] > 0.01 and obs_gt_avg[k] > 0.001:
                            ratio[k] = obs_gt_avg[k] / obs_pred_avg[k]
                    
                    # Apply correction to all cells of this class
                    class_mask = cls_test.ravel() == ic
                    pred[class_mask] *= ratio
                
                pred = np.clip(pred, 1e-4, None)
                pred /= pred.sum(axis=-1, keepdims=True)
            
            scores.append(kl_score(pred.reshape(H, W, 6), gt_test))
        round_scores.append(np.mean(scores))
    
    avg = np.mean(round_scores)
    detail = " ".join(f"{r}={s:.1f}" for r, s in zip(round_names, round_scores))
    print(f"  n_obs={n_obs:3d}  avg={avg:.2f}  {detail}", flush=True)

# Also try: observe one cell per class from EACH seed (more efficient use)
print("\n=== MULTI-SEED OBSERVATION (observe 1 per class per seed) ===\n", flush=True)
for n_per_class in [1, 2, 3, 5]:
    np.random.seed(42)
    round_scores = []
    
    for test_round in round_names:
        train_rounds = [r for r in round_names if r != test_round]
        X_train = np.vstack([e[0] for r in train_rounds for e in round_data[r]])
        Y_train = np.vstack([e[1].reshape(-1, 6) for r in train_rounds for e in round_data[r]])
        
        models = []
        for c in range(NUM_CLASSES):
            m = HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42)
            m.fit(X_train, Y_train[:, c])
            models.append(m)
        
        # Compute per-class correction using all seeds for observation
        all_obs = {ic: {'gt': [], 'pred': []} for ic in range(NUM_CLASSES)}
        total_obs = 0
        
        for X_test, gt_test, ig_test, cls_test in round_data[test_round]:
            H, W = ig_test.shape
            pred_flat = np.column_stack([m.predict(X_test) for m in models])
            pred_flat = np.clip(pred_flat, 1e-4, None)
            pred_flat /= pred_flat.sum(axis=-1, keepdims=True)
            
            for ic in range(NUM_CLASSES):
                if ic in [5]:  # skip mountain (no change)
                    continue
                class_cells = [(y, x) for y in range(H) for x in range(W) if cls_test[y, x] == ic]
                if len(class_cells) == 0:
                    continue
                chosen = [class_cells[i] for i in np.random.choice(len(class_cells), min(n_per_class, len(class_cells)), replace=False)]
                for y, x in chosen:
                    all_obs[ic]['gt'].append(gt_test[y, x])
                    all_obs[ic]['pred'].append(pred_flat[y * W + x])
                    total_obs += 1
        
        # Compute correction ratios
        corrections = {}
        for ic in range(NUM_CLASSES):
            if len(all_obs[ic]['gt']) < 2:
                corrections[ic] = np.ones(6)
                continue
            gt_avg = np.mean(all_obs[ic]['gt'], axis=0)
            pred_avg = np.mean(all_obs[ic]['pred'], axis=0)
            ratio = np.ones(6)
            for k in range(6):
                if pred_avg[k] > 0.01 and gt_avg[k] > 0.001:
                    ratio[k] = gt_avg[k] / pred_avg[k]
            corrections[ic] = ratio
        
        # Apply corrections and score
        scores = []
        for X_test, gt_test, ig_test, cls_test in round_data[test_round]:
            H, W = ig_test.shape
            pred = np.column_stack([m.predict(X_test) for m in models])
            pred = np.clip(pred, 1e-4, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            
            for ic in range(NUM_CLASSES):
                mask = cls_test.ravel() == ic
                pred[mask] *= corrections[ic]
            
            pred = np.clip(pred, 1e-4, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            scores.append(kl_score(pred.reshape(H, W, 6), gt_test))
        round_scores.append(np.mean(scores))
    
    avg = np.mean(round_scores)
    detail = " ".join(f"{r}={s:.1f}" for r, s in zip(round_names, round_scores))
    print(f"  n_per_class={n_per_class}  obs={total_obs:3d}  avg={avg:.2f}  {detail}", flush=True)
