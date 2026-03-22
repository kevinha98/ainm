"""Quick test: ultra-low clip floor values."""
import json, numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from src.settings import DATA_DIR, NUM_CLASSES
from src.models import build_class_grid

gt_files = {
    'R1': DATA_DIR / 'ground_truth_71451d74.json',
    'R2': DATA_DIR / 'ground_truth_76909e29.json',
    'R3': DATA_DIR / 'ground_truth_f1dac9a9.json',
    'R4': DATA_DIR / 'ground_truth_8e839974.json',
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
        entries.append((extract_features(ig), gt.reshape(-1, 6)))
    round_data[rname] = entries

round_names = sorted(round_data.keys())
clips = [1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4]
print("=== Ultra-low clip floor sweep ===\n", flush=True)
for clip_val in clips:
    round_scores = []
    for test_round in round_names:
        train_rounds = [r for r in round_names if r != test_round]
        X_train = np.vstack([e[0] for r in train_rounds for e in round_data[r]])
        Y_train = np.vstack([e[1] for r in train_rounds for e in round_data[r]])
        models = []
        for c in range(NUM_CLASSES):
            m = HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42)
            m.fit(X_train, Y_train[:, c])
            models.append(m)
        scores = []
        for X_test, Y_test in round_data[test_round]:
            gt = Y_test.reshape(40, 40, 6)
            pred = np.column_stack([m.predict(X_test) for m in models])
            pred = np.clip(pred, clip_val, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            scores.append(score_pred(pred.reshape(40,40,6), gt))
        round_scores.append(np.mean(scores))
    avg = np.mean(round_scores)
    print(f"  clip={clip_val:.0e}  avg={avg:.3f}  {' '.join(f'{s:.1f}' for s in round_scores)}", flush=True)
