"""Quick 6-round LOO temperature sweep (no calibration) to check if T=1.15 is still optimal."""
import json, numpy as np, sys
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path

NC = 6; CLIP_FLOOR = 0.0001
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}

def build_class_grid(ig):
    cls = np.zeros_like(ig)
    for raw, c in GRID_TO_CLASS.items(): cls[ig == raw] = c
    return cls

def extract_features(ig):
    cls = build_class_grid(ig); H, W = ig.shape
    ocean = (ig == 10); mountain = (ig == 5); settlement = (cls == 1); forest = (cls == 4); empty = (cls == 0)
    cls_oh = np.eye(NC)[cls.ravel()].reshape(H, W, NC)
    d_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    d_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    d_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20)
    d_m = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H,W), 20)
    k3 = np.ones((3,3)); k7 = np.ones((7,7)); k11 = np.ones((11,11))
    feat = np.concatenate([cls_oh, d_o[:,:,None], d_s[:,:,None], d_f[:,:,None], d_m[:,:,None],
        ndimage.convolve(settlement.astype(float), k3, mode='constant')[:,:,None],
        ndimage.convolve(settlement.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(forest.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(ocean.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(empty.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(settlement.astype(float), k11, mode='constant')[:,:,None],
        (ocean & (d_s > 5)).astype(float)[:,:,None]], axis=2)
    return feat.reshape(-1, feat.shape[2])

def kl_score(gt, pred):
    g = np.clip(gt.reshape(-1, NC), 1e-10, None); p = np.clip(pred.reshape(-1, NC), 1e-10, None)
    p /= p.sum(axis=-1, keepdims=True); g /= g.sum(axis=-1, keepdims=True)
    return 100 * np.exp(-np.mean(np.sum(g * np.log(g / p), axis=-1)))

all_data = {}
for gf in sorted(Path('data').glob('ground_truth_*.json')):
    rid = gf.stem.split('_')[-1][:8]; all_data[rid] = json.load(open(gf))

TEMPS = [0.9, 1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]
results = {t: [] for t in TEMPS}
round_list = sorted(all_data.keys())

for fold_idx, test_id in enumerate(round_list):
    train_ids = [k for k in round_list if k != test_id]
    X_tr, Y_tr = [], []
    for rid in train_ids:
        for si_str, sd in all_data[rid].items():
            ig = np.array(sd['initial_grid']); gt = np.array(sd.get('ground_truth', []))
            if gt.size == 0: continue
            X_tr.append(extract_features(ig)); Y_tr.append(gt.reshape(-1, NC))
    X_tr = np.vstack(X_tr); Y_tr = np.vstack(Y_tr)
    models = [HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42).fit(X_tr, Y_tr[:, c]) for c in range(NC)]
    
    test_data = all_data[test_id]
    for T in TEMPS:
        scores = []
        for si in range(5):
            ig = np.array(test_data[str(si)]['initial_grid']); gt_arr = np.array(test_data[str(si)]['ground_truth'])
            X = extract_features(ig); H, W = ig.shape
            pred = np.column_stack([m.predict(X) for m in models])
            pred = np.clip(pred, CLIP_FLOOR, None); pred /= pred.sum(axis=1, keepdims=True)
            if T != 1.0:
                log_p = np.log(np.clip(pred, 1e-12, None)); scaled = log_p / T
                scaled -= scaled.max(axis=1, keepdims=True)
                pred = np.exp(scaled); pred /= pred.sum(axis=1, keepdims=True)
            scores.append(kl_score(gt_arr, pred.reshape(H, W, NC)))
        results[T].append(np.mean(scores))
    print(f"Fold {fold_idx+1}/6 ({test_id[:4]}) done", flush=True)

print("\n6-round LOO Temperature Sweep (NO calibration):")
for T in TEMPS:
    avg = np.mean(results[T])
    per_round = [f'{s:.1f}' for s in results[T]]
    print(f"  T={T:.2f}: avg={avg:.2f}  per_round={per_round}")
