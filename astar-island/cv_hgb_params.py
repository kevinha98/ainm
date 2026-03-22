"""Quick test of HGB hyperparameter variants with 5 GT rounds.
Tests: higher iterations, different depths, different leaf sizes, ensemble."""
import json, numpy as np, time
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

DATA_DIR = Path("data")
NC = 6
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}
CLIP = 0.0001

def build_class_grid(ig):
    cls = np.zeros_like(ig)
    for raw, c in GRID_TO_CLASS.items(): cls[ig == raw] = c
    return cls

def extract_features(ig):
    cls = build_class_grid(ig); H, W = ig.shape
    ocean = (ig == 10); mountain = (ig == 5)
    settlement = (cls == 1); forest = (cls == 4); empty = (cls == 0)
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H,W), 20)
    k3, k7, k11 = np.ones((3,3)), np.ones((7,7)), np.ones((11,11))
    features = np.concatenate([
        np.eye(NC)[cls],
        dist_ocean[:,:,None], dist_settle[:,:,None], dist_forest[:,:,None], dist_mountain[:,:,None],
        ndimage.convolve(settlement.astype(float), k3, mode='constant')[:,:,None],
        ndimage.convolve(settlement.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(forest.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(ocean.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(empty.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(settlement.astype(float), k11, mode='constant')[:,:,None],
        is_coast[:,:,None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1])

def kl_score(gt, pred):
    g = np.clip(gt.reshape(-1, NC), 1e-10, None)
    p = np.clip(pred.reshape(-1, NC), 1e-10, None)
    p /= p.sum(axis=-1, keepdims=True); g /= g.sum(axis=-1, keepdims=True)
    return 100 * np.exp(-np.mean(np.sum(g * np.log(g / p), axis=-1)))

def load_all():
    rounds = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        with open(gf) as f: data = json.load(f)
        seeds = []
        for si in sorted(data.keys()):
            gt = np.array(data[si].get('ground_truth', []))
            ig = np.array(data[si].get('initial_grid', []))
            if gt.size > 0 and ig.size > 0: seeds.append((ig, gt))
        rounds.append(seeds)
    return rounds

t0 = time.time()
print("=== HGB PARAM SWEEP (5 rounds LOO) ===\n")
rounds = load_all()
nR = len(rounds)

configs = [
    ("BASELINE: iter=100 d=4 lr=.05 leaf=50", dict(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50)),
    ("iter=200 d=4 lr=.05 leaf=50", dict(max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=50)),
    ("iter=300 d=4 lr=.05 leaf=50", dict(max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=50)),
    ("iter=100 d=6 lr=.05 leaf=50", dict(max_iter=100, max_depth=6, learning_rate=0.05, min_samples_leaf=50)),
    ("iter=100 d=3 lr=.05 leaf=50", dict(max_iter=100, max_depth=3, learning_rate=0.05, min_samples_leaf=50)),
    ("iter=100 d=4 lr=.03 leaf=50", dict(max_iter=100, max_depth=4, learning_rate=0.03, min_samples_leaf=50)),
    ("iter=100 d=4 lr=.10 leaf=50", dict(max_iter=100, max_depth=4, learning_rate=0.10, min_samples_leaf=50)),
    ("iter=100 d=4 lr=.05 leaf=20", dict(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=20)),
    ("iter=100 d=4 lr=.05 leaf=100", dict(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=100)),
    ("iter=200 d=4 lr=.03 leaf=50", dict(max_iter=200, max_depth=4, learning_rate=0.03, min_samples_leaf=50)),
    ("iter=200 d=6 lr=.03 leaf=50", dict(max_iter=200, max_depth=6, learning_rate=0.03, min_samples_leaf=50)),
]

for name, params in configs:
    fold_scores = []
    for hold in range(nR):
        test_seeds = rounds[hold]
        train_data = [s for i, ss in enumerate(rounds) if i != hold for s in ss]
        X, Y = [], []
        for ig, gt in train_data:
            X.append(extract_features(ig)); Y.append(gt.reshape(-1, NC))
        X, Y = np.vstack(X), np.vstack(Y)
        
        models = [HistGradientBoostingRegressor(random_state=42, **params).fit(X, Y[:, c]) for c in range(NC)]
        
        scores = []
        for ig, gt in test_seeds:
            p = np.column_stack([m.predict(extract_features(ig)) for m in models])
            p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
            scores.append(kl_score(gt, p.reshape(ig.shape[0], ig.shape[1], NC)))
        fold_scores.append(np.mean(scores))
    
    avg = np.mean(fold_scores)
    folds_str = " ".join(f"{s:.1f}" for s in fold_scores)
    print(f"  {name:45s}: {avg:.2f} | {folds_str}", flush=True)

print(f"\nTotal: {time.time()-t0:.0f}s")
