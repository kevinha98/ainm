"""
v10: HGB Regressor with spatial features.
Trains HistGradientBoostingRegressor on all GT data,
predicts per-cell probability distributions.
Cross-round LOO: avg 91.0 (vs binned 90.4, basic 86.9)
"""
import json
import time
import sys
import numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

from src.settings import DATA_DIR, NUM_CLASSES, MAP_H, MAP_W, CLASS_NAMES
from src.api import AstarAPI
from src.models import build_class_grid, compute_stats


CLIP_FLOOR = 0.0001  # Tuned: lower clip = better KL score (91.36 vs 91.0 at 0.002)


def extract_features(ig):
    """Extract spatial features for each cell."""
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
        cls_onehot,                                    # 0-5: class one-hot
        dist_ocean[:, :, None],                        # 6
        dist_settle[:, :, None],                       # 7
        dist_forest[:, :, None],                       # 8
        dist_mountain[:, :, None],                     # 9
        n_settle_3[:, :, None],                        # 10
        n_settle_7[:, :, None],                        # 11
        n_forest_7[:, :, None],                        # 12
        n_ocean_7[:, :, None],                         # 13
        n_empty_7[:, :, None],                         # 14
        n_settle_11[:, :, None],                       # 15
        is_coast[:, :, None].astype(float),            # 16
    ], axis=-1)
    return features.reshape(-1, features.shape[-1])


def train_hgb_models(gt_dir=None):
    """Train HGB models on all available ground truth data."""
    if gt_dir is None:
        gt_dir = DATA_DIR
    
    gt_files = sorted(gt_dir.glob("ground_truth_*.json"))
    if not gt_files:
        return None, None
    
    X_all, Y_all, cls_all = [], [], []
    
    for gf in gt_files:
        with open(gf) as f:
            gt_data = json.load(f)
        
        for si_str in sorted(gt_data.keys()):
            gt = np.array(gt_data[si_str].get('ground_truth', []))
            ig = np.array(gt_data[si_str].get('initial_grid', []))
            if gt.size == 0 or ig.size == 0:
                continue
            
            X = extract_features(ig)
            Y = gt.reshape(-1, 6)
            cls = build_class_grid(ig).ravel()
            
            X_all.append(X)
            Y_all.append(Y)
            cls_all.append(cls)
    
    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)
    cls_all = np.concatenate(cls_all)
    
    print(f"  Training on {len(X_all)} cells from {len(gt_files)} GT files")
    
    # Per-class averages as fallback
    avg_class = {}
    for c in range(NUM_CLASSES):
        mask = cls_all == c
        avg_class[c] = Y_all[mask].mean(axis=0) if mask.any() else np.ones(6)/6
    
    # Train one HGB per output class
    models = []
    for oc in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, learning_rate=0.05,
            min_samples_leaf=50, random_state=42
        )
        m.fit(X_all, Y_all[:, oc])
        models.append(m)
    
    return models, avg_class


def predict_grid(models, ig, clip_floor=CLIP_FLOOR):
    """Predict probability distribution for each cell using HGB."""
    X = extract_features(ig)
    pred = np.column_stack([m.predict(X) for m in models])
    pred = np.clip(pred, clip_floor, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    H, W = ig.shape
    return pred.reshape(H, W, NUM_CLASSES)


def main():
    flags = set(sys.argv[1:])
    no_submit = "--no-submit" in flags
    
    t_total = time.time()
    print("=" * 65)
    print("  ASTAR ISLAND -- HGB Regressor v10")
    print("=" * 65)
    
    api = AstarAPI()
    
    # Step 1: Get round
    print("\n[1] ROUND DETECTION")
    rd = api.get_active_round()
    if rd is None:
        print("  No active round!")
        return
    
    round_id = rd['id']
    n_seeds = rd.get('seeds_count', len(rd.get('initial_states', [])))
    print(f"  Round {rd['round_number']}: {rd['status']} ({rd['map_width']}x{rd['map_height']}, {n_seeds} seeds)")
    print(f"  ID: {round_id[:12]}...")
    print(f"  Closes: {rd.get('closes_at', '?')}")
    
    # Step 2: Train HGB
    print("\n[2] TRAINING HGB MODELS")
    t_train = time.time()
    models, avg_class = train_hgb_models()
    if models is None:
        print("  ERROR: No ground truth data!")
        return
    print(f"  Training took {time.time()-t_train:.1f}s")
    
    # Step 3: Parse seeds and predict
    print(f"\n[3] PREDICTION")
    grids = []
    for st in rd["initial_states"]:
        grids.append(np.array(st["grid"]))
    
    final = {}
    for si in range(n_seeds):
        pred = predict_grid(models, grids[si])
        se = compute_stats(pred, grids[si])
        print(f"  Seed {si}: ent={se['ent']:.3f} conf={se['conf']:.3f}")
        final[si] = pred
    
    # Step 4: Save
    print(f"\n[4] SAVING")
    improved = []
    for si in range(n_seeds):
        improved.append({
            "seed_index": si,
            "probabilities": final[si].tolist(),
            "model": "hgb_v10",
        })
    with open(DATA_DIR / "improved_predictions.json", "w") as f:
        json.dump(improved, f)
    print("  Saved improved_predictions.json")
    
    # Step 5: Submit
    if not no_submit and rd["status"] == "active":
        print(f"\n[5] SUBMITTING")
        for si in range(n_seeds):
            p = final[si]
            p = np.clip(p, 1e-6, None)
            p /= p.sum(axis=-1, keepdims=True)
            ok, text = api.submit_prediction(round_id, si, p.tolist())
            status = "OK" if ok else f"FAIL: {text[:80]}"
            print(f"  Seed {si}: {status}")
            time.sleep(0.5)
        print("  All seeds submitted!")
    else:
        print(f"\n[5] SKIPPED SUBMISSION")
    
    tt = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  Total time: {tt:.0f}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
