"""
v11: HGB + Observation Calibration.
Uses 5 remaining observations to calibrate HGB predictions.
Each observation returns a 15x15 viewport of simulation outcome.
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


CLIP_FLOOR = 0.0001


def extract_features(ig):
    """Extract spatial features for HGB."""
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
    
    features = np.concatenate([
        cls_oh,
        dist_ocean[:,:,None], dist_settle[:,:,None], dist_forest[:,:,None], dist_mountain[:,:,None],
        n_s3[:,:,None], n_s7[:,:,None], n_f7[:,:,None], n_o7[:,:,None], n_e7[:,:,None], n_s11[:,:,None],
        is_coast[:,:,None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1])


def train_hgb_models():
    """Train HGB on all GT data."""
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    if not gt_files:
        return None
    
    X_all, Y_all = [], []
    for gf in gt_files:
        with open(gf) as f:
            gt_data = json.load(f)
        for si_str in sorted(gt_data.keys()):
            gt = np.array(gt_data[si_str].get('ground_truth', []))
            ig = np.array(gt_data[si_str].get('initial_grid', []))
            if gt.size == 0 or ig.size == 0:
                continue
            X_all.append(extract_features(ig))
            Y_all.append(gt.reshape(-1, 6))
    
    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)
    print(f"  Training on {len(X_all)} cells from {len(gt_files)} GT files")
    
    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, learning_rate=0.05,
            min_samples_leaf=50, random_state=42
        )
        m.fit(X_all, Y_all[:, c])
        models.append(m)
    return models


# Map raw grid value to class index consistent with build_class_grid
def raw_to_class(val):
    """Convert raw grid value to class index."""
    CLASS_MAP = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 5, 6: 2, 7: 3, 8: 4, 9: 4, 10: 0}
    return CLASS_MAP.get(val, 0)


def observe_and_calibrate(api, round_id, grids, models, n_obs=5):
    """Observe cells and compute per-class calibration factors.
    
    Strategy: observe one central position per seed to get max coverage.
    Each observation gives 15x15 = 225 cells of simulation outcome.
    """
    n_seeds = len(grids)
    H, W = grids[0].shape
    
    # First get uncalibrated HGB predictions
    preds = {}
    for si in range(n_seeds):
        X = extract_features(grids[si])
        pred = np.column_stack([m.predict(X) for m in models])
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        preds[si] = pred.reshape(H, W, 6)
    
    # Collect observed outcomes
    per_class_obs = {c: np.zeros(NUM_CLASSES) for c in range(NUM_CLASSES)}  # count -> outcome class
    per_class_pred = {c: np.zeros(NUM_CLASSES) for c in range(NUM_CLASSES)}
    per_class_n = {c: 0 for c in range(NUM_CLASSES)}
    
    # Choose observation centers for maximum coverage of different terrain
    # Use one observation per seed, different center positions
    centers = [
        (10, 10),  # top-left quadrant
        (10, 30),  # top-right quadrant
        (30, 10),  # bottom-left quadrant
        (30, 30),  # bottom-right quadrant
        (20, 20),  # center
    ]
    
    obs_count = 0
    for si in range(min(n_seeds, n_obs)):
        row, col = centers[si]
        print(f"  Observing seed {si} at ({row},{col})...", end=" ", flush=True)
        
        result = api.simulate(round_id, si, row, col, steps=50)
        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue
        
        viewport = np.array(result.get("viewport", result.get("grid", [])))
        if viewport.size == 0:
            print("empty viewport")
            continue
        
        vp_row = result.get("viewport_row", max(0, row - 7))
        vp_col = result.get("viewport_col", max(0, col - 7))
        vh, vw = viewport.shape
        print(f"got {vh}x{vw} viewport at ({vp_row},{vp_col})")
        obs_count += 1
        
        # Match observed outcomes to our predictions
        cls_grid = build_class_grid(grids[si])
        for vy in range(vh):
            for vx in range(vw):
                gy = vp_row + vy
                gx = vp_col + vx
                if gy < 0 or gy >= H or gx < 0 or gx >= W:
                    continue
                
                ic = cls_grid[gy, gx]
                observed_class = raw_to_class(viewport[vy, vx])
                
                per_class_obs[ic][observed_class] += 1
                per_class_pred[ic] += preds[si][gy, gx]
                per_class_n[ic] += 1
    
    print(f"\n  Observed {obs_count} viewports")
    
    if obs_count == 0:
        return preds
    
    # Compute calibration: ratio of observed frequencies to predicted averages
    calibrated = {}
    for si in range(n_seeds):
        pred = preds[si].copy().reshape(-1, 6)
        cls_grid = build_class_grid(grids[si])
        
        for ic in range(NUM_CLASSES):
            n = per_class_n[ic]
            if n < 5:
                continue
            
            obs_freq = per_class_obs[ic] / n
            pred_avg = per_class_pred[ic] / n
            
            # Multiplicative ratio with soft clipping
            ratio = np.ones(6)
            for k in range(6):
                if pred_avg[k] > 0.01:
                    r = obs_freq[k] / pred_avg[k]
                    ratio[k] = np.clip(r, 0.3, 3.0)  # Don't over-correct
            
            mask = cls_grid.ravel() == ic
            pred[mask] *= ratio
        
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        calibrated[si] = pred.reshape(H, W, 6)
    
    # Print calibration info
    for ic in range(NUM_CLASSES):
        n = per_class_n[ic]
        if n < 5:
            continue
        obs_freq = per_class_obs[ic] / n
        pred_avg = per_class_pred[ic] / n
        print(f"  {CLASS_NAMES[ic]:>12s} (n={n:4d}): obs={[f'{x:.3f}' for x in obs_freq]} pred={[f'{x:.3f}' for x in pred_avg]}")
    
    return calibrated


def main():
    flags = set(sys.argv[1:])
    no_submit = "--no-submit" in flags
    no_obs = "--no-obs" in flags
    
    t_total = time.time()
    print("=" * 65)
    print("  ASTAR ISLAND -- HGB + Observation Calibration v11")
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
    
    # Check budget
    budget = api.get_budget()
    used = budget.get('queries_used', 0)
    max_q = budget.get('queries_max', 50)
    remaining = max_q - used
    print(f"  Observation budget: {used}/{max_q} used, {remaining} remaining")
    
    # Step 2: Train HGB
    print("\n[2] TRAINING HGB MODELS")
    t_train = time.time()
    models = train_hgb_models()
    if models is None:
        print("  ERROR: No ground truth data!")
        return
    print(f"  Training took {time.time()-t_train:.1f}s")
    
    # Step 3: Parse seeds
    print(f"\n[3] PARSING SEEDS")
    grids = []
    for st in rd["initial_states"]:
        grids.append(np.array(st["grid"]))
    
    # Step 4: Predict + Calibrate
    if not no_obs and remaining > 0:
        n_obs = min(remaining, 5)  # Use up to 5 observations
        print(f"\n[4] PREDICTION + CALIBRATION ({n_obs} observations)")
        final = observe_and_calibrate(api, round_id, grids, models, n_obs=n_obs)
    else:
        print(f"\n[4] PREDICTION (no observations)")
        final = {}
        for si in range(n_seeds):
            X = extract_features(grids[si])
            pred = np.column_stack([m.predict(X) for m in models])
            pred = np.clip(pred, CLIP_FLOOR, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            final[si] = pred.reshape(grids[si].shape[0], grids[si].shape[1], 6)
    
    for si in range(n_seeds):
        se = compute_stats(final[si], grids[si])
        print(f"  Seed {si}: ent={se['ent']:.3f} conf={se['conf']:.3f}")
    
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
