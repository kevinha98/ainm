"""Resubmit R6 with dist_settle<=2 calibration using 5 remaining obs budget."""
import json, numpy as np, time, sys
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, ".")
from src.api import AstarAPI
from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from src.models import build_class_grid

NC = NUM_CLASSES
CLIP = 0.0001
VP = 15
SETTLE_DIST = 2.0

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
    features = np.concatenate([np.eye(NC)[cls],
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

api = AstarAPI()

# Step 1: Find active round
print("Step 1: Finding active round...", flush=True)
my_rounds = api.get_my_rounds()
round_id = None
for rd in my_rounds:
    if rd.get("status") == "active":
        round_id = rd["id"]
        budget_used = rd.get("queries_used", 0)
        budget_total = rd.get("queries_max", 50)
        print(f"  R6: {round_id}, budget={budget_used}/{budget_total}", flush=True)
        break

if not round_id:
    print("No active round!")
    sys.exit(0)

# Step 2: Test observation  
print("\nStep 2: Testing observation...", flush=True)
test_result = api.simulate(round_id, 0, 12, 12, steps=50)
if "error" in test_result:
    print(f"  Error: {test_result['error']}")
    print(f"  Detail: {test_result.get('detail', '')}")
    USE_NEW_OBS = False
elif "grid" in test_result:
    grid = np.array(test_result["grid"])
    print(f"  Success! viewport shape={grid.shape}")
    USE_NEW_OBS = True
else:
    print(f"  Unknown response: {list(test_result.keys())}")
    USE_NEW_OBS = False

# Step 3: Get R6 initial grids
print("\nStep 3: Fetching R6 data...", flush=True)
full_round = api._get(f"/rounds/{round_id}")
initial_states = full_round.get("initial_states", [])
n_seeds = len(initial_states)
grids = {si: np.array(state["grid"]) for si, state in enumerate(initial_states)}
H, W = grids[0].shape
print(f"  {n_seeds} seeds, grid {H}x{W}", flush=True)

# Step 4: Train HGB on all GT
print("\nStep 4: Training HGB...", flush=True)
X_all, Y_all = [], []
for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
    with open(gf) as f: data = json.load(f)
    for si in sorted(data.keys()):
        gt = np.array(data[si].get("ground_truth", []))
        ig = np.array(data[si].get("initial_grid", []))
        if gt.size > 0 and ig.size > 0:
            X_all.append(extract_features(ig))
            Y_all.append(gt.reshape(-1, NC))
X_all, Y_all = np.vstack(X_all), np.vstack(Y_all)
models = [HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05,
          min_samples_leaf=50, random_state=42).fit(X_all, Y_all[:,c]) for c in range(NC)]

# Step 5: HGB predictions
print("\nStep 5: Predicting R6...", flush=True)
preds = {}
for si in range(n_seeds):
    p = np.column_stack([m.predict(extract_features(grids[si])) for m in models])
    p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
    preds[si] = p.reshape(H, W, NC)

# Step 6: Compute settle masks
settle_masks = {}
for si in range(n_seeds):
    cls_g = build_class_grid(grids[si])
    settlement = (cls_g == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20)
    settle_masks[si] = dist_s <= SETTLE_DIST

if USE_NEW_OBS:
    # Make 4 more observations
    print("\nStep 7: Making observations...", flush=True)
    obs_data = [(0, 12, 12, grid)]
    for si, row, col in [(1, 0, 0), (2, 0, 25), (3, 25, 0), (4, 25, 25)]:
        result = api.simulate(round_id, si, row, col, steps=50)
        if "error" in result:
            print(f"  Error seed {si}: {result['error']}")
            continue
        vp = np.array(result.get("grid", []))
        if vp.ndim == 2:
            obs_data.append((si, row, col, vp))
            print(f"  Seed {si} ({row},{col}): {vp.shape}")
        time.sleep(0.3)
    print(f"  Total: {len(obs_data)} viewports", flush=True)

    # Build settle-split stats
    per_cls_obs = np.zeros((NC, NC)); per_cls_pred = np.zeros((NC, NC)); per_cls_n = np.zeros(NC)
    settle_obs = {True: np.zeros((NC, NC)), False: np.zeros((NC, NC))}
    settle_pred = {True: np.zeros((NC, NC)), False: np.zeros((NC, NC))}
    settle_n = {True: np.zeros(NC), False: np.zeros(NC)}

    for si, row, col, viewport in obs_data:
        cls = build_class_grid(grids[si])
        ns_mask = settle_masks[si]
        vh, vw = viewport.shape
        for vy in range(vh):
            for vx in range(vw):
                gy, gx = row + vy, col + vx
                if gy >= H or gx >= W: continue
                ic = cls[gy, gx]; oc = GRID_TO_CLASS.get(int(viewport[vy, vx]), 0)
                ns = bool(ns_mask[gy, gx])
                per_cls_obs[ic, oc] += 1; per_cls_pred[ic] += preds[si][gy, gx]; per_cls_n[ic] += 1
                settle_obs[ns][ic, oc] += 1; settle_pred[ns][ic] += preds[si][gy, gx]; settle_n[ns][ic] += 1

    total_cells = int(sum(per_cls_n))
    print(f"  {total_cells} cells from {len(obs_data)} obs", flush=True)

    # Cached 45-obs per-class data for fallback
    cached_cls_obs = {
        0: np.array([0.575, 0.222, 0.019, 0.026, 0.141, 0.017]) * 7358,
        1: np.array([0.5, 0.264, 0.013, 0.038, 0.178, 0.006]) * 314,
        4: np.array([0.49, 0.233, 0.023, 0.028, 0.214, 0.013]) * 2214,
        5: np.array([0.462, 0.226, 0.017, 0.034, 0.179, 0.081]) * 234,
    }
    cached_cls_pred = {
        0: np.array([0.854, 0.099, 0.006, 0.01, 0.031, 0.0]) * 7358,
        1: np.array([0.473, 0.28, 0.001, 0.023, 0.222, 0.0]) * 314,
        4: np.array([0.079, 0.12, 0.006, 0.011, 0.784, 0.0]) * 2214,
        5: np.array([0.007, 0.009, 0.0, 0.001, 0.004, 0.978]) * 234,
    }
    cached_cls_n = {0: 7358, 1: 314, 4: 2214, 5: 234}

    print("\nStep 8: Applying settle-split calibration...", flush=True)
    for si in range(n_seeds):
        pred = preds[si].copy().reshape(-1, NC)
        cls = build_class_grid(grids[si]).ravel()
        ns_flat = settle_masks[si].ravel()
        for ns in [True, False]:
            for ic in range(NC):
                n = settle_n[ns][ic]
                if n < 10:
                    if ic in cached_cls_n and cached_cls_n[ic] >= 10:
                        nn = cached_cls_n[ic]
                        of = cached_cls_obs[ic] / nn; pa = cached_cls_pred[ic] / nn
                    else: continue
                else:
                    of = settle_obs[ns][ic] / n; pa = settle_pred[ns][ic] / n
                ratio = np.where(pa > 0.01, np.clip(of / pa, 0.01, 100.0), 1.0)
                mask = (cls == ic) & (ns_flat == ns)
                pred[mask] *= ratio
        pred = np.clip(pred, CLIP, None); pred /= pred.sum(axis=-1, keepdims=True)
        preds[si] = pred.reshape(H, W, NC)
else:
    # Per-class only (cached data)
    print("\nApplying per-class calibration only...", flush=True)
    cached = {
        0: (np.array([0.575, 0.222, 0.019, 0.026, 0.141, 0.017]),
            np.array([0.854, 0.099, 0.006, 0.01, 0.031, 0.0])),
        1: (np.array([0.5, 0.264, 0.013, 0.038, 0.178, 0.006]),
            np.array([0.473, 0.28, 0.001, 0.023, 0.222, 0.0])),
        4: (np.array([0.49, 0.233, 0.023, 0.028, 0.214, 0.013]),
            np.array([0.079, 0.12, 0.006, 0.011, 0.784, 0.0])),
        5: (np.array([0.462, 0.226, 0.017, 0.034, 0.179, 0.081]),
            np.array([0.007, 0.009, 0.0, 0.001, 0.004, 0.978])),
    }
    for si in range(n_seeds):
        pred = preds[si].copy().reshape(-1, NC)
        cls = build_class_grid(grids[si]).ravel()
        for ic, (of, pa) in cached.items():
            ratio = np.where(pa > 0.01, np.clip(of / pa, 0.01, 100.0), 1.0)
            pred[cls == ic] *= ratio
        pred = np.clip(pred, CLIP, None); pred /= pred.sum(axis=-1, keepdims=True)
        preds[si] = pred.reshape(H, W, NC)

# Submit
print("\nStep 9: Submitting...", flush=True)
for si in range(n_seeds):
    plist = preds[si].tolist()
    result = api.submit_prediction(round_id, si, plist)
    print(f"  Seed {si}: {result}", flush=True)
    time.sleep(0.5)

print("\nDone!", flush=True)
