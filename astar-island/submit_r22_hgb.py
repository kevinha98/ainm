"""
Submit Round 22 using HGB model (LOO=90.8).
One-shot: train on all GT, predict, observe, blend, submit.
"""
import json, time, sys, numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, '.')
from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from src.api import AstarAPI

CLIP_FLOOR = 0.0001
TEMPERATURE = 1.0
HGB_PARAMS = dict(max_iter=200, max_depth=5, learning_rate=0.05, min_samples_leaf=50, random_state=42)

def build_class_grid(ig):
    cls = np.zeros_like(ig)
    for raw, c in GRID_TO_CLASS.items():
        cls[ig == raw] = c
    return cls

def extract_hgb_features(ig):
    cls = build_class_grid(ig)
    H, W = ig.shape
    ocean = (ig == 10); mountain = (ig == 5)
    settlement = (cls == 1); forest = (cls == 4); empty = (cls == 0)
    port = (cls == 2); ruin = (cls == 3)
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
        cls_oh, dist_ocean[:,:,None], dist_settle[:,:,None],
        dist_forest[:,:,None], dist_mountain[:,:,None],
        n_s3[:,:,None], n_s7[:,:,None], n_f7[:,:,None],
        n_o7[:,:,None], n_e7[:,:,None], n_s11[:,:,None],
        is_coast[:,:,None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1])

def train_all():
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    print(f"Training on {len(gt_files)} GT files...")
    X_all, Y_all = [], []
    for gf in gt_files:
        with open(gf) as f:
            data = json.load(f)
        for si_str in sorted(data.keys()):
            entry = data[si_str]
            if not isinstance(entry, dict): continue
            gt = np.array(entry['ground_truth'])
            ig = np.array(entry['initial_grid'])
            X_all.append(extract_hgb_features(ig))
            Y_all.append(gt.reshape(-1, 6))
    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)
    print(f"  {X_all.shape[0]} cells, {X_all.shape[1]} features")
    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(**HGB_PARAMS)
        m.fit(X_all, Y_all[:, c])
        models.append(m)
        print(f"  Class {c} ({CLASS_NAMES[c]}) trained")
    return models

def predict_grid(models, ig):
    X = extract_hgb_features(ig)
    pred = np.column_stack([m.predict(X) for m in models])
    cls = build_class_grid(ig)
    H, W = ig.shape
    mtn = (cls == 5).ravel()
    if mtn.any():
        pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
    pred[~mtn, 5] = 0.0
    s = pred[~mtn].sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1, s)
    pred[~mtn] /= s
    pred = np.clip(pred, CLIP_FLOOR, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred.reshape(H, W, 6)

def observe_and_blend(api, round_id, grids, hgb_preds):
    """Observe 45 viewports and blend obs frequencies with HGB predictions."""
    n_seeds = len(grids)
    H, W = grids[0].shape
    viewport_positions = [(r, c) for r in [0, 12, 25] for c in [0, 12, 25]]
    
    # Build class grids + spatial features for bucketing
    seed_cls = [build_class_grid(g) for g in grids]
    
    # Check budget
    try:
        my_rounds = api.get_my_rounds()
        remaining = 50
        for mr in my_rounds:
            if mr.get("id") == round_id:
                remaining = mr.get("queries_max", 50) - mr.get("queries_used", 0)
                break
    except:
        remaining = 50
    
    # Load saved observations
    obs_path = DATA_DIR / f"observations_{round_id[:8]}.json"
    raw_obs = []
    obs_used = 0
    if obs_path.exists():
        try:
            with open(obs_path) as f:
                raw_obs = json.load(f)
            obs_used = len(raw_obs)
            print(f"  Loaded {obs_used} saved observations")
        except:
            pass
    
    n_obs = min(remaining, 45)
    if obs_used >= 45:
        n_obs = 0
        print(f"  Already have {obs_used} obs, skipping live queries")
    
    # Live observations
    for si in range(n_seeds):
        for row, col in viewport_positions:
            if obs_used >= n_obs:
                break
            result = api.simulate(round_id, si, row, col, steps=50)
            if "error" in result:
                print(f"  Obs error s{si} ({row},{col}): {result.get('error')}")
                if result.get('error') == 'budget_exhausted':
                    n_obs = obs_used
                    break
                continue
            viewport = np.array(result.get("grid", []))
            if viewport.ndim != 2: continue
            obs_used += 1
            raw_obs.append({"seed": si, "row": row, "col": col, "grid": viewport.tolist()})
            time.sleep(0.3)
        if obs_used >= n_obs:
            break
    
    print(f"  Total observations: {obs_used}")
    
    # Save observations
    existing_count = 0
    if obs_path.exists():
        try:
            with open(obs_path) as f:
                existing_count = len(json.load(f))
        except: pass
    if len(raw_obs) > existing_count:
        with open(obs_path, "w") as f:
            json.dump(raw_obs, f)
    
    # Count per-class transitions from observations
    class_obs = {}  # ic -> counts[6]
    class_total = {}
    for obs_entry in raw_obs:
        si = obs_entry["seed"]
        row, col = obs_entry["row"], obs_entry["col"]
        viewport = np.array(obs_entry["grid"])
        if viewport.ndim != 2: continue
        cls = seed_cls[si]
        vh, vw = viewport.shape
        for vy in range(vh):
            for vx in range(vw):
                gy, gx = row + vy, col + vx
                if gy >= H or gx >= W: continue
                ic = int(cls[gy, gx])
                oc = GRID_TO_CLASS.get(int(viewport[vy, vx]), 0)
                class_obs.setdefault(ic, np.zeros(6))
                class_total.setdefault(ic, 0)
                class_obs[ic][oc] += 1
                class_total[ic] += 1
    
    # Log obs frequencies
    for ic in range(NUM_CLASSES):
        if ic in class_total and class_total[ic] > 0:
            freq = class_obs[ic] / class_total[ic]
            print(f"  Obs {CLASS_NAMES[ic]}: n={class_total[ic]:5d} freq={np.round(freq, 3).tolist()}")
    
    # For now: return HGB predictions directly (obs blending is marginal for HGB)
    # The HGB already captures the patterns; obs would just add noise with small samples
    return hgb_preds


if __name__ == "__main__":
    api = AstarAPI()
    
    # Get active round
    print("Fetching rounds...")
    rounds = api.get_rounds()
    active = None
    for r in rounds:
        if r["status"] == "active":
            active = r
            break
    
    if not active:
        print("No active round!")
        sys.exit(1)
    
    round_id = active["id"]
    print(f"Active: Round {active['round_number']} (id={round_id[:8]})")
    
    # Get full round data
    full = api._get(f"/rounds/{round_id}")
    n_seeds = len(full.get("initial_states", []))
    print(f"  {n_seeds} seeds")
    
    grids = [np.array(st["grid"]) for st in full["initial_states"]]
    
    # Train HGB on all GT
    models = train_all()
    
    # Predict with HGB
    print("Predicting...")
    hgb_preds = {}
    for si in range(n_seeds):
        hgb_preds[si] = predict_grid(models, grids[si])
        p = hgb_preds[si].reshape(-1, 6)
        ent = -np.sum(p * np.log(np.clip(p, 1e-10, None)), axis=1).mean()
        conf = p.max(axis=1).mean()
        print(f"  Seed {si}: entropy={ent:.3f}, confidence={conf:.3f}")
    
    # Observe and optionally blend
    print("Observing...")
    final_preds = observe_and_blend(api, round_id, grids, hgb_preds)
    
    # Submit
    print("Submitting...")
    for si in range(n_seeds):
        p = final_preds[si]
        p = np.clip(p, 1e-6, None)
        p = p / p.sum(axis=-1, keepdims=True)
        ok, text = api.submit_prediction(round_id, si, p.tolist())
        status = "OK" if ok else f"FAIL: {text[:80]}"
        print(f"  Seed {si}: {status}")
        time.sleep(0.5)
    
    print("\nDone! Submitted all seeds with HGB model (LOO=90.8)")
    
    # Also fetch R21 GT if available
    print("\nFetching R21 GT...")
    for r in rounds:
        if r["status"] == "completed":
            gt_path = DATA_DIR / f"ground_truth_{r['id'][:8]}.json"
            if not gt_path.exists():
                print(f"  Fetching GT for Round {r.get('round_number', '?')} ({r['id'][:8]})...")
                try:
                    full_r = api._get(f"/rounds/{r['id']}")
                    n_s = len(full_r.get("initial_states", []))
                    gt_data = {}
                    for si in range(n_s):
                        analysis = api.get_analysis(r['id'], si)
                        ig = full_r["initial_states"][si]["grid"]
                        gt = analysis.get("ground_truth", analysis.get("probabilities"))
                        if gt is not None and ig is not None:
                            gt_data[str(si)] = {"ground_truth": gt, "initial_grid": ig}
                        time.sleep(0.5)
                    if gt_data:
                        with open(gt_path, "w") as f:
                            json.dump(gt_data, f)
                        print(f"  Saved {gt_path.name} ({len(gt_data)} seeds)")
                except Exception as e:
                    print(f"  Error: {e}")
