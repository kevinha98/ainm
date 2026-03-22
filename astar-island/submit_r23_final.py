"""
ROUND 23 — FINAL ROUND SUBMISSION
Best HGB config from sweep: depth=6, clip=1e-6, iter=200, lr=0.05

Strategy:
1. Fetch R22 GT if available (adds training data)
2. Train HGB on ALL GT (21-22 rounds)
3. Predict all seeds
4. Observe 45 viewports (spend full 50-query budget)
5. Submit all seeds
"""
import json, time, sys, numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, '.')
from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from src.api import AstarAPI

# ── Best config from sweep (LOO=90.83 w/ 20 rounds, 91.15 w/ 21) ──
CLIP_FLOOR = 1e-6
TEMPERATURE = 1.0
HGB_PARAMS = dict(max_iter=200, max_depth=6, learning_rate=0.05,
                  min_samples_leaf=50, random_state=42)


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
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H, W), 20)
    k3, k7, k11 = np.ones((3, 3)), np.ones((7, 7)), np.ones((11, 11))
    n_s3 = ndimage.convolve(settlement.astype(float), k3, mode='constant')
    n_s7 = ndimage.convolve(settlement.astype(float), k7, mode='constant')
    n_f7 = ndimage.convolve(forest.astype(float), k7, mode='constant')
    n_o7 = ndimage.convolve(ocean.astype(float), k7, mode='constant')
    n_e7 = ndimage.convolve(empty.astype(float), k7, mode='constant')
    n_s11 = ndimage.convolve(settlement.astype(float), k11, mode='constant')
    cls_oh = np.zeros((H, W, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        cls_oh[:, :, c] = (cls == c).astype(float)
    features = np.concatenate([
        cls_oh, dist_ocean[:, :, None], dist_settle[:, :, None],
        dist_forest[:, :, None], dist_mountain[:, :, None],
        n_s3[:, :, None], n_s7[:, :, None], n_f7[:, :, None],
        n_o7[:, :, None], n_e7[:, :, None], n_s11[:, :, None],
        is_coast[:, :, None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1])


def train_all():
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    print(f"[TRAIN] {len(gt_files)} GT files, HGB params: depth={HGB_PARAMS['max_depth']}, "
          f"iter={HGB_PARAMS['max_iter']}, lr={HGB_PARAMS['learning_rate']}")
    X_all, Y_all = [], []
    for gf in gt_files:
        with open(gf) as f:
            data = json.load(f)
        for si_str in sorted(data.keys()):
            entry = data[si_str]
            if not isinstance(entry, dict):
                continue
            gt = np.array(entry['ground_truth'])
            ig = np.array(entry['initial_grid'])
            X_all.append(extract_hgb_features(ig))
            Y_all.append(gt.reshape(-1, 6))
    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)
    print(f"[TRAIN] {X_all.shape[0]} cells, {X_all.shape[1]} features")
    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(**HGB_PARAMS)
        m.fit(X_all, Y_all[:, c])
        models.append(m)
    print(f"[TRAIN] All 6 class models trained")
    return models


def predict_grid(models, ig):
    X = extract_hgb_features(ig)
    pred = np.column_stack([m.predict(X) for m in models])
    cls = build_class_grid(ig)
    mtn = (cls == 5).ravel()
    if mtn.any():
        pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
    pred[~mtn, 5] = 0.0
    s = pred[~mtn].sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1, s)
    pred[~mtn] /= s
    pred = np.clip(pred, CLIP_FLOOR, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    H, W = ig.shape
    return pred.reshape(H, W, 6)


def observe_viewports(api, round_id, grids, n_seeds):
    """Observe 45 viewports (9 per seed × 5 seeds), return raw observation data."""
    viewport_positions = [(r, c) for r in [0, 12, 25] for c in [0, 12, 25]]
    raw_obs = []
    obs_used = 0

    # Check budget
    try:
        my_rounds = api.get_my_rounds()
        remaining = 50
        for mr in my_rounds:
            if mr.get("id") == round_id:
                remaining = mr.get("queries_max", 50) - mr.get("queries_used", 0)
                break
        print(f"[OBS] Budget remaining: {remaining}")
    except:
        remaining = 50

    n_target = min(remaining, 45)
    for si in range(n_seeds):
        for row, col in viewport_positions:
            if obs_used >= n_target:
                break
            result = api.simulate(round_id, si, row, col, steps=50)
            if "error" in result:
                print(f"[OBS] Error s{si} ({row},{col}): {result.get('error')}")
                if result.get('error') == 'budget_exhausted':
                    break
                continue
            viewport = np.array(result.get("grid", []))
            if viewport.ndim != 2:
                continue
            obs_used += 1
            raw_obs.append({"seed": si, "row": row, "col": col,
                            "grid": viewport.tolist()})
            time.sleep(0.25)
        if obs_used >= n_target:
            break

    # Save observations
    obs_path = DATA_DIR / f"observations_{round_id[:8]}.json"
    with open(obs_path, "w") as f:
        json.dump(raw_obs, f)
    print(f"[OBS] {obs_used} viewports observed, saved to {obs_path.name}")
    return raw_obs


def fetch_r22_gt(api):
    """Try to fetch R22 GT if round is completed."""
    rounds = api.get_rounds()
    fetched = 0
    for r in rounds:
        if r["status"] != "completed":
            continue
        gt_path = DATA_DIR / f"ground_truth_{r['id'][:8]}.json"
        if gt_path.exists():
            continue
        rn = r.get('round_number', '?')
        print(f"[GT] Fetching GT for Round {rn} ({r['id'][:8]})...")
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
                time.sleep(0.3)
            if gt_data:
                with open(gt_path, "w") as f:
                    json.dump(gt_data, f)
                print(f"[GT] Saved {gt_path.name} ({len(gt_data)} seeds)")
                fetched += 1
        except Exception as e:
            print(f"[GT] Error: {e}")
    return fetched


def poll_for_round(api, target_round=23, max_wait=1200):
    """Poll API until target round is active. Returns round dict or None."""
    print(f"[POLL] Waiting for Round {target_round} to become active...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            rounds = api.get_rounds()
            for r in rounds:
                if r.get("round_number") == target_round and r["status"] == "active":
                    print(f"[POLL] Round {target_round} is ACTIVE! id={r['id'][:8]}")
                    return r
            # Also check if any new active round appeared
            for r in rounds:
                if r["status"] == "active":
                    rn = r.get("round_number", "?")
                    if rn >= 23:
                        print(f"[POLL] Found active Round {rn}! id={r['id'][:8]}")
                        return r
        except Exception as e:
            print(f"[POLL] API error: {e}")
        elapsed = int(time.time() - start)
        print(f"[POLL] {elapsed}s elapsed, waiting 15s...", end="\r")
        time.sleep(15)
    print(f"[POLL] Timeout after {max_wait}s")
    return None


if __name__ == "__main__":
    api = AstarAPI()

    # Step 0: Try to fetch any missing GT (especially R22)
    print("=" * 60)
    print("ROUND 23 — FINAL ROUND SUBMISSION")
    print(f"Config: depth={HGB_PARAMS['max_depth']}, iter={HGB_PARAMS['max_iter']}, "
          f"lr={HGB_PARAMS['learning_rate']}, clip={CLIP_FLOOR}")
    print("=" * 60)

    fetch_r22_gt(api)

    # Step 1: Check if R23 is already active, otherwise poll
    rounds = api.get_rounds()
    active = None
    for r in rounds:
        if r["status"] == "active" and r.get("round_number", 0) >= 23:
            active = r
            break

    if not active:
        print("\n[INFO] R23 not yet active. Polling...")
        active = poll_for_round(api, target_round=23, max_wait=1800)

    if not active:
        print("[FATAL] Could not find active R23. Exiting.")
        sys.exit(1)

    round_id = active["id"]
    round_num = active.get("round_number", 23)
    print(f"\n[ACTIVE] Round {round_num} (id={round_id[:8]})")

    # Step 2: Get round data
    full = api._get(f"/rounds/{round_id}")
    grids = [np.array(st["grid"]) for st in full["initial_states"]]
    n_seeds = len(grids)
    print(f"[DATA] {n_seeds} seeds, grid={grids[0].shape}")

    # Step 3: Fetch any newly available GT (R22 might have completed)
    fetch_r22_gt(api)

    # Step 4: Train HGB on ALL available GT
    t0 = time.time()
    models = train_all()
    train_time = time.time() - t0
    print(f"[TRAIN] Completed in {train_time:.0f}s")

    # Step 5: Predict with HGB
    print("[PREDICT] Generating predictions...")
    predictions = {}
    for si in range(n_seeds):
        predictions[si] = predict_grid(models, grids[si])
        p = predictions[si].reshape(-1, 6)
        ent = -np.sum(p * np.log(np.clip(p, 1e-10, None)), axis=1).mean()
        conf = p.max(axis=1).mean()
        print(f"  Seed {si}: entropy={ent:.3f}, confidence={conf:.3f}")

    # Step 6: Observe viewports (use full budget)
    raw_obs = observe_viewports(api, round_id, grids, n_seeds)

    # Step 7: Submit all seeds
    print("[SUBMIT] Submitting predictions...")
    submit_ok = 0
    for si in range(n_seeds):
        p = predictions[si]
        p = np.clip(p, CLIP_FLOOR, None)
        p = p / p.sum(axis=-1, keepdims=True)
        ok, text = api.submit_prediction(round_id, si, p.tolist())
        status = "OK" if ok else f"FAIL: {text[:80]}"
        print(f"  Seed {si}: {status}")
        if ok:
            submit_ok += 1
        time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print(f"ROUND {round_num} SUBMISSION COMPLETE: {submit_ok}/{n_seeds} seeds OK")
    gt_count = len(list(DATA_DIR.glob("ground_truth_*.json")))
    print(f"Trained on {gt_count} GT files | depth={HGB_PARAMS['max_depth']} "
          f"clip={CLIP_FLOOR} T={TEMPERATURE}")
    print(f"{'=' * 60}")
