"""
Auto-runner V3: HGB-based predictor (replaces LUT approach).

Strategy:
1. Train HGB regressors on ALL available GT data (20+ rounds, 160K+ cells)
2. For new round: extract features from initial grid -> predict 6-class distributions
3. Observe 45 viewports -> build observation LUT -> blend with HGB prior
4. Temperature scale -> clip -> normalize -> submit

LOO CV: HGB deeper (200i/d5) = 90.81 vs LUT approach = 72.47 (+18.34 points)
"""
import json
import time
import sys
import os
import logging
import traceback
import atexit
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from src.api import AstarAPI

# ── Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "auto_runner_v3.log"),
    ]
)
log = logging.getLogger("auto_runner_v3")

# ── Configuration
CLIP_FLOOR = 0.0001          # CV-validated clip (from cv_comprehensive)
TEMPERATURE = 1.0            # Will be tuned by sweep
OBS_BLEND_ALPHA = 0.3        # How much to trust observations vs HGB prior (obs weighted)
MIN_OBS_N = 15               # Min observation count to trust a bucket
HGB_PARAMS = dict(
    max_iter=200,
    max_depth=5,
    learning_rate=0.05,
    min_samples_leaf=50,
    random_state=42,
)

STATE_FILE = DATA_DIR / "auto_runner_v3_state.json"
LOCK_FILE = DATA_DIR / "auto_runner_v3.lock"


def build_class_grid(ig):
    cls = np.zeros_like(ig)
    for raw, c in GRID_TO_CLASS.items():
        cls[ig == raw] = c
    return cls


# ── Feature Extraction (17 features, same as cv_comprehensive) ──

def extract_hgb_features(ig):
    """Extract 17 features per cell for HGB prediction."""
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
        cls_oh,                          # 6: one-hot class
        dist_ocean[:,:,None],            # 1: distance to ocean
        dist_settle[:,:,None],           # 1: distance to settlement
        dist_forest[:,:,None],           # 1: distance to forest
        dist_mountain[:,:,None],         # 1: distance to mountain
        n_s3[:,:,None],                  # 1: settlement count 3x3
        n_s7[:,:,None],                  # 1: settlement count 7x7
        n_f7[:,:,None],                  # 1: forest count 7x7
        n_o7[:,:,None],                  # 1: ocean count 7x7
        n_e7[:,:,None],                  # 1: empty count 7x7
        n_s11[:,:,None],                 # 1: settlement count 11x11
        is_coast[:,:,None].astype(float),# 1: coastal flag
    ], axis=-1)  # 17 features
    return features.reshape(-1, features.shape[-1])


# ── HGB Training ──

def train_hgb_models():
    """Train 6 HGB regressors on all available GT data."""
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    if not gt_files:
        log.error("No ground truth files found!")
        return None

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
    log.info(f"Training HGB on {len(gt_files)} GT files, {X_all.shape[0]} cells, {X_all.shape[1]} features")

    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(**HGB_PARAMS)
        m.fit(X_all, Y_all[:, c])
        models.append(m)
        log.info(f"  Trained class {c} ({CLASS_NAMES[c]})")

    return models


def predict_hgb(models, ig):
    """Predict 6-class probabilities for a single grid using trained HGB models."""
    X = extract_hgb_features(ig)
    pred = np.column_stack([m.predict(X) for m in models])

    # Mountain fix
    cls = build_class_grid(ig)
    H, W = ig.shape
    mtn = (cls == 5).ravel()
    if mtn.any():
        pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
    pred[~mtn, 5] = 0.0
    s = pred[~mtn].sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1, s)
    pred[~mtn] /= s

    return pred.reshape(H, W, 6)


# ── Observation + Blending ──

def compute_spatial_features(cls, ig):
    """Compute settle_bin, near_forest, coastal, near_port."""
    H, W = cls.shape
    settlement = (cls == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    forest = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20.0)
    ocean = (ig == 10)
    dist_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    port = (cls == 2)
    dist_p = ndimage.distance_transform_edt(~port) if port.any() else np.full((H, W), 40.0)
    settle_bin = np.full((H, W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    coastal = (dist_o <= 1.5).astype(int)
    near_port = (dist_p <= 2.0).astype(int)
    return settle_bin, near_forest, coastal, near_port


def observe_and_predict(api, round_id, grids, hgb_models):
    """
    HGB prediction + optional observation-based refinement.
    1. Predict with HGB (trained on all past GT)
    2. Observe 45 viewports for current round data
    3. Build per-bucket observation frequencies
    4. Blend: where obs count is high, mix obs data into HGB prediction
    5. Temperature scale → clip → normalize
    """
    n_seeds = len(grids)
    H, W = grids[0].shape

    # Pre-compute HGB predictions for all seeds
    hgb_preds = {}
    for si in range(n_seeds):
        hgb_preds[si] = predict_hgb(hgb_models, grids[si])

    # Pre-compute spatial features
    seed_cls, seed_sb, seed_nf, seed_co, seed_np = [], [], [], [], []
    for si in range(n_seeds):
        cls = build_class_grid(grids[si])
        sb, nf, co, np_ = compute_spatial_features(cls, grids[si])
        seed_cls.append(cls)
        seed_sb.append(sb)
        seed_nf.append(nf)
        seed_co.append(co)
        seed_np.append(np_)

    # Check budget
    try:
        my_rounds = api.get_my_rounds()
        remaining = 50
        for mr in my_rounds:
            if mr.get("id") == round_id:
                remaining = mr.get("queries_max", 50) - mr.get("queries_used", 0)
                break
    except Exception:
        remaining = 50
    n_obs = min(remaining, 45)
    log.info(f"Observation budget: {remaining} remaining, using up to {n_obs}")

    viewport_positions = [(r, c) for r in [0, 12, 25] for c in [0, 12, 25]]

    # Tally observations
    obs_counts = {}
    obs_total = {}
    obs_used = 0
    raw_obs = []

    # Load saved observations
    obs_path = DATA_DIR / f"observations_{round_id[:8]}.json"
    if obs_path.exists():
        try:
            with open(obs_path) as f:
                saved_obs = json.load(f)
            if len(saved_obs) > 0:
                log.info(f"Loaded {len(saved_obs)} saved observations")
                for obs_entry in saved_obs:
                    si_prev = obs_entry["seed"]
                    row_prev = obs_entry["row"]
                    col_prev = obs_entry["col"]
                    viewport = np.array(obs_entry["grid"])
                    if viewport.ndim != 2:
                        continue
                    raw_obs.append(obs_entry)
                    obs_used += 1
                    vh, vw = viewport.shape
                    cls = seed_cls[si_prev]
                    sb = seed_sb[si_prev]
                    nf = seed_nf[si_prev]
                    co = seed_co[si_prev]
                    np_ = seed_np[si_prev]
                    for vy in range(vh):
                        for vx in range(vw):
                            gy, gx = row_prev + vy, col_prev + vx
                            if gy >= H or gx >= W:
                                continue
                            ic = int(cls[gy, gx])
                            key = (ic, int(sb[gy, gx]), int(nf[gy, gx]), int(co[gy, gx]), int(np_[gy, gx]))
                            oc = GRID_TO_CLASS.get(int(viewport[vy, vx]), 0)
                            obs_counts.setdefault(key, np.zeros(6))
                            obs_total.setdefault(key, 0)
                            obs_counts[key][oc] += 1
                            obs_total[key] += 1
                if obs_used >= 45:
                    log.info(f"Already have {obs_used} observations, skipping live queries")
                    n_obs = 0
        except Exception as e:
            log.warning(f"Failed to load saved observations: {e}")
            obs_used = 0
            raw_obs = []

    # Live observations
    for si in range(n_seeds):
        for row, col in viewport_positions:
            if obs_used >= n_obs:
                break
            result = api.simulate(round_id, si, row, col, steps=50)
            if "error" in result:
                log.warning(f"Obs error seed {si} ({row},{col}): {result.get('error')}")
                if result.get('error') == 'budget_exhausted':
                    n_obs = obs_used
                    break
                continue
            viewport = np.array(result.get("grid", []))
            if viewport.ndim != 2:
                continue
            obs_used += 1
            raw_obs.append({"seed": si, "row": row, "col": col, "grid": viewport.tolist()})
            vh, vw = viewport.shape
            cls = seed_cls[si]
            sb = seed_sb[si]
            nf = seed_nf[si]
            co = seed_co[si]
            np_ = seed_np[si]
            for vy in range(vh):
                for vx in range(vw):
                    gy, gx = row + vy, col + vx
                    if gy >= H or gx >= W:
                        continue
                    ic = int(cls[gy, gx])
                    key = (ic, int(sb[gy, gx]), int(nf[gy, gx]), int(co[gy, gx]), int(np_[gy, gx]))
                    oc = GRID_TO_CLASS.get(int(viewport[vy, vx]), 0)
                    obs_counts.setdefault(key, np.zeros(6))
                    obs_total.setdefault(key, 0)
                    obs_counts[key][oc] += 1
                    obs_total[key] += 1
            time.sleep(0.3)
        if obs_used >= n_obs:
            break

    log.info(f"Used {obs_used} observations, {sum(obs_total.values())} cell comparisons")

    # Save observations
    obs_path = DATA_DIR / f"observations_{round_id[:8]}.json"
    existing_count = 0
    if obs_path.exists():
        try:
            with open(obs_path) as f:
                existing_count = len(json.load(f))
        except Exception:
            pass
    if len(raw_obs) > existing_count:
        with open(obs_path, "w") as f:
            json.dump(raw_obs, f)
        log.info(f"Saved {len(raw_obs)} observations to {obs_path.name}")

    # Build observation LUT (frequency tables)
    obs_lut = {}
    for key, counts in obs_counts.items():
        n = obs_total[key]
        if n >= MIN_OBS_N:
            obs_lut[key] = counts / n

    log.info(f"Obs LUT: {len(obs_lut)} qualified buckets (min_n={MIN_OBS_N})")

    # Build final predictions: HGB + optional obs blend
    predictions = {}
    for si in range(n_seeds):
        pred = hgb_preds[si].copy()  # Start with HGB prediction
        cls = seed_cls[si]
        sb = seed_sb[si]
        nf = seed_nf[si]
        co = seed_co[si]
        np_ = seed_np[si]

        # Blend with observations where available
        if obs_lut and OBS_BLEND_ALPHA > 0:
            for y in range(H):
                for x in range(W):
                    if cls[y, x] == 5:  # Mountain stays fixed
                        continue
                    key = (int(cls[y, x]), int(sb[y, x]), int(nf[y, x]), int(co[y, x]), int(np_[y, x]))
                    if key in obs_lut:
                        obs_freq = obs_lut[key]
                        n = obs_total[key]
                        # Adaptive alpha: trust obs more when sample size is large
                        weight = min(OBS_BLEND_ALPHA * (n / 100), 0.5)
                        hgb_p = np.clip(pred[y, x], 1e-10, None)
                        obs_p = np.clip(obs_freq, 1e-10, None)
                        # Log-space blend
                        log_blend = (1 - weight) * np.log(hgb_p) + weight * np.log(obs_p)
                        pred[y, x] = np.exp(log_blend)

        # Mountain fix (ensure consistency)
        mtn = (cls == 5)
        if mtn.any():
            pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
        pred[~mtn, 5] = 0.0
        s = pred[~mtn].sum(axis=-1, keepdims=True)
        s = np.where(s == 0, 1, s)
        pred[~mtn] /= s

        # Temperature scaling
        if TEMPERATURE != 1.0:
            non_mtn = ~mtn
            if non_mtn.any():
                p = pred[non_mtn]
                p = np.clip(p, 1e-10, None)
                p = np.exp(np.log(p) / TEMPERATURE)
                pred[non_mtn] = p

        # Clip floor + normalize
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        predictions[si] = pred

    return predictions


# ── Lock management ──

def acquire_lock():
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            import psutil
            if psutil.pid_exists(old_pid):
                proc = psutil.Process(old_pid)
                if proc.is_running() and 'python' in proc.name().lower():
                    log.error(f"Another auto_runner_v3 is running (PID {old_pid}). Exiting.")
                    sys.exit(1)
            log.info(f"Removing stale lock (PID {old_pid})")
        except (ValueError, ImportError):
            log.info("Removing stale lock (cannot verify)")
        LOCK_FILE.unlink(missing_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))
    atexit.register(release_lock)
    log.info(f"Acquired lock (PID {os.getpid()})")


def release_lock():
    try:
        if LOCK_FILE.exists():
            if int(LOCK_FILE.read_text().strip()) == os.getpid():
                LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"submitted_rounds": [], "last_check": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def compute_stats(pred, ig):
    """Quick stats for logging."""
    p = pred.reshape(-1, 6)
    ent = -np.sum(p * np.log(np.clip(p, 1e-10, None)), axis=1).mean()
    conf = p.max(axis=1).mean()
    return {"ent": ent, "conf": conf}


# ── Fetch GT for completed rounds ──

def fetch_gt_for_round(api, round_id):
    gt_path = DATA_DIR / f"ground_truth_{round_id[:8]}.json"
    if gt_path.exists():
        return
    try:
        full_round = api._get(f"/rounds/{round_id}")
        if full_round.get("status") != "completed":
            return
        n_seeds = len(full_round.get("initial_states", []))
        gt_data = {}
        for si in range(n_seeds):
            try:
                analysis = api.get_analysis(round_id, si)
                ig = full_round["initial_states"][si]["grid"]
                gt = analysis.get("ground_truth", analysis.get("probabilities"))
                if gt is not None and ig is not None:
                    gt_data[str(si)] = {"ground_truth": gt, "initial_grid": ig}
                    log.info(f"  Fetched GT seed {si} for {round_id[:8]}")
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"  Failed to fetch GT seed {si}: {e}")
        if gt_data:
            with open(gt_path, "w") as f:
                json.dump(gt_data, f)
            log.info(f"Saved ground truth: {gt_path.name} ({len(gt_data)} seeds)")
    except Exception as e:
        log.warning(f"Failed to fetch GT for {round_id[:8]}: {e}")


# ── Submit a round ──

def submit_round(api, rd, hgb_models):
    round_id = rd['id']
    n_seeds = len(rd.get('initial_states', []))
    log.info(f"Processing round {rd['round_number']} (id={round_id[:8]}, {n_seeds} seeds)")

    grids = [np.array(st["grid"]) for st in rd["initial_states"]]
    final = observe_and_predict(api, round_id, grids, hgb_models)

    success = True
    for si in range(n_seeds):
        p = final[si]
        p = np.clip(p, 1e-6, None)
        p = p / p.sum(axis=-1, keepdims=True)

        se = compute_stats(p, grids[si])

        ok = False
        text = ""
        for attempt in range(3):
            ok, text = api.submit_prediction(round_id, si, p.tolist())
            if ok:
                break
            log.warning(f"  Seed {si}: attempt {attempt+1} failed: {text[:100]}")
            time.sleep(2 ** attempt)

        status = "OK" if ok else f"FAIL: {text[:100]}"
        log.info(f"  Seed {si}: {status} (ent={se['ent']:.3f} conf={se['conf']:.3f})")
        if not ok:
            success = False
        time.sleep(0.5)

    return success


# ── Main loop ──

def main():
    args = sys.argv[1:]
    poll_interval = 120

    for i, arg in enumerate(args):
        if arg == "--poll-interval" and i + 1 < len(args):
            poll_interval = int(args[i + 1])

    acquire_lock()
    api = AstarAPI()
    state = load_state()

    log.info(f"Auto-runner V3 (HGB) started (poll={poll_interval}s, T={TEMPERATURE}, clip={CLIP_FLOOR})")
    log.info(f"HGB params: {HGB_PARAMS}")
    log.info(f"Previously submitted: {[r[:8] for r in state['submitted_rounds']]}")

    # Train HGB models on startup
    log.info("Training HGB models on all GT data...")
    hgb_models = train_hgb_models()
    if hgb_models is None:
        log.error("Cannot start without GT data!")
        return
    log.info("HGB models ready!")

    gt_count = len(list(DATA_DIR.glob("ground_truth_*.json")))

    while True:
        try:
            state["last_check"] = datetime.now(timezone.utc).isoformat()

            rounds = api.get_rounds()
            active_round = None
            completed_rounds = []
            for r in rounds:
                if r["status"] == "active":
                    active_round = r
                elif r["status"] == "completed":
                    completed_rounds.append(r)

            # Fetch GT for completed rounds
            new_gt = False
            for cr in completed_rounds:
                gt_before = len(list(DATA_DIR.glob("ground_truth_*.json")))
                fetch_gt_for_round(api, cr["id"])
                if len(list(DATA_DIR.glob("ground_truth_*.json"))) > gt_before:
                    new_gt = True

            # Retrain HGB if new GT acquired
            if new_gt:
                new_count = len(list(DATA_DIR.glob("ground_truth_*.json")))
                log.info(f"New GT acquired ({gt_count} -> {new_count}), retraining HGB...")
                hgb_models = train_hgb_models()
                gt_count = new_count

            # Process active round
            if active_round and active_round["id"] not in state["submitted_rounds"]:
                log.info(f"NEW ROUND: {active_round['round_number']} (id={active_round['id'][:8]})")

                full_rd = api._get(f"/rounds/{active_round['id']}")
                my_rounds = api._get("/my-rounds")
                for mr in my_rounds:
                    if mr["id"] == active_round["id"]:
                        for k, v in mr.items():
                            if k not in full_rd:
                                full_rd[k] = v
                        break

                if full_rd.get("status") == "active":
                    success = submit_round(api, full_rd, hgb_models)
                    if success:
                        state["submitted_rounds"].append(active_round["id"])
                        save_state(state)
                        log.info(f"Round {active_round['round_number']} submitted!")
                    else:
                        log.warning(f"Round {active_round['round_number']} had submission failures")
                else:
                    log.info("Round closed before submission")

            save_state(state)

        except Exception as e:
            log.error(f"Error: {e}")
            log.error(traceback.format_exc())
            time.sleep(min(poll_interval * 2, 600))

        log.info(f"Next check in {poll_interval}s...")
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
