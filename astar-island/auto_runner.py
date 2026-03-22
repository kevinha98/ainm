"""
Auto-runner: Polls for new rounds and submits HGB + observation calibration.
Run as a persistent background process.
Usage: python auto_runner.py [--poll-interval 300] [--no-obs]
"""
import json
import time
import sys
import os
import logging
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from src.api import AstarAPI
from src.models import build_class_grid, compute_stats

# ── Logging setup
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "auto_runner.log"),
    ]
)
log = logging.getLogger("auto_runner")

# ── Configuration
CLIP_FLOOR = 0.0001
TEMPERATURE = 1.15  # Soften HGB preds before calibration (LOO: 97.22 vs 96.96 at T=1.0)
STATE_FILE = DATA_DIR / "auto_runner_state.json"


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"submitted_rounds": [], "last_check": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Feature extraction (same as v10)
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
    
    features = np.concatenate([
        cls_oh,
        dist_ocean[:,:,None], dist_settle[:,:,None], dist_forest[:,:,None], dist_mountain[:,:,None],
        n_s3[:,:,None], n_s7[:,:,None], n_f7[:,:,None], n_o7[:,:,None], n_e7[:,:,None], n_s11[:,:,None],
        is_coast[:,:,None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1])


# ── Train HGB models
def train_hgb():
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    if not gt_files:
        log.warning("No ground truth files found!")
        return None
    
    X_all, Y_all = [], []
    for gf in gt_files:
        with open(gf) as f:
            gt_data = json.load(f)
        for si_str in sorted(gt_data.keys()):
            entry = gt_data[si_str]
            # Handle multiple GT file formats
            if isinstance(entry, dict):
                gt = np.array(entry.get('ground_truth', []))
                ig = np.array(entry.get('initial_grid', []))
            elif isinstance(entry, list):
                # Direct array format (no initial_grid embedded) — skip, need initial_grid
                continue
            else:
                continue
            if gt.size == 0 or ig.size == 0: continue
            X_all.append(extract_features(ig))
            Y_all.append(gt.reshape(-1, 6))
    
    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)
    log.info(f"Training HGB on {len(X_all)} cells from {len(gt_files)} GT files")
    
    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, learning_rate=0.05,
            min_samples_leaf=50, random_state=42
        )
        m.fit(X_all, Y_all[:, c])
        models.append(m)
    return models


# ── Observation calibration
def observe_and_calibrate(api, round_id, grids, models, max_obs=45):
    """Use observations to calibrate HGB predictions.
    Strategy: systematic full-coverage viewports (9 per seed × 5 seeds = 45 obs).
    Each viewport is 15×15 at top-left (row, col).
    """
    n_seeds = len(grids)
    H, W = grids[0].shape
    
    # First get uncalibrated HGB predictions + temperature scaling
    preds = {}
    for si in range(n_seeds):
        X = extract_features(grids[si])
        pred = np.column_stack([m.predict(X) for m in models])
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        # Temperature scaling: soften predictions so calibration has more room
        log_p = np.log(pred)
        scaled = log_p / TEMPERATURE
        scaled -= scaled.max(axis=-1, keepdims=True)
        pred = np.exp(scaled)
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        preds[si] = pred.reshape(H, W, 6)
    
    # Check budget
    # Check budget from my-rounds data
    try:
        my_rounds = api.get_my_rounds()
        remaining = 50  # default
        for mr in my_rounds:
            if mr.get("id") == round_id:
                remaining = mr.get("queries_max", 50) - mr.get("queries_used", 0)
                break
    except Exception:
        remaining = 50  # Assume full budget on API error
    
    n_obs = min(remaining, max_obs)
    
    if n_obs <= 0:
        log.info("No observation budget remaining")
        return preds
    
    log.info(f"Using up to {n_obs} observations for calibration (budget remaining: {remaining})")
    
    # Calculate optimal viewport positions for full grid coverage
    vp_size = 15
    def grid_positions(dim):
        n = max(1, -(-dim // vp_size))  # ceil division: minimum viewports needed
        if n == 1:
            return [0]
        step = (dim - vp_size) / (n - 1)
        return [round(i * step) for i in range(n)]
    
    row_starts = grid_positions(H)
    col_starts = grid_positions(W)
    viewport_positions = [(r, c) for r in row_starts for c in col_starts]
    log.info(f"  Grid {H}x{W}: {len(viewport_positions)} viewport positions per seed")
    
    per_class_obs = {c: np.zeros(NUM_CLASSES) for c in range(NUM_CLASSES)}
    per_class_pred = {c: np.zeros(NUM_CLASSES) for c in range(NUM_CLASSES)}
    per_class_n = {c: 0 for c in range(NUM_CLASSES)}
    
    # Settlement proximity split: separate stats for near-settlement vs far cells
    SETTLE_DIST_THRESH = 2.0
    settle_obs = {True: {c: np.zeros(NUM_CLASSES) for c in range(NUM_CLASSES)},
                  False: {c: np.zeros(NUM_CLASSES) for c in range(NUM_CLASSES)}}
    settle_pred = {True: {c: np.zeros(NUM_CLASSES) for c in range(NUM_CLASSES)},
                   False: {c: np.zeros(NUM_CLASSES) for c in range(NUM_CLASSES)}}
    settle_n = {True: {c: 0 for c in range(NUM_CLASSES)},
                False: {c: 0 for c in range(NUM_CLASSES)}}
    
    # Pre-compute near-settlement masks for each seed (dist <= 2.0)
    settle_masks = {}
    for si in range(n_seeds):
        cls_grid = build_class_grid(grids[si])
        settlement = (cls_grid == 1)
        dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20)
        settle_masks[si] = dist_s <= SETTLE_DIST_THRESH
    
    obs_used = 0
    for si in range(n_seeds):
        for row, col in viewport_positions:
            if obs_used >= n_obs:
                break
            
            result = api.simulate(round_id, si, row, col, steps=50)
            if "error" in result:
                log.warning(f"Observation error at seed {si} ({row},{col}): {result.get('error')}")
                continue
            
            viewport = np.array(result.get("grid", []))
            if viewport.ndim != 2:
                continue
            
            vh, vw = viewport.shape
            obs_used += 1
            
            cls_grid = build_class_grid(grids[si])
            near_settle = settle_masks[si]
            for vy in range(vh):
                for vx in range(vw):
                    gy = row + vy
                    gx = col + vx
                    if gy >= H or gx >= W:
                        continue
                    
                    ic = cls_grid[gy, gx]
                    sim_val = viewport[vy, vx]
                    observed_class = GRID_TO_CLASS.get(sim_val, 0)
                    ns = bool(near_settle[gy, gx])
                    
                    per_class_obs[ic][observed_class] += 1
                    per_class_pred[ic] += preds[si][gy, gx]
                    per_class_n[ic] += 1
                    
                    settle_obs[ns][ic][observed_class] += 1
                    settle_pred[ns][ic] += preds[si][gy, gx]
                    settle_n[ns][ic] += 1
            
            time.sleep(0.3)
        
        if obs_used >= n_obs:
            break
    
    log.info(f"Used {obs_used} observations, got {sum(per_class_n.values())} cell comparisons")
    
    # Apply settlement-proximity calibration (per-class + near/far settlement)
    calibrated = {}
    for si in range(n_seeds):
        pred = preds[si].copy().reshape(-1, 6)
        cls_grid = build_class_grid(grids[si])
        near_settle = settle_masks[si].ravel()
        
        for ns in [True, False]:
            for ic in range(NUM_CLASSES):
                n = settle_n[ns][ic]
                if n < 10:
                    # Fall back to global per-class if not enough near/far data
                    n = per_class_n[ic]
                    if n < 10:
                        continue
                    obs_freq = per_class_obs[ic] / n
                    pred_avg = per_class_pred[ic] / n
                else:
                    obs_freq = settle_obs[ns][ic] / n
                    pred_avg = settle_pred[ns][ic] / n
                
                ratio = np.ones(6)
                for k in range(6):
                    if pred_avg[k] > 0.01:
                        r = obs_freq[k] / pred_avg[k]
                        ratio[k] = np.clip(r, 0.01, 100.0)
                
                mask = (cls_grid.ravel() == ic) & (near_settle == ns)
                pred[mask] *= ratio
        
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        calibrated[si] = pred.reshape(H, W, 6)
    
    # Log calibration
    for ic in range(NUM_CLASSES):
        n = per_class_n[ic]
        if n >= 10:
            obs_freq = per_class_obs[ic] / n
            pred_avg = per_class_pred[ic] / n
            log.info(f"  {CLASS_NAMES[ic]:>15s} (n={n:4d}): obs={np.round(obs_freq, 3).tolist()} pred={np.round(pred_avg, 3).tolist()}")
    for ns in [True, False]:
        label = "NEAR_SETTLE" if ns else "FAR_SETTLE"
        for ic in range(NUM_CLASSES):
            n = settle_n[ns][ic]
            if n >= 10:
                obs_freq = settle_obs[ns][ic] / n
                pred_avg = settle_pred[ns][ic] / n
                log.info(f"  {label} {CLASS_NAMES[ic]:>12s} (n={n:4d}): obs={np.round(obs_freq, 3).tolist()} pred={np.round(pred_avg, 3).tolist()}")
    
    return calibrated


# ── Fetch ground truth for completed rounds
def fetch_gt_for_round(api, round_id):
    """Download ground truth for a completed round via /analysis endpoint."""
    gt_path = DATA_DIR / f"ground_truth_{round_id[:8]}.json"
    if gt_path.exists():
        return  # Already have it
    
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
                    gt_data[str(si)] = {
                        "ground_truth": gt,
                        "initial_grid": ig,
                    }
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


# ── Main submission logic
def submit_round(api, rd, use_obs=True):
    """Train HGB, optionally observe, and submit predictions."""
    round_id = rd['id']
    n_seeds = len(rd.get('initial_states', []))
    
    log.info(f"Processing round {rd['round_number']} (id={round_id[:8]}, {n_seeds} seeds)")
    
    # Parse grids
    grids = [np.array(st["grid"]) for st in rd["initial_states"]]
    
    # Train HGB
    models = train_hgb()
    if models is None:
        log.error("Could not train models!")
        return False
    
    # Predict with optional observation calibration
    if use_obs:
        final = observe_and_calibrate(api, round_id, grids, models, max_obs=45)
    else:
        final = {}
        for si in range(n_seeds):
            X = extract_features(grids[si])
            pred = np.column_stack([m.predict(X) for m in models])
            pred = np.clip(pred, CLIP_FLOOR, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            # Temperature scaling
            log_p = np.log(pred)
            scaled = log_p / TEMPERATURE
            scaled -= scaled.max(axis=-1, keepdims=True)
            pred = np.exp(scaled)
            pred = np.clip(pred, CLIP_FLOOR, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            final[si] = pred.reshape(grids[si].shape[0], grids[si].shape[1], 6)
    
    # Submit (with one retry on failure)
    success = True
    for si in range(n_seeds):
        p = final[si]
        p = np.clip(p, 1e-6, None)
        p /= p.sum(axis=-1, keepdims=True)
        
        se = compute_stats(p, grids[si])
        ok, text = api.submit_prediction(round_id, si, p.tolist())
        
        if not ok:
            log.warning(f"  Seed {si}: FAIL on first try: {text[:80]}, retrying in 3s...")
            time.sleep(3)
            ok, text = api.submit_prediction(round_id, si, p.tolist())
        
        status = "OK" if ok else f"FAIL: {text[:80]}"
        log.info(f"  Seed {si}: {status} (ent={se['ent']:.3f} conf={se['conf']:.3f})")
        if not ok:
            success = False
        time.sleep(0.5)
    
    return success


# ── Main loop
def main():
    args = sys.argv[1:]
    poll_interval = 300  # 5 min default
    use_obs = True
    
    for i, arg in enumerate(args):
        if arg == "--poll-interval" and i + 1 < len(args):
            poll_interval = int(args[i + 1])
        elif arg == "--no-obs":
            use_obs = False
    
    api = AstarAPI()
    state = load_state()
    
    log.info(f"Auto-runner started (poll={poll_interval}s, obs={use_obs})")
    log.info(f"Previously submitted: {state['submitted_rounds']}")
    
    while True:
        try:
            now = datetime.now(timezone.utc).isoformat()
            state["last_check"] = now
            
            # Check for active round
            rounds = api.get_rounds()
            active_round = None
            completed_rounds = []
            
            for r in rounds:
                if r["status"] == "active":
                    active_round = r
                elif r["status"] == "completed":
                    completed_rounds.append(r)
            
            # Fetch GT for any completed rounds we don't have
            for cr in completed_rounds:
                fetch_gt_for_round(api, cr["id"])
            
            # Process active round
            if active_round and active_round["id"] not in state["submitted_rounds"]:
                log.info(f"NEW ROUND DETECTED: {active_round['round_number']} (id={active_round['id'][:8]})")
                
                # Fetch full round data
                full_rd = api._get(f"/rounds/{active_round['id']}")
                
                # Merge user data
                my_rounds = api._get("/my-rounds")
                for mr in my_rounds:
                    if mr["id"] == active_round["id"]:
                        for k, v in mr.items():
                            if k not in full_rd:
                                full_rd[k] = v
                        break
                
                if full_rd.get("status") == "active":
                    success = submit_round(api, full_rd, use_obs=use_obs)
                    if success:
                        state["submitted_rounds"].append(active_round["id"])
                        save_state(state)
                        log.info(f"Round {active_round['round_number']} submitted successfully!")
                    else:
                        log.warning(f"Round {active_round['round_number']} submission had failures")
                else:
                    log.info(f"Round closed before submission")
            elif active_round:
                log.debug(f"Round {active_round['round_number']} already submitted")
            else:
                log.debug("No active round")
            
            save_state(state)
        
        except Exception as e:
            log.error(f"Error in main loop: {e}")
            log.error(traceback.format_exc())
        
        log.info(f"Next check in {poll_interval}s...")
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
