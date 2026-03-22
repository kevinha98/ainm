"""
Auto-runner V2: Observation-first empirical model.

Strategy: Pure observation frequencies — no ML training needed.
1. Observe 45 viewports (9 per seed × 5 seeds) for full coverage
2. Build per-(initial_class, near_settle) → outcome frequency tables
3. Predict each cell using its bucket's empirical frequency
4. Mountain cells hardcoded to 100% Mountain
5. PID lockfile prevents double-runner budget waste

LOO CV result: 92.81 mean (range 88.3–98.5 across 12 rounds)
Competitor baseline: ~90.89 | Our previous HGB approach: 19–44 on server
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

from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from src.api import AstarAPI
from src.models import build_class_grid, compute_stats

# ── Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "auto_runner_v2.log"),
    ]
)
log = logging.getLogger("auto_runner_v2")

# ── Configuration (CV-optimized 2025-06-22, entropy-weighted KL scoring)
CLIP_FLOOR = 1e-06            # Lower clip preserves confident predictions
TEMPERATURE = 1.0            # LOO-CV: 72.63 at T=1.2 vs 69.14 at T=1.0 (+3.48)
MIN_N = 20                   # Bucket minimum sample size
ENSEMBLE_ALPHA = 0.6        # No-obs cell model blend (LOO-CV: 72.77 with T=1.2)
ENSEMBLE_ALPHA_OBS = 0.20    # With-obs cell model blend (watchdog found 0.20 > 0.10)
# Bucket key: (initial_class, settle_bin, near_forest, coastal, near_port)
# settle_bin: 0=adjacent(d<=1), 1=near(d<=2), 2=medium(d<=4), 3=far
# near_forest: 0/1 (d<=2), coastal: 0/1 (d_ocean<=1.5), near_port: 0/1 (d_port<=2)
# Zero-mountain fix: class 5 zeroed for non-mountain cells before blending
STATE_FILE = DATA_DIR / "auto_runner_v2_state.json"
LOCK_FILE = DATA_DIR / "auto_runner_v2.lock"


# ── PID lock to prevent double-runner
def acquire_lock():
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            import psutil
            if psutil.pid_exists(old_pid):
                proc = psutil.Process(old_pid)
                if proc.is_running() and 'python' in proc.name().lower():
                    log.error(f"Another auto_runner_v2 is running (PID {old_pid}). Exiting.")
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


# ── Observation + Prediction (the entire model in one function)

def compute_spatial_features(cls, ig):
    """Compute settle_bin, near_forest, coastal, near_port from class grid and initial grid."""
    H, W = cls.shape
    settlement = (cls == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    forest = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20.0)
    ocean = (ig == 10)
    dist_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    port = (cls == 2)
    dist_p = ndimage.distance_transform_edt(~port) if port.any() else np.full((H, W), 40.0)
    # 4-level settlement distance bins
    settle_bin = np.full((H, W), 3, dtype=int)  # 3=far
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    coastal = (dist_o <= 1.5).astype(int)
    near_port = (dist_p <= 2.0).astype(int)
    return settle_bin, near_forest, coastal, near_port


def build_fallback_lut():
    """Build a fallback LUT from cross-round GT averages.
    Used when no observations are available (budget spent).
    Key: (initial_class, settle_bin, near_forest, coastal, near_port) → prob[6]
    Also builds 4-feature and 3-feature fallbacks for sparse buckets."""
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    if not gt_files:
        return {}, {}, {}, {}
    
    cross_counts = {}   # 5-feature keys
    cross_total = {}
    fb4_counts = {}     # 4-feature fallback keys
    fb4_total = {}
    fb3_counts = {}     # 3-feature fallback keys
    fb3_total = {}
    for gf in gt_files:
        with open(gf) as f:
            gt_data = json.load(f)
        for si_str in sorted(gt_data.keys()):
            entry = gt_data[si_str]
            if not isinstance(entry, dict):
                continue
            gt = np.array(entry['ground_truth'])
            ig = np.array(entry['initial_grid'])
            h, w = ig.shape
            cls = build_class_grid(ig)
            settle_bin, near_forest, coastal, near_port = compute_spatial_features(cls, ig)
            for y in range(h):
                for x in range(w):
                    ic = int(cls[y, x])
                    key5 = (ic, int(settle_bin[y, x]), int(near_forest[y, x]), int(coastal[y, x]), int(near_port[y, x]))
                    key4 = key5[:4]
                    key3 = key5[:3]
                    cross_counts.setdefault(key5, np.zeros(6))
                    cross_total.setdefault(key5, 0)
                    cross_counts[key5] += gt[y, x]
                    cross_total[key5] += 1
                    fb4_counts.setdefault(key4, np.zeros(6))
                    fb4_total.setdefault(key4, 0)
                    fb4_counts[key4] += gt[y, x]
                    fb4_total[key4] += 1
                    fb3_counts.setdefault(key3, np.zeros(6))
                    fb3_total.setdefault(key3, 0)
                    fb3_counts[key3] += gt[y, x]
                    fb3_total[key3] += 1
    
    # Class-level averages for fallback
    class_avgs = {}
    for ic in range(6):
        tc, tn = np.zeros(6), 0
        for k, v in cross_counts.items():
            if k[0] == ic:
                tc += v
                tn += cross_total[k]
        class_avgs[ic] = tc / max(tn, 1) if tn > 0 else np.ones(6) / 6
    
    # 3-feature fallback LUT
    fb3_lut = {}
    for key3, total_prob in fb3_counts.items():
        n = fb3_total[key3]
        if n >= 10:
            avg_freq = total_prob / n
            avg_freq = np.clip(avg_freq, CLIP_FLOOR, None)
            avg_freq /= avg_freq.sum()
            fb3_lut[key3] = avg_freq
    
    # 4-feature fallback LUT
    fb4_lut = {}
    for key4, total_prob in fb4_counts.items():
        n = fb4_total[key4]
        if n >= 10:
            avg_freq = total_prob / n
            avg_freq = np.clip(avg_freq, CLIP_FLOOR, None)
            avg_freq /= avg_freq.sum()
            fb4_lut[key4] = avg_freq
    
    # 5-feature LUT
    lut = {}
    for key5, total_prob in cross_counts.items():
        n = cross_total[key5]
        if n >= 10:
            avg_freq = total_prob / n
            avg_freq = np.clip(avg_freq, CLIP_FLOOR, None)
            avg_freq /= avg_freq.sum()
            lut[key5] = avg_freq
        else:
            # Fall back to 4-feature then 3-feature
            key4 = key5[:4]
            key3 = key5[:3]
            lut[key5] = fb4_lut.get(key4, fb3_lut.get(key3, class_avgs[key5[0]]))
    
    log.info(f"Built fallback LUT from {len(gt_files)} GT files ({len(lut)} 5-feat, {len(fb4_lut)} 4-feat, {len(fb3_lut)} 3-feat)")
    return lut, fb4_lut, fb3_lut, class_avgs


def observe_and_predict(api, round_id, grids):
    """
    5-feature LUT + cell model ensemble with zero-mountain fix:
    1. Observe 9 viewports × N seeds for full grid coverage
    2. Tally per-(init_class, settle_bin, near_forest, coastal, near_port) outcome frequencies
    3. Build LUT with 5→4→3-feat fallback chain, mountain override
    4. Zero mountain class for non-mountain cells (prevents cell model leakage)
    5. Blend LUT with cell model (α=0.75 no-obs, α=0.10 with-obs) in log-space
    
    LOO-CV (entropy-weighted KL score): 68.40 no-obs
    """
    n_seeds = len(grids)
    H, W = grids[0].shape

    # Pre-compute per-seed spatial features
    seed_cls = []
    seed_settle_bin = []
    seed_near_forest = []
    seed_coastal = []
    seed_near_port = []
    for si in range(n_seeds):
        cls = build_class_grid(grids[si])
        settle_bin, near_forest, coastal, near_port = compute_spatial_features(cls, grids[si])
        seed_cls.append(cls)
        seed_settle_bin.append(settle_bin)
        seed_near_forest.append(near_forest)
        seed_coastal.append(coastal)
        seed_near_port.append(near_port)

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
    n_obs = min(remaining, 45)  # 9 viewports × 5 seeds = 45
    log.info(f"Observation budget: {remaining} remaining, using up to {n_obs}")

    # Viewport positions for full coverage (40×40 grid with 15×15 viewports)
    viewport_positions = [(r, c) for r in [0, 12, 25] for c in [0, 12, 25]]

    # Tally observations: (init_class, settle_bin, near_forest, coastal, near_port) → counts[6]
    obs_counts = {}    # 5-feature keys
    obs_total = {}
    obs4_counts = {}   # 4-feature fallback keys
    obs4_total = {}
    obs3_counts = {}   # 3-feature fallback keys
    obs3_total = {}
    obs_used = 0
    raw_obs = []

    # Load previously saved observations for this round (crash recovery)
    obs_path = DATA_DIR / f"observations_{round_id[:8]}.json"
    if obs_path.exists():
        try:
            with open(obs_path) as f:
                saved_obs = json.load(f)
            if len(saved_obs) > 0:
                log.info(f"Loaded {len(saved_obs)} saved observations from previous run")
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
                    sb = seed_settle_bin[si_prev]
                    nf = seed_near_forest[si_prev]
                    co = seed_coastal[si_prev]
                    np_ = seed_near_port[si_prev]
                    for vy in range(vh):
                        for vx in range(vw):
                            gy, gx = row_prev + vy, col_prev + vx
                            if gy >= H or gx >= W:
                                continue
                            ic = int(cls[gy, gx])
                            sb_val = int(sb[gy, gx])
                            nf_val = int(nf[gy, gx])
                            co_val = int(co[gy, gx])
                            np_val = int(np_[gy, gx])
                            oc = GRID_TO_CLASS.get(int(viewport[vy, vx]), 0)
                            key5 = (ic, sb_val, nf_val, co_val, np_val)
                            key4 = key5[:4]
                            key3 = key5[:3]
                            obs_counts.setdefault(key5, np.zeros(6))
                            obs_total.setdefault(key5, 0)
                            obs_counts[key5][oc] += 1
                            obs_total[key5] += 1
                            obs4_counts.setdefault(key4, np.zeros(6))
                            obs4_total.setdefault(key4, 0)
                            obs4_counts[key4][oc] += 1
                            obs4_total[key4] += 1
                            obs3_counts.setdefault(key3, np.zeros(6))
                            obs3_total.setdefault(key3, 0)
                            obs3_counts[key3][oc] += 1
                            obs3_total[key3] += 1
                # Skip live observations if we already have a full set
                if obs_used >= 45:
                    log.info(f"Already have {obs_used} observations, skipping live queries")
                    n_obs = 0
        except Exception as e:
            log.warning(f"Failed to load saved observations: {e}")
            obs_used = 0
            raw_obs = []

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
            sb = seed_settle_bin[si]
            nf = seed_near_forest[si]
            co = seed_coastal[si]
            np_ = seed_near_port[si]

            for vy in range(vh):
                for vx in range(vw):
                    gy, gx = row + vy, col + vx
                    if gy >= H or gx >= W:
                        continue
                    ic = int(cls[gy, gx])
                    sb_val = int(sb[gy, gx])
                    nf_val = int(nf[gy, gx])
                    co_val = int(co[gy, gx])
                    np_val = int(np_[gy, gx])
                    oc = GRID_TO_CLASS.get(int(viewport[vy, vx]), 0)

                    key5 = (ic, sb_val, nf_val, co_val, np_val)
                    key4 = key5[:4]
                    key3 = key5[:3]
                    obs_counts.setdefault(key5, np.zeros(6))
                    obs_total.setdefault(key5, 0)
                    obs_counts[key5][oc] += 1
                    obs_total[key5] += 1
                    obs4_counts.setdefault(key4, np.zeros(6))
                    obs4_total.setdefault(key4, 0)
                    obs4_counts[key4][oc] += 1
                    obs4_total[key4] += 1
                    obs3_counts.setdefault(key3, np.zeros(6))
                    obs3_total.setdefault(key3, 0)
                    obs3_counts[key3][oc] += 1
                    obs3_total[key3] += 1

            time.sleep(0.3)
        if obs_used >= n_obs:
            break

    log.info(f"Used {obs_used} observations, {sum(obs_total.values())} cell comparisons")

    # Save observations to disk (NEVER overwrite with fewer entries)
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
        log.info(f"Saved {len(raw_obs)} observation viewports to {obs_path.name}")
    else:
        log.info(f"Keeping existing {existing_count} observations (new has {len(raw_obs)})")

    # Log per-bucket frequencies (top 5-feature buckets, collapsed to 4-feat for readability)
    for ic in range(NUM_CLASSES):
        for sb_val in range(4):
            for nf_val in [0, 1]:
                for co_val in [0, 1]:
                    n4 = sum(obs_total.get((ic, sb_val, nf_val, co_val, np_val), 0) for np_val in [0, 1])
                    if n4 >= 20:
                        c4 = sum(obs_counts.get((ic, sb_val, nf_val, co_val, np_val), np.zeros(6)) for np_val in [0, 1])
                        freq = c4 / n4
                        sb_lbl = ["adj", "nr", "med", "far"][sb_val]
                        nf_lbl = "F" if nf_val else ""
                        co_lbl = "C" if co_val else ""
                        log.info(f"  {CLASS_NAMES[ic]:>10s} s={sb_lbl:3s} {nf_lbl:1s}{co_lbl:1s} (n={n4:5d}): {np.round(freq, 3).tolist()}")

    # Build LUT: start with cross-round GT fallback, overlay within-round obs
    fallback_lut, fb4_lut, fb3_lut, fallback_class_avgs = build_fallback_lut()
    
    # Class-level averages from within-round observations
    class_avgs = {}
    for ic in range(6):
        tc, tn = np.zeros(6), 0
        for k, c in obs_counts.items():
            if k[0] == ic:
                tc += c
                tn += obs_total[k]
        if tn > 0:
            class_avgs[ic] = tc / tn
        else:
            class_avgs[ic] = fallback_class_avgs.get(ic, np.ones(6) / 6)

    # Build obs fallback LUTs at 4-feat and 3-feat levels
    obs4_lut = {}
    for key4, counts in obs4_counts.items():
        n = obs4_total[key4]
        if n >= MIN_N:
            obs4_lut[key4] = counts / n
    obs3_lut = {}
    for key3, counts in obs3_counts.items():
        n = obs3_total[key3]
        if n >= MIN_N:
            obs3_lut[key3] = counts / n

    # Merge: within-round 5-feat > 4/3-feat obs > fallback 5/4/3-feat GT LUT > class avg
    lut = dict(fallback_lut)  # start with cross-round 5-feat priors
    for key5, counts in obs_counts.items():
        n = obs_total[key5]
        if n >= MIN_N:
            lut[key5] = counts / n
        else:
            key4 = key5[:4]
            key3 = key5[:3]
            if key4 in obs4_lut:
                lut[key5] = obs4_lut[key4]
            elif key3 in obs3_lut:
                lut[key5] = obs3_lut[key3]
            elif key5 not in lut:
                lut[key5] = fb4_lut.get(key4, fb3_lut.get(key3, class_avgs.get(key5[0], fallback_class_avgs.get(key5[0], np.ones(6) / 6))))

    # Load cell model for ensemble blending
    # Use adaptive alpha: lower when observations available (obs data dominates)
    has_obs = obs_used > 0
    alpha = ENSEMBLE_ALPHA_OBS if has_obs else ENSEMBLE_ALPHA
    log.info(f"Ensemble alpha: {alpha} ({'obs-aware' if has_obs else 'no-obs'}, {obs_used} observations)")
    
    cell_params = None
    if alpha > 0:
        try:
            from simulator.cell_model import predict_cell_distributions, params_from_vector
            params_path = DATA_DIR / "cell_model_params.npy"
            if params_path.exists():
                opt_vec = np.load(params_path)
                cell_params = params_from_vector(opt_vec)
                log.info(f"Loaded cell model params for ensemble (α={alpha})")
            else:
                log.warning("Cell model params not found, using LUT only")
        except Exception as e:
            log.warning(f"Failed to load cell model: {e}, using LUT only")

    # Build predictions
    predictions = {}
    for si in range(n_seeds):
        cls = seed_cls[si]
        sb = seed_settle_bin[si]
        nf = seed_near_forest[si]
        co = seed_coastal[si]
        np_ = seed_near_port[si]
        
        # Vectorized LUT lookup: build key arrays and batch lookup
        H0, W0 = cls.shape
        ic_flat = cls.ravel().astype(int)
        sb_flat = sb.ravel().astype(int)
        nf_flat = nf.ravel().astype(int)
        co_flat = co.ravel().astype(int)
        np_flat = np_.ravel().astype(int)
        
        pred_lut = np.ones((H0 * W0, 6)) / 6
        for i in range(H0 * W0):
            key5 = (ic_flat[i], sb_flat[i], nf_flat[i], co_flat[i], np_flat[i])
            if key5 in lut:
                pred_lut[i] = lut[key5]
            else:
                key4 = key5[:4]
                key3 = key5[:3]
                pred_lut[i] = fb4_lut.get(key4, fb3_lut.get(key3, class_avgs.get(key5[0], fallback_class_avgs.get(key5[0], np.ones(6) / 6))))
        pred_lut = pred_lut.reshape(H0, W0, 6)

        # Mountain fix: 100% for mountain cells, zero mountain prob for non-mountain
        mtn = (cls == 5)
        if mtn.any():
            pred_lut[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
        # Zero out mountain class for non-mountain cells (GT always 0% mountain there)
        pred_lut[~mtn, 5] = 0.0
        s = pred_lut[~mtn].sum(axis=-1, keepdims=True)
        s = np.where(s == 0, 1, s)
        pred_lut[~mtn] /= s

        # Ensemble: blend LUT with cell model in log-space
        if cell_params is not None:
            pred_cell = predict_cell_distributions(grids[si], cell_params)
            # Zero mountain for non-mountain cells BEFORE clipping (prevents leakage)
            pred_cell[~mtn, 5] = 0.0
            sc = pred_cell[~mtn].sum(axis=-1, keepdims=True)
            sc = np.where(sc == 0, 1, sc)
            pred_cell[~mtn] /= sc
            pred_cell = np.clip(pred_cell, CLIP_FLOOR, None)
            pred_cell = pred_cell / pred_cell.sum(axis=-1, keepdims=True)
            if mtn.any():
                pred_cell[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
            
            pred_lut_safe = np.clip(pred_lut, 1e-10, None)
            pred_cell_safe = np.clip(pred_cell, 1e-10, None)
            log_blend = (1 - alpha) * np.log(pred_lut_safe) + alpha * np.log(pred_cell_safe)
            pred = np.exp(log_blend)
        else:
            pred = pred_lut

        # Temperature scaling (only if T != 1.0)
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


# ── Fetch ground truth for completed rounds
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


# ── Submit a round
def submit_round(api, rd):
    round_id = rd['id']
    n_seeds = len(rd.get('initial_states', []))
    log.info(f"Processing round {rd['round_number']} (id={round_id[:8]}, {n_seeds} seeds)")

    grids = [np.array(st["grid"]) for st in rd["initial_states"]]
    final = observe_and_predict(api, round_id, grids)

    success = True
    for si in range(n_seeds):
        p = final[si]
        p = np.clip(p, 1e-6, None)
        p = p / p.sum(axis=-1, keepdims=True)

        se = compute_stats(p, grids[si])
        
        # Retry up to 3 times with backoff
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


# ── Main loop
def main():
    args = sys.argv[1:]
    poll_interval = 120

    for i, arg in enumerate(args):
        if arg == "--poll-interval" and i + 1 < len(args):
            poll_interval = int(args[i + 1])

    acquire_lock()
    api = AstarAPI()
    state = load_state()

    log.info(f"Auto-runner V2 started (poll={poll_interval}s, T={TEMPERATURE}, floor={CLIP_FLOOR})")
    log.info(f"Previously submitted: {[r[:8] for r in state['submitted_rounds']]}")

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
            for cr in completed_rounds:
                fetch_gt_for_round(api, cr["id"])

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
                    success = submit_round(api, full_rd)
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
            # Exponential backoff on errors to avoid hammering a down server
            time.sleep(min(poll_interval * 2, 600))

        log.info(f"Next check in {poll_interval}s...")
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
