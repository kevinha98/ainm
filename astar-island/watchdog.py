"""
IMMORTAL WATCHDOG: Never dies. Manages Mode A (live rounds) and Mode B (experiments).

Mode A: Polls API every 60s, submits new rounds via auto_runner_v2 logic.
Mode B: Runs local experiments between rounds (cell model reopt, feature eng, param sweeps).

Self-healing: catches ALL exceptions, logs them, sleeps, retries. No unhandled crash possible.
"""
import json
import time
import sys
import os
import logging
import traceback
import signal
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────
PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT / "src"))
DATA_DIR = PROJECT / "data"
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "watchdog.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("watchdog")

# ── Singleton lock ─────────────────────────────────────────
LOCK_FILE = DATA_DIR / "watchdog.lock"

def acquire_lock():
    """Single-instance enforcement. Kills stale holders."""
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            import psutil
            if psutil.pid_exists(old_pid):
                proc = psutil.Process(old_pid)
                if proc.is_running() and 'python' in proc.name().lower():
                    # Check if it's actually a watchdog
                    cmdline = ' '.join(proc.cmdline())
                    if 'watchdog' in cmdline:
                        log.error(f"Another watchdog running (PID {old_pid}). Exiting.")
                        sys.exit(1)
            log.info(f"Removing stale watchdog lock (PID {old_pid})")
        except Exception:
            log.info("Removing stale watchdog lock")
        LOCK_FILE.unlink(missing_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))
    log.info(f"Watchdog lock acquired (PID {os.getpid()})")

def release_lock():
    try:
        if LOCK_FILE.exists() and int(LOCK_FILE.read_text().strip()) == os.getpid():
            LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass

import atexit
atexit.register(release_lock)

# ── Also kill any orphan auto_runner_v2 processes ──────────
def kill_orphan_runners():
    """Kill any auto_runner_v2 processes not managed by us."""
    try:
        import psutil
        my_pid = os.getpid()
        for p in psutil.process_iter(['pid', 'cmdline']):
            try:
                c = p.info['cmdline']
                if c and p.info['pid'] != my_pid:
                    cmdstr = ' '.join(c)
                    if 'auto_runner' in cmdstr and 'python' in cmdstr.lower():
                        log.info(f"Killing orphan auto_runner PID {p.info['pid']}")
                        p.kill()
            except Exception:
                pass
        # Clear auto_runner lock
        ar_lock = DATA_DIR / "auto_runner_v2.lock"
        ar_lock.unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"Failed to clean orphans: {e}")

# ── State ──────────────────────────────────────────────────
STATE_FILE = DATA_DIR / "watchdog_state.json"

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "submitted_rounds": [],
        "experiments_completed": [],
        "current_best": {"alpha": 0.65, "temperature": 1.2, "clip": 1e-6},
        "mode_b_cycle": 0,
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ═══════════════════════════════════════════════════════════
# MODE A: Live round submission
# ═══════════════════════════════════════════════════════════
def mode_a_check_and_submit(state):
    """Check for active rounds and submit if new. Returns True if a round was processed."""
    from src.api import AstarAPI
    api = AstarAPI()

    rounds = api.get_rounds()
    active = [r for r in rounds if r['status'] == 'active']
    completed = [r for r in rounds if r['status'] == 'completed']

    # Fetch GT for any completed rounds we don't have
    for cr in completed:
        gt_path = DATA_DIR / f"ground_truth_{cr['id'][:8]}.json"
        if not gt_path.exists():
            try:
                fetch_gt(api, cr['id'])
            except Exception as e:
                log.warning(f"GT fetch failed for {cr['id'][:8]}: {e}")

    if not active:
        return False

    rd = active[0]
    round_id = rd['id']
    round_num = rd.get('round_number', '?')

    if round_id in state['submitted_rounds']:
        return False

    log.info(f"MODE A: New round {round_num} detected (id={round_id[:8]})")

    # Get full round data
    full_rd = api._get(f"/rounds/{round_id}")
    my_rounds = api._get("/my-rounds")
    for mr in my_rounds:
        if mr["id"] == round_id:
            for k, v in mr.items():
                if k not in full_rd:
                    full_rd[k] = v
            break

    if full_rd.get("status") != "active":
        log.info("Round already closed")
        return False

    # Import and run the submission logic
    import auto_runner_v2 as ar
    # Clear its lock so it doesn't block
    ar_lock = DATA_DIR / "auto_runner_v2.lock"
    ar_lock.unlink(missing_ok=True)

    success = ar.submit_round(api, full_rd)

    if success:
        state['submitted_rounds'].append(round_id)
        # Also update auto_runner state for consistency
        ar_state = ar.load_state()
        if round_id not in ar_state['submitted_rounds']:
            ar_state['submitted_rounds'].append(round_id)
            ar.save_state(ar_state)
        save_state(state)
        log.info(f"MODE A: Round {round_num} submitted successfully!")
    else:
        log.error(f"MODE A: Round {round_num} submission had failures")

    return True


def fetch_gt(api, round_id):
    """Fetch ground truth for a completed round."""
    gt_path = DATA_DIR / f"ground_truth_{round_id[:8]}.json"
    if gt_path.exists():
        return

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
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"GT seed {si} failed: {e}")
            time.sleep(2)

    if gt_data:
        with open(gt_path, "w") as f:
            json.dump(gt_data, f)
        log.info(f"Saved GT: {gt_path.name} ({len(gt_data)} seeds)")


# ═══════════════════════════════════════════════════════════
# MODE B: Local experiments
# ═══════════════════════════════════════════════════════════

# Competition scoring
def comp_score(pred, gt):
    eps = 1e-15
    gt_s = np.clip(gt, eps, None)
    pred_s = np.clip(pred, eps, None)
    entropy = -np.sum(gt * np.log(gt_s), axis=-1)
    kl = np.sum(gt * np.log(gt_s / pred_s), axis=-1)
    te = entropy.sum()
    if te < eps:
        return 100.0
    wkl = np.sum(entropy * kl) / te
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


def load_gt_data():
    """Load all ground truth data for experiments."""
    from src.settings import GRID_TO_CLASS
    from scipy.ndimage import distance_transform_edt

    all_entries = []
    for gf in sorted(DATA_DIR.glob('ground_truth_*.json')):
        rid = gf.stem.replace('ground_truth_', '')
        with open(gf) as f:
            data = json.load(f)
        for sk, entry in data.items():
            ig = np.array(entry['initial_grid'])
            gt = np.array(entry['ground_truth'])
            if ig.shape == (40, 40) and gt.shape == (40, 40, 6):
                mapped = np.vectorize(GRID_TO_CLASS.get)(ig)
                H, W = mapped.shape

                settlement = (mapped == 1)
                dist_s = distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
                forest = (mapped == 4)
                dist_f = distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20.0)
                ocean = (ig == 10)
                dist_o = distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
                port = (mapped == 2)
                dist_p = distance_transform_edt(~port) if port.any() else np.full((H, W), 40.0)

                settle_bin = np.full((H, W), 3, dtype=int)
                settle_bin[dist_s <= 4.0] = 2
                settle_bin[dist_s <= 2.0] = 1
                settle_bin[dist_s <= 1.0] = 0
                near_forest = (dist_f <= 2.0).astype(int)
                coastal = (dist_o <= 1.5).astype(int)
                near_port = (dist_p <= 2.0).astype(int)
                mtn = (mapped == 5)

                all_entries.append({
                    'rid': rid, 'ig': ig, 'gt': gt, 'seed': sk,
                    'mapped': mapped, 'mtn': mtn,
                    'feats': (settle_bin, near_forest, coastal, near_port),
                    'dist_s': dist_s, 'dist_f': dist_f,
                })
    return all_entries


def build_tallies(all_entries):
    """Build global and per-round tallies."""
    rounds_map = defaultdict(list)
    for i, e in enumerate(all_entries):
        rounds_map[e['rid']].append(i)

    global_tally = defaultdict(lambda: np.zeros(6))
    round_tally = {}
    for rid in rounds_map:
        rt = defaultdict(lambda: np.zeros(6))
        for idx in rounds_map[rid]:
            e = all_entries[idx]
            mapped = e['mapped']
            sb, nf, co, np_ = e['feats']
            gt = e['gt']
            for r in range(40):
                for c in range(40):
                    key = (int(mapped[r, c]), int(sb[r, c]), int(nf[r, c]), int(co[r, c]), int(np_[r, c]))
                    rt[key] += gt[r, c]
                    global_tally[key] += gt[r, c]
        round_tally[rid] = dict(rt)

    return dict(global_tally), round_tally, rounds_map


def loo_cv(all_entries, global_tally, round_tally, rounds_map,
           alpha=0.65, clip=1e-6, temperature=1.2, min_n=20, cell_preds=None):
    """Full LOO-CV. Returns (mean_score, std, per_round_dict)."""
    scores = []
    per_round = {}

    for hold_rid in rounds_map:
        # Build LUT excluding held-out round
        train_tally = {}
        for k, v in global_tally.items():
            sub = round_tally[hold_rid].get(k, np.zeros(6))
            rem = v - sub
            if rem.sum() > 0:
                train_tally[k] = rem

        fb4 = defaultdict(lambda: np.zeros(6))
        fb3 = defaultdict(lambda: np.zeros(6))
        ct = np.zeros((6, 6))
        for k, v in train_tally.items():
            fb4[k[:4]] += v
            fb3[k[:3]] += v
            ct[k[0]] += v

        lut = {k: v / v.sum() for k, v in train_tally.items() if v.sum() >= min_n}
        fb4_lut = {k: v / v.sum() for k, v in fb4.items() if v.sum() >= min_n}
        fb3_lut = {k: v / v.sum() for k, v in fb3.items() if v.sum() >= min_n}
        ca = {}
        for ci in range(6):
            s = ct[ci].sum()
            ca[ci] = ct[ci] / s if s > 0 else np.ones(6) / 6

        round_scores = []
        for idx in rounds_map[hold_rid]:
            e = all_entries[idx]
            mapped = e['mapped']
            sb, nf, co, np_ = e['feats']
            mtn = e['mtn']

            # LUT prediction
            pred = np.zeros((40, 40, 6))
            for r in range(40):
                for c in range(40):
                    ic = int(mapped[r, c])
                    key5 = (ic, int(sb[r, c]), int(nf[r, c]), int(co[r, c]), int(np_[r, c]))
                    if key5 in lut:
                        pred[r, c] = lut[key5]
                    elif key5[:4] in fb4_lut:
                        pred[r, c] = fb4_lut[key5[:4]]
                    elif key5[:3] in fb3_lut:
                        pred[r, c] = fb3_lut[key5[:3]]
                    else:
                        pred[r, c] = ca.get(ic, np.ones(6) / 6)

            # Zero mountain for non-mountain
            pred[~mtn, 5] = 0.0
            s = pred[~mtn].sum(axis=-1, keepdims=True)
            s = np.where(s == 0, 1, s)
            pred[~mtn] /= s
            pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])

            # Blend with cell model
            if alpha > 0 and cell_preds is not None:
                cell_zm = cell_preds[idx]
                lut_log = np.log(np.clip(pred, clip, None))
                cell_log = np.log(np.clip(cell_zm, clip, None))
                mixed = (1 - alpha) * lut_log + alpha * cell_log
                mixed -= mixed.max(axis=-1, keepdims=True)
                p = np.exp(mixed)
                p /= p.sum(axis=-1, keepdims=True)
            else:
                p = pred.copy()

            # Temperature
            if temperature != 1.0:
                non_mtn = ~mtn
                if non_mtn.any():
                    p_nm = np.clip(p[non_mtn], 1e-10, None)
                    p_nm = np.exp(np.log(p_nm) / temperature)
                    p_nm /= p_nm.sum(axis=-1, keepdims=True)
                    p[non_mtn] = p_nm

            p = np.clip(p, 1e-8, None)
            p /= p.sum(axis=-1, keepdims=True)

            sc = comp_score(p, e['gt'])
            round_scores.append(sc)
            scores.append(sc)

        per_round[hold_rid] = np.mean(round_scores)

    return np.mean(scores), np.std(scores), per_round


def mode_b_experiment_cycle(state):
    """Run one cycle of Mode B experiments. Returns after ~5 minutes of work."""
    cycle = state.get('mode_b_cycle', 0)
    log.info(f"MODE B: Starting experiment cycle {cycle}")

    # Load data
    all_entries = load_gt_data()
    n_seeds = len(all_entries)
    n_rounds = len(set(e['rid'] for e in all_entries))
    log.info(f"MODE B: {n_seeds} seeds from {n_rounds} rounds")

    if n_seeds < 10:
        log.info("MODE B: Not enough GT data yet, skipping")
        return

    global_tally, round_tally, rounds_map = build_tallies(all_entries)

    # Precompute cell model predictions
    try:
        from simulator.cell_model import predict_cell_distributions, params_from_vector
        cell_vec = np.load(DATA_DIR / 'cell_model_params.npy')
        cell_p = params_from_vector(cell_vec)
        cell_preds = []
        for e in all_entries:
            cd = predict_cell_distributions(e['ig'], cell_p)
            mtn = e['mtn']
            cd[~mtn, 5] = 0.0
            s = cd.sum(axis=-1, keepdims=True)
            s = np.where(s == 0, 1, s)
            cd /= s
            cell_preds.append(cd)
    except Exception as ex:
        log.warning(f"Cell model load failed: {ex}")
        cell_preds = None

    best = state.get('current_best', {'alpha': 0.65, 'temperature': 1.2, 'clip': 1e-6})
    best_alpha = best['alpha']
    best_temp = best['temperature']
    best_clip = best['clip']

    # Current baseline
    baseline, baseline_std, _ = loo_cv(
        all_entries, global_tally, round_tally, rounds_map,
        alpha=best_alpha, clip=best_clip, temperature=best_temp,
        cell_preds=cell_preds
    )
    log.info(f"MODE B: Current baseline = {baseline:.2f} +/- {baseline_std:.2f} (a={best_alpha}, T={best_temp})")

    improved = False
    log_lines = []

    # Cycle through different experiment types
    experiment_type = cycle % 5

    if experiment_type == 0:
        # Temperature fine-sweep around current best
        log.info("MODE B: [Exp] Temperature fine-sweep")
        t_range = np.arange(max(0.8, best_temp - 0.15), best_temp + 0.20, 0.05)
        for T in t_range:
            T = round(T, 2)
            if T == best_temp:
                continue
            mean, std, _ = loo_cv(
                all_entries, global_tally, round_tally, rounds_map,
                alpha=best_alpha, clip=best_clip, temperature=T,
                cell_preds=cell_preds
            )
            delta = mean - baseline
            log.info(f"  T={T:.2f}: {mean:.2f} +/- {std:.2f} (d={delta:+.2f})")
            log_lines.append(f"T={T:.2f}: {mean:.2f} d={delta:+.2f}")
            if mean > baseline + 0.05:
                log.info(f"  ** IMPROVED: T={T} -> {mean:.2f}")
                best_temp = T
                baseline = mean
                improved = True

    elif experiment_type == 1:
        # Alpha fine-sweep
        log.info("MODE B: [Exp] Alpha fine-sweep")
        a_range = np.arange(max(0.0, best_alpha - 0.15), min(1.0, best_alpha + 0.20), 0.05)
        for a in a_range:
            a = round(a, 2)
            if a == best_alpha:
                continue
            mean, std, _ = loo_cv(
                all_entries, global_tally, round_tally, rounds_map,
                alpha=a, clip=best_clip, temperature=best_temp,
                cell_preds=cell_preds
            )
            delta = mean - baseline
            log.info(f"  a={a:.2f}: {mean:.2f} +/- {std:.2f} (d={delta:+.2f})")
            log_lines.append(f"a={a:.2f}: {mean:.2f} d={delta:+.2f}")
            if mean > baseline + 0.05:
                log.info(f"  ** IMPROVED: a={a} -> {mean:.2f}")
                best_alpha = a
                baseline = mean
                improved = True

    elif experiment_type == 2:
        # Cell model param perturbation (cheap local search)
        if cell_preds is not None:
            log.info("MODE B: [Exp] Cell model local perturbation")
            from simulator.cell_model import predict_cell_distributions, params_from_vector
            best_cell_score = baseline
            best_cell_vec = cell_vec.copy()
            rng = np.random.default_rng(cycle)

            for trial in range(20):
                # Perturb 1-3 random params by +/-10%
                new_vec = cell_vec.copy()
                n_perturb = rng.integers(1, 4)
                idxs = rng.choice(len(new_vec), n_perturb, replace=False)
                for idx in idxs:
                    factor = rng.uniform(0.85, 1.15)
                    new_vec[idx] = max(1e-4, new_vec[idx] * factor)

                try:
                    new_params = params_from_vector(new_vec)
                    new_cell_preds = []
                    for e in all_entries:
                        cd = predict_cell_distributions(e['ig'], new_params)
                        mtn = e['mtn']
                        cd[~mtn, 5] = 0.0
                        s = cd.sum(axis=-1, keepdims=True)
                        s = np.where(s == 0, 1, s)
                        cd /= s
                        new_cell_preds.append(cd)

                    mean, std, _ = loo_cv(
                        all_entries, global_tally, round_tally, rounds_map,
                        alpha=best_alpha, clip=best_clip, temperature=best_temp,
                        cell_preds=new_cell_preds
                    )
                    if mean > best_cell_score + 0.05:
                        log.info(f"  Trial {trial}: {mean:.2f} ** IMPROVED (perturbed {idxs})")
                        best_cell_score = mean
                        best_cell_vec = new_vec.copy()
                        cell_preds = new_cell_preds
                        improved = True
                    elif trial % 5 == 0:
                        log.info(f"  Trial {trial}: {mean:.2f}")
                except Exception:
                    pass

            if best_cell_score > baseline + 0.05:
                np.save(DATA_DIR / 'cell_model_params.npy', best_cell_vec)
                np.save(DATA_DIR / 'cell_model_params_backup.npy', cell_vec)
                baseline = best_cell_score
                log.info(f"  Deployed new cell params: {best_cell_score:.2f}")
                log_lines.append(f"Cell reopt: {best_cell_score:.2f}")
        else:
            log.info("MODE B: [Exp] Cell model skipped (no params)")

    elif experiment_type == 3:
        # Settle distance threshold sweep
        log.info("MODE B: [Exp] Settle distance thresholds")
        from scipy.ndimage import distance_transform_edt
        from src.settings import GRID_TO_CLASS

        thresh_configs = [
            (1.0, 2.0, 3.0),   # tighter bins
            (1.0, 2.0, 4.0),   # current
            (1.5, 3.0, 5.0),   # wider bins
            (1.0, 3.0, 6.0),   # very wide mid
            (0.5, 1.5, 3.0),   # very tight
            (1.0, 2.5, 5.0),   # medium-wide
        ]

        best_thresh = (1.0, 2.0, 4.0)
        best_thresh_score = baseline

        for t1, t2, t3 in thresh_configs:
            if (t1, t2, t3) == (1.0, 2.0, 4.0):
                continue  # skip current

            # Recompute features with new thresholds
            test_entries = []
            for e in all_entries:
                mapped = e['mapped']
                H, W = mapped.shape
                settlement = (mapped == 1)
                dist_s = distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
                forest = (mapped == 4)
                dist_f = distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20.0)
                ocean = (e['ig'] == 10)
                dist_o = distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
                port = (mapped == 2)
                dist_p = distance_transform_edt(~port) if port.any() else np.full((H, W), 40.0)

                sb = np.full((H, W), 3, dtype=int)
                sb[dist_s <= t3] = 2
                sb[dist_s <= t2] = 1
                sb[dist_s <= t1] = 0
                nf = (dist_f <= 2.0).astype(int)
                co = (dist_o <= 1.5).astype(int)
                np_ = (dist_p <= 2.0).astype(int)

                te = dict(e)
                te['feats'] = (sb, nf, co, np_)
                test_entries.append(te)

            gt2, rt2, rm2 = build_tallies(test_entries)
            mean, std, _ = loo_cv(
                test_entries, gt2, rt2, rm2,
                alpha=best_alpha, clip=best_clip, temperature=best_temp,
                cell_preds=cell_preds
            )
            delta = mean - baseline
            log.info(f"  thresh=({t1},{t2},{t3}): {mean:.2f} +/- {std:.2f} (d={delta:+.2f})")
            log_lines.append(f"thresh=({t1},{t2},{t3}): {mean:.2f} d={delta:+.2f}")
            if mean > best_thresh_score + 0.05:
                best_thresh = (t1, t2, t3)
                best_thresh_score = mean
                improved = True
                log.info(f"  ** IMPROVED with thresholds ({t1},{t2},{t3})")

    elif experiment_type == 4:
        # Obs-alpha simulation (with-observations alpha tuning)
        log.info("MODE B: [Exp] Obs-alpha tuning (simulated perfect obs)")
        best_obs_alpha = 0.10
        best_obs_score = 0

        for obs_a in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
            obs_scores = []
            for hold_rid in rounds_map:
                # Use within-round GT as "perfect observations"
                obs_tally = round_tally[hold_rid]
                fb4 = defaultdict(lambda: np.zeros(6))
                fb3 = defaultdict(lambda: np.zeros(6))
                ct = np.zeros((6, 6))
                for k, v in obs_tally.items():
                    fb4[k[:4]] += v
                    fb3[k[:3]] += v
                    ct[k[0]] += v

                obs_lut = {k: v / v.sum() for k, v in obs_tally.items() if v.sum() >= 20}
                fb4l = {k: v / v.sum() for k, v in fb4.items() if v.sum() >= 20}
                fb3l = {k: v / v.sum() for k, v in fb3.items() if v.sum() >= 20}
                ca_o = {}
                for ci in range(6):
                    s = ct[ci].sum()
                    ca_o[ci] = ct[ci] / s if s > 0 else np.ones(6) / 6

                for idx in rounds_map[hold_rid]:
                    e = all_entries[idx]
                    mapped = e['mapped']
                    sb, nf, co, np_ = e['feats']
                    mtn = e['mtn']

                    pred = np.zeros((40, 40, 6))
                    for r in range(40):
                        for c in range(40):
                            ic = int(mapped[r, c])
                            key5 = (ic, int(sb[r, c]), int(nf[r, c]), int(co[r, c]), int(np_[r, c]))
                            if key5 in obs_lut:
                                pred[r, c] = obs_lut[key5]
                            elif key5[:4] in fb4l:
                                pred[r, c] = fb4l[key5[:4]]
                            elif key5[:3] in fb3l:
                                pred[r, c] = fb3l[key5[:3]]
                            else:
                                pred[r, c] = ca_o.get(ic, np.ones(6) / 6)

                    pred[~mtn, 5] = 0.0
                    s = pred[~mtn].sum(axis=-1, keepdims=True)
                    s = np.where(s == 0, 1, s)
                    pred[~mtn] /= s
                    pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])

                    if obs_a > 0 and cell_preds is not None:
                        cell_zm = cell_preds[idx]
                        lut_log = np.log(np.clip(pred, best_clip, None))
                        cell_log = np.log(np.clip(cell_zm, best_clip, None))
                        mixed = (1 - obs_a) * lut_log + obs_a * cell_log
                        mixed -= mixed.max(axis=-1, keepdims=True)
                        p = np.exp(mixed)
                        p /= p.sum(axis=-1, keepdims=True)
                    else:
                        p = pred.copy()

                    if best_temp != 1.0:
                        non_mtn = ~mtn
                        if non_mtn.any():
                            p_nm = np.clip(p[non_mtn], 1e-10, None)
                            p_nm = np.exp(np.log(p_nm) / best_temp)
                            p_nm /= p_nm.sum(axis=-1, keepdims=True)
                            p[non_mtn] = p_nm

                    p = np.clip(p, 1e-8, None)
                    p /= p.sum(axis=-1, keepdims=True)
                    obs_scores.append(comp_score(p, e['gt']))

            mean_o = np.mean(obs_scores)
            log.info(f"  obs_a={obs_a:.2f}: {mean_o:.2f}")
            log_lines.append(f"obs_a={obs_a:.2f}: {mean_o:.2f}")
            if mean_o > best_obs_score:
                best_obs_alpha = obs_a
                best_obs_score = mean_o

        log.info(f"  Best obs-alpha: {best_obs_alpha} -> {best_obs_score:.2f}")
        # Deploy if different from current
        import auto_runner_v2 as ar
        if best_obs_alpha != ar.ENSEMBLE_ALPHA_OBS:
            log.info(f"  NOTE: Consider updating ENSEMBLE_ALPHA_OBS to {best_obs_alpha}")

    # Log to overnight_log.md
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = PROJECT / "overnight_log.md"
    with open(log_file, "a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(f"| {ts} | Cycle {cycle} | {line} |\n")

    # Update state
    if improved:
        state['current_best'] = {
            'alpha': best_alpha,
            'temperature': best_temp,
            'clip': best_clip,
        }
        # Deploy to auto_runner_v2.py config
        deploy_best_params(best_alpha, best_temp, best_clip)
        log.info(f"MODE B: Deployed new best: a={best_alpha}, T={best_temp}, clip={best_clip}")

    state['mode_b_cycle'] = cycle + 1
    state['experiments_completed'].append({
        'cycle': cycle,
        'type': experiment_type,
        'baseline': round(baseline, 2),
        'improved': improved,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })
    # Keep only last 50 experiment records
    state['experiments_completed'] = state['experiments_completed'][-50:]
    save_state(state)

    log.info(f"MODE B: Cycle {cycle} complete (baseline={baseline:.2f}, improved={improved})")


def deploy_best_params(alpha, temperature, clip):
    """Update auto_runner_v2.py config in-place."""
    path = PROJECT / "auto_runner_v2.py"
    content = path.read_text(encoding='utf-8')

    import re
    content = re.sub(
        r'TEMPERATURE\s*=\s*[\d.]+',
        f'TEMPERATURE = {temperature}',
        content, count=1
    )
    content = re.sub(
        r'ENSEMBLE_ALPHA\s*=\s*[\d.]+',
        f'ENSEMBLE_ALPHA = {alpha}',
        content, count=1
    )
    content = re.sub(
        r'CLIP_FLOOR\s*=\s*[\d.e-]+',
        f'CLIP_FLOOR = {clip}',
        content, count=1
    )
    path.write_text(content, encoding='utf-8')
    log.info(f"Updated auto_runner_v2.py: T={temperature}, a={alpha}, clip={clip}")


# ═══════════════════════════════════════════════════════════
# MAIN LOOP — NEVER DIES
# ═══════════════════════════════════════════════════════════
def main():
    acquire_lock()
    kill_orphan_runners()

    state = load_state()
    poll_interval = 60  # seconds between Mode A checks
    mode_b_interval = 300  # Mode B cycle every ~5 min
    last_mode_b = 0

    log.info("=" * 60)
    log.info("WATCHDOG STARTED - IMMORTAL MODE")
    log.info(f"  PID: {os.getpid()}")
    log.info(f"  Poll: {poll_interval}s, Mode B: every {mode_b_interval}s")
    log.info(f"  Best params: {state.get('current_best', {})}")
    log.info(f"  Submitted rounds: {[r[:8] for r in state.get('submitted_rounds', [])]}")
    log.info("=" * 60)

    while True:
        try:
            # ── MODE A: Check for active rounds ──
            try:
                submitted = mode_a_check_and_submit(state)
                if submitted:
                    log.info("MODE A complete, back to polling")
            except Exception as e:
                log.error(f"MODE A error: {e}")
                log.error(traceback.format_exc())

            # ── MODE B: Run experiments if enough time has passed ──
            now = time.time()
            if now - last_mode_b >= mode_b_interval:
                try:
                    mode_b_experiment_cycle(state)
                    last_mode_b = time.time()
                except Exception as e:
                    log.error(f"MODE B error: {e}")
                    log.error(traceback.format_exc())
                    last_mode_b = time.time()  # Don't retry immediately

            # ── Sleep ──
            log.info(f"Sleeping {poll_interval}s... (next Mode B in {max(0, int(mode_b_interval - (time.time() - last_mode_b)))}s)")
            time.sleep(poll_interval)

        except KeyboardInterrupt:
            log.info("Keyboard interrupt, shutting down gracefully")
            break
        except Exception as e:
            # CATCH ABSOLUTELY EVERYTHING
            log.error(f"OUTER LOOP ERROR (will retry in 60s): {e}")
            log.error(traceback.format_exc())
            try:
                time.sleep(60)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
