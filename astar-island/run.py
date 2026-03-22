"""
Astar Island — Auto-Runner
===========================
Single entry point: detect round -> observe -> predict -> submit.
Usage:
    python run.py                  # Full auto pipeline
    python run.py --predict-only   # Skip observation (budget 0 or already done)
    python run.py --no-submit      # Predict but don't submit
"""
import json, time, sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src.settings import DATA_DIR, NUM_CLASSES, MAP_H, MAP_W, GRID_TO_CLASS, CLASS_NAMES
from src.api import AstarAPI
from src.observer import observe_round, build_observed_grid, compute_transition_matrix
from src.models import (
    build_class_grid, model_markov, model_monte_carlo, model_hgb,
    model_mrf, model_settlement_ca, model_observation_direct,
    ensemble, calibrate, sharpen, compute_stats
)

LOG_FILE = DATA_DIR / "experiment_log.json"


def load_round_data(api):
    """Get active round, save to disk."""
    rd = api.get_active_round()
    if rd is None:
        # Check completed rounds
        rounds = api.get_rounds()
        if rounds:
            rd = rounds[0]
            print(f"  No active round. Latest: Round {rd['round_number']} ({rd['status']})")
        else:
            print("  No rounds found!")
            return None

    n_seeds = rd.get('seeds_count', len(rd.get('initial_states', [])))
    print(f"  Round {rd['round_number']}: {rd['status']} ({rd['map_width']}x{rd['map_height']}, {n_seeds} seeds)")
    print(f"  ID: {rd['id'][:12]}...")
    print(f"  Weight: {rd.get('round_weight', '?')}")

    # Save
    with open(DATA_DIR / "round_info.json", "w") as f:
        json.dump(rd, f, indent=2)

    return rd


def parse_seeds(rd):
    """Extract grids, settlement sets, port sets from round data."""
    grids, all_s, all_p = [], [], []
    for st in rd["initial_states"]:
        g = np.array(st["grid"])
        grids.append(g)
        ss, ps = set(), set()
        for s in st.get("settlements", []):
            if s.get("alive", True):
                ss.add((s["y"], s["x"]))
                if s.get("has_port"):
                    ps.add((s["y"], s["x"]))
        all_s.append(ss)
        all_p.append(ps)
    return grids, all_s, all_p


def load_observations(round_id):
    """Load saved observation data if available."""
    obs_file = DATA_DIR / f"observations_{round_id[:8]}.json"
    if obs_file.exists():
        with open(obs_file) as f:
            return json.load(f)
    return None


def load_learned_transitions():
    """Load transition matrix learned from past rounds."""
    t_file = DATA_DIR / "learned_transitions.json"
    if t_file.exists():
        with open(t_file) as f:
            data = json.load(f)
        return np.array(data["matrix"]), data.get("total_cells", 0)
    return None, 0


def run_prediction(grids, all_s, all_p, observations=None, learned_T=None):
    """Run all models and return ensemble predictions."""
    n_seeds = len(grids)

    # Build observed grids if we have observation data
    obs_grids, obs_masks = [], []
    have_obs = False
    if observations:
        for si in range(n_seeds):
            si_str = str(si)
            if si_str in observations and observations[si_str]:
                og, om = build_observed_grid(observations[si_str])
                obs_grids.append(og)
                obs_masks.append(om)
                pct = om.sum() / om.size * 100
                print(f"  Seed {si}: {om.sum()}/{om.size} cells observed ({pct:.0f}%)")
                have_obs = True
            else:
                obs_grids.append(np.zeros((MAP_H, MAP_W), dtype=int))
                obs_masks.append(np.zeros((MAP_H, MAP_W), dtype=bool))
    else:
        for _ in range(n_seeds):
            obs_grids.append(np.zeros((MAP_H, MAP_W), dtype=int))
            obs_masks.append(np.zeros((MAP_H, MAP_W), dtype=bool))

    # Model weights (adaptive based on data availability)
    if have_obs:
        weights = {
            "observation": 0.90,  # Observations ARE ground truth — dominate
            "markov": 0.02,
            "mc": 0.02,
            "hgb": 0.02,
            "mrf_markov": 0.02,
            "settlement_ca": 0.02,
        }
    else:
        weights = {
            "markov": 0.20,
            "mc": 0.30,
            "hgb": 0.10,
            "mrf_markov": 0.10,
            "settlement_ca": 0.30,
        }

    # Train HGB on all seeds
    print("\n--- HistGradientBoosting ---")
    t0 = time.time()
    hgb_preds, _ = model_hgb(
        grids, all_s, all_p,
        obs_grids=obs_grids if have_obs else None,
        obs_masks=obs_masks if have_obs else None
    )
    print(f"    Done in {time.time()-t0:.1f}s")

    # Predict per seed
    final = {}
    all_stats = {}

    for si in range(n_seeds):
        g, ss, ps = grids[si], all_s[si], all_p[si]
        print(f"\n--- Seed {si} ({len(ss)} settlements, {len(ps)} ports) ---")
        seed_preds = {}
        sr = {}

        # Model 1: Markov
        t0 = time.time()
        pm = model_markov(g, ss, ps, learned_T=learned_T)
        sm = compute_stats(pm, g); sr["markov"] = sm
        seed_preds["markov"] = pm
        print(f"  Markov:     ent={sm['ent']:.3f} conf={sm['conf']:.3f} ({time.time()-t0:.1f}s)")

        # Model 2: MC
        t0 = time.time()
        pmc = model_monte_carlo(g, ss, ps, n_sims=1000, n_steps=50, recompute_every=25)
        smc = compute_stats(pmc, g); sr["mc"] = smc
        seed_preds["mc"] = pmc
        print(f"  MC(1000d):  ent={smc['ent']:.3f} conf={smc['conf']:.3f} ({time.time()-t0:.1f}s)")

        # Model 3: HGB
        shgb = compute_stats(hgb_preds[si], g); sr["hgb"] = shgb
        seed_preds["hgb"] = hgb_preds[si]
        print(f"  HGB:        ent={shgb['ent']:.3f} conf={shgb['conf']:.3f}")

        # Model 4: MRF on Markov
        t0 = time.time()
        pmrf = model_mrf(pm, g, iters=20, temp=0.20)
        smrf = compute_stats(pmrf, g); sr["mrf_markov"] = smrf
        seed_preds["mrf_markov"] = pmrf
        print(f"  MRF:        ent={smrf['ent']:.3f} conf={smrf['conf']:.3f} ({time.time()-t0:.1f}s)")

        # Model 5: Settlement CA
        t0 = time.time()
        pca = model_settlement_ca(g, ss, ps, n_steps=50, n_runs=200)
        sca = compute_stats(pca, g); sr["settlement_ca"] = sca
        seed_preds["settlement_ca"] = pca
        print(f"  Settle CA:  ent={sca['ent']:.3f} conf={sca['conf']:.3f} ({time.time()-t0:.1f}s)")

        # Model 6: Observation-direct (if available)
        full_coverage = False
        if have_obs and obs_masks[si].any():
            t0 = time.time()
            pobs = model_observation_direct(obs_grids[si], obs_masks[si], g)
            sobs = compute_stats(pobs, g); sr["observation"] = sobs
            seed_preds["observation"] = pobs
            full_coverage = obs_masks[si].all()
            print(f"  Obs-Direct: ent={sobs['ent']:.3f} conf={sobs['conf']:.3f} (100% coverage)" if full_coverage else f"  Obs-Direct: ent={sobs['ent']:.3f} conf={sobs['conf']:.3f} ({time.time()-t0:.1f}s)")

        if full_coverage:
            # 100% coverage: observations ARE ground truth — use directly, skip calibrate
            pe = pobs.copy()
            # Only sharpen mildly to push confidence higher
            pe = sharpen(pe, temperature=0.5, grid_np=g)
        else:
            # Ensemble + calibrate + sharpen
            pe = ensemble(seed_preds, weights)
            pe = calibrate(pe, g)
            pe = sharpen(pe, temperature=0.85, grid_np=g)

        se = compute_stats(pe, g); sr["ensemble"] = se
        print(f"  ENSEMBLE:   ent={se['ent']:.3f} conf={se['conf']:.3f} ll={se['ll']:.3f}")

        final[si] = pe
        all_stats[si] = sr

    return final, all_stats, weights


def save_predictions(final, round_id, model_version="v4"):
    """Save predictions to disk."""
    improved = []
    for si in range(len(final)):
        p = final[si]
        improved.append({
            "seed_index": si,
            "argmax_grid": np.argmax(p, axis=-1).tolist(),
            "confidence_grid": p.max(axis=-1).tolist(),
            "probabilities": p.tolist(),
            "score": None,
            "model": f"ensemble_{model_version}",
        })
    with open(DATA_DIR / "improved_predictions.json", "w") as f:
        json.dump(improved, f)
    print("  Saved improved_predictions.json")


def log_experiment(round_id, all_stats, weights, model_version="v4"):
    """Append experiment to log."""
    exp = {
        "timestamp": datetime.now().isoformat(),
        "round_id": round_id,
        "model_version": model_version,
        "seeds": {str(si): stats for si, stats in all_stats.items()},
        "weights": weights,
        "summary": {
            "avg_ent": float(np.mean([all_stats[i]["ensemble"]["ent"] for i in all_stats])),
            "avg_conf": float(np.mean([all_stats[i]["ensemble"]["conf"] for i in all_stats])),
            "avg_ll": float(np.mean([all_stats[i]["ensemble"]["ll"] for i in all_stats])),
        }
    }
    log = {"experiments": []}
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            log = json.load(f)
    log["experiments"].append(exp)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    return exp


def submit_predictions(api, final, round_id):
    """Submit all seeds to API."""
    for si in range(len(final)):
        p = final[si]
        p = np.clip(p, 1e-6, None)
        p /= p.sum(axis=-1, keepdims=True)
        ok, text = api.submit_prediction(round_id, si, p.tolist())
        status = "OK" if ok else f"FAIL: {text[:80]}"
        print(f"  Seed {si}: {status}")
        time.sleep(0.5)


def main():
    flags = set(sys.argv[1:])
    predict_only = "--predict-only" in flags
    no_submit = "--no-submit" in flags

    t_total = time.time()
    print("=" * 65)
    print("  ASTAR ISLAND -- Auto-Runner v5")
    print("=" * 65)

    api = AstarAPI()

    # Step 1: Get round
    print("\n[1] ROUND DETECTION")
    rd = load_round_data(api)
    if rd is None:
        return
    round_id = rd["id"]

    # Step 2: Check budget and observe
    budget = api.get_budget()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"\n[2] OBSERVATION (budget: {budget['queries_used']}/{budget['queries_max']})")

    observations = load_observations(round_id)
    if not predict_only and remaining > 0 and rd["status"] == "active":
        print(f"  {remaining} queries available -- observing...")
        observations = observe_round(api, rd, steps=50)
    elif observations:
        n_obs = sum(len(v) for v in observations.values())
        print(f"  Using {n_obs} cached observations")
    else:
        print(f"  No observations available (budget: {remaining})")

    # Step 3: Parse seeds
    print("\n[3] PARSING SEEDS")
    grids, all_s, all_p = parse_seeds(rd)
    for si in range(len(grids)):
        print(f"  Seed {si}: {len(all_s[si])} settlements, {len(all_p[si])} ports")

    # Step 4: Load learned transitions (from past rounds)
    learned_T, n_cells = load_learned_transitions()
    if learned_T is not None:
        print(f"\n[4] LEARNED TRANSITIONS ({n_cells} cells from past rounds)")
    else:
        print("\n[4] NO LEARNED TRANSITIONS (first round or no analysis yet)")

    # Step 5: Predict
    print("\n[5] PREDICTION")
    final, all_stats, weights = run_prediction(
        grids, all_s, all_p,
        observations=observations,
        learned_T=learned_T
    )

    # Step 6: Save
    print(f"\n[6] SAVING")
    save_predictions(final, round_id, model_version="v5")
    exp = log_experiment(round_id, all_stats, weights, model_version="v5")

    s = exp["summary"]
    print(f"\n  Entropy:    {s['avg_ent']:.4f}")
    print(f"  Confidence: {s['avg_conf']:.4f}")
    print(f"  Log-loss:   {s['avg_ll']:.4f}")

    # Step 7: Submit
    if not no_submit and rd["status"] == "active":
        print(f"\n[7] SUBMITTING")
        submit_predictions(api, final, round_id)
        print("  All seeds submitted!")
    else:
        print(f"\n[7] SKIPPED SUBMISSION (status={rd['status']}, no_submit={no_submit})")

    tt = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  Total time: {tt:.0f}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
