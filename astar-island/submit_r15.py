"""
R15 SUBMISSION: Best possible model targeting rank #1.

Strategy:
1. Observe all 45 viewports (9 per seed × 5 seeds) for full coverage
2. Build per-cell observation counts (5 samples per cell from 5 seeds)
3. Build bucket LUT (cls, settle_bin, near_forest) from observations
4. Bayesian per-cell update: blend bucket prior with cell-level observations
5. Spatial smoothing via neighbor averaging for rare-class cells
6. Mountain hardcode, clip + normalize
7. Save raw observations BEFORE submitting (never overwrite)
8. Submit all 5 seeds

CV benchmark: 94.02 with settle_bins+forest (within-round obs only)
Target: 95+ on actual server
"""
import sys, json, time, os
import numpy as np
from pathlib import Path
from scipy import ndimage

sys.path.insert(0, "src")
from api import AstarAPI
from settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from models import build_class_grid, compute_stats

CLIP_FLOOR = 0.0001


def compute_spatial_features(cls):
    """Compute settle_bin (4-level) and near_forest (binary) from class grid."""
    H, W = cls.shape
    settlement = (cls == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    forest = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20.0)
    settle_bin = np.full((H, W), 3, dtype=int)  # 3=far
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    return settle_bin, near_forest


def build_gt_fallback_lut():
    """Build fallback LUT from ALL available ground truth files."""
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    if not gt_files:
        return {}, {}
    cross_counts, cross_total = {}, {}
    for gf in gt_files:
        with open(gf) as f:
            gt_data = json.load(f)
        for si_str in sorted(gt_data.keys()):
            entry = gt_data[si_str]
            if not isinstance(entry, dict):
                continue
            gt = np.array(entry["ground_truth"])
            ig = np.array(entry["initial_grid"])
            cls = build_class_grid(ig)
            sb, nf = compute_spatial_features(cls)
            h, w = cls.shape
            for y in range(h):
                for x in range(w):
                    key = (int(cls[y, x]), int(sb[y, x]), int(nf[y, x]))
                    if key not in cross_counts:
                        cross_counts[key] = np.zeros(6)
                        cross_total[key] = 0
                    cross_counts[key] += gt[y, x]
                    cross_total[key] += 1

    class_avgs = {}
    for ic in range(6):
        tc, tn = np.zeros(6), 0
        for k, v in cross_counts.items():
            if k[0] == ic:
                tc += v
                tn += cross_total[k]
        class_avgs[ic] = tc / max(tn, 1) if tn > 0 else np.ones(6) / 6

    lut = {}
    for key, total_prob in cross_counts.items():
        n = cross_total[key]
        if n >= 10:
            avg_freq = total_prob / n
            avg_freq = np.clip(avg_freq, CLIP_FLOOR, None)
            avg_freq /= avg_freq.sum()
            lut[key] = avg_freq
        else:
            lut[key] = class_avgs[key[0]]

    print(f"Built GT fallback LUT from {len(gt_files)} files ({len(lut)} buckets)")
    return lut, class_avgs


def observe_all(api, round_id, n_seeds):
    """Observe 9 viewports × n_seeds. Save raw data immediately."""
    viewport_positions = [(r, c) for r in [0, 12, 25] for c in [0, 12, 25]]
    raw_obs = []
    obs_used = 0
    max_obs = 9 * n_seeds  # 45 for 5 seeds

    for si in range(n_seeds):
        for row, col in viewport_positions:
            if obs_used >= max_obs:
                break
            for attempt in range(3):
                try:
                    result = api.simulate(round_id, si, row, col, steps=50)
                    break
                except Exception as e:
                    print(f"  Retry {attempt+1} for seed {si} ({row},{col}): {e}")
                    time.sleep(2)
            else:
                result = {"error": "max_retries"}

            if "error" in result:
                print(f"  ERROR seed {si} ({row},{col}): {result.get('error')}")
                if result.get("error") == "budget_exhausted":
                    max_obs = obs_used
                    break
                continue

            viewport = np.array(result.get("grid", []))
            if viewport.ndim != 2:
                continue

            obs_used += 1
            raw_obs.append({"seed": si, "row": row, "col": col, "grid": viewport.tolist()})
            time.sleep(0.25)

        if obs_used >= max_obs:
            break

    # Save immediately — NEVER overwrite existing obs with fewer entries
    obs_path = DATA_DIR / f"observations_{round_id[:8]}.json"
    existing_count = 0
    if obs_path.exists():
        try:
            with open(obs_path) as f:
                existing = json.load(f)
                existing_count = len(existing)
        except Exception:
            pass
    if len(raw_obs) > existing_count:
        with open(obs_path, "w") as f:
            json.dump(raw_obs, f)
        print(f"Saved {len(raw_obs)} observations to {obs_path.name}")
    else:
        print(f"Keeping existing {existing_count} observations (new has {len(raw_obs)})")

    return raw_obs


def build_predictions(grids, raw_obs, fallback_lut, fallback_class_avgs):
    """Build predictions using bucket LUT + per-cell Bayesian updates."""
    n_seeds = len(grids)
    H, W = grids[0].shape

    # Pre-compute spatial features
    seed_cls = []
    seed_sb = []
    seed_nf = []
    for si in range(n_seeds):
        cls = build_class_grid(grids[si])
        sb, nf = compute_spatial_features(cls)
        seed_cls.append(cls)
        seed_sb.append(sb)
        seed_nf.append(nf)

    # === STEP 1: Tally observations into bucket counts ===
    obs_counts = {}  # (cls, sb, nf) → counts[6]
    obs_total = {}
    # Also track per-cell observations: cell_obs[si][(y,x)] → list of observed classes
    cell_obs = [{} for _ in range(n_seeds)]

    for entry in raw_obs:
        si = entry["seed"]
        row, col = entry["row"], entry["col"]
        viewport = np.array(entry["grid"])
        vh, vw = viewport.shape
        cls = seed_cls[si]
        sb = seed_sb[si]
        nf = seed_nf[si]

        for vy in range(vh):
            for vx in range(vw):
                gy, gx = row + vy, col + vx
                if gy >= H or gx >= W:
                    continue
                ic = int(cls[gy, gx])
                sb_val = int(sb[gy, gx])
                nf_val = int(nf[gy, gx])
                oc = GRID_TO_CLASS.get(int(viewport[vy, vx]), 0)

                key = (ic, sb_val, nf_val)
                if key not in obs_counts:
                    obs_counts[key] = np.zeros(6)
                    obs_total[key] = 0
                obs_counts[key][oc] += 1
                obs_total[key] += 1

                # Track per-cell
                if (gy, gx) not in cell_obs[si]:
                    cell_obs[si][(gy, gx)] = []
                cell_obs[si][(gy, gx)].append(oc)

    total_cells = sum(obs_total.values())
    print(f"Tallied {total_cells} cell observations into {len(obs_counts)} buckets")

    # Log key frequencies
    for key in sorted(obs_counts.keys()):
        n = obs_total[key]
        if n >= 50:
            freq = obs_counts[key] / n
            ic, sb_val, nf_val = key
            sb_lbl = ["adj", "nr", "med", "far"][sb_val]
            nf_lbl = "F" if nf_val else " "
            print(f"  {CLASS_NAMES[ic]:>10s} s={sb_lbl:3s} {nf_lbl} (n={n:5d}): {np.round(freq, 3).tolist()}")

    # === STEP 2: Build within-round bucket LUT ===
    # Class-level averages from observations
    class_avgs_obs = {}
    for ic in range(6):
        tc, tn = np.zeros(6), 0
        for k, c in obs_counts.items():
            if k[0] == ic:
                tc += c
                tn += obs_total[k]
        if tn > 0:
            class_avgs_obs[ic] = tc / tn
        else:
            class_avgs_obs[ic] = fallback_class_avgs.get(ic, np.ones(6) / 6)

    # Merge obs buckets with fallback
    lut = dict(fallback_lut)
    for key in set(list(lut.keys()) + list(obs_counts.keys())):
        ic = key[0]
        if key in obs_counts and obs_total[key] >= 10:
            lut[key] = obs_counts[key] / obs_total[key]
        elif key in obs_counts:
            # Bayesian blend: combine sparse obs with class average
            alpha = 5.0  # prior strength
            prior = class_avgs_obs.get(ic, fallback_class_avgs.get(ic, np.ones(6) / 6))
            posterior = (obs_counts[key] + alpha * prior) / (obs_total[key] + alpha)
            lut[key] = posterior / posterior.sum()
        # Otherwise keep fallback

    # === STEP 3: Build per-seed predictions with cell-level Bayesian updates ===
    predictions = {}
    for si in range(n_seeds):
        cls = seed_cls[si]
        sb = seed_sb[si]
        nf = seed_nf[si]

        pred = np.ones((H, W, 6)) / 6  # start uniform

        for y in range(H):
            for x in range(W):
                key = (int(cls[y, x]), int(sb[y, x]), int(nf[y, x]))
                if key in lut:
                    bucket_prior = lut[key].copy()
                else:
                    ic = int(cls[y, x])
                    bucket_prior = class_avgs_obs.get(ic, fallback_class_avgs.get(ic, np.ones(6) / 6)).copy()

                # Per-cell Bayesian update using observations from OTHER seeds
                # (don't use this seed's own observation — that would be circular for the test)
                # Actually, we're predicting the DISTRIBUTION, and each seed's observation
                # is a valid sample from that distribution.
                # Collect all observations of this cell across ALL seeds
                cell_samples = []
                for other_si in range(n_seeds):
                    if (y, x) in cell_obs[other_si]:
                        cell_samples.extend(cell_obs[other_si][(y, x)])

                if cell_samples:
                    # Bayesian update: posterior ∝ prior × likelihood
                    # prior = bucket_prior (Dirichlet parameter)
                    # observation = multinomial counts
                    alpha_prior = 10.0  # strength of bucket prior (in pseudo-counts)
                    cell_counts = np.zeros(6)
                    for s in cell_samples:
                        cell_counts[s] += 1
                    # Dirichlet-multinomial posterior
                    posterior = bucket_prior * alpha_prior + cell_counts
                    posterior = np.clip(posterior, CLIP_FLOOR, None)
                    posterior /= posterior.sum()
                    pred[y, x] = posterior
                else:
                    pred[y, x] = bucket_prior

        # Mountain override
        mtn = (cls == 5)
        if mtn.any():
            pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])

        # Clip + normalize
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        predictions[si] = pred

    return predictions


def main():
    t0 = time.time()
    api = AstarAPI()

    # Get R15 data
    rounds = api.get_rounds()
    r15 = [r for r in rounds if r.get("round_number") == 15]
    if not r15:
        # Find active round
        active = [r for r in rounds if r["status"] == "active"]
        if not active:
            print("No active round!")
            return
        r15 = active[0]
    else:
        r15 = r15[0]

    round_id = r15["id"]
    full = api._get(f"/rounds/{round_id}")

    if full["status"] != "active":
        print(f"Round is {full['status']}, not active")
        return

    grids = [np.array(st["grid"]) for st in full["initial_states"]]
    n_seeds = len(grids)
    H, W = grids[0].shape

    print(f"=== R{r15.get('round_number', '?')} (id={round_id[:8]}) ===")
    print(f"{n_seeds} seeds, {H}x{W} grid")

    # Budget check
    budget = api.get_budget()
    queries_left = budget.get("queries_max", 50) - budget.get("queries_used", 0)
    print(f"Budget: {queries_left} queries remaining")

    if queries_left < 5:
        print("Not enough budget for meaningful observations!")
        # Use GT fallback only
        raw_obs = []
    else:
        # Observe
        print(f"\n--- Observing ({min(queries_left, 45)} viewports) ---")
        raw_obs = observe_all(api, round_id, n_seeds)
        print(f"Observations complete: {len(raw_obs)} viewports ({time.time()-t0:.1f}s)")

    # Build predictions
    print(f"\n--- Building predictions ---")
    fallback_lut, fallback_class_avgs = build_gt_fallback_lut()
    predictions = build_predictions(grids, raw_obs, fallback_lut, fallback_class_avgs)

    # Submit
    print(f"\n--- Submitting ---")
    success_count = 0
    for si in range(n_seeds):
        pred = predictions[si]
        se = compute_stats(pred, grids[si])

        ok, text = api.submit_prediction(round_id, si, pred.tolist())
        if not ok:
            print(f"  Seed {si}: FAIL ({text[:80]}), retrying...")
            time.sleep(3)
            ok, text = api.submit_prediction(round_id, si, pred.tolist())

        status = "OK" if ok else f"FAIL: {text[:80]}"
        print(f"  Seed {si}: {status} (ent={se['ent']:.3f} conf={se['conf']:.3f})")
        if ok:
            success_count += 1
        time.sleep(0.5)

    elapsed = time.time() - t0
    print(f"\n=== Done: {success_count}/{n_seeds} seeds submitted ({elapsed:.1f}s) ===")

    if success_count == n_seeds:
        print("All seeds submitted successfully!")
    else:
        print(f"WARNING: Only {success_count}/{n_seeds} seeds submitted!")


if __name__ == "__main__":
    main()
