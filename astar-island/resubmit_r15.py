"""
R15 RESUBMISSION: Remove cell-level Bayesian (proven harmful by CV).

Changes from original submit_r15.py:
- REMOVED cell-level Bayesian updates (CV: alpha=10 → 88.19 vs bucket-only → 94.02)
- Changed CLIP_FLOOR from 0.0001 to 0.0005 (CV: +0.02)
- Changed min_n from 10 to 50 (CV: +0.02)
- Bucket-only LUT (settle_bins + near_forest) is the optimal model

Uses saved observations from data/observations_cc5442dd.json (no new queries needed).
"""
import sys, json, time
import numpy as np
from pathlib import Path
from scipy import ndimage

sys.path.insert(0, "src")
from api import AstarAPI
from settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from models import build_class_grid, compute_stats

CLIP_FLOOR = 0.0005
MIN_N = 50


def compute_spatial_features(cls_grid):
    H, W = cls_grid.shape
    settlement = (cls_grid == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    forest = (cls_grid == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20.0)
    settle_bin = np.full((H, W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    return settle_bin, near_forest


def build_gt_fallback_lut():
    """LUT from all available GT files for unseen bucket fallback."""
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
            cg = build_class_grid(ig)
            sb, nf = compute_spatial_features(cg)
            h, w = cg.shape
            for y in range(h):
                for x in range(w):
                    key = (int(cg[y, x]), int(sb[y, x]), int(nf[y, x]))
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
    for key in cross_counts:
        n = cross_total[key]
        if n >= 10:
            avg_freq = cross_counts[key] / n
            avg_freq = np.clip(avg_freq, CLIP_FLOOR, None)
            avg_freq /= avg_freq.sum()
            lut[key] = avg_freq
        else:
            lut[key] = class_avgs[key[0]]

    print(f"Built GT fallback LUT from {len(gt_files)} files ({len(lut)} buckets)")
    return lut, class_avgs


def build_predictions_bucket_only(grids, raw_obs, fallback_lut, fallback_class_avgs):
    """Pure bucket-only LUT — no cell-level Bayesian (CV-verified optimal)."""
    n_seeds = len(grids)
    H, W = grids[0].shape

    # Pre-compute spatial features
    seed_cls, seed_sb, seed_nf = [], [], []
    for si in range(n_seeds):
        cg = build_class_grid(grids[si])
        sb, nf = compute_spatial_features(cg)
        seed_cls.append(cg)
        seed_sb.append(sb)
        seed_nf.append(nf)

    # Tally observations into bucket counts
    obs_counts, obs_total = {}, {}
    for entry in raw_obs:
        si = entry["seed"]
        row, col = entry["row"], entry["col"]
        viewport = np.array(entry["grid"])
        vh, vw = viewport.shape
        cg = seed_cls[si]
        sb = seed_sb[si]
        nf = seed_nf[si]
        for vy in range(vh):
            for vx in range(vw):
                gy, gx = row + vy, col + vx
                if gy >= H or gx >= W:
                    continue
                ic = int(cg[gy, gx])
                sb_val = int(sb[gy, gx])
                nf_val = int(nf[gy, gx])
                oc = GRID_TO_CLASS.get(int(viewport[vy, vx]), 0)
                key = (ic, sb_val, nf_val)
                if key not in obs_counts:
                    obs_counts[key] = np.zeros(6)
                    obs_total[key] = 0
                obs_counts[key][oc] += 1
                obs_total[key] += 1

    total_cells = sum(obs_total.values())
    print(f"Tallied {total_cells} cell observations into {len(obs_counts)} buckets")

    # Class-level averages from observations
    class_avgs_obs = {}
    for ic in range(6):
        tc, tn = np.zeros(6), 0
        for k, c in obs_counts.items():
            if k[0] == ic:
                tc += c
                tn += obs_total[k]
        class_avgs_obs[ic] = tc / max(tn, 1) if tn > 0 else fallback_class_avgs.get(ic, np.ones(6)/6)

    # Build merged LUT: start from fallback, override with within-round obs
    lut = dict(fallback_lut)
    for key, counts in obs_counts.items():
        n = obs_total[key]
        if n >= MIN_N:
            # Large sample — use observation frequency directly
            lut[key] = counts / n
        elif key not in lut:
            # Small sample and no fallback — use class average
            lut[key] = class_avgs_obs.get(key[0], np.ones(6)/6)
        # Otherwise keep fallback LUT value for small-sample buckets

    # Log bucket info
    for key in sorted(obs_counts.keys()):
        n = obs_total[key]
        freq = obs_counts[key] / n
        ic, sb_val, nf_val = key
        sb_lbl = ["adj", "nr", "med", "far"][sb_val]
        nf_lbl = "F" if nf_val else " "
        source = "obs" if n >= MIN_N else "fb"
        final_freq = lut[key]
        print(f"  {CLASS_NAMES[ic]:>10s} s={sb_lbl:3s} {nf_lbl} n={n:5d} [{source}]: {np.round(final_freq, 3).tolist()}")

    # Build per-seed predictions (bucket-only, no cell-level updates)
    predictions = {}
    for si in range(n_seeds):
        cg = seed_cls[si]
        sb = seed_sb[si]
        nf = seed_nf[si]
        pred = np.ones((H, W, 6)) / 6

        for y in range(H):
            for x in range(W):
                key = (int(cg[y, x]), int(sb[y, x]), int(nf[y, x]))
                if key in lut:
                    pred[y, x] = lut[key]
                else:
                    ic = int(cg[y, x])
                    pred[y, x] = class_avgs_obs.get(ic, fallback_class_avgs.get(ic, np.ones(6)/6))

        # Mountain hardcode
        mtn = (cg == 5)
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

    # Get active round
    rounds = api.get_rounds()
    active = [r for r in rounds if r["status"] == "active"]
    if not active:
        print("No active round!")
        return
    r = active[0]
    round_id = r["id"]
    full = api._get(f"/rounds/{round_id}")
    grids = [np.array(st["grid"]) for st in full["initial_states"]]
    n_seeds = len(grids)
    H, W = grids[0].shape
    print(f"=== R{r.get('round_number', '?')} (id={round_id[:8]}) ===")
    print(f"{n_seeds} seeds, {H}x{W} grid")

    # Load saved observations (no new queries needed)
    obs_path = DATA_DIR / f"observations_{round_id[:8]}.json"
    if not obs_path.exists():
        print(f"ERROR: No saved observations at {obs_path}")
        return
    with open(obs_path) as f:
        raw_obs = json.load(f)
    print(f"Loaded {len(raw_obs)} saved observations from {obs_path.name}")

    # Build predictions
    print("\n--- Building predictions (BUCKET-ONLY, no cell-Bayesian) ---")
    fallback_lut, fallback_class_avgs = build_gt_fallback_lut()
    predictions = build_predictions_bucket_only(grids, raw_obs, fallback_lut, fallback_class_avgs)

    # Compare with old submission
    print("\n--- Comparison with old submission ---")
    old_preds = api._get(f"/my-predictions/{round_id}")
    if old_preds:
        print(f"  Old predictions: {len(old_preds)} seeds submitted")
        for op in old_preds:
            si = op.get("seed_index", "?")
            old_pred = np.array(op.get("prediction", []))
            if old_pred.ndim == 3:
                old_ent = -np.sum(old_pred * np.log(np.clip(old_pred, 1e-10, None)), axis=-1).mean()
                new_ent = -np.sum(predictions[si] * np.log(np.clip(predictions[si], 1e-10, None)), axis=-1).mean()
                print(f"  Seed {si}: old_ent={old_ent:.3f} → new_ent={new_ent:.3f}")

    # Submit
    print("\n--- Submitting (overwriting old predictions) ---")
    success_count = 0
    for si in range(n_seeds):
        pred = predictions[si]
        se = compute_stats(pred, grids[si])
        for attempt in range(3):
            ok, text = api.submit_prediction(round_id, si, pred.tolist())
            if ok:
                break
            print(f"  Seed {si}: retry {attempt+1} ({text[:60]})")
            time.sleep(3)
        status = "OK" if ok else f"FAIL: {text[:80]}"
        print(f"  Seed {si}: {status} (ent={se['ent']:.3f} conf={se['conf']:.3f})")
        if ok:
            success_count += 1
        time.sleep(0.5)

    elapsed = time.time() - t0
    print(f"\n=== Done: {success_count}/{n_seeds} seeds submitted ({elapsed:.1f}s) ===")


if __name__ == "__main__":
    main()
