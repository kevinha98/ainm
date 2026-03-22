"""Submit only seeds 3+4 with bucket-only model (ensure all 5 seeds are updated)."""
import sys, json, time
import numpy as np
from pathlib import Path
from scipy import ndimage

sys.path.insert(0, "src")
from api import AstarAPI
from settings import DATA_DIR, GRID_TO_CLASS, CLASS_NAMES
from models import build_class_grid, compute_stats

CLIP_FLOOR = 0.0005
MIN_N = 50
ROUND_ID = "cc5442dd-bc5d-418b-911b-7eb960cb0390"
SEEDS_TO_SUBMIT = [0, 1, 2]  # Ensure consistency with seeds 3-4


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


def main():
    api = AstarAPI()
    full = api._get(f"/rounds/{ROUND_ID}")
    grids = [np.array(st["grid"]) for st in full["initial_states"]]
    n_seeds = len(grids)
    H, W = grids[0].shape
    print(f"Round: {ROUND_ID[:8]}, {n_seeds} seeds, {H}x{W}")

    # Load saved observations
    obs_path = DATA_DIR / f"observations_{ROUND_ID[:8]}.json"
    with open(obs_path) as f:
        raw_obs = json.load(f)
    print(f"Loaded {len(raw_obs)} observations")

    # Build features
    seed_cls, seed_sb, seed_nf = [], [], []
    for si in range(n_seeds):
        cg = build_class_grid(grids[si])
        sb, nf = compute_spatial_features(cg)
        seed_cls.append(cg)
        seed_sb.append(sb)
        seed_nf.append(nf)

    # Build bucket LUT from observations
    obs_counts, obs_total = {}, {}
    for entry in raw_obs:
        si = entry["seed"]
        row, col = entry["row"], entry["col"]
        viewport = np.array(entry["grid"])
        vh, vw = viewport.shape
        cg, sb, nf = seed_cls[si], seed_sb[si], seed_nf[si]
        for vy in range(vh):
            for vx in range(vw):
                gy, gx = row + vy, col + vx
                if gy >= H or gx >= W:
                    continue
                key = (int(cg[gy, gx]), int(sb[gy, gx]), int(nf[gy, gx]))
                oc = GRID_TO_CLASS.get(int(viewport[vy, vx]), 0)
                if key not in obs_counts:
                    obs_counts[key] = np.zeros(6)
                    obs_total[key] = 0
                obs_counts[key][oc] += 1
                obs_total[key] += 1

    # Class averages
    class_avgs = {}
    for ic in range(6):
        tc, tn = np.zeros(6), 0
        for k, c in obs_counts.items():
            if k[0] == ic:
                tc += c
                tn += obs_total[k]
        class_avgs[ic] = tc / max(tn, 1) if tn > 0 else np.ones(6) / 6

    # GT fallback
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    cross_lut = {}
    for gf in gt_files:
        with open(gf) as f:
            gt_data = json.load(f)
        for si_str, entry in gt_data.items():
            if not isinstance(entry, dict):
                continue
            gt = np.array(entry["ground_truth"])
            ig = np.array(entry["initial_grid"])
            cg = build_class_grid(ig)
            sb2, nf2 = compute_spatial_features(cg)
            for y in range(cg.shape[0]):
                for x in range(cg.shape[1]):
                    key = (int(cg[y,x]), int(sb2[y,x]), int(nf2[y,x]))
                    if key not in cross_lut:
                        cross_lut[key] = {"sum": np.zeros(6), "n": 0}
                    cross_lut[key]["sum"] += gt[y, x]
                    cross_lut[key]["n"] += 1
    fallback = {k: v["sum"]/v["n"] for k, v in cross_lut.items() if v["n"] >= 5}

    # Build LUT
    lut = dict(fallback)
    for key, counts in obs_counts.items():
        n = obs_total[key]
        if n >= MIN_N:
            lut[key] = counts / n
    print(f"LUT: {len(lut)} buckets ({sum(1 for k in obs_counts if obs_total[k]>=MIN_N)} from obs)")

    # Submit seeds 3 and 4
    for si in SEEDS_TO_SUBMIT:
        cg, sb, nf = seed_cls[si], seed_sb[si], seed_nf[si]
        pred = np.ones((H, W, 6)) / 6
        for y in range(H):
            for x in range(W):
                key = (int(cg[y, x]), int(sb[y, x]), int(nf[y, x]))
                if key in lut:
                    pred[y, x] = lut[key]
                else:
                    pred[y, x] = class_avgs.get(int(cg[y, x]), np.ones(6) / 6)
        mtn = (cg == 5)
        if mtn.any():
            pred[mtn] = [0, 0, 0, 0, 0, 1.0]
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)

        se = compute_stats(pred, grids[si])
        for attempt in range(3):
            ok, text = api.submit_prediction(ROUND_ID, si, pred.tolist())
            if ok:
                break
            print(f"  Seed {si}: retry {attempt+1}: {text[:60]}")
            time.sleep(3)
        status = "OK" if ok else f"FAIL: {text[:80]}"
        print(f"Seed {si}: {status} (ent={se['ent']:.3f} conf={se['conf']:.3f})")
        time.sleep(0.5)

    print("Done!")


if __name__ == "__main__":
    main()
