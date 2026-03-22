"""Submit missing seed 4 for R15."""
import sys, json, time
import numpy as np
from pathlib import Path
from scipy import ndimage

sys.path.insert(0, "src")
from api import AstarAPI
from settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from models import build_class_grid, compute_stats

CLIP_FLOOR = 0.0001

def compute_spatial_features(cls):
    H, W = cls.shape
    settlement = (cls == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    forest = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20.0)
    settle_bin = np.full((H, W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    return settle_bin, near_forest

api = AstarAPI()
round_id = "cc5442dd-bc5d-418b-911b-7eb960cb0390"

# Check what we're missing
preds = api._get(f"/my-predictions/{round_id}")
print(f"Already submitted: {len(preds)} seeds")
submitted_seeds = set()
for p in preds:
    si = p.get("seed_index", p.get("seed", -1))
    submitted_seeds.add(si)
    print(f"  Seed {si}: submitted")

missing = [si for si in range(5) if si not in submitted_seeds]
print(f"Missing seeds: {missing}")

if not missing:
    print("All seeds submitted!")
    exit()

# Load saved observations
obs_path = DATA_DIR / f"observations_{round_id[:8]}.json"
with open(obs_path) as f:
    raw_obs = json.load(f)
print(f"\nLoaded {len(raw_obs)} observations")
for si in range(5):
    count = sum(1 for o in raw_obs if o["seed"] == si)
    print(f"  Seed {si}: {count} viewports")

# Get round data
full = api._get(f"/rounds/{round_id}")
grids = [np.array(st["grid"]) for st in full["initial_states"]]
n_seeds = len(grids)
H, W = grids[0].shape

# Pre-compute features for ALL seeds
seed_cls, seed_sb, seed_nf = [], [], []
for si in range(n_seeds):
    cls = build_class_grid(grids[si])
    sb, nf = compute_spatial_features(cls)
    seed_cls.append(cls)
    seed_sb.append(sb)
    seed_nf.append(nf)

# Tally ALL observations into buckets + per-cell
obs_counts, obs_total = {}, {}
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
            if (gy, gx) not in cell_obs[si]:
                cell_obs[si][(gy, gx)] = []
            cell_obs[si][(gy, gx)].append(oc)

print(f"\nTallied {sum(obs_total.values())} cell observations into {len(obs_counts)} buckets")

# Build fallback LUT
gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
fallback_counts, fallback_total = {}, {}
for gf in gt_files:
    with open(gf) as f:
        gt_data = json.load(f)
    for si_str in sorted(gt_data.keys()):
        entry = gt_data[si_str]
        if not isinstance(entry, dict): continue
        gt = np.array(entry["ground_truth"])
        ig = np.array(entry["initial_grid"])
        cls = build_class_grid(ig)
        sb, nf = compute_spatial_features(cls)
        h, w = cls.shape
        for y in range(h):
            for x in range(w):
                key = (int(cls[y, x]), int(sb[y, x]), int(nf[y, x]))
                if key not in fallback_counts:
                    fallback_counts[key] = np.zeros(6)
                    fallback_total[key] = 0
                fallback_counts[key] += gt[y, x]
                fallback_total[key] += 1

fallback_class_avgs = {}
for ic in range(6):
    tc, tn = np.zeros(6), 0
    for k, v in fallback_counts.items():
        if k[0] == ic: tc += v; tn += fallback_total[k]
    fallback_class_avgs[ic] = tc / max(tn, 1) if tn > 0 else np.ones(6) / 6

fallback_lut = {}
for key, tp in fallback_counts.items():
    n = fallback_total[key]
    avg = tp / n
    avg = np.clip(avg, CLIP_FLOOR, None)
    avg /= avg.sum()
    fallback_lut[key] = avg if n >= 10 else fallback_class_avgs[key[0]]

# Build within-round LUT
class_avgs_obs = {}
for ic in range(6):
    tc, tn = np.zeros(6), 0
    for k, c in obs_counts.items():
        if k[0] == ic: tc += c; tn += obs_total[k]
    class_avgs_obs[ic] = tc / tn if tn > 0 else fallback_class_avgs.get(ic, np.ones(6)/6)

lut = dict(fallback_lut)
for key in set(list(lut.keys()) + list(obs_counts.keys())):
    ic = key[0]
    if key in obs_counts and obs_total[key] >= 10:
        lut[key] = obs_counts[key] / obs_total[key]
    elif key in obs_counts:
        alpha = 5.0
        prior = class_avgs_obs.get(ic, fallback_class_avgs.get(ic, np.ones(6)/6))
        posterior = (obs_counts[key] + alpha * prior) / (obs_total[key] + alpha)
        lut[key] = posterior / posterior.sum()

# Submit missing seeds
for si in missing:
    cls = seed_cls[si]
    sb = seed_sb[si]
    nf = seed_nf[si]
    pred = np.ones((H, W, 6)) / 6
    for y in range(H):
        for x in range(W):
            key = (int(cls[y, x]), int(sb[y, x]), int(nf[y, x]))
            bucket_prior = lut.get(key, class_avgs_obs.get(int(cls[y, x]), np.ones(6)/6)).copy()
            cell_samples = []
            for other_si in range(n_seeds):
                if (y, x) in cell_obs[other_si]:
                    cell_samples.extend(cell_obs[other_si][(y, x)])
            if cell_samples:
                alpha_prior = 10.0
                cell_counts = np.zeros(6)
                for s in cell_samples:
                    cell_counts[s] += 1
                posterior = bucket_prior * alpha_prior + cell_counts
                posterior = np.clip(posterior, CLIP_FLOOR, None)
                posterior /= posterior.sum()
                pred[y, x] = posterior
            else:
                pred[y, x] = bucket_prior
    mtn = (cls == 5)
    if mtn.any():
        pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
    pred = np.clip(pred, CLIP_FLOOR, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)

    se = compute_stats(pred, grids[si])
    ok, text = api.submit_prediction(round_id, si, pred.tolist())
    if not ok:
        time.sleep(3)
        ok, text = api.submit_prediction(round_id, si, pred.tolist())
    print(f"  Seed {si}: {'OK' if ok else 'FAIL: '+text[:80]} (ent={se['ent']:.3f} conf={se['conf']:.3f})")

# Verify
preds_after = api._get(f"/my-predictions/{round_id}")
print(f"\nFinal: {len(preds_after)} seeds submitted")
