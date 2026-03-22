"""Submit R14 immediately using observation-first approach."""
import json, time, numpy as np
from scipy import ndimage
from src.api import AstarAPI
from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from src.models import build_class_grid, compute_stats

CLIP_FLOOR = 0.0001
SETTLE_DIST_THRESH = 2.0

api = AstarAPI()
rd = api.get_active_round()
round_id = rd['id']
n_seeds = len(rd['initial_states'])
grids = [np.array(st["grid"]) for st in rd["initial_states"]]
H, W = grids[0].shape

print(f"Round {rd['round_number']} (id={round_id[:8]}), {n_seeds} seeds, {H}x{W} grid")
budget = api.get_budget()
print(f"Budget: {budget['queries_used']}/{budget['queries_max']}")

# Pre-compute spatial features
seed_cls = []
seed_ns = []
for si in range(n_seeds):
    cls = build_class_grid(grids[si])
    settlement = (cls == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    seed_cls.append(cls)
    seed_ns.append((dist_s <= SETTLE_DIST_THRESH).astype(int))

# ── OBSERVE: 9 viewports × 5 seeds = 45 queries ──
obs_counts = {}
obs_total = {}
obs_used = 0
max_obs = 45

print(f"\nObserving ({max_obs} queries)...")
for si in range(n_seeds):
    for row in [0, 12, 25]:
        for col in [0, 12, 25]:
            if obs_used >= max_obs:
                break
            result = api.simulate(round_id, si, row, col, steps=50)
            if "error" in result:
                print(f"  ERROR seed {si} ({row},{col}): {result.get('error')}")
                if result.get('error') == 'budget_exhausted':
                    max_obs = obs_used
                    break
                continue
            viewport = np.array(result.get("grid", []))
            if viewport.ndim != 2:
                continue
            obs_used += 1
            vh, vw = viewport.shape
            cls = seed_cls[si]
            ns = seed_ns[si]
            for vy in range(vh):
                for vx in range(vw):
                    gy, gx = row + vy, col + vx
                    if gy >= H or gx >= W:
                        continue
                    ic = int(cls[gy, gx])
                    ns_val = int(ns[gy, gx])
                    oc = GRID_TO_CLASS.get(int(viewport[vy, vx]), 0)
                    key = (ic, ns_val)
                    if key not in obs_counts:
                        obs_counts[key] = np.zeros(6)
                        obs_total[key] = 0
                    obs_counts[key][oc] += 1
                    obs_total[key] += 1
            time.sleep(0.3)
        if obs_used >= max_obs:
            break
    if obs_used >= max_obs:
        break

print(f"Used {obs_used} observations, {sum(obs_total.values())} cell comparisons")

# Log frequencies
for ic in range(NUM_CLASSES):
    for ns_label, ns_val in [("NEAR", 1), ("FAR", 0)]:
        key = (ic, ns_val)
        if key in obs_counts and obs_total[key] >= 10:
            freq = obs_counts[key] / obs_total[key]
            print(f"  {ns_label:4s} {CLASS_NAMES[ic]:>15s} (n={obs_total[key]:5d}): {np.round(freq, 3).tolist()}")

# ── BUILD LUT ──
# Start with cross-round GT fallback
gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
fallback_counts = {}
fallback_total = {}
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
        c = build_class_grid(ig)
        s = (c == 1)
        d = ndimage.distance_transform_edt(~s) if s.any() else np.full((h, w), 20.0)
        n = (d <= SETTLE_DIST_THRESH).astype(int)
        for y in range(h):
            for x in range(w):
                key = (int(c[y, x]), int(n[y, x]))
                if key not in fallback_counts:
                    fallback_counts[key] = np.zeros(6)
                    fallback_total[key] = 0
                fallback_counts[key] += gt[y, x]
                fallback_total[key] += 1

lut = np.ones((6, 2, 6)) / 6
for (ic, ns), total_prob in fallback_counts.items():
    n = fallback_total[(ic, ns)]
    if n >= 10:
        lut[ic, ns] = np.clip(total_prob / n, CLIP_FLOOR, None)
        lut[ic, ns] /= lut[ic, ns].sum()

# Overwrite with within-round observations
for ic in range(6):
    tc, tn = np.zeros(6), 0
    for ns_val in [0, 1]:
        k = (ic, ns_val)
        if k in obs_counts:
            tc += obs_counts[k]
            tn += obs_total[k]
    if tn > 0:
        merged = tc / tn
        for ns_val in [0, 1]:
            k = (ic, ns_val)
            if k not in obs_counts or obs_total.get(k, 0) < 10:
                lut[ic, ns_val] = merged
for (ic, ns_val), counts in obs_counts.items():
    if obs_total[(ic, ns_val)] >= 10:
        lut[ic, ns_val] = counts / obs_total[(ic, ns_val)]

# ── PREDICT & SUBMIT ──
print(f"\nSubmitting predictions...")
for si in range(n_seeds):
    cls = seed_cls[si]
    ns = seed_ns[si]
    pred = lut[cls, ns].copy()
    
    # Mountain override
    mtn = (cls == 5)
    if mtn.any():
        pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
    
    pred = np.clip(pred, CLIP_FLOOR, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    
    se = compute_stats(pred, grids[si])
    ok, text = api.submit_prediction(round_id, si, pred.tolist())
    if not ok:
        print(f"  Seed {si}: FAIL first try: {text[:100]}, retrying...")
        time.sleep(3)
        ok, text = api.submit_prediction(round_id, si, pred.tolist())
    status = "OK" if ok else f"FAIL: {text[:100]}"
    print(f"  Seed {si}: {status} (ent={se['ent']:.3f})")

# Save observations
obs_path = DATA_DIR / f"observations_{round_id[:8]}.json"
with open(obs_path, "w") as f:
    json.dump({"obs_counts": {str(k): v.tolist() for k, v in obs_counts.items()},
               "obs_total": {str(k): v for k, v in obs_total.items()}}, f)
print(f"\nDone! Saved to {obs_path}")
