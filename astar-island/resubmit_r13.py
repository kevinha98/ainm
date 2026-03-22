"""Try to resubmit R13 with observation-first model (budget spent, no new obs)."""
import json, numpy as np
from scipy import ndimage
from src.api import AstarAPI
from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from src.models import build_class_grid, compute_stats

CLIP_FLOOR = 0.0001
SETTLE_DIST_THRESH = 2.0

api = AstarAPI()

# Get R13 round data
active = api.get_active_round()
if active is None:
    print("No active round")
    exit()

round_id = active['id']
print(f"Active round: {round_id[:8]}")
print(f"Seeds: {len(active['initial_states'])}")

# Budget check
budget = api.get_budget()
print(f"Budget: {json.dumps(budget)}")

# Test if we can submit (try seed 0 with a test prediction first)
grids = [np.array(st["grid"]) for st in active["initial_states"]]
H, W = grids[0].shape
print(f"Grid: {H}x{W}")

# Since budget is exhausted, we can't make new observations.
# But we CAN still submit new predictions!
# Strategy: use cross-round GT data to build average frequency tables,
# plus mountain override (which alone was worth huge points).

# Load ALL GT files to compute cross-round average frequencies by (class, near_settle)
gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
print(f"\nLoading {len(gt_files)} GT files for cross-round frequency estimation...")

cross_counts = {}  # (ic, ns) -> np.array(6) counts
cross_total = {}

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
        settlement = (cls == 1)
        dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((h, w), 20.0)
        near_settle = (dist_s <= SETTLE_DIST_THRESH).astype(int)
        
        for y in range(h):
            for x in range(w):
                ic = int(cls[y, x])
                ns = int(near_settle[y, x])
                key = (ic, ns)
                if key not in cross_counts:
                    cross_counts[key] = np.zeros(6)
                    cross_total[key] = 0
                cross_counts[key] += gt[y, x]
                cross_total[key] += 1

# Build LUT from cross-round averages
lut = np.ones((6, 2, 6)) / 6
for (ic, ns), total_prob in cross_counts.items():
    n = cross_total[(ic, ns)]
    if n >= 10:
        avg_freq = total_prob / n
        avg_freq = np.clip(avg_freq, CLIP_FLOOR, None)
        avg_freq /= avg_freq.sum()
        lut[ic, ns] = avg_freq

# Log the cross-round average frequencies
print("\nCross-round average frequencies (from GT):")
for ic in range(NUM_CLASSES):
    for ns_label, ns_val in [("NEAR", 1), ("FAR", 0)]:
        key = (ic, ns_val)
        if key in cross_total and cross_total[key] >= 10:
            freq = cross_counts[key] / cross_total[key]
            print(f"  {ns_label:4s} {CLASS_NAMES[ic]:>15s} (n={cross_total[key]:5d}): {np.round(freq, 3).tolist()}")

# Build and submit predictions
print("\nSubmitting predictions...")
for si in range(len(grids)):
    cls = build_class_grid(grids[si])
    settlement = (cls == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    near_settle = (dist_s <= SETTLE_DIST_THRESH).astype(int)
    
    pred = lut[cls, near_settle]  # (H, W, 6)
    
    # Mountain override
    mtn = (cls == 5)
    if mtn.any():
        pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
    
    # Clip + normalize
    pred = np.clip(pred, CLIP_FLOOR, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    
    se = compute_stats(pred, grids[si])
    ok, text = api.submit_prediction(round_id, si, pred.tolist())
    status = "OK" if ok else f"FAIL: {text[:100]}"
    print(f"  Seed {si}: {status} (ent={se['ent']:.3f})")

print("\nDone! Resubmitted with cross-round avg frequencies + mountain override")
