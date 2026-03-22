"""Resubmit R14 using saved observations + improved model (settle_bins + near_forest)."""
import sys, json, time
import numpy as np
from pathlib import Path
from scipy import ndimage

sys.path.insert(0, "src")
from api import AstarAPI
from settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from models import build_class_grid, compute_stats

# Import improved functions from auto_runner_v2
from auto_runner_v2 import compute_spatial_features, build_fallback_lut, CLIP_FLOOR

def main():
    api = AstarAPI()
    
    # Get R14 data
    rounds = api.get_rounds()
    r14 = [r for r in rounds if r.get("round_number") == 14][0]
    round_id = r14["id"]
    full = api._get(f"/rounds/{round_id}")
    
    if full["status"] != "active":
        print(f"R14 is {full['status']}, not active")
        return
    
    grids = [np.array(st["grid"]) for st in full["initial_states"]]
    n_seeds = len(grids)
    H, W = grids[0].shape
    print(f"R14: {n_seeds} seeds, {H}x{W} grid")
    
    # Load saved observations
    obs_path = DATA_DIR / f"observations_{round_id[:8]}.json"
    with open(obs_path) as f:
        raw_obs = json.load(f)
    print(f"Loaded {len(raw_obs)} saved observations")
    
    # Pre-compute spatial features for each seed
    seed_cls = []
    seed_settle_bin = []
    seed_near_forest = []
    for si in range(n_seeds):
        cls = build_class_grid(grids[si])
        sb, nf = compute_spatial_features(cls)
        seed_cls.append(cls)
        seed_settle_bin.append(sb)
        seed_near_forest.append(nf)
    
    # Re-tally observations with new bucket scheme
    obs_counts = {}
    obs_total = {}
    for obs_entry in raw_obs:
        si = obs_entry["seed"]
        row = obs_entry["row"]
        col = obs_entry["col"]
        viewport = np.array(obs_entry["grid"])
        vh, vw = viewport.shape
        
        cls = seed_cls[si]
        sb = seed_settle_bin[si]
        nf = seed_near_forest[si]
        
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
    
    total_cells = sum(obs_total.values())
    print(f"Re-tallied {total_cells} cell observations into {len(obs_counts)} buckets")
    
    # Log key bucket frequencies
    for key in sorted(obs_counts.keys()):
        n = obs_total[key]
        if n >= 20:
            freq = obs_counts[key] / n
            ic, sb_val, nf_val = key
            sb_lbl = ["adj", "nr", "med", "far"][sb_val]
            nf_lbl = "F" if nf_val else ""
            print(f"  {CLASS_NAMES[ic]:>10s} s={sb_lbl:3s} {nf_lbl:1s} (n={n:5d}): {np.round(freq, 3).tolist()}")
    
    # Build LUT: fallback from GT, overlay with within-round obs
    fallback_lut, fallback_class_avgs = build_fallback_lut()
    
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
    
    lut = dict(fallback_lut)
    for key in set(list(lut.keys()) + list(obs_counts.keys())):
        ic = key[0]
        if key in obs_counts and obs_total[key] >= 10:
            lut[key] = obs_counts[key] / obs_total[key]
        elif ic in class_avgs:
            lut[key] = class_avgs[ic]
    
    # Build predictions
    for si in range(n_seeds):
        cls = seed_cls[si]
        sb = seed_settle_bin[si]
        nf = seed_near_forest[si]
        
        pred = np.ones((H, W, 6)) / 6
        for y in range(H):
            for x in range(W):
                key = (int(cls[y, x]), int(sb[y, x]), int(nf[y, x]))
                if key in lut:
                    pred[y, x] = lut[key]
                else:
                    ic = int(cls[y, x])
                    pred[y, x] = class_avgs.get(ic, fallback_class_avgs.get(ic, np.ones(6) / 6))
        
        # Mountain override
        mtn = (cls == 5)
        if mtn.any():
            pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
        
        # Clip + normalize
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        
        # Compute stats
        se = compute_stats(pred, grids[si])
        
        # Submit
        ok, text = api.submit_prediction(round_id, si, pred.tolist())
        status = "OK" if ok else f"FAIL: {text[:100]}"
        print(f"  Seed {si}: {status} (ent={se['ent']:.3f} conf={se['conf']:.3f})")
        
        if not ok:
            time.sleep(2)
            ok, text = api.submit_prediction(round_id, si, pred.tolist())
            print(f"  Seed {si} retry: {'OK' if ok else 'FAIL: ' + text[:80]}")
        
        time.sleep(0.5)
    
    print("\nDone! R14 resubmitted with improved model.")

if __name__ == "__main__":
    main()
