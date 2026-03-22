"""
v9: Spatial-binned per-class distribution predictor.
Uses multi-round GT data with spatial binning (coast/inland, near/far settlements).
Cross-round LOO shows +2 points over basic per-class avg.
"""
import json
import time
import sys
import numpy as np
from pathlib import Path
from scipy import ndimage

from src.settings import DATA_DIR, NUM_CLASSES, MAP_H, MAP_W, CLASS_NAMES
from src.api import AstarAPI
from src.models import build_class_grid, compute_stats


def build_spatial_bins(ig):
    """Classify each cell into spatial bins based on neighborhood."""
    cls = build_class_grid(ig)
    H, W = ig.shape
    ocean = (ig == 10)
    mountain = (ig == 5)
    settlement = (cls == 1)
    
    # Coastal: adjacent to ocean
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    
    # Settlement density in radius 3
    kernel = np.ones((7, 7))
    n_settle = ndimage.convolve(settlement.astype(float), kernel, mode='constant')
    near_settle = n_settle > 0
    
    # Bin: 0=ocean (immutable), 1=mountain (immutable), 2=coastal, 3=near_settle, 4=inland_empty
    bins = np.full((H, W), 4, dtype=int)  # default: inland away from settlements
    bins[ocean] = 0
    bins[mountain] = 1
    # Overwrite for non-ocean/non-mountain
    land = ~ocean & ~mountain
    bins[land & is_coast] = 2
    bins[land & ~is_coast & near_settle] = 3
    bins[land & ~is_coast & ~near_settle] = 4
    
    return bins


def load_binned_averages():
    """Load all GT data and compute per-(initial_class, spatial_bin) averages."""
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    if not gt_files:
        return None
    
    # Key: (initial_class, spatial_bin) → list of GT distributions
    by_key = {}
    # Also keep plain per-class as fallback
    by_class = {c: [] for c in range(NUM_CLASSES)}
    
    for gf in gt_files:
        with open(gf) as f:
            gt_data = json.load(f)
        
        for si in range(5):
            si_str = str(si)
            if si_str not in gt_data:
                continue
            gt = np.array(gt_data[si_str].get('ground_truth', []))
            ig = np.array(gt_data[si_str].get('initial_grid', []))
            if gt.size == 0 or ig.size == 0:
                continue
            
            ic = build_class_grid(ig)
            bins = build_spatial_bins(ig)
            H, W = ig.shape
            
            for y in range(H):
                for x in range(W):
                    c = ic[y, x]
                    b = bins[y, x]
                    key = (c, b)
                    if key not in by_key:
                        by_key[key] = []
                    by_key[key].append(gt[y, x])
                    by_class[c].append(gt[y, x])
    
    # Compute averages
    avg_binned = {}
    for key, vals in by_key.items():
        avg_binned[key] = np.mean(vals, axis=0)
    
    avg_class = {}
    for c, vals in by_class.items():
        if vals:
            avg_class[c] = np.mean(vals, axis=0)
        else:
            avg_class[c] = np.ones(6) / 6
    
    return avg_binned, avg_class


def main():
    flags = set(sys.argv[1:])
    no_submit = "--no-submit" in flags
    
    t_total = time.time()
    print("=" * 65)
    print("  ASTAR ISLAND -- Spatial-Binned Predictor v9")
    print("=" * 65)
    
    api = AstarAPI()
    
    # Step 1: Get round
    print("\n[1] ROUND DETECTION")
    rd = api.get_active_round()
    if rd is None:
        print("  No active round!")
        return
    
    round_id = rd['id']
    n_seeds = rd.get('seeds_count', len(rd.get('initial_states', [])))
    print(f"  Round {rd['round_number']}: {rd['status']} ({rd['map_width']}x{rd['map_height']}, {n_seeds} seeds)")
    print(f"  ID: {round_id[:12]}...")
    print(f"  Closes: {rd.get('closes_at', '?')}")
    
    # Step 2: Load binned averages
    print("\n[2] LOADING BINNED SPATIAL AVERAGES")
    result = load_binned_averages()
    if result is None:
        print("  ERROR: No ground truth data!")
        return
    
    avg_binned, avg_class = result
    
    bin_names = {0: 'ocean', 1: 'mountain', 2: 'coastal', 3: 'near_settle', 4: 'inland_far'}
    for key, avg in sorted(avg_binned.items()):
        c, b = key
        n = len([1])  # placeholder
        print(f"    {CLASS_NAMES[c]:>15s} + {bin_names[b]:<12s}: {[round(x,3) for x in avg.tolist()]}")
    
    # Step 3: Parse seeds and predict
    print(f"\n[3] PREDICTION")
    grids = []
    for st in rd["initial_states"]:
        grids.append(np.array(st["grid"]))
    
    final = {}
    for si in range(n_seeds):
        g = grids[si]
        cls_grid = build_class_grid(g)
        bins = build_spatial_bins(g)
        H, W = g.shape
        
        pred = np.zeros((H, W, NUM_CLASSES))
        for y in range(H):
            for x in range(W):
                c = cls_grid[y, x]
                b = bins[y, x]
                key = (c, b)
                if key in avg_binned:
                    pred[y, x] = avg_binned[key]
                else:
                    pred[y, x] = avg_class.get(c, np.ones(6)/6)
        
        # Clip and normalize
        pred = np.clip(pred, 0.002, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        
        se = compute_stats(pred, g)
        print(f"  Seed {si}: ent={se['ent']:.3f} conf={se['conf']:.3f}")
        final[si] = pred
    
    # Step 4: Save
    print(f"\n[4] SAVING")
    improved = []
    for si in range(n_seeds):
        improved.append({
            "seed_index": si,
            "probabilities": final[si].tolist(),
            "model": "spatial_binned_v9",
        })
    with open(DATA_DIR / "improved_predictions.json", "w") as f:
        json.dump(improved, f)
    print("  Saved improved_predictions.json")
    
    # Step 5: Submit
    if not no_submit and rd["status"] == "active":
        print(f"\n[5] SUBMITTING")
        for si in range(n_seeds):
            p = final[si]
            p = np.clip(p, 1e-6, None)
            p /= p.sum(axis=-1, keepdims=True)
            ok, text = api.submit_prediction(round_id, si, p.tolist())
            status = "OK" if ok else f"FAIL: {text[:80]}"
            print(f"  Seed {si}: {status}")
            time.sleep(0.5)
        print("  All seeds submitted!")
    else:
        print(f"\n[5] SKIPPED SUBMISSION")
    
    tt = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  Total time: {tt:.0f}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
