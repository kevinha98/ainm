"""
v8: Multi-round per-class distribution predictor.
Uses per-class average GT distributions from ALL available prior rounds.
"""
import json
import time
import sys
import numpy as np
from pathlib import Path

from src.settings import DATA_DIR, NUM_CLASSES, MAP_H, MAP_W, CLASS_NAMES
from src.api import AstarAPI
from src.models import build_class_grid, compute_stats


def load_multi_round_averages():
    """Load pre-computed multi-round per-class averages."""
    avg_file = DATA_DIR / "multi_round_class_avg.json"
    if avg_file.exists():
        with open(avg_file) as f:
            data = json.load(f)
        class_avg = {}
        for k, v in data["class_avg_dist"].items():
            class_avg[int(k)] = np.array(v)
        print(f"  Loaded averages from {data['total_rounds']} rounds: {data['rounds_used']}")
        return class_avg
    return None


def load_single_round_gt(gt_file):
    """Load per-class averages from a single GT file."""
    with open(gt_file) as f:
        gt_data = json.load(f)
    
    by_class = {c: [] for c in range(NUM_CLASSES)}
    for si in range(5):
        si_str = str(si)
        if si_str not in gt_data:
            continue
        gt = np.array(gt_data[si_str].get('ground_truth', []))
        ig = np.array(gt_data[si_str].get('initial_grid', []))
        if gt.size == 0 or ig.size == 0:
            continue
        ic = build_class_grid(ig)
        for y in range(ig.shape[0]):
            for x in range(ig.shape[1]):
                by_class[ic[y, x]].append(gt[y, x])
    
    avg = {}
    for c in range(NUM_CLASSES):
        if by_class[c]:
            avg[c] = np.mean(by_class[c], axis=0)
        else:
            avg[c] = np.ones(6) / 6
    return avg


def main():
    flags = set(sys.argv[1:])
    no_submit = "--no-submit" in flags
    
    t_total = time.time()
    print("=" * 65)
    print("  ASTAR ISLAND -- Multi-Round Distribution Predictor v8")
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
    
    # Step 2: Load multi-round per-class averages  
    print("\n[2] LOADING MULTI-ROUND AVERAGES")
    class_avg = load_multi_round_averages()
    
    if class_avg is None:
        print("  No multi-round averages! Falling back to latest GT...")
        # Try to find any GT file
        gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
        if gt_files:
            class_avg = load_single_round_gt(gt_files[-1])
            print(f"  Loaded from {gt_files[-1].name}")
        else:
            print("  ERROR: No ground truth data available!")
            return
    
    for c in range(NUM_CLASSES):
        if c in class_avg:
            print(f"    {CLASS_NAMES[c]:>15s}: {[round(x,4) for x in class_avg[c].tolist()]}")
    
    # Step 3: Parse seeds and predict
    print(f"\n[3] PREDICTION")
    grids = []
    for st in rd["initial_states"]:
        grids.append(np.array(st["grid"]))
    
    final = {}
    for si in range(n_seeds):
        g = grids[si]
        cls_grid = build_class_grid(g)
        H, W = g.shape
        
        pred = np.zeros((H, W, NUM_CLASSES))
        for y in range(H):
            for x in range(W):
                c = cls_grid[y, x]
                pred[y, x] = class_avg.get(c, class_avg.get(0, np.ones(6)/6))
        
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
            "model": "distribution_v8_multi_round",
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
