"""Analyze Round 1 ground truth format and learn transitions."""
import json
import numpy as np
from src.settings import GRID_TO_CLASS, NUM_CLASSES, CLASS_NAMES, DATA_DIR

# Load ground truth
with open(DATA_DIR / "ground_truth_71451d74.json") as f:
    data = json.load(f)

print("=" * 60)
print("  ROUND 1 GROUND TRUTH ANALYSIS")
print("=" * 60)

# Per-seed scores
print("\n--- Per-Seed Scores ---")
total = 0
for si in range(5):
    score = data[str(si)]["score"]
    total += score
    print(f"  Seed {si}: {score:.4f}")
print(f"  TOTAL: {total:.4f}")
print(f"  AVG:   {total/5:.4f}")

# Ground truth format
s0 = data["0"]
gt0 = np.array(s0["ground_truth"])
print(f"\n--- Format ---")
print(f"  ground_truth shape: {gt0.shape}")
print(f"  Type: H x W x 6 probability tensor (one-hot for ground truth)")

# Class distributions per seed
print(f"\n--- Ground Truth Class Distributions ---")
for si in range(5):
    gt = np.array(data[str(si)]["ground_truth"])
    gt_cls = np.argmax(gt, axis=-1)
    dist = {CLASS_NAMES[c]: int((gt_cls == c).sum()) for c in range(NUM_CLASSES)}
    print(f"  Seed {si}: {dist}")

# Compare initial vs final
print(f"\n--- Initial vs Final (Seed 0) ---")
ig = np.array(data["0"].get("initial_grid", []))
if ig.size > 0:
    # initial_grid might be raw values (10, 11, etc) or class indices
    print(f"  initial_grid shape: {ig.shape}")
    unique_vals = sorted(set(ig.ravel().tolist()))
    print(f"  unique initial values: {unique_vals}")
    
    # Convert to classes
    if max(unique_vals) > 5:
        # Raw grid values
        init_cls = np.zeros_like(ig, dtype=int)
        for gv, cls in GRID_TO_CLASS.items():
            init_cls[ig == gv] = cls
    else:
        init_cls = ig.astype(int)
    
    gt_cls = np.argmax(gt0, axis=-1)
    
    # Transition matrix
    print(f"\n--- TRANSITION MATRIX (initial -> final, Seed 0) ---")
    counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            counts[init_cls[i, j], gt_cls[i, j]] += 1
    
    print(f"  {'':>20s}", end="")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:>10s}", end="")
    print("    (total)")
    
    for r in range(NUM_CLASSES):
        total_r = counts[r].sum()
        print(f"  {CLASS_NAMES[r]:>20s}", end="")
        for c in range(NUM_CLASSES):
            if total_r > 0:
                pct = counts[r, c] / total_r * 100
                print(f"  {pct:>9.1f}%", end="")
            else:
                print(f"  {'n/a':>10s}", end="")
        print(f"    (n={total_r})")
    
    # Changes
    changed = init_cls != gt_cls
    print(f"\n  Cells changed: {changed.sum()} / {ig.size} ({changed.sum()/ig.size*100:.1f}%)")
    
    # What changed the most?
    print(f"\n--- Top Changes ---")
    change_counts = {}
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if init_cls[i, j] != gt_cls[i, j]:
                key = f"{CLASS_NAMES[init_cls[i,j]]} -> {CLASS_NAMES[gt_cls[i,j]]}"
                change_counts[key] = change_counts.get(key, 0) + 1
    
    for k, v in sorted(change_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {v:4d} cells: {k}")

# Build FULL transition matrix across all seeds
print(f"\n\n{'='*60}")
print(f"  FULL TRANSITION MATRIX (ALL 5 SEEDS)")
print(f"{'='*60}")

# Load initial states from round_info
with open(DATA_DIR / "round_info.json") as f:
    rd = json.load(f)

total_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for si in range(5):
    init_grid = np.array(rd["initial_states"][si]["grid"])
    gt = np.array(data[str(si)]["ground_truth"])
    gt_cls = np.argmax(gt, axis=-1)
    
    init_cls = np.zeros_like(init_grid, dtype=int)
    for gv, cls in GRID_TO_CLASS.items():
        init_cls[init_grid == gv] = cls
    
    for i in range(init_grid.shape[0]):
        for j in range(init_grid.shape[1]):
            total_counts[init_cls[i, j], gt_cls[i, j]] += 1

print(f"\n  {'':>20s}", end="")
for c in range(NUM_CLASSES):
    print(f"  {CLASS_NAMES[c]:>10s}", end="")
print("    (total)")

T = np.zeros((NUM_CLASSES, NUM_CLASSES))
for r in range(NUM_CLASSES):
    total_r = total_counts[r].sum()
    print(f"  {CLASS_NAMES[r]:>20s}", end="")
    for c in range(NUM_CLASSES):
        if total_r > 0:
            pct = total_counts[r, c] / total_r * 100
            T[r, c] = total_counts[r, c] / total_r
            print(f"  {pct:>9.1f}%", end="")
        else:
            print(f"  {'n/a':>10s}", end="")
    print(f"    (n={total_r})")

total_cells = total_counts.sum()
print(f"\n  Total cells: {total_cells}")
print(f"  Total changed: {total_cells - np.trace(total_counts)} ({(total_cells - np.trace(total_counts))/total_cells*100:.1f}%)")

# Save learned transitions
learned = {
    "matrix": T.tolist(),
    "counts": total_counts.tolist(),
    "total_cells": int(total_cells),
    "round_id": "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    "source": "Round 1 ground truth (5 seeds)",
}
with open(DATA_DIR / "learned_transitions.json", "w") as f:
    json.dump(learned, f, indent=2)
print(f"\n  Saved learned_transitions.json ({total_cells} cells)")
