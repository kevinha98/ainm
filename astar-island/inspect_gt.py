"""Inspect GT data format."""
import json, numpy as np
from pathlib import Path

gt_files = sorted(Path("data").glob("ground_truth_*.json"))
print(f"Total GT files: {len(gt_files)}")

gf = gt_files[0]
print(f"File: {gf.name}")
with open(gf) as f:
    data = json.load(f)

keys = list(data.keys())
print(f"Seeds: {keys}")
s0 = data[keys[0]]
print(f"Keys per seed: {list(s0.keys())}")

ig = np.array(s0["initial_grid"])
gt = np.array(s0["ground_truth"])
print(f"initial_grid shape: {ig.shape}, dtype: {ig.dtype}")
print(f"ground_truth shape: {gt.shape}, dtype: {gt.dtype}")
print(f"initial_grid unique values: {np.unique(ig)}")
print(f"GT class probs sample (cell 20,20): {np.round(gt[20, 20], 4)}")
print(f"GT class probs sample (cell 0,0): {np.round(gt[0, 0], 4)}")
print(f"GT sums check: min={gt.sum(axis=-1).min():.4f} max={gt.sum(axis=-1).max():.4f}")

# Check what fraction is hard labels vs soft
hard = (gt.max(axis=-1) > 0.99).mean()
soft = (gt.max(axis=-1) < 0.8).mean()
print(f"Hard labels (>99%): {hard:.1%}")
print(f"Soft/uncertain (<80%): {soft:.1%}")
