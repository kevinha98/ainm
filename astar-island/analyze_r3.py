"""Analyze what makes R3 hard — why is LOO CV only 82 vs 94+ on other rounds?"""
import json
import numpy as np
from src.settings import DATA_DIR, NUM_CLASSES, CLASS_NAMES, GRID_TO_CLASS
from src.models import build_class_grid

gt_files = {
    "R1": DATA_DIR / "ground_truth_71451d74.json",
    "R2": DATA_DIR / "ground_truth_76909e29.json",
    "R3": DATA_DIR / "ground_truth_f1dac9a9.json",
    "R4": DATA_DIR / "ground_truth_8e839974.json",
}
all_gt = {}
for name, path in gt_files.items():
    with open(path) as f:
        all_gt[name] = json.load(f)

def kl_score(pred, gt):
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    kl = np.sum(gt_safe * np.log(gt_safe / (pred + 1e-15)), axis=-1).mean()
    return 100 * np.exp(-kl)

print("=== PER-CLASS AVERAGES BY ROUND ===\n")
for rname, rdata in sorted(all_gt.items()):
    by_class = {c: [] for c in range(NUM_CLASSES)}
    class_counts = {c: 0 for c in range(NUM_CLASSES)}
    
    for si_str in sorted(rdata.keys()):
        gt = np.array(rdata[si_str]['ground_truth'])
        ig = np.array(rdata[si_str]['initial_grid'])
        cls = build_class_grid(ig)
        for c in range(NUM_CLASSES):
            mask = cls == c
            if mask.any():
                by_class[c].extend(gt[mask].tolist())
                class_counts[c] += mask.sum()
    
    print(f"\n{rname}:")
    for c in range(NUM_CLASSES):
        if by_class[c]:
            avg = np.mean(by_class[c], axis=0)
            print(f"  {CLASS_NAMES[c]:>20s} (n={class_counts[c]:5d}): {[round(x,4) for x in avg.tolist()]}")

# Compare R3 vs others per-class
print("\n\n=== R3 vs OTHER ROUNDS — Per-class avg DEVIATION ===\n")
others = {c: [] for c in range(NUM_CLASSES)}
r3_avgs = {c: None for c in range(NUM_CLASSES)}

for rname, rdata in all_gt.items():
    by_class = {c: [] for c in range(NUM_CLASSES)}
    for si_str in sorted(rdata.keys()):
        gt = np.array(rdata[si_str]['ground_truth'])
        ig = np.array(rdata[si_str]['initial_grid'])
        cls = build_class_grid(ig)
        for c in range(NUM_CLASSES):
            mask = cls == c
            if mask.any():
                by_class[c].extend(gt[mask].tolist())
    
    for c in range(NUM_CLASSES):
        if by_class[c]:
            avg = np.mean(by_class[c], axis=0)
            if rname == "R3":
                r3_avgs[c] = avg
            else:
                others[c].append(avg)

for c in range(NUM_CLASSES):
    if r3_avgs[c] is not None and others[c]:
        others_avg = np.mean(others[c], axis=0)
        diff = r3_avgs[c] - others_avg
        big_diffs = [(CLASS_NAMES[k], diff[k]) for k in range(6) if abs(diff[k]) > 0.01]
        print(f"  {CLASS_NAMES[c]:>20s}: ", end="")
        if big_diffs:
            for name, d in big_diffs:
                print(f"{name}={d:+.3f}", end="  ")
            print()
        else:
            print("(similar)")

# Check GT entropy differences
print("\n\n=== GT ENTROPY BY ROUND ===\n")
for rname, rdata in sorted(all_gt.items()):
    all_ent = []
    for si_str in sorted(rdata.keys()):
        gt = np.array(rdata[si_str]['ground_truth'])
        gt_safe = np.clip(gt, 1e-15, None)
        ent = -np.sum(gt_safe * np.log(gt_safe), axis=-1)
        all_ent.append(ent.mean())
    avg_ent = np.mean(all_ent)
    print(f"  {rname}: mean entropy = {avg_ent:.4f}")

# Check map composition
print("\n\n=== MAP COMPOSITION BY ROUND ===\n")
for rname, rdata in sorted(all_gt.items()):
    all_counts = np.zeros(NUM_CLASSES)
    total = 0
    for si_str in sorted(rdata.keys()):
        ig = np.array(rdata[si_str]['initial_grid'])
        cls = build_class_grid(ig)
        for c in range(NUM_CLASSES):
            all_counts[c] += (cls == c).sum()
        total += cls.size
    fracs = all_counts / total
    print(f"  {rname}: " + " ".join(f"{CLASS_NAMES[c]}={fracs[c]:.3f}" for c in range(NUM_CLASSES)))
