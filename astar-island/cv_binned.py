"""Cross-round LOO validation for spatial-binned per-class approach."""
import json
import numpy as np
from scipy import ndimage
from src.settings import DATA_DIR, NUM_CLASSES, CLASS_NAMES
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

def score_pred(pred, gt):
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    kl = np.sum(gt_safe * np.log(gt_safe / (pred + 1e-15)), axis=-1).mean()
    return 100 * np.exp(-kl)

def build_spatial_bins(ig):
    cls = build_class_grid(ig)
    ocean = (ig == 10)
    mountain = (ig == 5)
    settlement = (cls == 1)
    
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    kernel = np.ones((7, 7))
    n_settle = ndimage.convolve(settlement.astype(float), kernel, mode='constant')
    near_settle = n_settle > 0
    
    H, W = ig.shape
    bins = np.full((H, W), 4, dtype=int)
    bins[ocean] = 0
    bins[mountain] = 1
    land = ~ocean & ~mountain
    bins[land & is_coast] = 2
    bins[land & ~is_coast & near_settle] = 3
    bins[land & ~is_coast & ~near_settle] = 4
    return bins

print("=== CROSS-ROUND LOO: SPATIAL BINNED ===\n")
round_names = sorted(all_gt.keys())

all_results = []
for test_round in round_names:
    train_rounds = [r for r in round_names if r != test_round]
    
    # Build per-class avg (baseline)
    by_class = {c: [] for c in range(NUM_CLASSES)}
    # Build binned avg
    by_key = {}
    
    for r in train_rounds:
        for si in range(5):
            si_str = str(si)
            if si_str not in all_gt[r]:
                continue
            gt = np.array(all_gt[r][si_str]['ground_truth'])
            ig = np.array(all_gt[r][si_str]['initial_grid'])
            if gt.size == 0 or ig.size == 0:
                continue
            ic = build_class_grid(ig)
            bins = build_spatial_bins(ig)
            for y in range(ig.shape[0]):
                for x in range(ig.shape[1]):
                    c = ic[y, x]
                    b = bins[y, x]
                    by_class[c].append(gt[y, x])
                    key = (c, b)
                    if key not in by_key:
                        by_key[key] = []
                    by_key[key].append(gt[y, x])
    
    avg_class = {c: np.mean(v, axis=0) if v else np.ones(6)/6 for c, v in by_class.items()}
    avg_binned = {k: np.mean(v, axis=0) for k, v in by_key.items()}
    
    scores_basic = []
    scores_binned = []
    
    for si in range(5):
        si_str = str(si)
        if si_str not in all_gt[test_round]:
            continue
        gt = np.array(all_gt[test_round][si_str]['ground_truth'])
        ig = np.array(all_gt[test_round][si_str]['initial_grid'])
        ic = build_class_grid(ig)
        bins = build_spatial_bins(ig)
        H, W = ig.shape
        
        pred_basic = np.zeros((H, W, 6))
        pred_binned = np.zeros((H, W, 6))
        
        for y in range(H):
            for x in range(W):
                c = ic[y, x]
                b = bins[y, x]
                pred_basic[y, x] = avg_class.get(c, avg_class[0])
                key = (c, b)
                pred_binned[y, x] = avg_binned.get(key, avg_class.get(c, avg_class[0]))
        
        pred_basic = np.clip(pred_basic, 0.002, None)
        pred_basic /= pred_basic.sum(axis=-1, keepdims=True)
        pred_binned = np.clip(pred_binned, 0.002, None)
        pred_binned /= pred_binned.sum(axis=-1, keepdims=True)
        
        scores_basic.append(score_pred(pred_basic, gt))
        scores_binned.append(score_pred(pred_binned, gt))
    
    avg_b = np.mean(scores_basic)
    avg_s = np.mean(scores_binned)
    print(f"{test_round}: basic={avg_b:.2f}  binned={avg_s:.2f}  diff={avg_s-avg_b:+.2f}")
    all_results.append({'round': test_round, 'basic': avg_b, 'binned': avg_s})

print(f"\nOverall:")
print(f"  Basic:  {np.mean([r['basic'] for r in all_results]):.2f}")
print(f"  Binned: {np.mean([r['binned'] for r in all_results]):.2f}")
print(f"  Delta:  {np.mean([r['binned']-r['basic'] for r in all_results]):+.2f}")
