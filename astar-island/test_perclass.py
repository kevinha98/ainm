"""Quick test: per-class avg + Bayesian vs global avg + Bayesian."""
import json
import numpy as np
from src.settings import GRID_TO_CLASS, CLASS_NAMES, NUM_CLASSES
from src.models import build_class_grid

with open('data/ground_truth_71451d74.json') as f:
    gt_data = json.load(f)

def score_pred(pred, gt):
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    kl = np.sum(gt * np.log((gt + 1e-15) / (pred + 1e-15)), axis=-1).mean()
    return 100 * np.exp(-kl)

# Build per-class averages
all_gt_by_class = {c: [] for c in range(NUM_CLASSES)}
for si in range(5):
    gt = np.array(gt_data[str(si)]['ground_truth'])
    ig = np.array(gt_data[str(si)]['initial_grid'])
    ic = build_class_grid(ig)
    for y in range(40):
        for x in range(40):
            all_gt_by_class[ic[y, x]].append(gt[y, x])

class_avg = {c: np.mean(v, axis=0) if v else np.ones(6)/6 for c, v in all_gt_by_class.items()}

# Global average
all_gt = np.vstack([np.array(gt_data[str(si)]['ground_truth']).reshape(-1, 6) for si in range(5)])
global_avg = all_gt.mean(axis=0)

print("Per-class avg:")
for c in range(NUM_CLASSES):
    print(f"  {CLASS_NAMES[c]:>20s}: {[round(x,3) for x in class_avg[c].tolist()]}")
print(f"\nGlobal avg: {[round(x,3) for x in global_avg.tolist()]}")

# Leave-one-out CV: for each seed, use other 4 seed averages
for test_seed in range(5):
    gt = np.array(gt_data[str(test_seed)]['ground_truth'])
    ig = np.array(gt_data[str(test_seed)]['initial_grid'])
    ic = build_class_grid(ig)
    gt_argmax = np.argmax(gt, axis=-1)
    
    # Per-class avg + no obs
    pred_class = np.zeros((40, 40, 6))
    for y in range(40):
        for x in range(40):
            pred_class[y, x] = class_avg[ic[y, x]]
    
    s_class_base = score_pred(pred_class, gt)
    s_global_base = score_pred(np.tile(global_avg, (40, 40, 1)), gt)
    
    # With observations (argmax, strength=3)
    pred_class_obs = pred_class.copy()
    pred_global_obs = np.tile(global_avg, (40, 40, 1)).copy()
    for y in range(40):
        for x in range(40):
            obs_cls = gt_argmax[y, x]
            for p in [pred_class_obs, pred_global_obs]:
                p[y, x, obs_cls] += 3.0 / 4.0
                p[y, x] = np.clip(p[y, x], 0.002, None)
                p[y, x] /= p[y, x].sum()
    
    s_class_obs = score_pred(pred_class_obs, gt)
    s_global_obs = score_pred(pred_global_obs, gt)
    
    print(f"\nSeed {test_seed}: class_base={s_class_base:.1f}, global_base={s_global_base:.1f} | class+obs={s_class_obs:.1f}, global+obs={s_global_obs:.1f}")
