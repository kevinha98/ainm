"""Leave-one-out cross-validation for per-class avg approach."""
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

print("=== PROPER Leave-One-Seed-Out Cross-Validation ===")

for test_seed in range(5):
    # Build per-class averages from OTHER seeds only
    train_seeds = [s for s in range(5) if s != test_seed]
    
    all_gt_by_class = {c: [] for c in range(NUM_CLASSES)}
    for si in train_seeds:
        gt = np.array(gt_data[str(si)]['ground_truth'])
        ig = np.array(gt_data[str(si)]['initial_grid'])
        ic = build_class_grid(ig)
        for y in range(40):
            for x in range(40):
                all_gt_by_class[ic[y, x]].append(gt[y, x])
    
    class_avg = {c: np.mean(v, axis=0) if v else np.ones(6)/6 for c, v in all_gt_by_class.items()}
    
    # Global average from train seeds
    all_gt = []
    for si in train_seeds:
        all_gt.append(np.array(gt_data[str(si)]['ground_truth']).reshape(-1, 6))
    global_avg = np.vstack(all_gt).mean(axis=0)
    
    # Predict test seed
    gt_test = np.array(gt_data[str(test_seed)]['ground_truth'])
    ig_test = np.array(gt_data[str(test_seed)]['initial_grid'])
    ic_test = build_class_grid(ig_test)
    gt_argmax = np.argmax(gt_test, axis=-1)
    
    # Per-class avg, no obs
    pred_pc = np.zeros((40, 40, 6))
    for y in range(40):
        for x in range(40):
            pred_pc[y, x] = class_avg[ic_test[y, x]]
    
    s_pc = score_pred(pred_pc, gt_test)
    s_global = score_pred(np.tile(global_avg, (40, 40, 1)), gt_test)
    
    # With observations (argmax, strength=3)
    pred_pc_obs = pred_pc.copy()
    for y in range(40):
        for x in range(40):
            obs_cls = gt_argmax[y, x]
            pred_pc_obs[y, x, obs_cls] += 3.0 / 4.0
            pred_pc_obs[y, x] = np.clip(pred_pc_obs[y, x], 0.002, None)
            pred_pc_obs[y, x] /= pred_pc_obs[y, x].sum()
    s_pc_obs = score_pred(pred_pc_obs, gt_test)
    
    # With random observations
    rng = np.random.RandomState(test_seed)
    scores_random = []
    for trial in range(10):
        pred_rnd = pred_pc.copy()
        for y in range(40):
            for x in range(40):
                obs_cls = rng.choice(6, p=gt_test[y, x])
                pred_rnd[y, x, obs_cls] += 3.0 / 4.0
                pred_rnd[y, x] = np.clip(pred_rnd[y, x], 0.002, None)
                pred_rnd[y, x] /= pred_rnd[y, x].sum()
        scores_random.append(score_pred(pred_rnd, gt_test))
    
    s_rnd = np.mean(scores_random)
    
    actual = gt_data[str(test_seed)]['score']
    print(f"Seed {test_seed}: per-class={s_pc:.1f}, +obs_argmax={s_pc_obs:.1f}, +obs_random={s_rnd:.1f}, global={s_global:.1f}, actual_R1={actual:.1f}")

print("\nKey: per-class uses averages from OTHER seeds only (leave-one-out)")
