"""Test optimal Bayesian update strength for observations."""
import json
import numpy as np
from src.settings import GRID_TO_CLASS, CLASS_NAMES, NUM_CLASSES

with open('data/ground_truth_71451d74.json') as f:
    gt_data = json.load(f)

def score_pred(pred, gt):
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    kl = np.sum(gt * np.log((gt + 1e-15) / (pred + 1e-15)), axis=-1).mean()
    return 100 * np.exp(-kl)

# Global average distribution from all 5 seeds
all_gt = []
for si in range(5):
    all_gt.append(np.array(gt_data[str(si)]['ground_truth']).reshape(-1, 6))
avg_dist = np.vstack(all_gt).mean(axis=0)
print(f'Global avg: {[round(x, 3) for x in avg_dist.tolist()]}')

# Test on each seed
for seed_test in range(5):
    gt = np.array(gt_data[str(seed_test)]['ground_truth'])
    gt_argmax = np.argmax(gt, axis=-1)
    actual_score = gt_data[str(seed_test)]['score']
    
    print(f'\n=== Seed {seed_test} (actual R1 score: {actual_score:.1f}) ===')
    
    # Test different Bayesian strength values
    # Observation = argmax of GT (best case — observation matches most likely outcome)
    for strength in [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
        pred = np.tile(avg_dist, (40, 40, 1)).copy()
        
        if strength > 0:
            for y in range(40):
                for x in range(40):
                    obs_cls = gt_argmax[y, x]
                    pred[y, x, obs_cls] += strength / (strength + 1)
                    pred[y, x] = np.clip(pred[y, x], 0.002, None)
                    pred[y, x] /= pred[y, x].sum()
        
        s = score_pred(pred, gt)
        print(f'  strength={strength:5.1f}: score={s:.2f}')

# Now test with RANDOM draws (simulating one stochastic observation)
print('\n\n=== Random Draws (simulating real observations) ===')
rng = np.random.RandomState(42)
for seed_test in [0, 1]:
    gt = np.array(gt_data[str(seed_test)]['ground_truth'])
    print(f'\nSeed {seed_test}:')
    
    for strength in [0, 1.0, 2.0, 3.0, 5.0, 10.0]:
        scores = []
        for trial in range(10):
            pred = np.tile(avg_dist, (40, 40, 1)).copy()
            for y in range(40):
                for x in range(40):
                    obs_cls = rng.choice(6, p=gt[y, x])
                    if strength > 0:
                        pred[y, x, obs_cls] += strength / (strength + 1)
                        pred[y, x] = np.clip(pred[y, x], 0.002, None)
                        pred[y, x] /= pred[y, x].sum()
            scores.append(score_pred(pred, gt))
        print(f'  strength={strength:4.1f}: avg={np.mean(scores):.2f} std={np.std(scores):.2f}')
