"""Analyze ground truth distributions and neighborhood effects."""
import json
import numpy as np
from src.settings import GRID_TO_CLASS, CLASS_NAMES

with open('data/ground_truth_71451d74.json') as f:
    data = json.load(f)
with open('data/round_info.json') as f:
    rd = json.load(f)

gt = np.array(data['0']['ground_truth'])
ig = np.array(rd['initial_states'][0]['grid'])
init_cls = np.zeros_like(ig, dtype=int)
for gv, cls in GRID_TO_CLASS.items():
    init_cls[ig == gv] = cls

# Settlement analysis by neighbor count
settle_cells = np.argwhere(init_cls == 1)
print(f'Settlements in seed 0: {len(settle_cells)}')
for sy, sx in settle_cells:
    neighbors = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = sy+dy, sx+dx
            if 0 <= ny < 40 and 0 <= nx < 40 and init_cls[ny, nx] == 1:
                neighbors += 1
    gt_dist = gt[sy, sx]
    argmax = CLASS_NAMES[np.argmax(gt_dist)]
    print(f'  [{sy:2d},{sx:2d}] nbrs={neighbors} gt={[round(x,3) for x in gt_dist.tolist()]} -> {argmax}')

# Ground truth keys
print(f'\nGT keys per seed: {list(data["0"].keys())}')
print(f'Score field: {data["0"]["score"]}')

# Cross-seed analysis: are GT distributions stable for same terrain type?
print('\n--- Cross-seed GT for Plains cells ---')
for si in range(5):
    gt_s = np.array(data[str(si)]['ground_truth'])
    ig_s = np.array(rd['initial_states'][si]['grid'])
    init_cls_s = np.zeros_like(ig_s, dtype=int)
    for gv, cls in GRID_TO_CLASS.items():
        init_cls_s[ig_s == gv] = cls
    
    plains = (init_cls_s == 0) & (ig_s != 10)  # Plains but not ocean
    if plains.any():
        avg_gt = gt_s[plains].mean(axis=0)
        print(f'  Seed {si}: {plains.sum()} plains, avg gt={[round(x,3) for x in avg_gt.tolist()]}')

print('\n--- Cross-seed GT for Forest cells ---')
for si in range(5):
    gt_s = np.array(data[str(si)]['ground_truth'])
    ig_s = np.array(rd['initial_states'][si]['grid'])
    init_cls_s = np.zeros_like(ig_s, dtype=int)
    for gv, cls in GRID_TO_CLASS.items():
        init_cls_s[ig_s == gv] = cls
    
    forest = (init_cls_s == 4)
    if forest.any():
        avg_gt = gt_s[forest].mean(axis=0)
        print(f'  Seed {si}: {forest.sum()} forests, avg gt={[round(x,3) for x in avg_gt.tolist()]}')

print('\n--- Cross-seed GT for Settlement cells ---')
for si in range(5):
    gt_s = np.array(data[str(si)]['ground_truth'])
    ig_s = np.array(rd['initial_states'][si]['grid'])
    init_cls_s = np.zeros_like(ig_s, dtype=int)
    for gv, cls in GRID_TO_CLASS.items():
        init_cls_s[ig_s == gv] = cls
    
    settle = (init_cls_s == 1)
    if settle.any():
        avg_gt = gt_s[settle].mean(axis=0)
        print(f'  Seed {si}: {settle.sum()} settlements, avg gt={[round(x,3) for x in avg_gt.tolist()]}')

# Now: what would different prediction strategies score?
# Using the formula score ≈ 100 * exp(-KL(gt||pred))
print('\n\n=== STRATEGY COMPARISON (seed 0) ===')
gt0 = np.array(data['0']['ground_truth'])

def score_pred(pred, gt):
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    kl = np.sum(gt * np.log((gt + 1e-15) / (pred + 1e-15)), axis=-1).mean()
    return 100 * np.exp(-kl)

# 1. Perfect (pred=gt)
print(f'  Perfect (pred=gt): {score_pred(gt0, gt0):.2f}')

# 2. Uniform
uniform = np.ones_like(gt0) / 6
print(f'  Uniform: {score_pred(uniform, gt0):.2f}')

# 3. Observation on argmax (99.9%)
obs999 = np.zeros_like(gt0) + 0.0002
obs999[np.arange(40)[:, None], np.arange(40)[None, :], np.argmax(gt0, axis=-1)] = 0.999
print(f'  Obs 99.9% on argmax: {score_pred(obs999, gt0):.2f}')

# 4. Observation on argmax (80%)
obs80 = np.zeros_like(gt0) + 0.04
obs80[np.arange(40)[:, None], np.arange(40)[None, :], np.argmax(gt0, axis=-1)] = 0.80
print(f'  Obs 80% on argmax: {score_pred(obs80, gt0):.2f}')

# 5. Observation on argmax (60%)  
obs60 = np.zeros_like(gt0) + 0.08
obs60[np.arange(40)[:, None], np.arange(40)[None, :], np.argmax(gt0, axis=-1)] = 0.60
print(f'  Obs 60% on argmax: {score_pred(obs60, gt0):.2f}')

# 6. Learned transition matrix per initial class
with open('data/learned_transitions.json') as f:
    T = np.array(json.load(f)['matrix'])
t_pred = np.zeros_like(gt0)
for y in range(40):
    for x in range(40):
        t_pred[y, x] = T[init_cls[y, x]]
print(f'  Global transition matrix: {score_pred(t_pred, gt0):.2f}')

# 7. HGB that learns per-cell features?
# This is our existing model - train on R1 ground truth directly
