"""Quick dry-run test for auto_runner_v3 HGB model.
Tests: train on all GT → predict one round → measure LOO score.
"""
import json, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')
from auto_runner_v3 import (
    extract_hgb_features, train_hgb_models, predict_hgb,
    build_class_grid, CLIP_FLOOR, TEMPERATURE, HGB_PARAMS
)
from src.settings import DATA_DIR, GRID_TO_CLASS

def kl_score(gt, pred, clip=1e-10):
    gt = np.clip(gt, clip, None)
    pred = np.clip(pred, clip, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt = gt / gt.sum(axis=-1, keepdims=True)
    kl = np.sum(gt * np.log(gt / pred), axis=-1)
    return 100 * np.exp(-kl.mean())

print("=== auto_runner_v3 Dry Run ===")
print(f"HGB params: {HGB_PARAMS}")
print(f"T={TEMPERATURE}, clip={CLIP_FLOOR}")

# Load all data
gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
print(f"Found {len(gt_files)} GT files")

all_rounds = []
for gf in gt_files:
    rid = gf.stem.replace("ground_truth_", "")
    with open(gf) as f:
        data = json.load(f)
    seeds = []
    for si_str in sorted(data.keys()):
        entry = data[si_str]
        if isinstance(entry, dict):
            gt = np.array(entry['ground_truth'])
            ig = np.array(entry['initial_grid'])
            seeds.append((ig, gt))
    all_rounds.append((rid, seeds))

print(f"Loaded {len(all_rounds)} rounds, {sum(len(s) for _,s in all_rounds)} seeds")

# Quick LOO on 3 rounds to verify
from sklearn.ensemble import HistGradientBoostingRegressor

scores = []
for hold_idx in [0, 10, 14]:  # Test 3 diverse rounds (easy, medium, hard)
    if hold_idx >= len(all_rounds):
        continue
    hold_rid, hold_seeds = all_rounds[hold_idx]
    
    # Train on everything except held-out
    X_train, Y_train = [], []
    for i, (rid, seeds) in enumerate(all_rounds):
        if i == hold_idx:
            continue
        for ig, gt in seeds:
            X_train.append(extract_hgb_features(ig))
            Y_train.append(gt.reshape(-1, 6))
    X_train = np.vstack(X_train)
    Y_train = np.vstack(Y_train)
    
    # Train models
    models = []
    for c in range(6):
        m = HistGradientBoostingRegressor(**HGB_PARAMS)
        m.fit(X_train, Y_train[:, c])
        models.append(m)
    
    # Predict held-out
    seed_scores = []
    for ig, gt in hold_seeds:
        pred = predict_hgb(models, ig)
        # Apply temperature
        if TEMPERATURE != 1.0:
            cls = build_class_grid(ig)
            mtn = (cls == 5)
            if (~mtn).any():
                p = pred[~mtn]
                p = np.clip(p, 1e-10, None)
                p = np.exp(np.log(p) / TEMPERATURE)
                pred[~mtn] = p
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        s = kl_score(gt, pred)
        seed_scores.append(s)
    
    mean_s = np.mean(seed_scores)
    scores.append(mean_s)
    print(f"  R{hold_idx+1} ({hold_rid[:8]}): {mean_s:.2f} (seeds: {[f'{x:.1f}' for x in seed_scores]})")

print(f"\nMean of 3 test rounds: {np.mean(scores):.2f}")
print("PASS - HGB model working correctly" if np.mean(scores) > 85 else "FAIL - Check model!")
