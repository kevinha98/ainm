"""
Comprehensive cross-round LOO CV with 4 rounds of GT data.
Tests HGB model and several improvement ideas.
"""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("data")
NUM_CLASSES = 6
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
CLIP_FLOOR = 0.0001


def build_class_grid(ig):
    cls = np.zeros_like(ig)
    for raw, c in GRID_TO_CLASS.items():
        cls[ig == raw] = c
    return cls


def extract_features(ig):
    cls = build_class_grid(ig)
    H, W = ig.shape
    ocean = (ig == 10); mountain = (ig == 5)
    settlement = (cls == 1); forest = (cls == 4); empty = (cls == 0)
    
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H,W), 20)
    
    k3, k7, k11 = np.ones((3,3)), np.ones((7,7)), np.ones((11,11))
    n_s3 = ndimage.convolve(settlement.astype(float), k3, mode='constant')
    n_s7 = ndimage.convolve(settlement.astype(float), k7, mode='constant')
    n_f7 = ndimage.convolve(forest.astype(float), k7, mode='constant')
    n_o7 = ndimage.convolve(ocean.astype(float), k7, mode='constant')
    n_e7 = ndimage.convolve(empty.astype(float), k7, mode='constant')
    n_s11 = ndimage.convolve(settlement.astype(float), k11, mode='constant')
    
    cls_oh = np.zeros((H, W, NUM_CLASSES))
    for c in range(NUM_CLASSES): cls_oh[:,:,c] = (cls==c).astype(float)
    
    features = np.concatenate([
        cls_oh,
        dist_ocean[:,:,None], dist_settle[:,:,None], dist_forest[:,:,None], dist_mountain[:,:,None],
        n_s3[:,:,None], n_s7[:,:,None], n_f7[:,:,None], n_o7[:,:,None], n_e7[:,:,None], n_s11[:,:,None],
        is_coast[:,:,None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1])


def kl_score(gt, pred, clip=1e-10):
    """Compute score ≈ 100 * exp(-mean_KL(gt||pred))."""
    gt = np.clip(gt, clip, None)
    pred = np.clip(pred, clip, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt = gt / gt.sum(axis=-1, keepdims=True)
    kl = np.sum(gt * np.log(gt / pred), axis=-1)
    return 100 * np.exp(-kl.mean())


def load_all_rounds():
    """Load all GT data, returns list of (round_id, list of (ig, gt) per seed)."""
    rounds = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        rid = gf.stem.replace("ground_truth_", "")
        with open(gf) as f:
            data = json.load(f)
        seeds = []
        for si_str in sorted(data.keys()):
            gt = np.array(data[si_str].get('ground_truth', []))
            ig = np.array(data[si_str].get('initial_grid', []))
            if gt.size > 0 and ig.size > 0:
                seeds.append((ig, gt))
        rounds.append((rid, seeds))
    return rounds


def train_hgb(train_data, hgb_params=None):
    """Train HGB on given data."""
    if hgb_params is None:
        hgb_params = dict(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42)
    
    X_all, Y_all = [], []
    for ig, gt in train_data:
        X_all.append(extract_features(ig))
        Y_all.append(gt.reshape(-1, 6))
    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)
    
    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(**hgb_params)
        m.fit(X_all, Y_all[:, c])
        models.append(m)
    return models


def predict_hgb(models, ig, clip=CLIP_FLOOR):
    X = extract_features(ig)
    pred = np.column_stack([m.predict(X) for m in models])
    pred = np.clip(pred, clip, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred.reshape(ig.shape[0], ig.shape[1], 6)


def loo_cv(rounds, model_fn, label=""):
    """Leave-one-round-out cross validation."""
    scores = []
    for hold_idx in range(len(rounds)):
        rid, test_seeds = rounds[hold_idx]
        train_data = []
        for i, (_, seeds) in enumerate(rounds):
            if i != hold_idx:
                train_data.extend(seeds)
        
        result = model_fn(train_data, test_seeds)
        
        seed_scores = []
        for (ig, gt), pred in zip(test_seeds, result):
            s = kl_score(gt, pred)
            seed_scores.append(s)
        
        avg = np.mean(seed_scores)
        scores.append(avg)
        print(f"  R{hold_idx+1} ({rid[:8]}): {avg:.2f} (seeds: {[f'{s:.1f}' for s in seed_scores]})")
    
    mean = np.mean(scores)
    print(f"  {'LOO avg':>15s}: {mean:.2f}")
    return mean, scores


# ── Model functions (return list of predictions for test seeds)

def model_hgb_base(train_data, test_seeds):
    models = train_hgb(train_data)
    return [predict_hgb(models, ig) for ig, gt in test_seeds]


def model_hgb_deeper(train_data, test_seeds):
    models = train_hgb(train_data, dict(max_iter=200, max_depth=5, learning_rate=0.03, min_samples_leaf=30, random_state=42))
    return [predict_hgb(models, ig) for ig, gt in test_seeds]


def model_hgb_wider(train_data, test_seeds):
    models = train_hgb(train_data, dict(max_iter=300, max_depth=3, learning_rate=0.03, min_samples_leaf=80, random_state=42))
    return [predict_hgb(models, ig) for ig, gt in test_seeds]


def model_per_class_avg(train_data, test_seeds):
    """Simple per-class average baseline."""
    all_cls, all_gt = [], []
    for ig, gt in train_data:
        all_cls.append(build_class_grid(ig).ravel())
        all_gt.append(gt.reshape(-1, 6))
    all_cls = np.concatenate(all_cls)
    all_gt = np.vstack(all_gt)
    
    avgs = {}
    for c in range(NUM_CLASSES):
        mask = all_cls == c
        avgs[c] = all_gt[mask].mean(axis=0) if mask.any() else np.ones(6)/6
    
    results = []
    for ig, gt in test_seeds:
        cls = build_class_grid(ig)
        pred = np.zeros((ig.shape[0], ig.shape[1], 6))
        for c in range(NUM_CLASSES):
            mask = cls == c
            pred[mask] = avgs[c]
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        results.append(pred)
    return results


def model_hgb_log_target(train_data, test_seeds):
    """HGB trained on log(probability) targets — might optimize KL better."""
    X_all, Y_all = [], []
    for ig, gt in train_data:
        X_all.append(extract_features(ig))
        y = gt.reshape(-1, 6)
        y = np.clip(y, 1e-6, None)  # Avoid log(0)
        Y_all.append(np.log(y))
    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)
    
    params = dict(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42)
    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(**params)
        m.fit(X_all, Y_all[:, c])
        models.append(m)
    
    results = []
    for ig, gt in test_seeds:
        X = extract_features(ig)
        log_pred = np.column_stack([m.predict(X) for m in models])
        pred = np.exp(log_pred)  # Back to probability space
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        results.append(pred.reshape(ig.shape[0], ig.shape[1], 6))
    return results


def model_hgb_ensemble_3(train_data, test_seeds):
    """Ensemble of 3 HGB configs, averaged."""
    configs = [
        dict(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42),
        dict(max_iter=200, max_depth=3, learning_rate=0.03, min_samples_leaf=80, random_state=123),
        dict(max_iter=150, max_depth=5, learning_rate=0.04, min_samples_leaf=30, random_state=7),
    ]
    all_preds = []
    for cfg in configs:
        models = train_hgb(train_data, cfg)
        preds = [predict_hgb(models, ig) for ig, gt in test_seeds]
        all_preds.append(preds)
    
    results = []
    for si in range(len(test_seeds)):
        avg = np.mean([all_preds[ci][si] for ci in range(len(configs))], axis=0)
        avg = np.clip(avg, CLIP_FLOOR, None)
        avg /= avg.sum(axis=-1, keepdims=True)
        results.append(avg)
    return results


def model_hgb_with_raw_grid(train_data, test_seeds):
    """HGB with additional feature: raw grid value (0 vs 10 vs 11 for 'empty')."""
    def extract_ext(ig):
        base = extract_features(ig)
        raw_norm = ig.ravel()[:, None] / 11.0  # Normalize raw grid value
        return np.hstack([base, raw_norm])
    
    X_all, Y_all = [], []
    for ig, gt in train_data:
        X_all.append(extract_ext(ig))
        Y_all.append(gt.reshape(-1, 6))
    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)
    
    params = dict(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42)
    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(**params)
        m.fit(X_all, Y_all[:, c])
        models.append(m)
    
    results = []
    for ig, gt in test_seeds:
        X = extract_ext(ig)
        pred = np.column_stack([m.predict(X) for m in models])
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        results.append(pred.reshape(ig.shape[0], ig.shape[1], 6))
    return results


def model_hgb_position_features(train_data, test_seeds):
    """HGB with row/col normalized position features."""
    def extract_pos(ig):
        base = extract_features(ig)
        H, W = ig.shape
        rows = np.repeat(np.arange(H), W) / H  # 0..1
        cols = np.tile(np.arange(W), H) / W  # 0..1
        return np.hstack([base, rows[:, None], cols[:, None]])
    
    X_all, Y_all = [], []
    for ig, gt in train_data:
        X_all.append(extract_pos(ig))
        Y_all.append(gt.reshape(-1, 6))
    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)
    
    params = dict(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42)
    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(**params)
        m.fit(X_all, Y_all[:, c])
        models.append(m)
    
    results = []
    for ig, gt in test_seeds:
        X = extract_pos(ig)
        pred = np.column_stack([m.predict(X) for m in models])
        pred = np.clip(pred, CLIP_FLOOR, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        results.append(pred.reshape(ig.shape[0], ig.shape[1], 6))
    return results


# ── Main
if __name__ == "__main__":
    print("=" * 70)
    print("  COMPREHENSIVE CROSS-ROUND LOO CV (4 rounds)")
    print("=" * 70)
    
    rounds = load_all_rounds()
    print(f"\nLoaded {len(rounds)} rounds:")
    for rid, seeds in rounds:
        n_cells = sum(ig.shape[0]*ig.shape[1] for ig, _ in seeds)
        print(f"  {rid[:8]}: {len(seeds)} seeds, {n_cells} cells")
    
    experiments = [
        ("Per-class avg baseline", model_per_class_avg),
        ("HGB base (100iter/d4)", model_hgb_base),
        ("HGB deeper (200iter/d5)", model_hgb_deeper),
        ("HGB wider (300iter/d3)", model_hgb_wider),
        ("HGB log-target", model_hgb_log_target),
        ("HGB ensemble (3 configs)", model_hgb_ensemble_3),
        ("HGB + raw grid feature", model_hgb_with_raw_grid),
        ("HGB + position features", model_hgb_position_features),
    ]
    
    results = {}
    for name, fn in experiments:
        print(f"\n{'─'*60}")
        print(f"Testing: {name}")
        print(f"{'─'*60}")
        avg, per_round = loo_cv(rounds, fn, name)
        results[name] = (avg, per_round)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY (sorted by LOO avg)")
    print(f"{'='*70}")
    sorted_results = sorted(results.items(), key=lambda x: -x[1][0])
    for name, (avg, rounds_scores) in sorted_results:
        r_str = " ".join(f"{s:.1f}" for s in rounds_scores)
        delta = avg - results["Per-class avg baseline"][0]
        print(f"  {avg:6.2f} ({delta:+5.2f}) | {name:30s} | [{r_str}]")
