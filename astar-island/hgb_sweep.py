"""
HGB parameter sweep with 20 rounds LOO CV.
Systematic exploration of the HGB model which scores 90.74 LOO (vs LUT 72.47).
Tests: temperature, clip floor, features, HGB hyperparams.
"""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime

DATA_DIR = Path("data")
NUM_CLASSES = 6
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}


def build_class_grid(ig):
    cls = np.zeros_like(ig)
    for raw, c in GRID_TO_CLASS.items():
        cls[ig == raw] = c
    return cls


def extract_features_base(ig):
    """Base 17-feature extraction (same as cv_comprehensive)."""
    cls = build_class_grid(ig)
    H, W = ig.shape
    ocean = (ig == 10); mountain = (ig == 5)
    settlement = (cls == 1); forest = (cls == 4); empty = (cls == 0)
    port = (cls == 2); ruin = (cls == 3)
    
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
        cls_oh,                          # 6 features: one-hot class
        dist_ocean[:,:,None],            # 1: distance to ocean
        dist_settle[:,:,None],           # 1: distance to settlement
        dist_forest[:,:,None],           # 1: distance to forest
        dist_mountain[:,:,None],         # 1: distance to mountain
        n_s3[:,:,None],                  # 1: settlement count 3x3
        n_s7[:,:,None],                  # 1: settlement count 7x7
        n_f7[:,:,None],                  # 1: forest count 7x7
        n_o7[:,:,None],                  # 1: ocean count 7x7
        n_e7[:,:,None],                  # 1: empty count 7x7
        n_s11[:,:,None],                 # 1: settlement count 11x11
        is_coast[:,:,None].astype(float),# 1: coastal flag
    ], axis=-1)  # Total: 17 features
    return features.reshape(-1, features.shape[-1])


def extract_features_extended(ig):
    """Extended features: base + port/ruin distances + density ratios + position."""
    cls = build_class_grid(ig)
    H, W = ig.shape
    ocean = (ig == 10); mountain = (ig == 5)
    settlement = (cls == 1); forest = (cls == 4); empty = (cls == 0)
    port = (cls == 2); ruin = (cls == 3)
    
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H,W), 20)
    dist_port = ndimage.distance_transform_edt(~port) if port.any() else np.full((H,W), 40)
    dist_ruin = ndimage.distance_transform_edt(~ruin) if ruin.any() else np.full((H,W), 40)
    
    k3, k5, k7, k11 = np.ones((3,3)), np.ones((5,5)), np.ones((7,7)), np.ones((11,11))
    n_s3 = ndimage.convolve(settlement.astype(float), k3, mode='constant')
    n_s5 = ndimage.convolve(settlement.astype(float), k5, mode='constant')
    n_s7 = ndimage.convolve(settlement.astype(float), k7, mode='constant')
    n_f3 = ndimage.convolve(forest.astype(float), k3, mode='constant')
    n_f7 = ndimage.convolve(forest.astype(float), k7, mode='constant')
    n_o7 = ndimage.convolve(ocean.astype(float), k7, mode='constant')
    n_e7 = ndimage.convolve(empty.astype(float), k7, mode='constant')
    n_s11 = ndimage.convolve(settlement.astype(float), k11, mode='constant')
    n_p3 = ndimage.convolve(port.astype(float), k3, mode='constant')
    n_r3 = ndimage.convolve(ruin.astype(float), k3, mode='constant')
    
    # Density ratios (settlement / total non-ocean in neighborhood)
    total7 = ndimage.convolve((~ocean).astype(float), k7, mode='constant')
    settle_density7 = np.where(total7 > 0, n_s7 / total7, 0)
    forest_density7 = np.where(total7 > 0, n_f7 / total7, 0)
    
    # Row/col position (normalized)
    rows = np.repeat(np.arange(H), W).reshape(H, W) / max(H-1, 1)
    cols = np.tile(np.arange(W), H).reshape(H, W) / max(W-1, 1)
    
    # Raw grid value (distinguishes empty vs ocean-empty)
    raw_norm = ig / 11.0
    
    cls_oh = np.zeros((H, W, NUM_CLASSES))
    for c in range(NUM_CLASSES): cls_oh[:,:,c] = (cls==c).astype(float)
    
    features = np.concatenate([
        cls_oh,                          # 6
        dist_ocean[:,:,None],            # 1
        dist_settle[:,:,None],           # 1
        dist_forest[:,:,None],           # 1
        dist_mountain[:,:,None],         # 1
        dist_port[:,:,None],             # 1 NEW
        dist_ruin[:,:,None],             # 1 NEW
        n_s3[:,:,None],                  # 1
        n_s5[:,:,None],                  # 1 NEW
        n_s7[:,:,None],                  # 1
        n_f3[:,:,None],                  # 1 NEW
        n_f7[:,:,None],                  # 1
        n_o7[:,:,None],                  # 1
        n_e7[:,:,None],                  # 1
        n_s11[:,:,None],                 # 1
        n_p3[:,:,None],                  # 1 NEW
        n_r3[:,:,None],                  # 1 NEW
        is_coast[:,:,None].astype(float),# 1
        settle_density7[:,:,None],       # 1 NEW
        forest_density7[:,:,None],       # 1 NEW
        rows[:,:,None],                  # 1 NEW
        cols[:,:,None],                  # 1 NEW
        raw_norm[:,:,None],              # 1 NEW
    ], axis=-1)  # Total: 28 features
    return features.reshape(-1, features.shape[-1])


def kl_score(gt, pred, clip=1e-10):
    gt = np.clip(gt, clip, None)
    pred = np.clip(pred, clip, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt = gt / gt.sum(axis=-1, keepdims=True)
    kl = np.sum(gt * np.log(gt / pred), axis=-1)
    return 100 * np.exp(-kl.mean())


def load_all_rounds():
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


def loo_cv_score(rounds, predict_fn):
    """Quick LOO CV returning just the mean score."""
    scores = []
    for hold_idx in range(len(rounds)):
        train_data = []
        for i, (_, seeds) in enumerate(rounds):
            if i != hold_idx:
                train_data.extend(seeds)
        _, test_seeds = rounds[hold_idx]
        preds = predict_fn(train_data, test_seeds)
        seed_scores = [kl_score(gt, pred) for (ig, gt), pred in zip(test_seeds, preds)]
        scores.append(np.mean(seed_scores))
    return np.mean(scores), np.std(scores), scores


def make_hgb_predictor(hgb_params, feat_fn, clip_floor=0.0001, temperature=1.0, mountain_fix=True):
    """Create a predict function with given HGB params, features, clip, temp."""
    def predict_fn(train_data, test_seeds):
        X_train, Y_train = [], []
        for ig, gt in train_data:
            X_train.append(feat_fn(ig))
            Y_train.append(gt.reshape(-1, 6))
        X_train = np.vstack(X_train)
        Y_train = np.vstack(Y_train)
        
        models = []
        for c in range(NUM_CLASSES):
            m = HistGradientBoostingRegressor(**hgb_params)
            m.fit(X_train, Y_train[:, c])
            models.append(m)
        
        results = []
        for ig, gt in test_seeds:
            X = feat_fn(ig)
            pred = np.column_stack([m.predict(X) for m in models])
            
            # Mountain fix
            if mountain_fix:
                cls = build_class_grid(ig)
                mtn = (cls == 5).ravel()
                if mtn.any():
                    pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
                pred[~mtn, 5] = 0.0
                s = pred[~mtn].sum(axis=-1, keepdims=True)
                s = np.where(s == 0, 1, s)
                pred[~mtn] /= s
            
            # Temperature
            if temperature != 1.0:
                p = np.clip(pred, 1e-10, None)
                log_p = np.log(p) / temperature
                pred = np.exp(log_p)
            
            pred = np.clip(pred, clip_floor, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            results.append(pred.reshape(ig.shape[0], ig.shape[1], 6))
        return results
    return predict_fn


def log_result(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"| {ts} | {msg} |"
    with open("overnight_log.md", "a") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    print("Loading data...")
    rounds = load_all_rounds()
    print(f"Loaded {len(rounds)} rounds, {sum(len(s) for _,s in rounds)} seeds")
    
    best_score = 0
    best_config = ""
    
    # ── BLOCK 1: Baseline HGB with base features ──
    print("\n=== BLOCK 1: HGB with base features (17-feat) ===")
    params_base = dict(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42)
    fn = make_hgb_predictor(params_base, extract_features_base, clip_floor=0.0001, temperature=1.0)
    mean, std, per_round = loo_cv_score(rounds, fn)
    log_result(f"HGB base 17feat T=1.0 clip=1e-4 | LOO={mean:.2f}+/-{std:.2f}")
    best_score = mean
    best_config = "HGB base 17feat T=1.0 clip=1e-4"
    
    # ── BLOCK 2: HGB with extended features (28-feat) ──
    print("\n=== BLOCK 2: HGB with extended features (28-feat) ===")
    fn = make_hgb_predictor(params_base, extract_features_extended, clip_floor=0.0001, temperature=1.0)
    mean, std, per_round = loo_cv_score(rounds, fn)
    log_result(f"HGB ext 28feat T=1.0 clip=1e-4 | LOO={mean:.2f}+/-{std:.2f}")
    if mean > best_score:
        best_score = mean
        best_config = "HGB ext 28feat T=1.0 clip=1e-4"
        log_result(f"NEW BEST: {best_config} -> {best_score:.2f}")
    
    # ── BLOCK 3: Temperature sweep ──
    print("\n=== BLOCK 3: Temperature sweep ===")
    for temp in [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]:
        fn = make_hgb_predictor(params_base, extract_features_base, clip_floor=0.0001, temperature=temp)
        mean, std, _ = loo_cv_score(rounds, fn)
        delta = mean - best_score
        marker = " **NEW BEST**" if mean > best_score else ""
        log_result(f"HGB T={temp} | LOO={mean:.2f}+/-{std:.2f} delta={delta:+.2f}{marker}")
        if mean > best_score:
            best_score = mean
            best_config = f"HGB T={temp}"
    
    # ── BLOCK 4: Clip floor sweep ──
    print("\n=== BLOCK 4: Clip floor sweep ===")
    for clip in [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 0.01]:
        fn = make_hgb_predictor(params_base, extract_features_base, clip_floor=clip, temperature=1.0)
        mean, std, _ = loo_cv_score(rounds, fn)
        delta = mean - best_score
        marker = " **NEW BEST**" if mean > best_score else ""
        log_result(f"HGB clip={clip} | LOO={mean:.2f}+/-{std:.2f} delta={delta:+.2f}{marker}")
        if mean > best_score:
            best_score = mean
            best_config = f"HGB clip={clip}"
    
    # ── BLOCK 5: HGB depth sweep ──
    print("\n=== BLOCK 5: HGB depth sweep ===")
    for depth in [3, 4, 5, 6, 7]:
        params = dict(max_iter=100, max_depth=depth, learning_rate=0.05, min_samples_leaf=50, random_state=42)
        fn = make_hgb_predictor(params, extract_features_base, clip_floor=0.0001, temperature=1.0)
        mean, std, _ = loo_cv_score(rounds, fn)
        delta = mean - best_score
        marker = " **NEW BEST**" if mean > best_score else ""
        log_result(f"HGB depth={depth} | LOO={mean:.2f}+/-{std:.2f} delta={delta:+.2f}{marker}")
        if mean > best_score:
            best_score = mean
            best_config = f"HGB depth={depth}"
    
    # ── BLOCK 6: HGB iterations sweep ──
    print("\n=== BLOCK 6: HGB iterations sweep ===")
    for iters in [50, 100, 150, 200, 300]:
        params = dict(max_iter=iters, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42)
        fn = make_hgb_predictor(params, extract_features_base, clip_floor=0.0001, temperature=1.0)
        mean, std, _ = loo_cv_score(rounds, fn)
        delta = mean - best_score
        marker = " **NEW BEST**" if mean > best_score else ""
        log_result(f"HGB iter={iters} | LOO={mean:.2f}+/-{std:.2f} delta={delta:+.2f}{marker}")
        if mean > best_score:
            best_score = mean
            best_config = f"HGB iter={iters}"
    
    # ── BLOCK 7: Min samples leaf sweep ──
    print("\n=== BLOCK 7: Min samples leaf sweep ===")
    for leaf in [10, 20, 30, 50, 75, 100, 150]:
        params = dict(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=leaf, random_state=42)
        fn = make_hgb_predictor(params, extract_features_base, clip_floor=0.0001, temperature=1.0)
        mean, std, _ = loo_cv_score(rounds, fn)
        delta = mean - best_score
        marker = " **NEW BEST**" if mean > best_score else ""
        log_result(f"HGB leaf={leaf} | LOO={mean:.2f}+/-{std:.2f} delta={delta:+.2f}{marker}")
        if mean > best_score:
            best_score = mean
            best_config = f"HGB leaf={leaf}"

    # ── BLOCK 8: Learning rate sweep ──
    print("\n=== BLOCK 8: Learning rate sweep ===")
    for lr in [0.01, 0.03, 0.05, 0.08, 0.1]:
        params = dict(max_iter=100, max_depth=4, learning_rate=lr, min_samples_leaf=50, random_state=42)
        fn = make_hgb_predictor(params, extract_features_base, clip_floor=0.0001, temperature=1.0)
        mean, std, _ = loo_cv_score(rounds, fn)
        delta = mean - best_score
        marker = " **NEW BEST**" if mean > best_score else ""
        log_result(f"HGB lr={lr} | LOO={mean:.2f}+/-{std:.2f} delta={delta:+.2f}{marker}")
        if mean > best_score:
            best_score = mean
            best_config = f"HGB lr={lr}"

    # ── BLOCK 9: HGB ensemble (3 configs) ──
    print("\n=== BLOCK 9: HGB 3-config ensemble ===")
    def ensemble_predict(train_data, test_seeds):
        configs = [
            dict(max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42),
            dict(max_iter=200, max_depth=3, learning_rate=0.03, min_samples_leaf=80, random_state=123),
            dict(max_iter=150, max_depth=5, learning_rate=0.04, min_samples_leaf=30, random_state=7),
        ]
        all_preds = []
        for cfg in configs:
            fn = make_hgb_predictor(cfg, extract_features_base, clip_floor=0.0001, temperature=1.0)
            preds = fn(train_data, test_seeds)
            all_preds.append(preds)
        results = []
        for si in range(len(test_seeds)):
            avg = np.mean([all_preds[ci][si] for ci in range(len(configs))], axis=0)
            avg = np.clip(avg, 0.0001, None)
            avg /= avg.sum(axis=-1, keepdims=True)
            results.append(avg)
        return results
    mean, std, _ = loo_cv_score(rounds, ensemble_predict)
    delta = mean - best_score
    marker = " **NEW BEST**" if mean > best_score else ""
    log_result(f"HGB ensemble 3-config | LOO={mean:.2f}+/-{std:.2f} delta={delta:+.2f}{marker}")
    if mean > best_score:
        best_score = mean
        best_config = "HGB ensemble 3-config"
    
    # ── BLOCK 10: Log-target HGB ──
    print("\n=== BLOCK 10: Log-target HGB ===")
    def log_target_predict(train_data, test_seeds):
        X_all, Y_all = [], []
        for ig, gt in train_data:
            X_all.append(extract_features_base(ig))
            y = gt.reshape(-1, 6)
            y = np.clip(y, 1e-6, None)
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
            X = extract_features_base(ig)
            log_pred = np.column_stack([m.predict(X) for m in models])
            pred = np.exp(log_pred)
            cls = build_class_grid(ig)
            mtn = (cls == 5).ravel()
            if mtn.any():
                pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
            pred[~mtn, 5] = 0.0
            s = pred[~mtn].sum(axis=-1, keepdims=True)
            s = np.where(s == 0, 1, s)
            pred[~mtn] /= s
            pred = np.clip(pred, 0.0001, None)
            pred /= pred.sum(axis=-1, keepdims=True)
            results.append(pred.reshape(ig.shape[0], ig.shape[1], 6))
        return results
    mean, std, _ = loo_cv_score(rounds, log_target_predict)
    delta = mean - best_score
    marker = " **NEW BEST**" if mean > best_score else ""
    log_result(f"HGB log-target | LOO={mean:.2f}+/-{std:.2f} delta={delta:+.2f}{marker}")
    if mean > best_score:
        best_score = mean
        best_config = "HGB log-target"
    
    print(f"\n{'='*60}")
    print(f"FINAL BEST: {best_config} -> LOO = {best_score:.2f}")
    print(f"{'='*60}")
    
    # Save best config
    import json as _json
    with open("hgb_sweep_best.json", "w") as f:
        _json.dump({"best_score": best_score, "best_config": best_config}, f, indent=2)
