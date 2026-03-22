"""
Test improved observation calibration strategies in LOO CV.
Simulates observations by sampling from GT distribution.
"""
import json, numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("data")
NUM_CLASSES = 6
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}
CLIP = 0.0001

def build_class_grid(ig):
    cls = np.zeros_like(ig)
    for raw, c in GRID_TO_CLASS.items(): cls[ig == raw] = c
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
        cls_oh, dist_ocean[:,:,None], dist_settle[:,:,None],
        dist_forest[:,:,None], dist_mountain[:,:,None],
        n_s3[:,:,None], n_s7[:,:,None], n_f7[:,:,None],
        n_o7[:,:,None], n_e7[:,:,None], n_s11[:,:,None],
        is_coast[:,:,None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1])

def kl_score(gt, pred):
    gt = np.clip(gt, 1e-10, None); pred = np.clip(pred, 1e-10, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt = gt / gt.sum(axis=-1, keepdims=True)
    kl = np.sum(gt * np.log(gt / pred), axis=-1)
    return 100 * np.exp(-kl.mean())

def load_all_rounds():
    rounds = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        rid = gf.stem.replace("ground_truth_", "")
        with open(gf) as f: data = json.load(f)
        seeds = []
        for si_str in sorted(data.keys()):
            gt = np.array(data[si_str].get('ground_truth', []))
            ig = np.array(data[si_str].get('initial_grid', []))
            if gt.size > 0 and ig.size > 0: seeds.append((ig, gt))
        rounds.append((rid, seeds))
    return rounds

def train_hgb(data):
    X, Y = [], []
    for ig, gt in data:
        X.append(extract_features(ig)); Y.append(gt.reshape(-1,6))
    X, Y = np.vstack(X), np.vstack(Y)
    models = [HistGradientBoostingRegressor(
        max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42
    ).fit(X, Y[:,c]) for c in range(6)]
    return models

def predict_hgb(models, ig):
    X = extract_features(ig)
    p = np.column_stack([m.predict(X) for m in models])
    p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
    return p.reshape(ig.shape[0], ig.shape[1], 6)

def sample_observation(gt, rng):
    """Sample one observation per cell from GT distribution (simulates API observe)."""
    H, W, C = gt.shape
    flat = gt.reshape(-1, C)
    flat = np.clip(flat, 1e-10, None)
    flat /= flat.sum(axis=-1, keepdims=True)
    obs = np.zeros(H * W, dtype=int)
    for i in range(H * W):
        obs[i] = rng.choice(C, p=flat[i])
    return obs.reshape(H, W)


# ── Calibration strategies ──

def calibrate_per_class(preds, grids, gt_list, n_obs_seeds, rng):
    """Current strategy: per-class multiplicative correction.
    Uses n_obs_seeds seeds for observations."""
    n_seeds = len(preds)
    H, W = grids[0].shape
    
    # Aggregate per-class stats from observed seeds
    per_class_obs = {c: np.zeros(6) for c in range(6)}
    per_class_pred = {c: np.zeros(6) for c in range(6)}
    per_class_n = {c: 0 for c in range(6)}
    
    for si in range(min(n_obs_seeds, n_seeds)):
        obs_grid = sample_observation(gt_list[si], rng)
        cls_grid = build_class_grid(grids[si])
        for y in range(H):
            for x in range(W):
                ic = cls_grid[y, x]
                oc = obs_grid[y, x]
                per_class_obs[ic][oc] += 1
                per_class_pred[ic] += preds[si][y, x]
                per_class_n[ic] += 1
    
    # Apply calibration to ALL seeds
    calibrated = []
    for si in range(n_seeds):
        pred = preds[si].copy().reshape(-1, 6)
        cls_grid = build_class_grid(grids[si])
        for ic in range(6):
            n = per_class_n[ic]
            if n < 10: continue
            obs_freq = per_class_obs[ic] / n
            pred_avg = per_class_pred[ic] / n
            ratio = np.ones(6)
            for k in range(6):
                if pred_avg[k] > 0.01:
                    ratio[k] = np.clip(obs_freq[k] / pred_avg[k], 0.3, 3.0)
            mask = cls_grid.ravel() == ic
            pred[mask] *= ratio
        pred = np.clip(pred, CLIP, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        calibrated.append(pred.reshape(H, W, 6))
    return calibrated


def calibrate_bayesian_cell(preds, grids, gt_list, n_obs_seeds, rng, concentration=3.0):
    """Per-cell Bayesian calibration using Dirichlet conjugate prior.
    Prior = HGB prediction * concentration
    Posterior = Prior + observed counts across seeds."""
    n_seeds = len(preds)
    H, W = grids[0].shape
    
    # Collect observations per cell per seed
    cell_counts = np.zeros((n_seeds, H, W, 6))
    observed_seeds = min(n_obs_seeds, n_seeds)
    
    for si in range(observed_seeds):
        obs_grid = sample_observation(gt_list[si], rng)
        for y in range(H):
            for x in range(W):
                cell_counts[si, y, x, obs_grid[y, x]] += 1
    
    # For each seed, compute posterior
    calibrated = []
    for si in range(n_seeds):
        prior = preds[si] * concentration  # Dirichlet prior from HGB
        
        # Sum observations from OTHER seeds (not si itself to avoid data leakage in CV)
        obs_count = np.zeros((H, W, 6))
        for osi in range(observed_seeds):
            if osi != si:
                obs_count += cell_counts[osi]
        
        posterior = prior + obs_count
        posterior = np.clip(posterior, CLIP, None)
        posterior /= posterior.sum(axis=-1, keepdims=True)
        calibrated.append(posterior)
    return calibrated


def calibrate_hybrid(preds, grids, gt_list, n_obs_seeds, rng, cell_conc=2.0):
    """Hybrid: per-class correction THEN per-cell Bayesian update."""
    # First apply per-class correction
    class_cal = calibrate_per_class(preds, grids, gt_list, n_obs_seeds, rng)
    # Then apply per-cell Bayesian on the class-corrected predictions
    return calibrate_bayesian_cell(class_cal, grids, gt_list, n_obs_seeds, rng, cell_conc)


def calibrate_per_class_bayesian(preds, grids, gt_list, n_obs_seeds, rng, alpha=50):
    """Per-class Bayesian: Dirichlet prior from HGB class average, updated with observations.
    Uses Bayesian shrinkage so rare classes get less aggressive correction."""
    n_seeds = len(preds)
    H, W = grids[0].shape
    
    # Aggregate per-class observations
    per_class_obs = {c: np.zeros(6) for c in range(6)}
    per_class_n = {c: 0 for c in range(6)}
    
    for si in range(min(n_obs_seeds, n_seeds)):
        obs_grid = sample_observation(gt_list[si], rng)
        cls_grid = build_class_grid(grids[si])
        for y in range(H):
            for x in range(W):
                ic = cls_grid[y, x]
                per_class_obs[ic][obs_grid[y, x]] += 1
                per_class_n[ic] += 1
    
    # Compute per-class HGB mean (as Dirichlet prior)
    per_class_pred_mean = {c: np.zeros(6) for c in range(6)}
    per_class_pred_n = {c: 0 for c in range(6)}
    for si in range(n_seeds):
        cls_grid = build_class_grid(grids[si])
        for c in range(6):
            mask = cls_grid == c
            if mask.any():
                per_class_pred_mean[c] += preds[si][mask].sum(axis=0)
                per_class_pred_n[c] += mask.sum()
    
    for c in range(6):
        if per_class_pred_n[c] > 0:
            per_class_pred_mean[c] /= per_class_pred_n[c]
        else:
            per_class_pred_mean[c] = np.ones(6) / 6
    
    # Bayesian estimate: prior + observations
    per_class_posterior = {}
    for c in range(6):
        prior = per_class_pred_mean[c] * alpha
        obs = per_class_obs[c]
        posterior = prior + obs
        posterior = np.clip(posterior, CLIP, None)
        per_class_posterior[c] = posterior / posterior.sum()
    
    # Apply correction ratio
    calibrated = []
    for si in range(n_seeds):
        pred = preds[si].copy().reshape(-1, 6)
        cls_grid = build_class_grid(grids[si])
        for ic in range(6):
            n = per_class_n[ic]
            if n < 10: continue
            posterior = per_class_posterior[ic]
            pred_avg = per_class_pred_mean[ic]
            ratio = np.ones(6)
            for k in range(6):
                if pred_avg[k] > 0.01:
                    ratio[k] = np.clip(posterior[k] / pred_avg[k], 0.3, 3.0)
            mask = cls_grid.ravel() == ic
            pred[mask] *= ratio
        pred = np.clip(pred, CLIP, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        calibrated.append(pred.reshape(H, W, 6))
    return calibrated


# ── LOO CV ──

def run_loo_cv(rounds, calib_fn, label, n_mc=5):
    """LOO CV with observation calibration.
    n_mc: number of Monte Carlo repetitions (observations are stochastic)."""
    all_scores = []
    
    for mc in range(n_mc):
        rng = np.random.default_rng(42 + mc)
        round_scores = []
        
        for hold_idx in range(len(rounds)):
            _, test_seeds = rounds[hold_idx]
            train_data = []
            for i, (_, seeds) in enumerate(rounds):
                if i != hold_idx: train_data.extend(seeds)
            
            models = train_hgb(train_data)
            
            grids = [ig for ig, _ in test_seeds]
            gts = [gt for _, gt in test_seeds]
            preds = [predict_hgb(models, ig) for ig in grids]
            
            if calib_fn is not None:
                preds = calib_fn(preds, grids, gts, 5, rng)
            
            s = np.mean([kl_score(gt, p) for gt, p in zip(gts, preds)])
            round_scores.append(s)
        
        all_scores.append(round_scores)
    
    # Average over MC runs
    avg_scores = np.mean(all_scores, axis=0)
    overall = np.mean(avg_scores)
    std = np.std(np.mean(all_scores, axis=1))
    
    r_str = " ".join(f"{s:.2f}" for s in avg_scores)
    print(f"  {label:35s}: avg={overall:.2f} ±{std:.2f} [{r_str}]")
    return overall


if __name__ == "__main__":
    print("=" * 75)
    print("  OBSERVATION CALIBRATION STRATEGIES — LOO CV")
    print("=" * 75)
    
    rounds = load_all_rounds()
    print(f"Loaded {len(rounds)} rounds\n")
    
    # No calibration baseline
    print("--- UNCALIBRATED ---")
    run_loo_cv(rounds, None, "HGB base (no obs)", n_mc=1)
    
    print("\n--- PER-CLASS CALIBRATION ---")
    run_loo_cv(rounds, lambda p,g,gt,n,rng: calibrate_per_class(p,g,gt,5,rng),
               "Per-class mult (5 seeds)")
    run_loo_cv(rounds, lambda p,g,gt,n,rng: calibrate_per_class(p,g,gt,3,rng),
               "Per-class mult (3 seeds)")
    
    print("\n--- PER-CLASS BAYESIAN ---")
    for alpha in [20, 50, 100, 200]:
        run_loo_cv(rounds, lambda p,g,gt,n,rng,a=alpha: calibrate_per_class_bayesian(p,g,gt,5,rng,a),
                   f"Per-class Bayes (a={alpha}, 5 seeds)")
    
    print("\n--- PER-CELL BAYESIAN ---")
    for conc in [1, 2, 3, 5, 10]:
        run_loo_cv(rounds, lambda p,g,gt,n,rng,c=conc: calibrate_bayesian_cell(p,g,gt,5,rng,c),
                   f"Per-cell Bayes (conc={conc}, 5 seeds)")
    
    print("\n--- HYBRID (class then cell) ---")
    for conc in [1, 2, 3, 5]:
        run_loo_cv(rounds, lambda p,g,gt,n,rng,c=conc: calibrate_hybrid(p,g,gt,5,rng,c),
                   f"Hybrid (class+cell conc={conc})")
