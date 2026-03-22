"""
Test observation calibration strategies — VECTORIZED version for speed.
N_MC=5 Monte Carlo reps per strategy.
"""
import json, numpy as np, time
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("data")
NC = 6
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}
CLIP = 0.0001

def build_class_grid(ig):
    cls = np.zeros_like(ig)
    for raw, c in GRID_TO_CLASS.items(): cls[ig == raw] = c
    return cls

def extract_features(ig):
    cls = build_class_grid(ig); H, W = ig.shape
    ocean = (ig == 10); mountain = (ig == 5)
    settlement = (cls == 1); forest = (cls == 4); empty = (cls == 0)
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H,W), 20)
    k3, k7, k11 = np.ones((3,3)), np.ones((7,7)), np.ones((11,11))
    features = np.concatenate([
        np.eye(NC)[cls],
        dist_ocean[:,:,None], dist_settle[:,:,None], dist_forest[:,:,None], dist_mountain[:,:,None],
        ndimage.convolve(settlement.astype(float), k3, mode='constant')[:,:,None],
        ndimage.convolve(settlement.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(forest.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(ocean.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(empty.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(settlement.astype(float), k11, mode='constant')[:,:,None],
        is_coast[:,:,None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1])

def kl_score(gt, pred):
    g = np.clip(gt.reshape(-1, NC), 1e-10, None)
    p = np.clip(pred.reshape(-1, NC), 1e-10, None)
    p /= p.sum(axis=-1, keepdims=True); g /= g.sum(axis=-1, keepdims=True)
    return 100 * np.exp(-np.mean(np.sum(g * np.log(g / p), axis=-1)))

def load_all():
    rounds = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        rid = gf.stem.replace("ground_truth_", "")
        with open(gf) as f: data = json.load(f)
        seeds = []
        for si in sorted(data.keys()):
            gt = np.array(data[si].get('ground_truth', []))
            ig = np.array(data[si].get('initial_grid', []))
            if gt.size > 0 and ig.size > 0: seeds.append((ig, gt))
        rounds.append((rid, seeds))
    return rounds

def train_hgb(data):
    X, Y = [], []
    for ig, gt in data: X.append(extract_features(ig)); Y.append(gt.reshape(-1,NC))
    X, Y = np.vstack(X), np.vstack(Y)
    return [HistGradientBoostingRegressor(max_iter=100,max_depth=4,learning_rate=0.05,min_samples_leaf=50,random_state=42).fit(X,Y[:,c]) for c in range(NC)]

def predict_hgb(models, ig):
    X = extract_features(ig)
    p = np.column_stack([m.predict(X) for m in models])
    p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
    return p.reshape(ig.shape[0], ig.shape[1], NC)

def sample_obs_vectorized(gt, rng):
    """Sample one observation per cell — fully vectorized."""
    flat = np.clip(gt.reshape(-1, NC), 1e-10, None)
    flat /= flat.sum(axis=-1, keepdims=True)
    cumsum = np.cumsum(flat, axis=-1)
    u = rng.random(len(flat))
    return (u[:, None] < cumsum).argmax(axis=-1).reshape(gt.shape[:2])


# ── Calibration strategies ──

def calib_none(preds, grids, gts, rng):
    return preds

def calib_per_class(preds, grids, gts, rng, n_obs_seeds=5):
    """Per-class multiplicative correction (current auto_runner strategy)."""
    n_seeds = len(preds); H, W = grids[0].shape
    per_cls_obs = np.zeros((NC, NC)); per_cls_pred = np.zeros((NC, NC)); per_cls_n = np.zeros(NC)
    
    for si in range(min(n_obs_seeds, n_seeds)):
        obs = sample_obs_vectorized(gts[si], rng)
        cls = build_class_grid(grids[si])
        for ic in range(NC):
            mask = cls == ic
            if not mask.any(): continue
            # Count observed classes for this initial class
            obs_flat = obs[mask]
            for oc in range(NC): per_cls_obs[ic, oc] += np.sum(obs_flat == oc)
            per_cls_pred[ic] += preds[si][mask].sum(axis=0)
            per_cls_n[ic] += mask.sum()
    
    cal = []
    for si in range(n_seeds):
        p = preds[si].copy().reshape(-1, NC)
        cls = build_class_grid(grids[si]).ravel()
        for ic in range(NC):
            n = per_cls_n[ic]
            if n < 10: continue
            obs_freq = per_cls_obs[ic] / n
            pred_avg = per_cls_pred[ic] / n
            ratio = np.where(pred_avg > 0.01, np.clip(obs_freq / pred_avg, 0.3, 3.0), 1.0)
            p[cls == ic] *= ratio
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        cal.append(p.reshape(H, W, NC))
    return cal

def calib_per_class_bayes(preds, grids, gts, rng, alpha=50, n_obs_seeds=5):
    """Per-class Bayesian (Dirichlet shrinkage)."""
    n_seeds = len(preds); H, W = grids[0].shape
    per_cls_obs = np.zeros((NC, NC)); per_cls_n = np.zeros(NC)
    per_cls_pred = np.zeros((NC, NC)); per_cls_pred_n = np.zeros(NC)
    
    for si in range(min(n_obs_seeds, n_seeds)):
        obs = sample_obs_vectorized(gts[si], rng)
        cls = build_class_grid(grids[si])
        for ic in range(NC):
            mask = cls == ic
            if not mask.any(): continue
            for oc in range(NC): per_cls_obs[ic, oc] += np.sum(obs[mask] == oc)
            per_cls_n[ic] += mask.sum()
    
    for si in range(n_seeds):
        cls = build_class_grid(grids[si])
        for ic in range(NC):
            mask = cls == ic
            if not mask.any(): continue
            per_cls_pred[ic] += preds[si][mask].sum(axis=0)
            per_cls_pred_n[ic] += mask.sum()
    
    cal = []
    for si in range(n_seeds):
        p = preds[si].copy().reshape(-1, NC)
        cls = build_class_grid(grids[si]).ravel()
        for ic in range(NC):
            if per_cls_n[ic] < 10 or per_cls_pred_n[ic] < 1: continue
            prior = (per_cls_pred[ic] / per_cls_pred_n[ic]) * alpha
            posterior = (prior + per_cls_obs[ic]) / (alpha + per_cls_n[ic])
            pred_avg = per_cls_pred[ic] / per_cls_pred_n[ic]
            ratio = np.where(pred_avg > 0.01, np.clip(posterior / pred_avg, 0.3, 3.0), 1.0)
            p[cls == ic] *= ratio
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        cal.append(p.reshape(H, W, NC))
    return cal

def calib_cell_bayes(preds, grids, gts, rng, conc=3.0, n_obs_seeds=5):
    """Per-cell Bayesian: Dirichlet(HGB_pred * conc + obs_counts)."""
    n_seeds = len(preds); H, W = grids[0].shape
    
    # Collect observations per seed
    obs_one_hot = []
    for si in range(min(n_obs_seeds, n_seeds)):
        obs = sample_obs_vectorized(gts[si], rng)
        oh = np.eye(NC)[obs]  # H×W×NC one-hot
        obs_one_hot.append(oh)
    
    cal = []
    for si in range(n_seeds):
        prior = preds[si] * conc  # H×W×NC
        obs_count = np.zeros((H, W, NC))
        for osi in range(len(obs_one_hot)):
            if osi != si:
                obs_count += obs_one_hot[osi]
        posterior = prior + obs_count
        posterior = np.clip(posterior, CLIP, None)
        posterior /= posterior.sum(axis=-1, keepdims=True)
        cal.append(posterior)
    return cal

def calib_hybrid(preds, grids, gts, rng, cell_conc=2.0, n_obs_seeds=5):
    """Per-class correction first, then per-cell Bayesian."""
    class_cal = calib_per_class(preds, grids, gts, rng, n_obs_seeds)
    return calib_cell_bayes(class_cal, grids, gts, rng, cell_conc, n_obs_seeds)


# ── LOO CV engine ──
def run_cv(rounds, calib_fn, label, n_mc=5):
    all_avgs = []
    for mc in range(n_mc):
        rng = np.random.default_rng(42 + mc)
        rscores = []
        for hold in range(len(rounds)):
            _, test = rounds[hold]
            train = [s for i,(_, seeds) in enumerate(rounds) if i!=hold for s in seeds]
            models = train_hgb(train)
            grids = [ig for ig, _ in test]; gts = [gt for _, gt in test]
            preds = [predict_hgb(models, ig) for ig in grids]
            preds = calib_fn(preds, grids, gts, rng)
            rscores.append(np.mean([kl_score(gt, p) for gt, p in zip(gts, preds)]))
        all_avgs.append(np.mean(rscores))
    
    avg = np.mean(all_avgs); std = np.std(all_avgs)
    print(f"  {label:40s}: {avg:.2f} ±{std:.2f}")
    return avg


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("  OBSERVATION CALIBRATION STRATEGIES — VECTORIZED LOO CV")
    print("=" * 70)
    rounds = load_all()
    
    print("\n--- BASELINES ---")
    run_cv(rounds, calib_none, "HGB (no obs)", n_mc=1)
    
    print("\n--- PER-CLASS MULT (current strategy) ---")
    run_cv(rounds, lambda p,g,gt,r: calib_per_class(p,g,gt,r,5), "Per-class mult (5 seeds)")
    run_cv(rounds, lambda p,g,gt,r: calib_per_class(p,g,gt,r,3), "Per-class mult (3 seeds)")
    
    print("\n--- PER-CLASS BAYESIAN ---")
    for a in [20, 50, 100, 200]:
        run_cv(rounds, lambda p,g,gt,r,a=a: calib_per_class_bayes(p,g,gt,r,a,5), f"Per-class Bayes alpha={a}")
    
    print("\n--- PER-CELL BAYESIAN ---")
    for c in [0.5, 1, 2, 3, 5, 10]:
        run_cv(rounds, lambda p,g,gt,r,c=c: calib_cell_bayes(p,g,gt,r,c,5), f"Cell Bayes conc={c}")
    
    print("\n--- HYBRID (class + cell) ---")
    for c in [0.5, 1, 2, 3, 5]:
        run_cv(rounds, lambda p,g,gt,r,c=c: calib_hybrid(p,g,gt,r,c,5), f"Hybrid cell_conc={c}")
    
    print(f"\nTotal time: {time.time()-t0:.0f}s")
