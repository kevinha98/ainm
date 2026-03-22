"""
Observation calibration strategy sweep — with cached HGB models for speed.
Trains LOO HGB models once, then tries all calibration strategies.
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

def sample_obs(gt, rng):
    flat = np.clip(gt.reshape(-1, NC), 1e-10, None)
    flat /= flat.sum(axis=-1, keepdims=True)
    cumsum = np.cumsum(flat, axis=-1)
    u = rng.random(len(flat))
    return (u[:, None] < cumsum).argmax(axis=-1).reshape(gt.shape[:2])


# ── Pre-compute LOO predictions ──
def precompute_loo(rounds):
    """Train LOO HGB models and cache predictions."""
    loo_preds = {}  # hold_idx -> list of pred arrays (one per test seed)
    
    for hold_idx in range(len(rounds)):
        _, test_seeds = rounds[hold_idx]
        train_data = [s for i,(_, seeds) in enumerate(rounds) if i!=hold_idx for s in seeds]
        
        X, Y = [], []
        for ig, gt in train_data: X.append(extract_features(ig)); Y.append(gt.reshape(-1,NC))
        X, Y = np.vstack(X), np.vstack(Y)
        
        models = [HistGradientBoostingRegressor(max_iter=100,max_depth=4,learning_rate=0.05,
                  min_samples_leaf=50,random_state=42).fit(X,Y[:,c]) for c in range(NC)]
        
        preds = []
        for ig, gt in test_seeds:
            Xt = extract_features(ig)
            p = np.column_stack([m.predict(Xt) for m in models])
            p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
            preds.append(p.reshape(ig.shape[0], ig.shape[1], NC))
        
        loo_preds[hold_idx] = preds
        rids = [r[0][:8] for r in rounds]
        s = np.mean([kl_score(gt, p) for (ig,gt), p in zip(test_seeds, preds)])
        print(f"  LOO fold {hold_idx} ({rids[hold_idx]}): {s:.2f}")
    
    return loo_preds


# ── Calibration strategies ──

def calib_per_class(preds, grids, gts, rng, n_obs_seeds=5):
    n = len(preds); H, W = grids[0].shape
    cls_obs = np.zeros((NC, NC)); cls_pred = np.zeros((NC, NC)); cls_n = np.zeros(NC)
    for si in range(min(n_obs_seeds, n)):
        obs = sample_obs(gts[si], rng); cls = build_class_grid(grids[si])
        for ic in range(NC):
            m = cls == ic
            if not m.any(): continue
            for oc in range(NC): cls_obs[ic,oc] += np.sum(obs[m]==oc)
            cls_pred[ic] += preds[si][m].sum(axis=0); cls_n[ic] += m.sum()
    cal = []
    for si in range(n):
        p = preds[si].copy().reshape(-1, NC); c = build_class_grid(grids[si]).ravel()
        for ic in range(NC):
            if cls_n[ic] < 10: continue
            of = cls_obs[ic]/cls_n[ic]; pa = cls_pred[ic]/cls_n[ic]
            r = np.where(pa > 0.01, np.clip(of/pa, 0.3, 3.0), 1.0)
            p[c==ic] *= r
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        cal.append(p.reshape(H,W,NC))
    return cal

def calib_per_class_bayes(preds, grids, gts, rng, alpha=50, n_obs_seeds=5):
    n = len(preds); H, W = grids[0].shape
    cls_obs = np.zeros((NC, NC)); cls_n = np.zeros(NC)
    cls_pred = np.zeros((NC, NC)); cls_pred_n = np.zeros(NC)
    for si in range(min(n_obs_seeds, n)):
        obs = sample_obs(gts[si], rng); cls = build_class_grid(grids[si])
        for ic in range(NC):
            m = cls == ic
            if not m.any(): continue
            for oc in range(NC): cls_obs[ic,oc] += np.sum(obs[m]==oc)
            cls_n[ic] += m.sum()
    for si in range(n):
        cls = build_class_grid(grids[si])
        for ic in range(NC):
            m = cls == ic
            if not m.any(): continue
            cls_pred[ic] += preds[si][m].sum(axis=0); cls_pred_n[ic] += m.sum()
    cal = []
    for si in range(n):
        p = preds[si].copy().reshape(-1, NC); c = build_class_grid(grids[si]).ravel()
        for ic in range(NC):
            if cls_n[ic] < 10 or cls_pred_n[ic] < 1: continue
            prior_mean = cls_pred[ic] / cls_pred_n[ic]
            posterior = (prior_mean * alpha + cls_obs[ic]) / (alpha + cls_n[ic])
            r = np.where(prior_mean > 0.01, np.clip(posterior / prior_mean, 0.3, 3.0), 1.0)
            p[c==ic] *= r
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        cal.append(p.reshape(H,W,NC))
    return cal

def calib_cell_bayes(preds, grids, gts, rng, conc=3.0, n_obs_seeds=5):
    n = len(preds); H, W = grids[0].shape
    obs_oh = []
    for si in range(min(n_obs_seeds, n)):
        obs = sample_obs(gts[si], rng)
        obs_oh.append(np.eye(NC)[obs])
    cal = []
    for si in range(n):
        prior = preds[si] * conc
        obs_count = np.zeros((H,W,NC))
        for osi in range(len(obs_oh)):
            if osi != si: obs_count += obs_oh[osi]
        post = prior + obs_count
        post = np.clip(post, CLIP, None); post /= post.sum(axis=-1, keepdims=True)
        cal.append(post)
    return cal

def calib_hybrid(preds, grids, gts, rng, cell_conc=2.0, n_obs_seeds=5):
    c1 = calib_per_class(preds, grids, gts, rng, n_obs_seeds)
    return calib_cell_bayes(c1, grids, gts, rng, cell_conc, n_obs_seeds)

def calib_cell_bayes_all_seeds(preds, grids, gts, rng, conc=3.0, n_obs_seeds=5):
    """Per-cell Bayesian where observations include the test seed itself."""
    n = len(preds); H, W = grids[0].shape
    obs_oh = []
    for si in range(min(n_obs_seeds, n)):
        obs = sample_obs(gts[si], rng)
        obs_oh.append(np.eye(NC)[obs])
    cal = []
    for si in range(n):
        prior = preds[si] * conc
        obs_count = sum(obs_oh)  # All seeds, including si
        post = prior + obs_count
        post = np.clip(post, CLIP, None); post /= post.sum(axis=-1, keepdims=True)
        cal.append(post)
    return cal


# ── Test runner ──
def test_calib(rounds, loo_preds, calib_fn, label, n_mc=5):
    all_avgs = []
    for mc in range(n_mc):
        rng = np.random.default_rng(42 + mc)
        rscores = []
        for hold in range(len(rounds)):
            _, test = rounds[hold]
            grids = [ig for ig,_ in test]; gts = [gt for _,gt in test]
            preds = [p.copy() for p in loo_preds[hold]]
            cal = calib_fn(preds, grids, gts, rng)
            rscores.append(np.mean([kl_score(gt, p) for gt, p in zip(gts, cal)]))
        all_avgs.append(np.mean(rscores))
    avg = np.mean(all_avgs); std = np.std(all_avgs)
    print(f"  {label:40s}: {avg:.2f} ±{std:.2f}")
    return avg


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("  OBS CALIBRATION SWEEP (cached HGB models)")
    print("=" * 70)
    rounds = load_all()
    
    print("\n[1] Pre-computing LOO HGB predictions...")
    loo_preds = precompute_loo(rounds)
    print(f"  Done in {time.time()-t0:.0f}s")
    
    # Baseline
    uncal = np.mean([np.mean([kl_score(gt,p) for (_,gt),p in zip(rounds[h][1], loo_preds[h])]) for h in range(4)])
    print(f"\n  {'HGB (no obs)':40s}: {uncal:.2f}")
    
    print("\n[2] Calibration strategies (5 MC each):\n")
    
    print("--- PER-CLASS MULT ---")
    test_calib(rounds, loo_preds, lambda p,g,gt,r: calib_per_class(p,g,gt,r,5), "per-class mult (5 seeds)")
    test_calib(rounds, loo_preds, lambda p,g,gt,r: calib_per_class(p,g,gt,r,3), "per-class mult (3 seeds)")
    
    print("\n--- PER-CLASS BAYESIAN ---")
    for a in [10, 20, 50, 100, 200, 500]:
        test_calib(rounds, loo_preds, lambda p,g,gt,r,a=a: calib_per_class_bayes(p,g,gt,r,a,5), f"per-class bayes alpha={a}")
    
    print("\n--- PER-CELL BAYESIAN ---")
    for c in [0.5, 1, 2, 3, 5, 10, 20]:
        test_calib(rounds, loo_preds, lambda p,g,gt,r,c=c: calib_cell_bayes(p,g,gt,r,c,5), f"cell bayes conc={c}")
    
    print("\n--- PER-CELL BAYESIAN (incl test seed) ---")
    for c in [0.5, 1, 2, 3, 5, 10]:
        test_calib(rounds, loo_preds, lambda p,g,gt,r,c=c: calib_cell_bayes_all_seeds(p,g,gt,r,c,5), f"cell bayes+all conc={c}")
    
    print("\n--- HYBRID (class then cell) ---")
    for c in [0.5, 1, 2, 3, 5]:
        test_calib(rounds, loo_preds, lambda p,g,gt,r,c=c: calib_hybrid(p,g,gt,r,c,5), f"hybrid cell_conc={c}")
    
    print(f"\nTotal time: {time.time()-t0:.0f}s")
