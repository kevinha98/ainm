"""Quick test of Bayes + blend strategies (continued from cv_calib_sweep.py)."""
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

t0 = time.time()
print("=== BAYES + BLEND STRATEGIES ===\n")
rounds = load_all()
nR = len(rounds)
VP = 15

def grid_positions(dim):
    n = max(1, -(-dim // VP))
    if n == 1: return [0]
    step = (dim - VP) / (n - 1)
    return [round(i * step) for i in range(n)]

print("[1] Pre-training LOO models...", flush=True)
cached = {}
for hold in range(nR):
    _, test_seeds = rounds[hold]
    train = [s for i,(_, ss) in enumerate(rounds) if i!=hold for s in ss]
    X, Y = [], []
    for ig, gt in train: X.append(extract_features(ig)); Y.append(gt.reshape(-1, NC))
    X, Y = np.vstack(X), np.vstack(Y)
    models = [HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05,
              min_samples_leaf=50, random_state=42).fit(X, Y[:,c]) for c in range(NC)]
    grids = [ig for ig,_ in test_seeds]; gts = [gt for _,gt in test_seeds]
    H, W = grids[0].shape
    preds = []
    for ig, gt in test_seeds:
        p = np.column_stack([m.predict(extract_features(ig)) for m in models])
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        preds.append(p.reshape(H, W, NC))
    rows = grid_positions(H); cols = grid_positions(W)
    vps = [(r,c) for r in rows for c in cols]
    cached[hold] = {"grids": grids, "gts": gts, "preds": preds, "vps": vps, "H": H, "W": W}
print(f"  Done in {time.time()-t0:.0f}s\n", flush=True)


def run_calib(name, calib_fn, n_mc=5):
    mc_scores = []
    for mc in range(n_mc):
        rng = np.random.default_rng(mc * 1000 + 42)
        fold_scores = []
        for hold in range(nR):
            c = cached[hold]
            grids, gts, preds = c["grids"], c["gts"], c["preds"]
            vps, H, W = c["vps"], c["H"], c["W"]
            per_cls_obs = np.zeros((NC, NC)); per_cls_pred = np.zeros((NC, NC)); per_cls_n = np.zeros(NC)
            obs_used = 0
            for si in range(len(grids)):
                obs_grid = sample_obs(gts[si], rng)
                for row, col in vps:
                    if obs_used >= 45: break
                    cls = build_class_grid(grids[si])
                    for vy in range(min(VP, H-row)):
                        for vx in range(min(VP, W-col)):
                            gy, gx = row+vy, col+vx
                            if gy >= H or gx >= W: continue
                            ic = cls[gy, gx]; oc = obs_grid[gy, gx]
                            per_cls_obs[ic, oc] += 1
                            per_cls_pred[ic] += preds[si][gy, gx]
                            per_cls_n[ic] += 1
                    obs_used += 1
                if obs_used >= 45: break
            cal = calib_fn(preds, grids, per_cls_obs, per_cls_pred, per_cls_n)
            fold_scores.append(np.mean([kl_score(gts[si], cal[si]) for si in range(len(grids))]))
        mc_scores.append(np.mean(fold_scores))
    avg = np.mean(mc_scores); std = np.std(mc_scores)
    
    rng = np.random.default_rng(42)
    folds = []
    for hold in range(nR):
        c = cached[hold]
        grids, gts, preds = c["grids"], c["gts"], c["preds"]
        vps, H, W = c["vps"], c["H"], c["W"]
        per_cls_obs = np.zeros((NC, NC)); per_cls_pred = np.zeros((NC, NC)); per_cls_n = np.zeros(NC)
        obs_used = 0
        for si in range(len(grids)):
            obs_grid = sample_obs(gts[si], rng)
            for row, col in vps:
                if obs_used >= 45: break
                cls = build_class_grid(grids[si])
                for vy in range(min(VP, H-row)):
                    for vx in range(min(VP, W-col)):
                        gy, gx = row+vy, col+vx
                        if gy >= H or gx >= W: continue
                        ic = cls[gy, gx]; oc = obs_grid[gy, gx]
                        per_cls_obs[ic, oc] += 1
                        per_cls_pred[ic] += preds[si][gy, gx]
                        per_cls_n[ic] += 1
                obs_used += 1
            if obs_used >= 45: break
        cal = calib_fn(preds, grids, per_cls_obs, per_cls_pred, per_cls_n)
        folds.append(np.mean([kl_score(gts[si], cal[si]) for si in range(len(grids))]))
    folds_str = " ".join(f"{f:.1f}" for f in folds)
    print(f"  {name:40s}: {avg:.2f} +/-{std:.2f} | {folds_str}", flush=True)
    return avg


def mult(clip_lo, clip_hi):
    def fn(preds, grids, cls_obs, cls_pred, cls_n):
        cal = []
        for si in range(len(preds)):
            p = preds[si].copy().reshape(-1, NC)
            cls = build_class_grid(grids[si]).ravel()
            for ic in range(NC):
                if cls_n[ic] < 10: continue
                of = cls_obs[ic]/cls_n[ic]; pa = cls_pred[ic]/cls_n[ic]
                r = np.where(pa > 0.01, np.clip(of/pa, clip_lo, clip_hi), 1.0)
                p[cls==ic] *= r
            p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
            cal.append(p.reshape(grids[0].shape[0], grids[0].shape[1], NC))
        return cal
    return fn


def bayes(alpha, clip_lo, clip_hi):
    def fn(preds, grids, cls_obs, cls_pred, cls_n):
        cal = []
        for si in range(len(preds)):
            p = preds[si].copy().reshape(-1, NC)
            cls = build_class_grid(grids[si]).ravel()
            for ic in range(NC):
                if cls_n[ic] < 10: continue
                prior = cls_pred[ic] / cls_n[ic]
                post = (prior * alpha + cls_obs[ic]) / (alpha + cls_n[ic])
                r = np.where(prior > 0.01, np.clip(post/prior, clip_lo, clip_hi), 1.0)
                p[cls==ic] *= r
            p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
            cal.append(p.reshape(grids[0].shape[0], grids[0].shape[1], NC))
        return cal
    return fn


def interp(alpha_lo, alpha_hi, clip_lo, clip_hi):
    """Adaptive alpha: use smaller alpha (more obs-driven) when obs/pred gap is large."""
    def fn(preds, grids, cls_obs, cls_pred, cls_n):
        cal = []
        for si in range(len(preds)):
            p = preds[si].copy().reshape(-1, NC)
            cls = build_class_grid(grids[si]).ravel()
            for ic in range(NC):
                if cls_n[ic] < 10: continue
                prior = cls_pred[ic] / cls_n[ic]
                obs_dist = cls_obs[ic] / cls_n[ic]
                # KL divergence between obs and pred for this class
                kl = np.sum(np.where(obs_dist > 0.01, obs_dist * np.log(obs_dist / np.clip(prior, 1e-10, None)), 0))
                # Higher KL -> trust observations more -> lower alpha
                alpha = alpha_hi * np.exp(-kl * 2) + alpha_lo * (1 - np.exp(-kl * 2))
                post = (prior * alpha + cls_obs[ic]) / (alpha + cls_n[ic])
                r = np.where(prior > 0.01, np.clip(post/prior, clip_lo, clip_hi), 1.0)
                p[cls==ic] *= r
            p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
            cal.append(p.reshape(grids[0].shape[0], grids[0].shape[1], NC))
        return cal
    return fn


# Run tests
print("[2] Testing strategies (5 MC each):\n", flush=True)

# Reference
run_calib("mult [0.3, 3] BASELINE", mult(0.3, 3.0))
run_calib("mult [0.01, 100] BEST-CLIP", mult(0.01, 100.0))

# Bayes with wide clip
for a in [5, 10, 20, 50, 100]:
    run_calib(f"bayes a={a:3d} [0.01, 100]", bayes(a, 0.01, 100.0))

# Bayes with no clip
for a in [5, 10, 20]:
    run_calib(f"bayes a={a:3d} NO CLIP", bayes(a, 0.0, 1000.0))

# Adaptive alpha
run_calib("adapt a=[5,100] [0.01, 100]", interp(5, 100, 0.01, 100.0))
run_calib("adapt a=[1,50] [0.01, 100]", interp(1, 50, 0.01, 100.0))
run_calib("adapt a=[10,200] [0.01, 100]", interp(10, 200, 0.01, 100.0))

print(f"\nTotal: {time.time()-t0:.0f}s")
