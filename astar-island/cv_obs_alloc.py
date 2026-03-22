"""Test observation allocation strategies.
Current: 9 viewports * 5 seeds = 45 obs (uniform across seeds)
Test: Can we do better with different allocations?"""
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
        with open(gf) as f: data = json.load(f)
        seeds = []
        for si in sorted(data.keys()):
            gt = np.array(data[si].get('ground_truth', []))
            ig = np.array(data[si].get('initial_grid', []))
            if gt.size > 0 and ig.size > 0: seeds.append((ig, gt))
        rounds.append(seeds)
    return rounds

def sample_obs(gt, rng):
    flat = np.clip(gt.reshape(-1, NC), 1e-10, None)
    flat /= flat.sum(axis=-1, keepdims=True)
    cumsum = np.cumsum(flat, axis=-1)
    u = rng.random(len(flat))
    return (u[:, None] < cumsum).argmax(axis=-1).reshape(gt.shape[:2])

t0 = time.time()
print("=== OBSERVATION ALLOCATION STRATEGY SWEEP ===\n")
rounds = load_all()
nR = len(rounds)
VP = 15

def grid_positions(dim):
    n = max(1, -(-dim // VP))
    if n == 1: return [0]
    step = (dim - VP) / (n - 1)
    return [round(i * step) for i in range(n)]

# Pre-train LOO models
print("[1] Pre-training LOO models...", flush=True)
cached = {}
for hold in range(nR):
    _, test_seeds = [(None, rs) for rs in [rounds[hold]]][0]
    train = [s for i, ss in enumerate(rounds) if i!=hold for s in ss]
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


def test_allocation(name, max_obs, seeds_to_use, n_mc=5):
    """Test a specific observation allocation strategy."""
    mc_scores = []
    for mc in range(n_mc):
        rng = np.random.default_rng(mc * 1000 + 42)
        fold_scores = []
        for hold in range(nR):
            c = cached[hold]
            grids, gts, preds = c["grids"], c["gts"], c["preds"]
            vps, H, W = c["vps"], c["H"], c["W"]
            n_seeds = len(grids)
            
            per_cls_obs = np.zeros((NC, NC)); per_cls_pred = np.zeros((NC, NC)); per_cls_n = np.zeros(NC)
            obs_used = 0
            
            for si in seeds_to_use(n_seeds):
                obs_grid = sample_obs(gts[si], rng)
                for row, col in vps:
                    if obs_used >= max_obs: break
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
                if obs_used >= max_obs: break
            
            # Apply calibration with optimal clip
            cal = []
            for si in range(n_seeds):
                p = preds[si].copy().reshape(-1, NC)
                cls = build_class_grid(grids[si]).ravel()
                for ic in range(NC):
                    if per_cls_n[ic] < 10: continue
                    of = per_cls_obs[ic]/per_cls_n[ic]; pa = per_cls_pred[ic]/per_cls_n[ic]
                    r = np.where(pa > 0.01, np.clip(of/pa, 0.01, 100.0), 1.0)
                    p[cls==ic] *= r
                p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
                cal.append(p.reshape(H, W, NC))
            
            fold_scores.append(np.mean([kl_score(gts[si], cal[si]) for si in range(n_seeds)]))
        mc_scores.append(np.mean(fold_scores))
    
    avg = np.mean(mc_scores); std = np.std(mc_scores)
    
    # Per-fold detail
    rng = np.random.default_rng(42)
    folds = []
    for hold in range(nR):
        c = cached[hold]
        grids, gts, preds = c["grids"], c["gts"], c["preds"]
        vps, H, W = c["vps"], c["H"], c["W"]
        per_cls_obs = np.zeros((NC, NC)); per_cls_pred = np.zeros((NC, NC)); per_cls_n = np.zeros(NC)
        obs_used = 0
        for si in seeds_to_use(len(grids)):
            obs_grid = sample_obs(gts[si], rng)
            for row, col in vps:
                if obs_used >= max_obs: break
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
            if obs_used >= max_obs: break
        cal = []
        for si in range(len(grids)):
            p = preds[si].copy().reshape(-1, NC)
            cls = build_class_grid(grids[si]).ravel()
            for ic in range(NC):
                if per_cls_n[ic] < 10: continue
                of = per_cls_obs[ic]/per_cls_n[ic]; pa = per_cls_pred[ic]/per_cls_n[ic]
                r = np.where(pa > 0.01, np.clip(of/pa, 0.01, 100.0), 1.0)
                p[cls==ic] *= r
            p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
            cal.append(p.reshape(H, W, NC))
        folds.append(np.mean([kl_score(gts[si], cal[si]) for si in range(len(grids))]))
    
    folds_str = " ".join(f"{f:.1f}" for f in folds)
    print(f"  {name:45s}: {avg:.2f} +/-{std:.2f} | {folds_str}", flush=True)
    return avg


print("[2] Testing observation allocations (clip [0.01, 100]):\n", flush=True)

# Current: 9*5 = 45 obs uniform across all seeds
test_allocation("CURRENT: 45 obs, 5 seeds (9 per seed)", 45, lambda n: range(n))

# Fewer seeds, more viewports per seed
test_allocation("45 obs, 4 seeds (skip seed 4)", 45, lambda n: range(min(4, n)))
test_allocation("45 obs, 3 seeds (skip seeds 3,4)", 45, lambda n: range(min(3, n)))
test_allocation("36 obs, 4 seeds (9 per seed)", 36, lambda n: range(min(4, n)))

# More observations using remaining budget
test_allocation("50 obs, 5 seeds (10 per seed)", 50, lambda n: range(n))

# Repeat seeds for multiple stochastic samples
def repeat_seeds(n):
    """Cycle through seeds multiple times for more stochastic samples."""
    result = []
    for cycle in range(3):  # up to 3 cycles
        for si in range(n):
            result.append(si)
    return result

test_allocation("45 obs, repeat seeds x3", 45, repeat_seeds)

# Only odd/even seeds
test_allocation("45 obs, odd seeds only", 45, lambda n: [i for i in range(n) if i % 2 == 1])
test_allocation("45 obs, even seeds only", 45, lambda n: [i for i in range(n) if i % 2 == 0])

print(f"\nTotal: {time.time()-t0:.0f}s")
