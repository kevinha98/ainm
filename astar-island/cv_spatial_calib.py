"""Test spatial + class calibration: split obs by (class, region quadrant).
Hypothesis: different parts of the map may have different dynamics."""
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
print("=== SPATIAL CALIBRATION SWEEP ===\n")
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
    test_seeds = rounds[hold]
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


def gather_obs(hold, rng, max_obs=45):
    c = cached[hold]
    grids, gts, preds = c["grids"], c["gts"], c["preds"]
    vps, H, W = c["vps"], c["H"], c["W"]
    
    # Per-class overall
    per_cls_obs = np.zeros((NC, NC))
    per_cls_pred = np.zeros((NC, NC))
    per_cls_n = np.zeros(NC)
    
    # Per-class + coastal (split by is_coast)
    coast_obs = {True: np.zeros((NC, NC)), False: np.zeros((NC, NC))}
    coast_pred = {True: np.zeros((NC, NC)), False: np.zeros((NC, NC))}
    coast_n = {True: np.zeros(NC), False: np.zeros(NC)}
    
    # Per-class + quadrant
    quad_obs = {}; quad_pred = {}; quad_n = {}
    for q in range(4):
        quad_obs[q] = np.zeros((NC, NC)); quad_pred[q] = np.zeros((NC, NC)); quad_n[q] = np.zeros(NC)
    
    obs_used = 0
    for si in range(len(grids)):
        obs_grid = sample_obs(gts[si], rng)
        cls = build_class_grid(grids[si])
        ocean = (grids[si] == 10)
        is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
        
        for row, col in vps:
            if obs_used >= max_obs: break
            for vy in range(min(VP, H-row)):
                for vx in range(min(VP, W-col)):
                    gy, gx = row+vy, col+vx
                    if gy >= H or gx >= W: continue
                    ic = cls[gy, gx]; oc = obs_grid[gy, gx]
                    
                    per_cls_obs[ic, oc] += 1
                    per_cls_pred[ic] += preds[si][gy, gx]
                    per_cls_n[ic] += 1
                    
                    cst = bool(is_coast[gy, gx])
                    coast_obs[cst][ic, oc] += 1
                    coast_pred[cst][ic] += preds[si][gy, gx]
                    coast_n[cst][ic] += 1
                    
                    q = (0 if gy < H//2 else 2) + (0 if gx < W//2 else 1)
                    quad_obs[q][ic, oc] += 1
                    quad_pred[q][ic] += preds[si][gy, gx]
                    quad_n[q][ic] += 1
            obs_used += 1
        if obs_used >= max_obs: break
    
    return {
        "per_cls": (per_cls_obs, per_cls_pred, per_cls_n),
        "coast": (coast_obs, coast_pred, coast_n),
        "quad": (quad_obs, quad_pred, quad_n),
    }


def apply_per_class(preds, grids, cls_obs, cls_pred, cls_n, clip_lo=0.01, clip_hi=100.0):
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


def apply_coastal(preds, grids, coast_obs, coast_pred, coast_n, clip_lo=0.01, clip_hi=100.0):
    cal = []
    for si in range(len(preds)):
        H, W = grids[si].shape
        p = preds[si].copy().reshape(-1, NC)
        cls = build_class_grid(grids[si]).ravel()
        ocean = (grids[si] == 10)
        is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
        is_coast_flat = is_coast.ravel()
        
        for cst in [True, False]:
            for ic in range(NC):
                n = coast_n[cst][ic]
                if n < 10: continue
                of = coast_obs[cst][ic]/n; pa = coast_pred[cst][ic]/n
                r = np.where(pa > 0.01, np.clip(of/pa, clip_lo, clip_hi), 1.0)
                mask = (cls == ic) & (is_coast_flat == cst)
                p[mask] *= r
        
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        cal.append(p.reshape(H, W, NC))
    return cal


def apply_quadrant(preds, grids, quad_obs, quad_pred, quad_n, clip_lo=0.01, clip_hi=100.0):
    cal = []
    for si in range(len(preds)):
        H, W = grids[si].shape
        p = preds[si].copy().reshape(-1, NC)
        cls = build_class_grid(grids[si]).ravel()
        gy_flat = np.repeat(np.arange(H), W)
        gx_flat = np.tile(np.arange(W), H)
        q_flat = (np.where(gy_flat < H//2, 0, 2) + np.where(gx_flat < W//2, 0, 1))
        
        for q in range(4):
            for ic in range(NC):
                n = quad_n[q][ic]
                if n < 10: continue
                of = quad_obs[q][ic]/n; pa = quad_pred[q][ic]/n
                r = np.where(pa > 0.01, np.clip(of/pa, clip_lo, clip_hi), 1.0)
                mask = (cls == ic) & (q_flat == q)
                p[mask] *= r
        
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        cal.append(p.reshape(H, W, NC))
    return cal


def test_strategy(name, calib_fn, n_mc=5):
    mc_scores = []
    for mc in range(n_mc):
        rng = np.random.default_rng(mc * 1000 + 42)
        fold_scores = []
        for hold in range(nR):
            c = cached[hold]
            obs = gather_obs(hold, rng)
            cal = calib_fn(c["preds"], c["grids"], obs)
            fold_scores.append(np.mean([kl_score(c["gts"][si], cal[si]) for si in range(len(c["gts"]))]))
        mc_scores.append(np.mean(fold_scores))
    
    avg = np.mean(mc_scores); std = np.std(mc_scores)
    
    rng = np.random.default_rng(42)
    folds = []
    for hold in range(nR):
        c = cached[hold]
        obs = gather_obs(hold, rng)
        cal = calib_fn(c["preds"], c["grids"], obs)
        folds.append(np.mean([kl_score(c["gts"][si], cal[si]) for si in range(len(c["gts"]))]))
    
    folds_str = " ".join(f"{f:.1f}" for f in folds)
    print(f"  {name:40s}: {avg:.2f} +/-{std:.2f} | {folds_str}", flush=True)
    return avg


print("[2] Testing spatial calibration strategies:\n", flush=True)

# Baseline: per-class multiplicative
test_strategy("per-class (baseline)", 
    lambda p, g, obs: apply_per_class(p, g, *obs["per_cls"]))

# Split by coastal vs inland
test_strategy("per-class + coastal split",
    lambda p, g, obs: apply_coastal(p, g, *obs["coast"]))

# Split by quadrant
test_strategy("per-class + quadrant split",
    lambda p, g, obs: apply_quadrant(p, g, *obs["quad"]))

# Use both: coastal first, then class
def apply_hybrid_coast_class(preds, grids, obs):
    # First pass: coastal calibration
    cal1 = apply_coastal(preds, grids, *obs["coast"])
    # Then collect per-class stats on calibrated predictions... no, that's double-counting
    # Better: just use the finer-grained coastal calibration
    return cal1

test_strategy("coastal only",
    lambda p, g, obs: apply_hybrid_coast_class(p, g, obs))

print(f"\nTotal: {time.time()-t0:.0f}s")
