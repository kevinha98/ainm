"""Test two-pass calibration: per-class first, then settle-split on residual."""
import json, numpy as np, time
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("data"); NC = 6; CLIP = 0.0001; VP = 15
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}
SETTLE_DIST = 2.0

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
    features = np.concatenate([np.eye(NC)[cls],
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
    g = np.clip(gt.reshape(-1, NC), 1e-10, None); p = np.clip(pred.reshape(-1, NC), 1e-10, None)
    p /= p.sum(axis=-1, keepdims=True); g /= g.sum(axis=-1, keepdims=True)
    return 100 * np.exp(-np.mean(np.sum(g * np.log(g / p), axis=-1)))

def sample_obs(gt, rng):
    flat = np.clip(gt.reshape(-1, NC), 1e-10, None); flat /= flat.sum(axis=-1, keepdims=True)
    return (rng.random(len(flat))[:, None] < np.cumsum(flat, axis=-1)).argmax(axis=-1).reshape(gt.shape[:2])

def grid_positions(dim):
    n = max(1, -(-dim // VP)); step = (dim - VP) / max(n - 1, 1) if n > 1 else 0
    return [round(i * step) for i in range(n)]

t0 = time.time()
rounds = []
for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
    with open(gf) as f: data = json.load(f)
    seeds = [(np.array(data[si]["initial_grid"]), np.array(data[si]["ground_truth"]))
             for si in sorted(data.keys()) if data[si].get("ground_truth")]
    rounds.append(seeds)
nR = len(rounds)

print("Training LOO models...", flush=True)
cached = {}
for hold in range(nR):
    train = [s for i, ss in enumerate(rounds) if i!=hold for s in ss]
    X = np.vstack([extract_features(ig) for ig, _ in train])
    Y = np.vstack([gt.reshape(-1, NC) for _, gt in train])
    models = [HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05,
              min_samples_leaf=50, random_state=42).fit(X, Y[:,c]) for c in range(NC)]
    grids = [ig for ig,_ in rounds[hold]]; gts = [gt for _,gt in rounds[hold]]
    H, W = grids[0].shape
    preds = []
    for ig, _ in rounds[hold]:
        p = np.column_stack([m.predict(extract_features(ig)) for m in models])
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        preds.append(p.reshape(H, W, NC))
    cached[hold] = {"grids": grids, "gts": gts, "preds": preds, "H": H, "W": W}
print(f"Done in {time.time()-t0:.0f}s\n", flush=True)


def gather_obs(hold, rng, preds_to_use):
    """Gather observation data. preds_to_use can be the original or already-calibrated preds."""
    c = cached[hold]; grids, gts = c["grids"], c["gts"]; H, W = c["H"], c["W"]
    vps = [(r,cc) for r in grid_positions(H) for cc in grid_positions(W)]
    
    cls_obs = np.zeros((NC, NC)); cls_pred = np.zeros((NC, NC)); cls_n = np.zeros(NC)
    settle_obs = {True: np.zeros((NC, NC)), False: np.zeros((NC, NC))}
    settle_pred = {True: np.zeros((NC, NC)), False: np.zeros((NC, NC))}
    settle_n = {True: np.zeros(NC), False: np.zeros(NC)}
    
    obs_used = 0
    for si in range(len(grids)):
        obs_grid = sample_obs(gts[si], rng)
        cls = build_class_grid(grids[si])
        settlement = (cls == 1)
        dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
        near_settle = dist_s <= SETTLE_DIST
        
        for row, col in vps:
            if obs_used >= 45: break
            for vy in range(min(VP, H-row)):
                for vx in range(min(VP, W-col)):
                    gy, gx = row+vy, col+vx
                    if gy >= H or gx >= W: continue
                    ic = int(cls[gy,gx]); oc = obs_grid[gy, gx]
                    pp = preds_to_use[si][gy, gx]
                    ns = bool(near_settle[gy, gx])
                    cls_obs[ic, oc] += 1; cls_pred[ic] += pp; cls_n[ic] += 1
                    settle_obs[ns][ic, oc] += 1; settle_pred[ns][ic] += pp; settle_n[ns][ic] += 1
            obs_used += 1
        if obs_used >= 45: break
    
    return cls_obs, cls_pred, cls_n, settle_obs, settle_pred, settle_n


def apply_per_class(preds, grids, cls_obs, cls_pred, cls_n, clip_lo=0.01, clip_hi=100.0):
    H, W = grids[0].shape
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
        cal.append(p.reshape(H, W, NC))
    return cal


def apply_settle_split(preds, grids, settle_obs, settle_pred, settle_n,
                       cls_obs=None, cls_pred=None, cls_n=None, clip_lo=0.01, clip_hi=100.0):
    H, W = grids[0].shape
    cal = []
    for si in range(len(preds)):
        p = preds[si].copy().reshape(-1, NC)
        cls = build_class_grid(grids[si])
        settlement = (cls == 1)
        dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
        ns_flat = (dist_s.ravel() <= SETTLE_DIST)
        cls = cls.ravel()
        for ns in [True, False]:
            for ic in range(NC):
                n = settle_n[ns][ic]
                if n < 10:
                    if cls_n is not None and cls_n[ic] >= 10:
                        of = cls_obs[ic]/cls_n[ic]; pa = cls_pred[ic]/cls_n[ic]
                    else: continue
                else:
                    of = settle_obs[ns][ic] / n; pa = settle_pred[ns][ic] / n
                r = np.where(pa > 0.01, np.clip(of/pa, clip_lo, clip_hi), 1.0)
                mask = (cls == ic) & (ns_flat == ns)
                p[mask] *= r
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        cal.append(p.reshape(H, W, NC))
    return cal


n_trials = 20
strategies = {}

# Strategy 1: per-class only (baseline)
print("Strategy 1: Per-class only...", flush=True)
fold_scores = []
for hold in range(nR):
    c = cached[hold]
    trial_scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial*100+hold)
        co, cp, cn, so, sp, sn = gather_obs(hold, rng, c["preds"])
        cal = apply_per_class(c["preds"], c["grids"], co, cp, cn)
        trial_scores.append(np.mean([kl_score(c["gts"][si], cal[si]) for si in range(len(c["gts"]))]))
    fold_scores.append(np.mean(trial_scores))
strategies["per-class"] = np.mean(fold_scores)
print(f"  {strategies['per-class']:.2f}", flush=True)

# Strategy 2: settle-split only (current)
print("Strategy 2: Settle-split only...", flush=True)
fold_scores = []
for hold in range(nR):
    c = cached[hold]
    trial_scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial*100+hold)
        co, cp, cn, so, sp, sn = gather_obs(hold, rng, c["preds"])
        cal = apply_settle_split(c["preds"], c["grids"], so, sp, sn, co, cp, cn)
        trial_scores.append(np.mean([kl_score(c["gts"][si], cal[si]) for si in range(len(c["gts"]))]))
    fold_scores.append(np.mean(trial_scores))
strategies["settle-split"] = np.mean(fold_scores)
print(f"  {strategies['settle-split']:.2f}", flush=True)

# Strategy 3: Two-pass: per-class first, then settle-split
print("Strategy 3: Two-pass (per-class -> settle)...", flush=True)
fold_scores = []
for hold in range(nR):
    c = cached[hold]
    trial_scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial*100+hold)
        # Pass 1: per-class calibration
        co1, cp1, cn1, _, _, _ = gather_obs(hold, rng, c["preds"])
        cal1 = apply_per_class(c["preds"], c["grids"], co1, cp1, cn1)
        # Pass 2: gather NEW obs stats against calibrated preds, then settle-split
        rng2 = np.random.RandomState(trial*100+hold+50)  # different seed for pass 2
        co2, cp2, cn2, so2, sp2, sn2 = gather_obs(hold, rng2, cal1)
        cal2 = apply_settle_split(cal1, c["grids"], so2, sp2, sn2, co2, cp2, cn2)
        trial_scores.append(np.mean([kl_score(c["gts"][si], cal2[si]) for si in range(len(c["gts"]))]))
    fold_scores.append(np.mean(trial_scores))
strategies["two-pass(cls->settle)"] = np.mean(fold_scores)
print(f"  {strategies['two-pass(cls->settle)']:.2f}", flush=True)

# Strategy 4: Two-pass: settle first, then per-class  
print("Strategy 4: Two-pass (settle -> per-class)...", flush=True)
fold_scores = []
for hold in range(nR):
    c = cached[hold]
    trial_scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial*100+hold)
        co1, cp1, cn1, so1, sp1, sn1 = gather_obs(hold, rng, c["preds"])
        cal1 = apply_settle_split(c["preds"], c["grids"], so1, sp1, sn1, co1, cp1, cn1)
        rng2 = np.random.RandomState(trial*100+hold+50)
        co2, cp2, cn2, _, _, _ = gather_obs(hold, rng2, cal1)
        cal2 = apply_per_class(cal1, c["grids"], co2, cp2, cn2)
        trial_scores.append(np.mean([kl_score(c["gts"][si], cal2[si]) for si in range(len(c["gts"]))]))
    fold_scores.append(np.mean(trial_scores))
strategies["two-pass(settle->cls)"] = np.mean(fold_scores)
print(f"  {strategies['two-pass(settle->cls)']:.2f}", flush=True)

# Strategy 5: Iterative settle-split (apply twice)
print("Strategy 5: Iterative settle-split (2x)...", flush=True)
fold_scores = []
for hold in range(nR):
    c = cached[hold]
    trial_scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial*100+hold)
        co, cp, cn, so, sp, sn = gather_obs(hold, rng, c["preds"])
        cal1 = apply_settle_split(c["preds"], c["grids"], so, sp, sn, co, cp, cn)
        # Same obs data, re-apply with calibrated predictions as input
        rng2 = np.random.RandomState(trial*100+hold+50)
        co2, cp2, cn2, so2, sp2, sn2 = gather_obs(hold, rng2, cal1)
        cal2 = apply_settle_split(cal1, c["grids"], so2, sp2, sn2, co2, cp2, cn2)
        trial_scores.append(np.mean([kl_score(c["gts"][si], cal2[si]) for si in range(len(c["gts"]))]))
    fold_scores.append(np.mean(trial_scores))
strategies["iterative-settle(2x)"] = np.mean(fold_scores)
print(f"  {strategies['iterative-settle(2x)']:.2f}", flush=True)

print(f"\n{'Strategy':35s} {'Score':>8s} {'Delta':>8s}")
base = strategies["settle-split"]
for name, score in sorted(strategies.items(), key=lambda x: -x[1]):
    delta = score - base
    print(f"{name:35s} {score:8.2f} {'+' if delta>=0 else ''}{delta:.2f}")
print(f"\nTotal time: {time.time()-t0:.0f}s")
