"""Test label smoothing vs temperature scaling interaction."""
import json, numpy as np, time
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("data"); NC = 6; CLIP = 0.0001; VP = 15
GRID_TO_CLASS = {0:0,1:1,2:2,3:3,4:4,5:5,10:0,11:0}
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

def apply_temperature(flat, T):
    log_p = np.log(np.clip(flat, 1e-10, None))
    scaled = log_p / T
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp_p = np.exp(scaled)
    prob = exp_p / exp_p.sum(axis=-1, keepdims=True)
    return np.clip(prob, CLIP, None)

def run_calibrated(preds_list, grids, gts, n_trials=20):
    H, W = grids[0].shape; n_seeds = len(grids)
    vps = [(r,c) for r in grid_positions(H) for c in grid_positions(W)]
    trial_scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial*100)
        cls_obs = np.zeros((NC,NC)); cls_pred = np.zeros((NC,NC)); cls_n = np.zeros(NC)
        so = {True: np.zeros((NC,NC)), False: np.zeros((NC,NC))}
        sp = {True: np.zeros((NC,NC)), False: np.zeros((NC,NC))}
        sn = {True: np.zeros(NC), False: np.zeros(NC)}
        ou = 0
        for si in range(n_seeds):
            og = sample_obs(gts[si], rng)
            cls = build_class_grid(grids[si]); se = (cls == 1)
            ds = ndimage.distance_transform_edt(~se) if se.any() else np.full((H,W),20)
            nm = ds <= SETTLE_DIST
            for row, col in vps:
                if ou >= 45: break
                for vy in range(min(VP, H-row)):
                    for vx in range(min(VP, W-col)):
                        gy, gx = row+vy, col+vx
                        if gy >= H or gx >= W: continue
                        ic = int(cls[gy,gx]); oc = og[gy,gx]; ns = bool(nm[gy,gx])
                        pp = preds_list[si][gy,gx]
                        cls_obs[ic,oc]+=1; cls_pred[ic]+=pp; cls_n[ic]+=1
                        so[ns][ic,oc]+=1; sp[ns][ic]+=pp; sn[ns][ic]+=1
                ou += 1
            if ou >= 45: break
        cal = []
        for si in range(n_seeds):
            p = preds_list[si].copy().reshape(-1, NC)
            cls = build_class_grid(grids[si]); se = (cls==1)
            ds = ndimage.distance_transform_edt(~se) if se.any() else np.full((H,W),20)
            nf = (ds.ravel() <= SETTLE_DIST); cls = cls.ravel()
            for ns in [True, False]:
                for ic in range(NC):
                    n = sn[ns][ic]
                    if n < 10:
                        n = cls_n[ic]
                        if n < 10: continue
                        of_ = cls_obs[ic]/n; pa_ = cls_pred[ic]/n
                    else:
                        of_ = so[ns][ic]/n; pa_ = sp[ns][ic]/n
                    r = np.where(pa_ > .01, np.clip(of_/pa_, .01, 100.), 1.)
                    mask = (cls==ic) & (nf==ns)
                    p[mask] *= r
            p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
            cal.append(p.reshape(H, W, NC))
        trial_scores.append(np.mean([kl_score(gts[si], cal[si]) for si in range(n_seeds)]))
    return np.mean(trial_scores)

t0 = time.time()
rounds = []
for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
    with open(gf) as f: data = json.load(f)
    seeds = [(np.array(data[si]["initial_grid"]), np.array(data[si]["ground_truth"]))
             for si in sorted(data.keys()) if data[si].get("ground_truth")]
    rounds.append(seeds)
nR = len(rounds)

cached = {}
for hold in range(nR):
    train = [s for i, ss in enumerate(rounds) if i!=hold for s in ss]
    X = np.vstack([extract_features(ig) for ig, _ in train])
    Y = np.vstack([gt.reshape(-1, NC) for _, gt in train])
    models = [HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05,
              min_samples_leaf=50, random_state=42).fit(X, Y[:,c]) for c in range(NC)]
    H, W = rounds[hold][0][0].shape
    preds_raw = []
    for ig, _ in rounds[hold]:
        p = np.column_stack([m.predict(extract_features(ig)) for m in models])
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        preds_raw.append(p)  # flat
    cached[hold] = {"grids": [ig for ig,_ in rounds[hold]], "gts": [gt for _,gt in rounds[hold]],
                     "preds_raw": preds_raw, "H": H, "W": W}
print(f"Models trained in {time.time()-t0:.0f}s\n", flush=True)

# Test 1: Label smoothing (blend with uniform)
print("=== Label smoothing (blend with uniform) + calibration ===\n", flush=True)
for alpha in [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
    fold_scores = []
    for hold in range(nR):
        c = cached[hold]; H, W = c["H"], c["W"]
        preds = []
        for p_raw in c["preds_raw"]:
            p = (1 - alpha) * p_raw + alpha * (1.0 / NC)
            p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
            preds.append(p.reshape(H, W, NC))
        fold_scores.append(run_calibrated(preds, c["grids"], c["gts"]))
    avg = np.mean(fold_scores)
    fstr = " ".join(f"R{i+1}={s:.2f}" for i, s in enumerate(fold_scores))
    print(f"  alpha={alpha:.2f}: {avg:.2f}  [{fstr}]", flush=True)

# Test 2: Temperature + small label smoothing combo
print("\n=== T=1.15 + label smoothing combo ===\n", flush=True)
for alpha in [0.0, 0.005, 0.01, 0.02, 0.05]:
    fold_scores = []
    for hold in range(nR):
        c = cached[hold]; H, W = c["H"], c["W"]
        preds = []
        for p_raw in c["preds_raw"]:
            p = apply_temperature(p_raw, 1.15)
            p = (1 - alpha) * p + alpha * (1.0 / NC)
            p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
            preds.append(p.reshape(H, W, NC))
        fold_scores.append(run_calibrated(preds, c["grids"], c["gts"]))
    avg = np.mean(fold_scores)
    fstr = " ".join(f"R{i+1}={s:.2f}" for i, s in enumerate(fold_scores))
    print(f"  T=1.15 + alpha={alpha:.3f}: {avg:.2f}  [{fstr}]", flush=True)

# Test 3: Additive + multiplicative calibration combo
print("\n=== Additive calibration after multiplicative (with T=1.15) ===\n", flush=True)
for add_weight in [0.0, 0.1, 0.2, 0.3, 0.5]:
    fold_scores = []
    for hold in range(nR):
        c = cached[hold]; grids, gts = c["grids"], c["gts"]; H, W = c["H"], c["W"]
        preds = []
        for p_raw in c["preds_raw"]:
            p = apply_temperature(p_raw, 1.15)
            p /= p.sum(axis=-1, keepdims=True)
            preds.append(p.reshape(H, W, NC))
        
        vps = [(r,cc) for r in grid_positions(H) for cc in grid_positions(W)]
        n_seeds = len(grids)
        trial_scores = []
        for trial in range(20):
            rng = np.random.RandomState(trial*100+hold)
            cls_obs = np.zeros((NC,NC)); cls_pred = np.zeros((NC,NC)); cls_n = np.zeros(NC)
            so = {True: np.zeros((NC,NC)), False: np.zeros((NC,NC))}
            sp = {True: np.zeros((NC,NC)), False: np.zeros((NC,NC))}
            sn = {True: np.zeros(NC), False: np.zeros(NC)}
            ou = 0
            for si in range(n_seeds):
                og = sample_obs(gts[si], rng)
                cls = build_class_grid(grids[si]); se = (cls == 1)
                ds = ndimage.distance_transform_edt(~se) if se.any() else np.full((H,W),20)
                nm = ds <= SETTLE_DIST
                for row, col in vps:
                    if ou >= 45: break
                    for vy in range(min(VP, H-row)):
                        for vx in range(min(VP, W-col)):
                            gy, gx = row+vy, col+vx
                            if gy >= H or gx >= W: continue
                            ic = int(cls[gy,gx]); oc = og[gy,gx]; ns = bool(nm[gy,gx])
                            pp = preds[si][gy,gx]
                            cls_obs[ic,oc]+=1; cls_pred[ic]+=pp; cls_n[ic]+=1
                            so[ns][ic,oc]+=1; sp[ns][ic]+=pp; sn[ns][ic]+=1
                    ou += 1
                if ou >= 45: break
            cal = []
            for si in range(n_seeds):
                p = preds[si].copy().reshape(-1, NC)
                cls = build_class_grid(grids[si]); se = (cls==1)
                ds = ndimage.distance_transform_edt(~se) if se.any() else np.full((H,W),20)
                nf = (ds.ravel() <= SETTLE_DIST); cls = cls.ravel()
                for ns in [True, False]:
                    for ic in range(NC):
                        n = sn[ns][ic]
                        if n < 10:
                            n = cls_n[ic]
                            if n < 10: continue
                            of_ = cls_obs[ic]/n; pa_ = cls_pred[ic]/n
                        else:
                            of_ = so[ns][ic]/n; pa_ = sp[ns][ic]/n
                        # Multiplicative ratio
                        ratio_mult = np.where(pa_ > .01, np.clip(of_/pa_, .01, 100.), 1.)
                        # Additive diff
                        diff_add = of_ - pa_
                        mask = (cls==ic) & (nf==ns)
                        # Combined: multiply then add
                        p[mask] = p[mask] * ratio_mult * (1 - add_weight) + (p[mask] + diff_add) * add_weight
                p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
                cal.append(p.reshape(H, W, NC))
            trial_scores.append(np.mean([kl_score(gts[si], cal[si]) for si in range(n_seeds)]))
        fold_scores.append(np.mean(trial_scores))
    avg = np.mean(fold_scores)
    fstr = " ".join(f"R{i+1}={s:.2f}" for i, s in enumerate(fold_scores))
    print(f"  add_weight={add_weight:.1f}: {avg:.2f}  [{fstr}]", flush=True)

print(f"\nTotal time: {time.time()-t0:.0f}s")
