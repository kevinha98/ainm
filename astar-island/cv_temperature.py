"""Test temperature scaling on HGB predictions before calibration."""
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

def apply_temperature(preds_list, T):
    """Apply temperature scaling to probability predictions."""
    result = []
    for p in preds_list:
        H, W = p.shape[:2]
        flat = p.reshape(-1, NC)
        # Convert to logits, scale, convert back
        log_p = np.log(np.clip(flat, 1e-10, None))
        scaled = log_p / T
        # Softmax
        scaled -= scaled.max(axis=-1, keepdims=True)
        exp_p = np.exp(scaled)
        prob = exp_p / exp_p.sum(axis=-1, keepdims=True)
        prob = np.clip(prob, CLIP, None)
        prob /= prob.sum(axis=-1, keepdims=True)
        result.append(prob.reshape(H, W, NC))
    return result

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

n_trials = 20

# Test temperature without calibration first
print("=== Temperature scaling WITHOUT calibration ===\n", flush=True)
for T in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
    fold_scores = []
    for hold in range(nR):
        c = cached[hold]
        scaled = apply_temperature(c["preds"], T)
        fold_scores.append(np.mean([kl_score(c["gts"][si], scaled[si]) for si in range(len(c["gts"]))]))
    avg = np.mean(fold_scores)
    fstr = " ".join(f"R{i+1}={s:.2f}" for i, s in enumerate(fold_scores))
    print(f"  T={T:.1f}: {avg:.2f}  [{fstr}]", flush=True)

# Test temperature WITH settle-split calibration
print("\n=== Temperature scaling WITH settle-split calibration ===\n", flush=True)
for T in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
    fold_scores = []
    for hold in range(nR):
        c = cached[hold]; grids, gts = c["grids"], c["gts"]; H, W = c["H"], c["W"]
        scaled_preds = apply_temperature(c["preds"], T)
        vps = [(r,cc) for r in grid_positions(H) for cc in grid_positions(W)]
        trial_scores = []
        for trial in range(n_trials):
            rng = np.random.RandomState(trial*100+hold)
            cls_obs = np.zeros((NC,NC)); cls_pred = np.zeros((NC,NC)); cls_n = np.zeros(NC)
            settle_obs = {True: np.zeros((NC,NC)), False: np.zeros((NC,NC))}
            settle_pred = {True: np.zeros((NC,NC)), False: np.zeros((NC,NC))}
            settle_n = {True: np.zeros(NC), False: np.zeros(NC)}
            obs_used = 0
            for si in range(len(grids)):
                obs_grid = sample_obs(gts[si], rng)
                cls = build_class_grid(grids[si]); settlement = (cls == 1)
                dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W),20)
                ns_mask = dist_s <= SETTLE_DIST
                for row, col in vps:
                    if obs_used >= 45: break
                    for vy in range(min(VP, H-row)):
                        for vx in range(min(VP, W-col)):
                            gy, gx = row+vy, col+vx
                            if gy >= H or gx >= W: continue
                            ic = int(cls[gy,gx]); oc = obs_grid[gy,gx]; ns = bool(ns_mask[gy,gx])
                            pp = scaled_preds[si][gy,gx]
                            cls_obs[ic,oc]+=1; cls_pred[ic]+=pp; cls_n[ic]+=1
                            settle_obs[ns][ic,oc]+=1; settle_pred[ns][ic]+=pp; settle_n[ns][ic]+=1
                    obs_used += 1
                if obs_used >= 45: break
            cal = []
            for si in range(len(grids)):
                p = scaled_preds[si].copy().reshape(-1, NC)
                cls = build_class_grid(grids[si]); settlement = (cls==1)
                dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W),20)
                ns_flat = (dist_s.ravel() <= SETTLE_DIST); cls = cls.ravel()
                for ns in [True, False]:
                    for ic in range(NC):
                        n = settle_n[ns][ic]
                        if n < 10:
                            n = cls_n[ic]
                            if n < 10: continue
                            of=cls_obs[ic]/n; pa=cls_pred[ic]/n
                        else:
                            of=settle_obs[ns][ic]/n; pa=settle_pred[ns][ic]/n
                        r = np.where(pa > .01, np.clip(of/pa, .01, 100.), 1.)
                        mask = (cls==ic) & (ns_flat==ns)
                        p[mask] *= r
                p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
                cal.append(p.reshape(H, W, NC))
            trial_scores.append(np.mean([kl_score(gts[si], cal[si]) for si in range(len(gts))]))
        fold_scores.append(np.mean(trial_scores))
    avg = np.mean(fold_scores)
    fstr = " ".join(f"R{i+1}={s:.2f}" for i, s in enumerate(fold_scores))
    print(f"  T={T:.1f}: {avg:.2f}  [{fstr}]", flush=True)

print(f"\nTotal time: {time.time()-t0:.0f}s")
