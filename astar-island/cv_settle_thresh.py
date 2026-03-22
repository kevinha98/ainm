"""Test finer distance-to-settlement thresholds for calibration split."""
import json, numpy as np, time
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("data"); NC = 6; CLIP = 0.0001; VP = 15
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}

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

for thresh in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    fold_scores = []
    for hold in range(nR):
        c = cached[hold]; grids, gts, preds = c["grids"], c["gts"], c["preds"]
        H, W = c["H"], c["W"]
        vps = [(r,cc) for r in grid_positions(H) for cc in grid_positions(W)]
        tscore = []
        for trial in range(20):
            rng = np.random.RandomState(trial*100+hold)
            cls_obs = np.zeros((NC,NC)); cls_pred = np.zeros((NC,NC)); cls_n = np.zeros(NC)
            bk_obs = {}; bk_pred = {}; bk_n = {}
            obs_used = 0
            for si in range(len(grids)):
                obs_grid = sample_obs(gts[si], rng)
                cls = build_class_grid(grids[si]); settle = (cls == 1)
                dist_s = ndimage.distance_transform_edt(~settle) if settle.any() else np.full((H,W), 20)
                for row, col in vps:
                    if obs_used >= 45: break
                    for vy in range(min(VP, H-row)):
                        for vx in range(min(VP, W-col)):
                            gy, gx = row+vy, col+vx
                            if gy >= H or gx >= W: continue
                            ic = int(cls[gy,gx]); oc = obs_grid[gy, gx]; pp = preds[si][gy, gx]
                            cls_obs[ic,oc] += 1; cls_pred[ic] += pp; cls_n[ic] += 1
                            bk = dist_s[gy,gx] <= thresh
                            key = (ic, bk)
                            if key not in bk_obs: bk_obs[key]=np.zeros(NC); bk_pred[key]=np.zeros(NC); bk_n[key]=0
                            bk_obs[key][oc] += 1; bk_pred[key] += pp; bk_n[key] += 1
                    obs_used += 1
                if obs_used >= 45: break
            cr = {}
            for ic in range(NC):
                if cls_n[ic]>=10:
                    of=cls_obs[ic]/cls_n[ic]; pa=cls_pred[ic]/cls_n[ic]
                    cr[ic] = np.where(pa>.01, np.clip(of/pa, 0.01, 100.0), 1.0)
            br = {}
            for key in bk_obs:
                if bk_n[key]>=10:
                    of=bk_obs[key]/bk_n[key]; pa=bk_pred[key]/bk_n[key]
                    br[key] = np.where(pa>.01, np.clip(of/pa, 0.01, 100.0), 1.0)
            cal = []
            for si in range(len(preds)):
                p = preds[si].copy().reshape(-1, NC)
                cls = build_class_grid(grids[si]); settle = (cls==1)
                dist_s = ndimage.distance_transform_edt(~settle) if settle.any() else np.full((H,W),20)
                flat_cls = cls.ravel(); flat_near = dist_s.ravel() <= thresh
                for ic in range(NC):
                    for bk_val in [True, False]:
                        m = (flat_cls==ic) & (flat_near == bk_val)
                        if m.any():
                            key = (ic, bk_val)
                            if key in br: p[m] *= br[key]
                            elif ic in cr: p[m] *= cr[ic]
                p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
                cal.append(p.reshape(H,W,NC))
            tscore.append(np.mean([kl_score(gts[si], cal[si]) for si in range(len(gts))]))
        fold_scores.append(np.mean(tscore))
    avg = np.mean(fold_scores)
    fstr = " ".join(f"R{i+1}={s:.2f}" for i,s in enumerate(fold_scores))
    print(f"dist_settle<={thresh:.1f}: {avg:.2f}  [{fstr}]", flush=True)

print(f"\nDone in {time.time()-t0:.0f}s")
