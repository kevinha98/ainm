"""Deep-dive on near_settle calibration split: test different sizes and combinations."""
import json, numpy as np, time
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("data")
NC = 6
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}
CLIP = 0.0001
VP = 15

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

def grid_positions(dim):
    n = max(1, -(-dim // VP))
    if n == 1: return [0]
    step = (dim - VP) / (n - 1)
    return [round(i * step) for i in range(n)]

t0 = time.time()
print("=== NEAR_SETTLE DEEP DIVE ===\n")
rounds = load_all()
nR = len(rounds)

print("[1] Pre-training LOO models...", flush=True)
cached = {}
for hold in range(nR):
    train = [s for i, ss in enumerate(rounds) if i!=hold for s in ss]
    X = np.vstack([extract_features(ig) for ig, _ in train])
    Y = np.vstack([gt.reshape(-1, NC) for _, gt in train])
    models = [HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05,
              min_samples_leaf=50, random_state=42).fit(X, Y[:,c]) for c in range(NC)]
    test_seeds = rounds[hold]
    grids = [ig for ig,_ in test_seeds]; gts = [gt for _,gt in test_seeds]
    H, W = grids[0].shape
    preds = []
    for ig, gt in test_seeds:
        p = np.column_stack([m.predict(extract_features(ig)) for m in models])
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        preds.append(p.reshape(H, W, NC))
    cached[hold] = {"grids": grids, "gts": gts, "preds": preds, "H": H, "W": W}
print(f"  Done in {time.time()-t0:.0f}s\n", flush=True)


def run_calibration(split_fn, n_trials=20, clip_lo=0.01, clip_hi=100.0, min_bucket=10):
    """Generic calibration with arbitrary split function.
    split_fn(ig) -> dict of label -> mask(H,W)"""
    scores = []
    for hold in range(nR):
        c = cached[hold]
        grids, gts, preds = c["grids"], c["gts"], c["preds"]
        H, W = c["H"], c["W"]
        rows = grid_positions(H); cols = grid_positions(W)
        vps = [(r,cc) for r in rows for cc in cols]
        
        trial_scores = []
        for trial in range(n_trials):
            rng = np.random.RandomState(trial * 100 + hold)
            
            per_cls_obs = np.zeros((NC, NC))
            per_cls_pred = np.zeros((NC, NC))
            per_cls_n = np.zeros(NC)
            bucket_obs = {}; bucket_pred = {}; bucket_n = {}
            
            obs_used = 0
            for si in range(len(grids)):
                obs_grid = sample_obs(gts[si], rng)
                cls = build_class_grid(grids[si]).ravel()
                masks = split_fn(grids[si])
                
                for row, col in vps:
                    if obs_used >= 45: break
                    for vy in range(min(VP, H-row)):
                        for vx in range(min(VP, W-col)):
                            gy, gx = row+vy, col+vx
                            if gy >= H or gx >= W: continue
                            ic = int(cls[gy*W + gx]); oc = obs_grid[gy, gx]
                            pp = preds[si][gy, gx]
                            per_cls_obs[ic, oc] += 1; per_cls_pred[ic] += pp; per_cls_n[ic] += 1
                            for bk, mask in masks.items():
                                if mask[gy, gx]:
                                    key = (ic, bk)
                                    if key not in bucket_obs:
                                        bucket_obs[key] = np.zeros(NC)
                                        bucket_pred[key] = np.zeros(NC)
                                        bucket_n[key] = 0
                                    bucket_obs[key][oc] += 1; bucket_pred[key] += pp; bucket_n[key] += 1
                                    break
                    obs_used += 1
                if obs_used >= 45: break
            
            cls_ratio = {}
            for ic in range(NC):
                if per_cls_n[ic] >= min_bucket:
                    of = per_cls_obs[ic]/per_cls_n[ic]; pa = per_cls_pred[ic]/per_cls_n[ic]
                    cls_ratio[ic] = np.where(pa > 0.01, np.clip(of/pa, clip_lo, clip_hi), 1.0)
            
            bkt_ratio = {}
            for key in bucket_obs:
                if bucket_n[key] >= min_bucket:
                    of = bucket_obs[key]/bucket_n[key]; pa = bucket_pred[key]/bucket_n[key]
                    bkt_ratio[key] = np.where(pa > 0.01, np.clip(of/pa, clip_lo, clip_hi), 1.0)
            
            cal_preds = []
            for si in range(len(preds)):
                p = preds[si].copy().reshape(-1, NC)
                cls = build_class_grid(grids[si]).ravel()
                masks = split_fn(grids[si])
                for ic in range(NC):
                    ic_mask = cls == ic
                    for bk, mask in masks.items():
                        key = (ic, bk)
                        m = ic_mask & mask.ravel()
                        if m.any():
                            if key in bkt_ratio: p[m] *= bkt_ratio[key]
                            elif ic in cls_ratio: p[m] *= cls_ratio[ic]
                p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
                cal_preds.append(p.reshape(H, W, NC))
            
            trial_scores.append(np.mean([kl_score(gts[si], cal_preds[si]) for si in range(len(gts))]))
        scores.append(np.mean(trial_scores))
    return np.mean(scores), scores


# ═══════════════════════════════════════════
# Test 1: near_settle with different kernel sizes
# ═══════════════════════════════════════════
print("[2] Testing near_settle kernel sizes...\n", flush=True)
for sz in [3, 5, 7, 9, 11]:
    def make_split(ig, _sz=sz):
        cls = build_class_grid(ig)
        settle = (cls == 1)
        near = ndimage.maximum_filter(settle.astype(float), size=_sz) > 0
        return {True: near, False: ~near}
    
    t1 = time.time()
    score, per_fold = run_calibration(make_split)
    dt = time.time() - t1
    fold_str = " ".join(f"R{i+1}={s:.2f}" for i, s in enumerate(per_fold))
    print(f"  near_settle size={sz:2d}: {score:.2f}  [{fold_str}]  [{dt:.0f}s]", flush=True)

# ═══════════════════════════════════════════
# Test 2: distance-band settlement split
# ═══════════════════════════════════════════
print("\n[3] Testing distance-band settlement splits...\n", flush=True)
for threshold in [2, 3, 5, 8]:
    def make_split(ig, _t=threshold):
        cls = build_class_grid(ig)
        settle = (cls == 1)
        dist = ndimage.distance_transform_edt(~settle) if settle.any() else np.full(ig.shape, 20)
        return {"near": dist <= _t, "far": dist > _t}
    
    t1 = time.time()
    score, per_fold = run_calibration(make_split)
    dt = time.time() - t1
    fold_str = " ".join(f"R{i+1}={s:.2f}" for i, s in enumerate(per_fold))
    print(f"  dist_settle<={threshold}: {score:.2f}  [{fold_str}]  [{dt:.0f}s]", flush=True)

# ═══════════════════════════════════════════
# Test 3: near_settle + coastal combination
# ═══════════════════════════════════════════
print("\n[4] Testing combinations...\n", flush=True)

# near_settle + coastal (4 buckets)
def split_settle_coast(ig, settle_sz=5):
    cls = build_class_grid(ig)
    H, W = ig.shape
    settle = (cls == 1)
    near_s = ndimage.maximum_filter(settle.astype(float), size=settle_sz) > 0
    ocean = (ig == 10)
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    return {
        "ns_c": near_s & is_coast,
        "ns_i": near_s & ~is_coast,
        "fs_c": ~near_s & is_coast,
        "fs_i": ~near_s & ~is_coast,
    }

t1 = time.time()
score, per_fold = run_calibration(split_settle_coast)
fold_str = " ".join(f"R{i+1}={s:.2f}" for i, s in enumerate(per_fold))
print(f"  settle5 + coastal: {score:.2f}  [{fold_str}]  [{time.time()-t1:.0f}s]", flush=True)

# near_settle + dist_ocean (4 buckets)
def split_settle_ocean(ig, settle_sz=5, ocean_thresh=3):
    cls = build_class_grid(ig)
    settle = (cls == 1)
    near_s = ndimage.maximum_filter(settle.astype(float), size=settle_sz) > 0
    ocean = (ig == 10)
    dist_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full(ig.shape, 20)
    near_o = dist_o <= ocean_thresh
    return {
        "ns_no": near_s & near_o,
        "ns_fo": near_s & ~near_o,
        "fs_no": ~near_s & near_o,
        "fs_fo": ~near_s & ~near_o,
    }

t1 = time.time()
score, per_fold = run_calibration(split_settle_ocean)
fold_str = " ".join(f"R{i+1}={s:.2f}" for i, s in enumerate(per_fold))
print(f"  settle5 + ocean3: {score:.2f}  [{fold_str}]  [{time.time()-t1:.0f}s]", flush=True)

# 3-way settle split: near_settle, near_coast, far
def split_three_way(ig, settle_sz=5):
    cls = build_class_grid(ig)
    H, W = ig.shape
    settle = (cls == 1)
    near_s = ndimage.maximum_filter(settle.astype(float), size=settle_sz) > 0
    ocean = (ig == 10)
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    return {
        "settle": near_s,
        "coast": ~near_s & is_coast,
        "inland": ~near_s & ~is_coast,
    }

t1 = time.time()
score, per_fold = run_calibration(split_three_way)
fold_str = " ".join(f"R{i+1}={s:.2f}" for i, s in enumerate(per_fold))
print(f"  3-way (settle/coast/inland): {score:.2f}  [{fold_str}]  [{time.time()-t1:.0f}s]", flush=True)

# ═══════════════════════════════════════════
# Test 4: min_bucket sensitivity with near_settle
# ═══════════════════════════════════════════
print("\n[5] Testing min_bucket sizes...\n", flush=True)

def make_ns5(ig):
    cls = build_class_grid(ig)
    settle = (cls == 1)
    near = ndimage.maximum_filter(settle.astype(float), size=5) > 0
    return {True: near, False: ~near}

for mb in [5, 10, 20, 50, 100]:
    t1 = time.time()
    score, _ = run_calibration(make_ns5, min_bucket=mb)
    print(f"  min_bucket={mb:3d}: {score:.2f}  [{time.time()-t1:.0f}s]", flush=True)

print(f"\nTotal time: {time.time()-t0:.0f}s")
