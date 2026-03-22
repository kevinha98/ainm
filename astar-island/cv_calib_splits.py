"""Test more calibration split strategies beyond coastal.
Hypothesis: other spatial splits may capture additional dynamics."""
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
print("=== CALIBRATION SPLIT SWEEP ===\n")
rounds = load_all()
nR = len(rounds)

print(f"Loaded {nR} rounds\n")
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

def compute_split_masks(ig, split_type):
    """Return dict of mask_label -> bool mask (H,W) for given split."""
    cls = build_class_grid(ig)
    H, W = ig.shape
    ocean = (ig == 10)
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    
    if split_type == "coastal":
        return {True: is_coast, False: ~is_coast}
    
    elif split_type == "dist_ocean_2":
        # Near ocean (dist <= 3) vs far from ocean
        dist = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
        near = dist <= 3
        return {True: near, False: ~near}
    
    elif split_type == "dist_ocean_3":
        # Three bands: ocean-adjacent (<=1), near (2-5), far (>5)
        dist = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
        return {"adj": dist <= 1.5, "near": (dist > 1.5) & (dist <= 5), "far": dist > 5}
    
    elif split_type == "near_forest":
        # Near forest (in 3x3 neighborhood) vs not
        forest = (cls == 4)
        near = ndimage.maximum_filter(forest.astype(float), size=3) > 0
        return {True: near, False: ~near}
    
    elif split_type == "near_settle":
        # Near settlement (in 5x5) vs not
        settle = (cls == 1)
        near = ndimage.maximum_filter(settle.astype(float), size=5) > 0
        return {True: near, False: ~near}
    
    elif split_type == "forest_density":
        # High vs low forest density in 7x7
        forest = (cls == 4)
        k7 = np.ones((7,7))
        fd = ndimage.convolve(forest.astype(float), k7, mode='constant') / 49.0
        return {"high": fd > 0.3, "low": fd <= 0.3}
    
    elif split_type == "edge":
        # Biome edge (multiple class types in 3x3) vs interior
        unique_count = np.zeros((H, W))
        for c in range(NC):
            has = ndimage.maximum_filter((cls == c).astype(float), size=3) > 0
            unique_count += has
        edge = unique_count >= 2
        return {True: edge, False: ~edge}
    
    elif split_type == "predicted_confident":
        # Can't use here — needs prediction. Skip.
        return None
    
    elif split_type == "coastal_x_forest":
        # 2D split: coastal × near-forest
        forest = (cls == 4)
        near_f = ndimage.maximum_filter(forest.astype(float), size=5) > 0
        return {
            "coast_forest": is_coast & near_f,
            "coast_noforest": is_coast & ~near_f,
            "inland_forest": ~is_coast & near_f,
            "inland_noforest": ~is_coast & ~near_f,
        }
    
    return None


def run_sweep(split_type, n_trials=20, clip_lo=0.01, clip_hi=100.0):
    """Run LOO CV with a given split type."""
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
            
            # Collect observation stats per (class, split_bucket)
            # Also collect per-class-only stats as fallback
            per_cls_obs = np.zeros((NC, NC))
            per_cls_pred = np.zeros((NC, NC))
            per_cls_n = np.zeros(NC)
            
            bucket_obs = {}; bucket_pred = {}; bucket_n = {}
            
            obs_used = 0
            for si in range(len(grids)):
                obs_grid = sample_obs(gts[si], rng)
                cls = build_class_grid(grids[si]).ravel()
                split_masks = compute_split_masks(grids[si], split_type)
                if split_masks is None:
                    return None
                
                for row, col in vps:
                    if obs_used >= 45: break
                    for vy in range(min(VP, H-row)):
                        for vx in range(min(VP, W-col)):
                            gy, gx = row+vy, col+vx
                            if gy >= H or gx >= W: continue
                            ic = int(cls[gy*W + gx])
                            oc = obs_grid[gy, gx]
                            pp = preds[si][gy, gx]
                            
                            per_cls_obs[ic, oc] += 1
                            per_cls_pred[ic] += pp
                            per_cls_n[ic] += 1
                            
                            for bk, mask in split_masks.items():
                                if mask[gy, gx]:
                                    key = (ic, bk)
                                    if key not in bucket_obs:
                                        bucket_obs[key] = np.zeros(NC)
                                        bucket_pred[key] = np.zeros(NC)
                                        bucket_n[key] = 0
                                    bucket_obs[key][oc] += 1
                                    bucket_pred[key] += pp
                                    bucket_n[key] += 1
                                    break  # each cell belongs to exactly one bucket
                    obs_used += 1
                if obs_used >= 45: break
            
            # Compute per-class fallback ratios
            cls_ratio = {}
            for ic in range(NC):
                if per_cls_n[ic] >= 10:
                    of = per_cls_obs[ic] / per_cls_n[ic]
                    pa = per_cls_pred[ic] / per_cls_n[ic]
                    cls_ratio[ic] = np.where(pa > 0.01, np.clip(of/pa, clip_lo, clip_hi), 1.0)
            
            # Compute bucket ratios
            bkt_ratio = {}
            for key in bucket_obs:
                ic, bk = key
                if bucket_n[key] >= 10:
                    of = bucket_obs[key] / bucket_n[key]
                    pa = bucket_pred[key] / bucket_n[key]
                    bkt_ratio[key] = np.where(pa > 0.01, np.clip(of/pa, clip_lo, clip_hi), 1.0)
            
            # Apply calibration
            cal_preds = []
            for si in range(len(preds)):
                p = preds[si].copy().reshape(-1, NC)
                cls = build_class_grid(grids[si]).ravel()
                split_masks = compute_split_masks(grids[si], split_type)
                
                for ic in range(NC):
                    ic_mask = cls == ic
                    for bk, mask in split_masks.items():
                        key = (ic, bk)
                        m = ic_mask & mask.ravel()
                        if m.any():
                            if key in bkt_ratio:
                                p[m] *= bkt_ratio[key]
                            elif ic in cls_ratio:
                                p[m] *= cls_ratio[ic]
                
                p = np.clip(p, CLIP, None); p/= p.sum(axis=-1, keepdims=True)
                cal_preds.append(p.reshape(H, W, NC))
            
            fold_score = np.mean([kl_score(gts[si], cal_preds[si]) for si in range(len(gts))])
            trial_scores.append(fold_score)
        
        scores.append(np.mean(trial_scores))
    
    return np.mean(scores)


# Run the sweep
splits_to_test = [
    "coastal",           # baseline: coastal vs inland 
    "dist_ocean_2",      # near ocean (<=3) vs far
    "dist_ocean_3",      # 3-band ocean distance
    "near_forest",       # adjacent to forest vs not
    "near_settle",       # near settlement vs not
    "forest_density",    # high vs low forest density
    "edge",              # biome edge vs interior
    "coastal_x_forest",  # 2D: coastal × near-forest (4 buckets)
]

# First run per-class only baseline
print("[2] Running per-class-only baseline...", flush=True)
base_scores = []
for hold in range(nR):
    c = cached[hold]
    grids, gts, preds = c["grids"], c["gts"], c["preds"]
    H, W = c["H"], c["W"]
    rows = grid_positions(H); cols = grid_positions(W)
    vps = [(r,cc) for r in rows for cc in cols]
    trial_scores = []
    for trial in range(20):
        rng = np.random.RandomState(trial * 100 + hold)
        cls_obs = np.zeros((NC, NC)); cls_pred = np.zeros((NC, NC)); cls_n = np.zeros(NC)
        obs_used = 0
        for si in range(len(grids)):
            obs_grid = sample_obs(gts[si], rng)
            cls = build_class_grid(grids[si]).ravel()
            for row, col in vps:
                if obs_used >= 45: break
                for vy in range(min(VP, H-row)):
                    for vx in range(min(VP, W-col)):
                        gy, gx = row+vy, col+vx
                        if gy >= H or gx >= W: continue
                        ic = int(cls[gy*W + gx])
                        oc = obs_grid[gy, gx]
                        cls_obs[ic, oc] += 1; cls_pred[ic] += preds[si][gy, gx]; cls_n[ic] += 1
                obs_used += 1
            if obs_used >= 45: break
        cal = []
        for si in range(len(preds)):
            p = preds[si].copy().reshape(-1, NC)
            cls = build_class_grid(grids[si]).ravel()
            for ic in range(NC):
                if cls_n[ic] < 10: continue
                of = cls_obs[ic]/cls_n[ic]; pa = cls_pred[ic]/cls_n[ic]
                r = np.where(pa > 0.01, np.clip(of/pa, 0.01, 100.0), 1.0)
                p[cls==ic] *= r
            p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
            cal.append(p.reshape(H, W, NC))
        trial_scores.append(np.mean([kl_score(gts[si], cal[si]) for si in range(len(gts))]))
    base_scores.append(np.mean(trial_scores))
baseline = np.mean(base_scores)
print(f"  per-class only:  {baseline:.2f}\n", flush=True)

print("[3] Testing split strategies...\n", flush=True)
results = {}
for split in splits_to_test:
    t1 = time.time()
    score = run_sweep(split)
    dt = time.time() - t1
    if score is not None:
        delta = score - baseline
        results[split] = score
        print(f"  {split:25s}: {score:.2f}  ({'+' if delta>=0 else ''}{delta:.2f})  [{dt:.0f}s]", flush=True)
    else:
        print(f"  {split:25s}: SKIPPED", flush=True)

print(f"\n=== SUMMARY ===")
print(f"{'Strategy':30s} {'Score':>8s} {'Delta':>8s}")
print(f"{'per-class only':30s} {baseline:8.2f} {'baseline':>8s}")
for split, score in sorted(results.items(), key=lambda x: -x[1]):
    delta = score - baseline
    print(f"{split:30s} {score:8.2f} {'+' if delta>=0 else ''}{delta:.2f}")
print(f"\nTotal time: {time.time()-t0:.0f}s")
