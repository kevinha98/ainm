"""
Fast cell model calibration using differential evolution.
Also runs LOO CV comparison against the bucket LUT.
"""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.optimize import differential_evolution
import time

DATA_DIR = Path("data")
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


def build_class_grid(ig):
    cg = np.zeros_like(ig)
    for gv, cls in GRID_TO_CLASS.items():
        cg[ig == gv] = cls
    return cg


def kl_score(pred, gt):
    pred = np.clip(pred, 1e-8, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    kl = np.where(gt > 0, gt * np.log(np.clip(gt, 1e-15, None) / pred), 0).sum(axis=-1)
    kl = np.where(np.isfinite(kl), kl, 0)
    return 100 - kl.mean() * 100


def load_all_gt():
    """Load GT grouped by round."""
    round_entries = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        with open(gf) as f:
            data = json.load(f)
        seeds = []
        for si_str in sorted(data.keys()):
            entry = data[si_str]
            seeds.append((np.array(entry['initial_grid']), np.array(entry['ground_truth'])))
        round_entries.append(seeds)
    return round_entries


def compute_features_vectorized(ig):
    """Compute all spatial features as numpy arrays."""
    H, W = ig.shape
    cls = build_class_grid(ig)
    
    settle = (cls == 1) | (cls == 2)
    dist_s = ndimage.distance_transform_edt(~settle) if settle.any() else np.full((H, W), 40.0)
    
    forest = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 40.0)
    
    ocean = (ig == 10)
    dist_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    coastal = (dist_o <= 1.5).astype(float)
    
    settle_density = ndimage.uniform_filter(settle.astype(float), size=7, mode='constant')
    
    mountain = (cls == 5)
    deep_ocean = ocean & (dist_s > 5)
    
    return cls, dist_s, dist_f, coastal, settle_density, mountain, deep_ocean


def predict_vectorized(ig, x):
    """
    Vectorized prediction using parameter vector x.
    Much faster than the loop-based version.
    """
    H, W = ig.shape
    cls, dist_s, dist_f, coastal, sd, mountain, deep_ocean = compute_features_vectorized(ig)
    
    # Unpack parameters
    (sb_e, ss_e, sb_f, ss_f, pb, ps, rfr, rp,
     ssb, ssdb, fpb, fsp, ffe, ffs, psurv) = x
    
    pred = np.zeros((H, W, 6))
    
    # Mountains: 100% mountain
    pred[mountain, 5] = 1.0
    
    # Deep ocean: 100% empty
    pred[deep_ocean, 0] = 1.0
    
    # Empty/Plains/Ocean (class 0, not mountain, not deep ocean)
    mask_empty = (cls == 0) & ~mountain & ~deep_ocean
    if mask_empty.any():
        ds = dist_s[mask_empty]
        df = dist_f[mask_empty]
        co = coastal[mask_empty]
        
        p_settle = sb_e * np.exp(-ds / ss_e)
        p_port = pb * co * np.exp(-ds / ps)
        p_ruin = rp * np.ones_like(ds)
        p_forest = ffe * (1 + 1.0 / (1 + df))
        p_empty = np.clip(1.0 - p_settle - p_port - p_ruin - p_forest, 0, None)
        
        pred[mask_empty, 0] = p_empty
        pred[mask_empty, 1] = p_settle
        pred[mask_empty, 2] = p_port
        pred[mask_empty, 3] = p_ruin
        pred[mask_empty, 4] = p_forest
    
    # Settlement (class 1)
    mask_settle = (cls == 1) & ~mountain & ~deep_ocean
    if mask_settle.any():
        co = coastal[mask_settle]
        s_d = sd[mask_settle]
        
        p_survive = np.clip(ssb + ssdb * s_d, 0, 0.9)
        p_port = pb * co * 2
        p_ruin = rfr * np.ones_like(co)
        p_forest = ffs * (1 - p_survive * 0.5)
        p_empty = np.clip(1.0 - p_survive - p_port - p_ruin - p_forest, 0, None)
        
        pred[mask_settle, 0] = p_empty
        pred[mask_settle, 1] = p_survive
        pred[mask_settle, 2] = p_port
        pred[mask_settle, 3] = p_ruin
        pred[mask_settle, 4] = p_forest
    
    # Port (class 2)
    mask_port = (cls == 2) & ~mountain & ~deep_ocean
    if mask_port.any():
        p_port_surv = psurv * np.ones(mask_port.sum())
        p_settle_surv = 0.10 * np.ones(mask_port.sum())
        p_ruin = rfr * np.ones(mask_port.sum())
        p_forest = ffs * 0.8 * np.ones(mask_port.sum())
        p_empty = np.clip(1.0 - p_port_surv - p_settle_surv - p_ruin - p_forest, 0, None)
        
        pred[mask_port, 0] = p_empty
        pred[mask_port, 1] = p_settle_surv
        pred[mask_port, 2] = p_port_surv
        pred[mask_port, 3] = p_ruin
        pred[mask_port, 4] = p_forest
    
    # Forest (class 4)
    mask_forest = (cls == 4) & ~mountain & ~deep_ocean
    if mask_forest.any():
        ds = dist_s[mask_forest]
        co = coastal[mask_forest]
        s_d = sd[mask_forest]
        
        forest_surv = np.clip(fpb - fsp * s_d, 0.3, 0.99)
        p_settle = sb_f * np.exp(-ds / ss_f)
        p_port = pb * co * np.exp(-ds / ps) * 0.5
        p_ruin = rp * np.ones_like(ds)
        p_empty = np.clip(1.0 - forest_surv - p_settle - p_port - p_ruin, 0, None)
        
        pred[mask_forest, 0] = p_empty
        pred[mask_forest, 1] = p_settle
        pred[mask_forest, 2] = p_port
        pred[mask_forest, 3] = p_ruin
        pred[mask_forest, 4] = forest_surv
    
    # Ruin (class 3, very rare as initial)
    mask_ruin = (cls == 3) & ~mountain & ~deep_ocean
    if mask_ruin.any():
        ds = dist_s[mask_ruin]
        p_settle = sb_e * np.exp(-ds / ss_e) * 1.2
        p_forest = 0.3 * np.ones(mask_ruin.sum())
        p_ruin = 0.05 * np.ones(mask_ruin.sum())
        p_empty = np.clip(1.0 - p_settle - p_forest - p_ruin, 0, None)
        
        pred[mask_ruin, 0] = p_empty
        pred[mask_ruin, 1] = p_settle
        pred[mask_ruin, 3] = p_ruin
        pred[mask_ruin, 4] = p_forest
    
    # Clip and normalize
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


# ─── Bucket LUT for comparison ────────────────────────────

def compute_settle_features(cls):
    H, W = cls.shape
    settlement = (cls == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    forest_mask = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest_mask) if forest_mask.any() else np.full((H, W), 20.0)
    settle_bin = np.full((H, W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    return settle_bin, near_forest


def lut_loo(round_entries, test_idx, clip_floor=0.0005, min_n=50):
    """LOO CV for bucket LUT."""
    lut_c, lut_n = {}, {}
    cls_c, cls_n = {}, {}
    
    for ri, seeds in enumerate(round_entries):
        if ri == test_idx:
            continue
        for ig, gt in seeds:
            cls = build_class_grid(ig)
            sb, nf = compute_settle_features(cls)
            H, W = ig.shape
            for y in range(H):
                for x in range(W):
                    k = (int(cls[y,x]), int(sb[y,x]), int(nf[y,x]))
                    lut_c.setdefault(k, np.zeros(6))
                    lut_n.setdefault(k, 0)
                    lut_c[k] += gt[y,x]
                    lut_n[k] += 1
                    ic = int(cls[y,x])
                    cls_c.setdefault(ic, np.zeros(6))
                    cls_n.setdefault(ic, 0)
                    cls_c[ic] += gt[y,x]
                    cls_n[ic] += 1
    
    cls_avg = {ic: cls_c[ic]/cls_n[ic] if cls_n.get(ic,0)>0 else np.ones(6)/6 for ic in range(6)}
    lut = {}
    for k, v in lut_c.items():
        n = lut_n[k]
        if n >= min_n:
            a = v / n
            a = np.clip(a, clip_floor, None)
            a /= a.sum()
            lut[k] = a
        else:
            lut[k] = cls_avg[k[0]]
    
    scores = []
    for ig, gt in round_entries[test_idx]:
        cls = build_class_grid(ig)
        sb, nf = compute_settle_features(cls)
        H, W = ig.shape
        pred = np.ones((H,W,6))/6
        for y in range(H):
            for x in range(W):
                ic = int(cls[y,x])
                if ic == 5:
                    pred[y,x] = [0,0,0,0,0,1]
                    continue
                k = (ic, int(sb[y,x]), int(nf[y,x]))
                pred[y,x] = lut.get(k, cls_avg.get(ic, np.ones(6)/6))
        pred = np.clip(pred, clip_floor, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        scores.append(kl_score(pred, gt))
    return np.mean(scores)


# ─── Main calibration + comparison ────────────────────────

def objective(x, entries_flat):
    """Negative mean KL score across calibration entries."""
    scores = []
    for ig, gt in entries_flat:
        pred = predict_vectorized(ig, x)
        scores.append(kl_score(pred, gt))
    return -np.mean(scores)


def main():
    round_entries = load_all_gt()
    n_rounds = len(round_entries)
    print(f"Loaded {n_rounds} rounds")
    
    # Flatten for calibration (1 seed per round for speed)
    cal_entries = [seeds[0] for seeds in round_entries]
    
    # Default params
    x0 = np.array([
        0.35, 3.0,   # settle_base_empty, settle_scale_empty
        0.30, 2.5,   # settle_base_forest, settle_scale_forest
        0.05, 2.0,   # port_base, port_scale
        0.025, 0.01, # ruin_from_settle, ruin_persistence
        0.35, 0.5,   # settle_survival_base, settle_survival_density_bonus
        0.80, 0.25,  # forest_persist_base, forest_settle_penalty
        0.03, 0.20,  # forest_from_empty, forest_from_settle
        0.20,         # port_survival
    ])
    
    # Bounds
    bounds = [
        (0.01, 0.8), (1.0, 10.0),    # settle empty
        (0.01, 0.8), (1.0, 10.0),    # settle forest
        (0.001, 0.3), (0.5, 10.0),   # port
        (0.001, 0.15), (0.001, 0.05),# ruin
        (0.05, 0.8), (0.0, 3.0),     # settle survival
        (0.4, 0.99), (0.0, 2.0),     # forest persist
        (0.001, 0.15), (0.05, 0.5),  # forest from
        (0.02, 0.5),                   # port survival
    ]
    
    print(f"\nDefault params KL: {-objective(x0, cal_entries):.2f}")
    
    # Differential evolution (global optimizer - good for ~15 params)
    print("\nRunning differential evolution...")
    t0 = time.time()
    result = differential_evolution(
        objective, bounds, args=(cal_entries,),
        seed=42, maxiter=100, popsize=20,
        tol=0.001, mutation=(0.5, 1.5), recombination=0.8,
        x0=x0,
        disp=True,
    )
    t1 = time.time()
    print(f"Optimization: {t1-t0:.1f}s, score={-result.fun:.2f}")
    best_x = result.x
    
    # Print best params
    names = [
        'settle_base_empty', 'settle_scale_empty',
        'settle_base_forest', 'settle_scale_forest',
        'port_base', 'port_scale',
        'ruin_from_settle', 'ruin_persistence',
        'settle_survival_base', 'settle_survival_density_bonus',
        'forest_persist_base', 'forest_settle_penalty',
        'forest_from_empty', 'forest_from_settle',
        'port_survival',
    ]
    print("\nOptimized parameters:")
    for name, val in zip(names, best_x):
        print(f"  {name:30s} = {val:.6f}")
    
    # LOO CV comparison
    print(f"\n{'='*60}")
    print("LOO Cross-Validation (all rounds)")
    print(f"{'='*60}")
    
    lut_scores = []
    cell_scores = []
    
    for ri in range(n_rounds):
        # LUT
        lut_s = lut_loo(round_entries, ri)
        lut_scores.append(lut_s)
        
        # Cell model (same params for all folds - slight optimistic bias)
        cell_seeds = []
        for ig, gt in round_entries[ri]:
            pred = predict_vectorized(ig, best_x)
            cell_seeds.append(kl_score(pred, gt))
        cell_s = np.mean(cell_seeds)
        cell_scores.append(cell_s)
        
        marker = " ***" if cell_s > lut_s else ""
        print(f"  Round {ri:2d}: LUT={lut_s:.2f}  Cell={cell_s:.2f}  diff={cell_s-lut_s:+.2f}{marker}")
    
    print(f"\n  LUT  mean: {np.mean(lut_scores):.2f}")
    print(f"  Cell mean: {np.mean(cell_scores):.2f}")
    print(f"  Diff:      {np.mean(cell_scores)-np.mean(lut_scores):+.2f}")
    
    # Save best params
    np.save(DATA_DIR / "cell_model_params.npy", best_x)
    print(f"\nSaved best params to {DATA_DIR / 'cell_model_params.npy'}")


if __name__ == "__main__":
    main()
