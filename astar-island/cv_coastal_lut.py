"""
Enhanced bucket LUT: adds coastal feature to the existing (init_class, settle_bin, near_forest).

GT analysis showed coastal cells have significantly different distributions:
  - (0, 2, 0, coastal=1) → [0.947, 0.026, 0.016, ...] (mostly empty, some ports)
  - (0, 2, 0, coastal=0) → [0.839, 0.115, 0.000, ...] (more settlements, NO ports)
  
Ports ONLY form on coastal cells. Adding coastal as a 4th feature should improve
the port prediction significantly.

This runs LOO CV to test if the 4-feature LUT beats the 3-feature LUT.
"""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
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


def load_rounds():
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


def compute_features(cls, ig):
    H, W = cls.shape
    settlement = (cls == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    forest_mask = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest_mask) if forest_mask.any() else np.full((H, W), 20.0)
    ocean = (ig == 10)
    dist_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    
    settle_bin = np.full((H, W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    coastal = (dist_o <= 1.5).astype(int)
    
    return settle_bin, near_forest, coastal


def lut_loo(round_entries, test_idx, n_features=3, clip_floor=0.0005, min_n=50):
    """
    LOO CV for bucket LUT.
    n_features=3: (init_class, settle_bin, near_forest)
    n_features=4: (init_class, settle_bin, near_forest, coastal)
    """
    lut_c, lut_n = {}, {}
    cls_c, cls_n = {}, {}
    
    for ri, seeds in enumerate(round_entries):
        if ri == test_idx:
            continue
        for ig, gt in seeds:
            cls = build_class_grid(ig)
            sb, nf, co = compute_features(cls, ig)
            H, W = ig.shape
            for y in range(H):
                for x in range(W):
                    ic = int(cls[y, x])
                    if n_features == 4:
                        k = (ic, int(sb[y, x]), int(nf[y, x]), int(co[y, x]))
                    else:
                        k = (ic, int(sb[y, x]), int(nf[y, x]))
                    lut_c.setdefault(k, np.zeros(6))
                    lut_n.setdefault(k, 0)
                    lut_c[k] += gt[y, x]
                    lut_n[k] += 1
                    cls_c.setdefault(ic, np.zeros(6))
                    cls_n.setdefault(ic, 0)
                    cls_c[ic] += gt[y, x]
                    cls_n[ic] += 1
    
    cls_avg = {ic: cls_c[ic] / cls_n[ic] if cls_n.get(ic, 0) > 0 else np.ones(6) / 6 for ic in range(6)}
    
    # Build 3-feature fallback for when 4-feature bucket has too few samples
    fallback_lut = {}
    if n_features == 4:
        fb_c, fb_n = {}, {}
        for k4, v in lut_c.items():
            k3 = k4[:3]
            fb_c.setdefault(k3, np.zeros(6))
            fb_n.setdefault(k3, 0)
            fb_c[k3] += v
            fb_n[k3] += lut_n[k4]
        for k3, v in fb_c.items():
            n = fb_n[k3]
            if n >= min_n:
                a = v / n
                a = np.clip(a, clip_floor, None)
                a /= a.sum()
                fallback_lut[k3] = a
    
    lut = {}
    for k, v in lut_c.items():
        n = lut_n[k]
        if n >= min_n:
            a = v / n
            a = np.clip(a, clip_floor, None)
            a /= a.sum()
            lut[k] = a
    
    # Predict test round
    scores = []
    for ig, gt in round_entries[test_idx]:
        cls = build_class_grid(ig)
        sb, nf, co = compute_features(cls, ig)
        H, W = ig.shape
        pred = np.ones((H, W, 6)) / 6
        for y in range(H):
            for x in range(W):
                ic = int(cls[y, x])
                if ic == 5:
                    pred[y, x] = [0, 0, 0, 0, 0, 1]
                    continue
                if n_features == 4:
                    k = (ic, int(sb[y, x]), int(nf[y, x]), int(co[y, x]))
                    if k in lut:
                        pred[y, x] = lut[k]
                    else:
                        # Fall back to 3-feature
                        k3 = k[:3]
                        pred[y, x] = fallback_lut.get(k3, cls_avg.get(ic, np.ones(6) / 6))
                else:
                    k = (ic, int(sb[y, x]), int(nf[y, x]))
                    pred[y, x] = lut.get(k, cls_avg.get(ic, np.ones(6) / 6))
        
        pred = np.clip(pred, clip_floor, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        scores.append(kl_score(pred, gt))
    
    return np.mean(scores)


def main():
    round_entries = load_rounds()
    n_rounds = len(round_entries)
    print(f"Loaded {n_rounds} rounds")
    
    # Test various LUT configurations
    configs = [
        ("3-feat (current)", 3, 0.0005, 50),
        ("4-feat coast", 4, 0.0005, 50),
        ("4-feat coast min_n=30", 4, 0.0005, 30),
        ("4-feat coast min_n=20", 4, 0.0005, 20),
        ("3-feat min_n=30", 3, 0.0005, 30),
    ]
    
    for name, nf, cf, mn in configs:
        print(f"\n=== {name} ===")
        scores = []
        t0 = time.time()
        for ri in range(n_rounds):
            s = lut_loo(round_entries, ri, n_features=nf, clip_floor=cf, min_n=mn)
            scores.append(s)
        t1 = time.time()
        print(f"  Scores: {[f'{s:.2f}' for s in scores]}")
        print(f"  Mean: {np.mean(scores):.3f}  ({t1-t0:.1f}s)")
    
    # Compare best 3-feat vs best 4-feat
    print(f"\n{'='*60}")
    print("Head-to-head comparison:")
    print(f"{'='*60}")
    
    for ri in range(n_rounds):
        s3 = lut_loo(round_entries, ri, n_features=3, clip_floor=0.0005, min_n=50)
        s4 = lut_loo(round_entries, ri, n_features=4, clip_floor=0.0005, min_n=30)
        diff = s4 - s3
        marker = " ***" if diff > 0.01 else " ---" if diff < -0.01 else ""
        print(f"  Round {ri:2d}: 3-feat={s3:.3f}  4-feat={s4:.3f}  diff={diff:+.3f}{marker}")


if __name__ == "__main__":
    main()
