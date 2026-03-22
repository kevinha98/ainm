"""
Systematic CV comparison of LUT improvements over 4-feat coastal baseline.
Tests: Bayesian blending, additional features (density, near_port), fine settle bins.
"""
import json
import numpy as np
from pathlib import Path
from scipy import ndimage

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


def compute_features(cls, ig, extended=False):
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
    if not extended:
        return settle_bin, near_forest, coastal, None, None
    # Settlement density in 5x5 neighborhood
    from scipy.ndimage import uniform_filter
    settle_float = settlement.astype(float)
    density = uniform_filter(settle_float, size=5, mode='constant')
    settle_density = np.zeros((H, W), dtype=int)
    settle_density[density > 0.05] = 1
    settle_density[density > 0.15] = 2
    # Near port
    port = (cls == 2)
    dist_p = ndimage.distance_transform_edt(~port) if port.any() else np.full((H, W), 40.0)
    near_port = (dist_p <= 2.0).astype(int)
    return settle_bin, near_forest, coastal, settle_density, near_port


def generic_lut_cv(round_entries, test_idx, key_fn, fallback_fns, clip_floor=0.0005, min_n=50):
    """Generic LOO CV for any key function with fallback chain.
    key_fn(cls, ig, y, x, feats) -> tuple key
    fallback_fns: list of (key_fn, min_n) for fallback levels
    """
    # Build all LUT levels
    levels_c = [{} for _ in range(len(fallback_fns) + 1)]
    levels_n = [{} for _ in range(len(fallback_fns) + 1)]
    cls_c, cls_n = {}, {}

    for ri, seeds in enumerate(round_entries):
        if ri == test_idx:
            continue
        for ig, gt in seeds:
            cls = build_class_grid(ig)
            feats = compute_features(cls, ig, extended=True)
            H, W = ig.shape
            for y in range(H):
                for x in range(W):
                    ic = int(cls[y, x])
                    primary_key = key_fn(cls, ig, y, x, feats)
                    levels_c[0].setdefault(primary_key, np.zeros(6))
                    levels_n[0].setdefault(primary_key, 0)
                    levels_c[0][primary_key] += gt[y, x]
                    levels_n[0][primary_key] += 1
                    for li, fb_fn in enumerate(fallback_fns):
                        fb_key = fb_fn(cls, ig, y, x, feats)
                        levels_c[li+1].setdefault(fb_key, np.zeros(6))
                        levels_n[li+1].setdefault(fb_key, 0)
                        levels_c[li+1][fb_key] += gt[y, x]
                        levels_n[li+1][fb_key] += 1
                    cls_c.setdefault(ic, np.zeros(6)); cls_n.setdefault(ic, 0)
                    cls_c[ic] += gt[y, x]; cls_n[ic] += 1

    cls_avg = {ic: cls_c[ic] / cls_n[ic] if cls_n.get(ic,0)>0 else np.ones(6)/6 for ic in range(6)}
    luts = []
    for li in range(len(levels_c)):
        lut = {}
        mn = min_n
        for k, v in levels_c[li].items():
            n = levels_n[li][k]
            if n >= mn:
                a = v / n; a = np.clip(a, clip_floor, None); a /= a.sum()
                lut[k] = a
        luts.append(lut)

    scores = []
    for ig, gt in round_entries[test_idx]:
        cls = build_class_grid(ig)
        feats = compute_features(cls, ig, extended=True)
        H, W = ig.shape
        pred = np.ones((H, W, 6)) / 6
        for y in range(H):
            for x in range(W):
                ic = int(cls[y, x])
                if ic == 5:
                    pred[y, x] = [0,0,0,0,0,1]; continue
                found = False
                pk = key_fn(cls, ig, y, x, feats)
                if pk in luts[0]:
                    pred[y, x] = luts[0][pk]; found = True
                if not found:
                    for li, fb_fn in enumerate(fallback_fns):
                        fk = fb_fn(cls, ig, y, x, feats)
                        if fk in luts[li+1]:
                            pred[y, x] = luts[li+1][fk]; found = True; break
                if not found:
                    pred[y, x] = cls_avg.get(ic, np.ones(6)/6)
        pred = np.clip(pred, clip_floor, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        scores.append(kl_score(pred, gt))
    return np.mean(scores)


# Key functions for different models
def key_4feat(cls, ig, y, x, feats):
    sb, nf, co = feats[0], feats[1], feats[2]
    return (int(cls[y,x]), int(sb[y,x]), int(nf[y,x]), int(co[y,x]))

def key_3feat(cls, ig, y, x, feats):
    sb, nf = feats[0], feats[1]
    return (int(cls[y,x]), int(sb[y,x]), int(nf[y,x]))

def key_class(cls, ig, y, x, feats):
    return (int(cls[y,x]),)

def key_5feat_density(cls, ig, y, x, feats):
    sb, nf, co, sd = feats[0], feats[1], feats[2], feats[3]
    return (int(cls[y,x]), int(sb[y,x]), int(nf[y,x]), int(co[y,x]), int(sd[y,x]))

def key_5feat_port(cls, ig, y, x, feats):
    sb, nf, co, sd, np_ = feats[0], feats[1], feats[2], feats[3], feats[4]
    return (int(cls[y,x]), int(sb[y,x]), int(nf[y,x]), int(co[y,x]), int(np_[y,x]))


# ── Bayesian blending model
def model_bayesian(round_entries, test_idx, clip_floor=0.0005, prior_weight=20):
    lut_c, lut_n = {}, {}
    cls_c, cls_n = {}, {}
    fb3_c, fb3_n = {}, {}
    for ri, seeds in enumerate(round_entries):
        if ri == test_idx: continue
        for ig, gt in seeds:
            cls = build_class_grid(ig)
            sb, nf, co, _, _ = compute_features(cls, ig)
            H, W = ig.shape
            for y in range(H):
                for x in range(W):
                    ic = int(cls[y,x])
                    k4 = (ic, int(sb[y,x]), int(nf[y,x]), int(co[y,x]))
                    k3 = k4[:3]
                    lut_c.setdefault(k4, np.zeros(6)); lut_n.setdefault(k4, 0)
                    lut_c[k4] += gt[y,x]; lut_n[k4] += 1
                    fb3_c.setdefault(k3, np.zeros(6)); fb3_n.setdefault(k3, 0)
                    fb3_c[k3] += gt[y,x]; fb3_n[k3] += 1
                    cls_c.setdefault(ic, np.zeros(6)); cls_n.setdefault(ic, 0)
                    cls_c[ic] += gt[y,x]; cls_n[ic] += 1
    cls_avg = {ic: cls_c[ic]/cls_n[ic] if cls_n.get(ic,0)>0 else np.ones(6)/6 for ic in range(6)}
    # Hierarchical Bayesian: 3-feat blended with class, then 4-feat blended with 3-feat
    fb3 = {}
    for k3, v in fb3_c.items():
        n = fb3_n[k3]
        prior = cls_avg.get(k3[0], np.ones(6)/6)
        b = (prior_weight * prior + v) / (prior_weight + n)
        b = np.clip(b, clip_floor, None); b /= b.sum()
        fb3[k3] = b
    lut = {}
    for k4, v in lut_c.items():
        n = lut_n[k4]
        k3 = k4[:3]
        prior = fb3.get(k3, cls_avg.get(k4[0], np.ones(6)/6))
        b = (prior_weight * prior + v) / (prior_weight + n)
        b = np.clip(b, clip_floor, None); b /= b.sum()
        lut[k4] = b
    scores = []
    for ig, gt in round_entries[test_idx]:
        cls = build_class_grid(ig)
        sb, nf, co, _, _ = compute_features(cls, ig)
        H, W = ig.shape
        pred = np.ones((H, W, 6)) / 6
        for y in range(H):
            for x in range(W):
                ic = int(cls[y,x])
                if ic == 5: pred[y,x] = [0,0,0,0,0,1]; continue
                k4 = (ic, int(sb[y,x]), int(nf[y,x]), int(co[y,x]))
                if k4 in lut: pred[y,x] = lut[k4]
                else:
                    k3 = k4[:3]
                    pred[y,x] = fb3.get(k3, cls_avg.get(ic, np.ones(6)/6))
        pred = np.clip(pred, clip_floor, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        scores.append(kl_score(pred, gt))
    return np.mean(scores)


def main():
    round_entries = load_rounds()
    n_rounds = len(round_entries)
    print(f"Loaded {n_rounds} rounds\n")

    # --- Test all models ---
    configs = {
        "4-feat LUT (baseline)": lambda ri: generic_lut_cv(
            round_entries, ri, key_4feat, [key_3feat], min_n=50),
        "4-feat min_n=30": lambda ri: generic_lut_cv(
            round_entries, ri, key_4feat, [key_3feat], min_n=30),
        "4-feat min_n=100": lambda ri: generic_lut_cv(
            round_entries, ri, key_4feat, [key_3feat], min_n=100),
        "Bayesian pw=5": lambda ri: model_bayesian(round_entries, ri, prior_weight=5),
        "Bayesian pw=10": lambda ri: model_bayesian(round_entries, ri, prior_weight=10),
        "Bayesian pw=20": lambda ri: model_bayesian(round_entries, ri, prior_weight=20),
        "Bayesian pw=50": lambda ri: model_bayesian(round_entries, ri, prior_weight=50),
        "Bayesian pw=100": lambda ri: model_bayesian(round_entries, ri, prior_weight=100),
        "5-feat +density": lambda ri: generic_lut_cv(
            round_entries, ri, key_5feat_density, [key_4feat, key_3feat], min_n=50),
        "5-feat +near_port": lambda ri: generic_lut_cv(
            round_entries, ri, key_5feat_port, [key_4feat, key_3feat], min_n=50),
        "5-feat +density mn=30": lambda ri: generic_lut_cv(
            round_entries, ri, key_5feat_density, [key_4feat, key_3feat], min_n=30),
    }

    results = {}
    per_round = {}
    for name, fn in configs.items():
        print(f"Testing: {name}...", end=" ", flush=True)
        scores = [fn(ri) for ri in range(n_rounds)]
        m = np.mean(scores)
        results[name] = m
        per_round[name] = scores
        print(f"CV={m:.3f}")

    print("\n" + "=" * 70)
    print(f"{'Model':<30s} {'Mean':>8s} {'Min':>7s} {'Max':>7s} {'Delta':>7s}")
    print("-" * 70)
    baseline = results["4-feat LUT (baseline)"]
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        d = score - baseline
        s = per_round[name]
        print(f"{name:<30s} {score:8.3f} {min(s):7.2f} {max(s):7.2f} {d:+7.3f}")

    # Per-round detail for best
    best_name = max(results, key=results.get)
    if best_name != "4-feat LUT (baseline)":
        print(f"\nPer-round: {best_name} vs baseline:")
        wins = 0
        for ri in range(n_rounds):
            b = per_round["4-feat LUT (baseline)"][ri]
            t = per_round[best_name][ri]
            d = t - b
            wins += d > 0
            print(f"  R{ri:2d}: {b:.2f} -> {t:.2f}  ({d:+.2f})")
        print(f"  Wins: {wins}/{n_rounds}")


if __name__ == "__main__":
    main()