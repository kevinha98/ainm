"""Quick CV tests for clip floor, min_n, and cross-round fallback."""
import json, numpy as np
from pathlib import Path
from scipy import ndimage

DATA_DIR = Path("data")
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}

def build_class_grid(ig):
    ig = np.array(ig)
    return np.vectorize(lambda v: GRID_TO_CLASS.get(v, 0))(ig)

def score_pred(pred, gt):
    pred = np.clip(pred, 1e-12, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    kl = np.sum(gt_safe * np.log(gt_safe / pred), axis=-1).mean()
    return 100 * np.exp(-kl)

def compute_features(ig):
    cg = build_class_grid(ig)
    H, W = cg.shape
    settlement = (cg == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20.0)
    forest = (cg == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20.0)
    settle_bin = np.full((H,W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    return cg, settle_bin, near_forest

def sim_obs(test_data, seed=42):
    oc, ot = {}, {}
    rng = np.random.RandomState(seed)
    for si_str in sorted(test_data.keys()):
        entry = test_data[si_str]
        if not isinstance(entry, dict): continue
        gt = np.array(entry["ground_truth"])
        ig = np.array(entry["initial_grid"])
        H, W = ig.shape
        cg, sb, nf = compute_features(ig)
        for row in [0, 12, 25]:
            for col in [0, 12, 25]:
                re2, ce = min(row+15, H), min(col+15, W)
                sub_gt = gt[row:re2, col:ce].reshape(-1, 6)
                cumprob = np.cumsum(np.clip(sub_gt, 0, None), axis=1)
                cumprob /= cumprob[:, -1:] + 1e-15
                u = rng.random(len(sub_gt))
                obs_cls = (u[:, None] > cumprob).sum(axis=1).clip(0, 5)
                idx = 0
                for vy in range(re2 - row):
                    for vx in range(ce - col):
                        gy, gx = row + vy, col + vx
                        key = (int(cg[gy, gx]), int(sb[gy, gx]), int(nf[gy, gx]))
                        if key not in oc:
                            oc[key] = np.zeros(6)
                            ot[key] = 0
                        oc[key][obs_cls[idx]] += 1
                        ot[key] += 1
                        idx += 1
    return oc, ot

def predict(test_data, lut, class_avgs, cross_lut=None, floor=0.0001):
    seed_scores = []
    for si_str in sorted(test_data.keys()):
        entry = test_data[si_str]
        if not isinstance(entry, dict): continue
        gt = np.array(entry["ground_truth"])
        ig = np.array(entry["initial_grid"])
        H, W = ig.shape
        cg, sb, nf = compute_features(ig)
        pred = np.ones((H, W, 6)) / 6
        for y in range(H):
            for x in range(W):
                key = (int(cg[y, x]), int(sb[y, x]), int(nf[y, x]))
                if key in lut:
                    pred[y, x] = lut[key]
                elif cross_lut and key in cross_lut:
                    pred[y, x] = cross_lut[key]
                else:
                    pred[y, x] = class_avgs.get(int(cg[y, x]), np.ones(6) / 6)
        mtn = (cg == 5)
        if mtn.any():
            pred[mtn] = [0, 0, 0, 0, 0, 1.0]
        pred = np.clip(pred, floor, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        seed_scores.append(score_pred(pred, gt))
    return seed_scores

def build_lut(oc, ot, min_n=10):
    class_avgs = {}
    for ic in range(6):
        tc, tn = np.zeros(6), 0
        for k, c in oc.items():
            if k[0] == ic:
                tc += c
                tn += ot[k]
        class_avgs[ic] = tc / max(tn, 1) if tn > 0 else np.ones(6) / 6
    lut = {}
    for k, c in oc.items():
        lut[k] = c / ot[k] if ot[k] >= min_n else class_avgs[k[0]]
    return lut, class_avgs


def main():
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    all_gt = {}
    for gf in gt_files:
        rid = gf.stem.replace("ground_truth_", "")
        with open(gf) as f:
            all_gt[rid] = json.load(f)
    rids = sorted(all_gt.keys())
    print(f"Loaded {len(all_gt)} GT files\n")

    # Pre-compute observations
    all_obs = {rid: sim_obs(all_gt[rid]) for rid in rids}

    # ==== TEST 1: min_n sweep ====
    print("min_n sweep (no cross-round fallback):")
    for min_n in [1, 3, 5, 10, 15, 20, 30, 50]:
        round_scores = []
        for rid in rids:
            oc, ot = all_obs[rid]
            lut, ca = build_lut(oc, ot, min_n)
            scores = predict(all_gt[rid], lut, ca)
            round_scores.append(np.mean(scores))
        print(f"  min_n={min_n:3d}: {np.mean(round_scores):.3f}")

    # ==== TEST 2: clip floor sweep ====
    print("\nClip floor sweep (min_n=10):")
    for floor in [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        round_scores = []
        for rid in rids:
            oc, ot = all_obs[rid]
            lut, ca = build_lut(oc, ot, 10)
            scores = predict(all_gt[rid], lut, ca, floor=floor)
            round_scores.append(np.mean(scores))
        print(f"  floor={floor:.0e}: {np.mean(round_scores):.3f}")

    # ==== TEST 3: Cross-round fallback (LOO) ====
    print("\nLOO: within-round obs + cross-round GT fallback:")
    for min_n in [5, 10, 20]:
        round_scores = []
        for test_rid in rids:
            # Build cross-round GT LUT
            cross_oc, cross_ot = {}, {}
            for other_rid in rids:
                if other_rid == test_rid:
                    continue
                for si_str, entry in all_gt[other_rid].items():
                    if not isinstance(entry, dict):
                        continue
                    gt = np.array(entry["ground_truth"])
                    ig = np.array(entry["initial_grid"])
                    H, W = ig.shape
                    cg, sb, nf = compute_features(ig)
                    for y in range(H):
                        for x in range(W):
                            key = (int(cg[y,x]), int(sb[y,x]), int(nf[y,x]))
                            if key not in cross_oc:
                                cross_oc[key] = np.zeros(6)
                                cross_ot[key] = 0
                            cross_oc[key] += gt[y, x]
                            cross_ot[key] += 1
            cross_lut = {k: cross_oc[k] / cross_ot[k] for k in cross_oc}

            oc, ot = all_obs[test_rid]
            lut, ca = build_lut(oc, ot, min_n)
            scores = predict(all_gt[test_rid], lut, ca, cross_lut=cross_lut)
            round_scores.append(np.mean(scores))
        print(f"  min_n={min_n:3d}: {np.mean(round_scores):.3f}  worst={min(round_scores):.1f}")

    # ==== TEST 4: Blending within-round + cross-round ====
    print("\nBlending: within-round obs weighted with cross-round GT:")
    for blend_w in [0.0, 0.1, 0.2, 0.3, 0.5]:
        round_scores = []
        for test_rid in rids:
            # Cross-round GT counts
            cross_oc, cross_ot = {}, {}
            for other_rid in rids:
                if other_rid == test_rid:
                    continue
                for si_str, entry in all_gt[other_rid].items():
                    if not isinstance(entry, dict):
                        continue
                    gt = np.array(entry["ground_truth"])
                    ig = np.array(entry["initial_grid"])
                    H, W = ig.shape
                    cg, sb, nf = compute_features(ig)
                    for y in range(H):
                        for x in range(W):
                            key = (int(cg[y,x]), int(sb[y,x]), int(nf[y,x]))
                            if key not in cross_oc:
                                cross_oc[key] = np.zeros(6)
                                cross_ot[key] = 0
                            cross_oc[key] += gt[y, x]
                            cross_ot[key] += 1

            oc, ot = all_obs[test_rid]
            # Blend: (1-w)*within_freq + w*cross_freq
            blended_lut = {}
            ca_within = {}
            for ic in range(6):
                tc, tn = np.zeros(6), 0
                for k, c in oc.items():
                    if k[0] == ic:
                        tc += c
                        tn += ot[k]
                ca_within[ic] = tc / max(tn, 1) if tn > 0 else np.ones(6) / 6

            for k in set(list(oc.keys()) + list(cross_oc.keys())):
                within_freq = oc.get(k, np.zeros(6))
                within_n = ot.get(k, 0)
                cross_freq = cross_oc.get(k, np.zeros(6))
                cross_n = cross_ot.get(k, 0)
                if within_n >= 10:
                    wf = within_freq / within_n
                else:
                    wf = ca_within.get(k[0], np.ones(6)/6)
                if cross_n > 0:
                    cf = cross_freq / cross_n
                else:
                    cf = wf
                blended = (1 - blend_w) * wf + blend_w * cf
                blended /= blended.sum()
                blended_lut[k] = blended

            scores = predict(all_gt[test_rid], blended_lut, ca_within)
            round_scores.append(np.mean(scores))
        print(f"  blend={blend_w:.1f}: {np.mean(round_scores):.3f}")

    print("\nDone")


if __name__ == "__main__":
    main()
