"""Verify temperature improvement using EDT features (same as auto_runner_v2.py)."""
import json, sys, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, "src")
DATA_DIR = Path("data")
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}

# Load GT
all_entries = []
for gf in sorted(DATA_DIR.glob('ground_truth_*.json')):
    rid = gf.stem.replace('ground_truth_', '')
    with open(gf) as f:
        data = json.load(f)
    for sk, entry in data.items():
        ig = np.array(entry['initial_grid'])
        gt = np.array(entry['ground_truth'])
        if ig.shape == (40,40) and gt.shape == (40,40,6):
            all_entries.append({'rid': rid, 'ig': ig, 'gt': gt})
print(f"Seeds: {len(all_entries)}")

# EDT features (same as auto_runner_v2.py)
def compute_features_edt(cls, ig):
    H, W = cls.shape
    settlement = (cls == 1)
    dist_s = distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20.0)
    forest = (cls == 4)
    dist_f = distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20.0)
    ocean = (ig == 10)
    dist_o = distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 40.0)
    port = (cls == 2)
    dist_p = distance_transform_edt(~port) if port.any() else np.full((H,W), 40.0)
    settle_bin = np.full((H,W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    coastal = (dist_o <= 1.5).astype(int)
    near_port = (dist_p <= 2.0).astype(int)
    return settle_bin, near_forest, coastal, near_port

def comp_score(pred, gt):
    eps = 1e-15
    gt_s = np.clip(gt, eps, None)
    pred_s = np.clip(pred, eps, None)
    entropy = -np.sum(gt * np.log(gt_s), axis=-1)
    kl = np.sum(gt * np.log(gt_s / pred_s), axis=-1)
    te = entropy.sum()
    if te < eps: return 100.0
    wkl = np.sum(entropy * kl) / te
    return max(0, min(100, 100 * np.exp(-3 * wkl)))

# Precompute
from simulator.cell_model import predict_cell_distributions, params_from_vector
cell_vec = np.load(DATA_DIR / 'cell_model_params.npy')
cell_p = params_from_vector(cell_vec)

print("Precomputing EDT features and cell model...")
for e in all_entries:
    ig = e['ig']
    mapped = np.vectorize(GRID_TO_CLASS.get)(ig)
    e['mapped'] = mapped
    e['feats'] = compute_features_edt(mapped, ig)
    e['mtn'] = (mapped == 5)
    cd = predict_cell_distributions(ig, cell_p)
    mtn = e['mtn']
    cd[~mtn, 5] = 0.0
    s = cd.sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1, s)
    cd /= s
    e['cell_zm'] = cd

rounds_map = defaultdict(list)
for i, e in enumerate(all_entries):
    rounds_map[e['rid']].append(i)

# Build tallies
global_tally = defaultdict(lambda: np.zeros(6))
round_tally = {}
for rid in rounds_map:
    rt = defaultdict(lambda: np.zeros(6))
    for idx in rounds_map[rid]:
        e = all_entries[idx]
        mapped = e['mapped']
        sb, nf, co, np_ = e['feats']
        gt = e['gt']
        for r in range(40):
            for c in range(40):
                key = (int(mapped[r,c]), int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]))
                rt[key] += gt[r,c]
                global_tally[key] += gt[r,c]
    round_tally[rid] = dict(rt)

print(f"Buckets: {len(global_tally)}")

# LOO-CV sweep
def loo_score(alpha, clip, temperature, min_n=20):
    scores = []
    for hold_rid in rounds_map:
        train_tally = {}
        for k, v in global_tally.items():
            sub = round_tally[hold_rid].get(k, np.zeros(6))
            rem = v - sub
            if rem.sum() > 0:
                train_tally[k] = rem
        fb4 = defaultdict(lambda: np.zeros(6))
        fb3 = defaultdict(lambda: np.zeros(6))
        ct = np.zeros((6,6))
        for k, v in train_tally.items():
            fb4[k[:4]] += v; fb3[k[:3]] += v; ct[k[0]] += v
        lut = {k: v/v.sum() for k,v in train_tally.items() if v.sum() >= min_n}
        fb4l = {k: v/v.sum() for k,v in fb4.items() if v.sum() >= min_n}
        fb3l = {k: v/v.sum() for k,v in fb3.items() if v.sum() >= min_n}
        ca = {ci: ct[ci]/ct[ci].sum() if ct[ci].sum() > 0 else np.ones(6)/6 for ci in range(6)}
        
        for idx in rounds_map[hold_rid]:
            e = all_entries[idx]
            mapped = e['mapped']; sb, nf, co, np_ = e['feats']; mtn = e['mtn']
            pred = np.zeros((40,40,6))
            for r in range(40):
                for c in range(40):
                    ic = int(mapped[r,c])
                    key5 = (ic, int(sb[r,c]), int(nf[r,c]), int(co[r,c]), int(np_[r,c]))
                    if key5 in lut: pred[r,c] = lut[key5]
                    elif key5[:4] in fb4l: pred[r,c] = fb4l[key5[:4]]
                    elif key5[:3] in fb3l: pred[r,c] = fb3l[key5[:3]]
                    else: pred[r,c] = ca.get(ic, np.ones(6)/6)
            # Zero mountain
            pred[~mtn, 5] = 0.0
            s = pred.sum(axis=-1, keepdims=True); s = np.where(s==0,1,s); pred /= s
            pred[mtn] = np.array([0,0,0,0,0,1.0])
            
            if alpha > 0:
                cell_zm = e['cell_zm']
                llog = np.log(np.clip(pred, clip, None))
                clog = np.log(np.clip(cell_zm, clip, None))
                mixed = (1-alpha)*llog + alpha*clog
                mixed -= mixed.max(axis=-1, keepdims=True)
                p = np.exp(mixed)
                p /= p.sum(axis=-1, keepdims=True)
            else:
                p = pred.copy()
            
            if temperature != 1.0:
                nm = ~mtn
                if nm.any():
                    pnm = np.clip(p[nm], 1e-10, None)
                    pnm = np.exp(np.log(pnm) / temperature)
                    pnm /= pnm.sum(axis=-1, keepdims=True)
                    p[nm] = pnm
            
            p = np.clip(p, 1e-8, None); p /= p.sum(axis=-1, keepdims=True)
            scores.append(comp_score(p, e['gt']))
    return np.mean(scores), np.std(scores)

# Test key configs with EDT features
print("\n=== EDT-based verification ===")
configs = [
    ("Current deployed (a=0.75, T=1.0)", 0.75, 1e-6, 1.0),
    ("T=1.1, a=0.75", 0.75, 1e-6, 1.1),
    ("T=1.15, a=0.75", 0.75, 1e-6, 1.15),
    ("T=1.2, a=0.75", 0.75, 1e-6, 1.2),
    ("T=1.25, a=0.75", 0.75, 1e-6, 1.25),
    ("T=1.3, a=0.75", 0.75, 1e-6, 1.3),
    ("T=1.2, a=0.65", 0.65, 1e-6, 1.2),
    ("T=1.2, a=0.60", 0.60, 1e-6, 1.2),
    ("T=1.2, a=0.70", 0.70, 1e-6, 1.2),
    ("T=1.2, a=0.50", 0.50, 1e-6, 1.2),
    ("LUT only T=1.2", 0.0, 1e-6, 1.2),
    ("LUT only T=1.0", 0.0, 1e-6, 1.0),
]
for name, a, clip, T in configs:
    mean, std = loo_score(a, clip, T)
    print(f"  {name:35s} -> {mean:.2f} +/- {std:.2f}")
