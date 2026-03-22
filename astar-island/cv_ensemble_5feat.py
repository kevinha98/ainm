"""Test 5-feat near_port LUT + cell model ensemble."""
import json, numpy as np, sys
from pathlib import Path
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from simulator.cell_model import predict_cell_distributions, params_from_vector

DATA_DIR = Path("data")
GRID_TO_CLASS = {0:0,1:1,2:2,3:3,4:4,5:5,10:0,11:0}

def build_class_grid(ig):
    cg = np.zeros_like(ig)
    for gv, cls in GRID_TO_CLASS.items(): cg[ig==gv] = cls
    return cg

def kl_score(pred, gt):
    pred = np.clip(pred, 1e-8, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    kl = np.where(gt>0, gt*np.log(np.clip(gt,1e-15,None)/pred), 0).sum(axis=-1)
    kl = np.where(np.isfinite(kl), kl, 0)
    return 100 - kl.mean()*100

def compute_features(cls, ig):
    H,W = cls.shape
    s = (cls==1); ds = ndimage.distance_transform_edt(~s) if s.any() else np.full((H,W),20.0)
    f = (cls==4); df = ndimage.distance_transform_edt(~f) if f.any() else np.full((H,W),20.0)
    o = (ig==10); do = ndimage.distance_transform_edt(~o) if o.any() else np.full((H,W),40.0)
    p = (cls==2); dp = ndimage.distance_transform_edt(~p) if p.any() else np.full((H,W),40.0)
    sb = np.full((H,W),3,dtype=int)
    sb[ds<=4.0]=2; sb[ds<=2.0]=1; sb[ds<=1.0]=0
    nf = (df<=2.0).astype(int)
    co = (do<=1.5).astype(int)
    np_ = (dp<=2.0).astype(int)
    return sb, nf, co, np_

def load_rounds():
    rounds = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        with open(gf) as f: data = json.load(f)
        seeds = []
        for si in sorted(data.keys()):
            e = data[si]
            seeds.append((np.array(e['initial_grid']), np.array(e['ground_truth'])))
        rounds.append(seeds)
    return rounds

def run_cv(rounds, opt_params, alpha, use_5feat=True):
    n = len(rounds)
    CF = 0.0005
    all_scores = []
    per_round_scores = []
    
    for ti in range(n):
        lut5_c, lut5_n = {}, {}
        lut4_c, lut4_n = {}, {}
        fb3_c, fb3_n = {}, {}
        cls_c, cls_n = {}, {}
        for ri, seeds in enumerate(rounds):
            if ri == ti: continue
            for ig, gt in seeds:
                cls = build_class_grid(ig)
                sb, nf, co, np_ = compute_features(cls, ig)
                H,W = ig.shape
                for y in range(H):
                    for x in range(W):
                        ic = int(cls[y,x])
                        k5 = (ic,int(sb[y,x]),int(nf[y,x]),int(co[y,x]),int(np_[y,x]))
                        k4 = k5[:4]
                        k3 = k5[:3]
                        if use_5feat:
                            lut5_c.setdefault(k5,np.zeros(6)); lut5_n.setdefault(k5,0)
                            lut5_c[k5] += gt[y,x]; lut5_n[k5] += 1
                        lut4_c.setdefault(k4,np.zeros(6)); lut4_n.setdefault(k4,0)
                        lut4_c[k4] += gt[y,x]; lut4_n[k4] += 1
                        fb3_c.setdefault(k3,np.zeros(6)); fb3_n.setdefault(k3,0)
                        fb3_c[k3] += gt[y,x]; fb3_n[k3] += 1
                        cls_c.setdefault(ic,np.zeros(6)); cls_n.setdefault(ic,0)
                        cls_c[ic] += gt[y,x]; cls_n[ic] += 1

        ca = {ic: cls_c[ic]/cls_n[ic] if cls_n.get(ic,0)>0 else np.ones(6)/6 for ic in range(6)}
        def build_lut(c, n, mn):
            l = {}
            for k, v in c.items():
                if n[k] >= mn:
                    a = v/n[k]; a = np.clip(a,CF,None); a/=a.sum(); l[k] = a
            return l
        lut5 = build_lut(lut5_c, lut5_n, 50) if use_5feat else {}
        lut4 = build_lut(lut4_c, lut4_n, 50)
        fb3 = build_lut(fb3_c, fb3_n, 50)

        round_scores = []
        for ig, gt in rounds[ti]:
            cls = build_class_grid(ig)
            sb, nf, co, np_ = compute_features(cls, ig)
            H,W = ig.shape

            pred_lut = np.ones((H,W,6))/6
            for y in range(H):
                for x in range(W):
                    ic = int(cls[y,x])
                    if ic == 5: pred_lut[y,x] = [0,0,0,0,0,1]; continue
                    k5 = (ic,int(sb[y,x]),int(nf[y,x]),int(co[y,x]),int(np_[y,x]))
                    k4 = k5[:4]; k3 = k5[:3]
                    if use_5feat and k5 in lut5:
                        pred_lut[y,x] = lut5[k5]
                    elif k4 in lut4:
                        pred_lut[y,x] = lut4[k4]
                    elif k3 in fb3:
                        pred_lut[y,x] = fb3[k3]
                    else:
                        pred_lut[y,x] = ca.get(ic, np.ones(6)/6)
            pred_lut = np.clip(pred_lut, CF, None)
            pred_lut /= pred_lut.sum(axis=-1, keepdims=True)

            if alpha == 0:
                pred = pred_lut
            else:
                pred_cell = predict_cell_distributions(ig, opt_params)
                pred_cell = np.clip(pred_cell, CF, None)
                pred_cell /= pred_cell.sum(axis=-1, keepdims=True)
                log_blend = (1-alpha)*np.log(pred_lut) + alpha*np.log(pred_cell)
                pred = np.exp(log_blend)
                pred = np.clip(pred, CF, None)
                pred /= pred.sum(axis=-1, keepdims=True)

            s = kl_score(pred, gt)
            all_scores.append(s)
            round_scores.append(s)
        per_round_scores.append(np.mean(round_scores))
    
    return np.mean(all_scores), per_round_scores

if __name__ == "__main__":
    rounds = load_rounds()
    print(f"Loaded {len(rounds)} rounds\n")

    opt_vec = np.load("data/cell_model_params.npy")
    opt_params = params_from_vector(opt_vec)

    configs = [
        ("4-feat LUT only",     False, 0.0),
        ("5-feat LUT only",     True,  0.0),
        ("4-feat + cell a=0.3", False, 0.3),
        ("4-feat + cell a=0.4", False, 0.4),
        ("5-feat + cell a=0.2", True,  0.2),
        ("5-feat + cell a=0.3", True,  0.3),
        ("5-feat + cell a=0.4", True,  0.4),
        ("5-feat + cell a=0.5", True,  0.5),
    ]

    print(f"{'Config':<30s} {'Mean':>8s} {'Min':>7s} {'Max':>7s}")
    print("-" * 55)
    for name, use5, alpha in configs:
        cv, prs = run_cv(rounds, opt_params, alpha, use_5feat=use5)
        print(f"{name:<30s} {cv:8.3f} {min(prs):7.2f} {max(prs):7.2f}")
    print("Done")
