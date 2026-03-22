"""Test ensemble of 4-feat LUT + optimized cell model at different blend weights."""
import json, numpy as np, sys
from pathlib import Path
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from simulator.cell_model import predict_cell_distributions, CellParams, params_from_vector

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
    s = (cls==1)
    ds = ndimage.distance_transform_edt(~s) if s.any() else np.full((H,W),20.0)
    f = (cls==4)
    df = ndimage.distance_transform_edt(~f) if f.any() else np.full((H,W),20.0)
    o = (ig==10)
    do = ndimage.distance_transform_edt(~o) if o.any() else np.full((H,W),40.0)
    sb = np.full((H,W),3,dtype=int)
    sb[ds<=4.0]=2; sb[ds<=2.0]=1; sb[ds<=1.0]=0
    nf = (df<=2.0).astype(int)
    co = (do<=1.5).astype(int)
    return sb, nf, co

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

def run_ensemble_cv(rounds, opt_params, alpha):
    n = len(rounds)
    CF = 0.0005
    all_scores = []
    
    for ti in range(n):
        lut_c, lut_n = {}, {}
        fb3_c, fb3_n = {}, {}
        cls_c, cls_n = {}, {}
        for ri, seeds in enumerate(rounds):
            if ri == ti: continue
            for ig, gt in seeds:
                cls = build_class_grid(ig)
                sb, nf, co = compute_features(cls, ig)
                H,W = ig.shape
                for y in range(H):
                    for x in range(W):
                        ic = int(cls[y,x])
                        k4 = (ic,int(sb[y,x]),int(nf[y,x]),int(co[y,x]))
                        k3 = k4[:3]
                        lut_c.setdefault(k4,np.zeros(6)); lut_n.setdefault(k4,0)
                        lut_c[k4] += gt[y,x]; lut_n[k4] += 1
                        fb3_c.setdefault(k3,np.zeros(6)); fb3_n.setdefault(k3,0)
                        fb3_c[k3] += gt[y,x]; fb3_n[k3] += 1
                        cls_c.setdefault(ic,np.zeros(6)); cls_n.setdefault(ic,0)
                        cls_c[ic] += gt[y,x]; cls_n[ic] += 1

        ca = {ic: cls_c[ic]/cls_n[ic] if cls_n.get(ic,0)>0 else np.ones(6)/6 for ic in range(6)}
        fb3 = {}
        for k3,v in fb3_c.items():
            if fb3_n[k3] >= 50:
                a = v/fb3_n[k3]; a = np.clip(a,CF,None); a/=a.sum(); fb3[k3] = a
        lut = {}
        for k4,v in lut_c.items():
            if lut_n[k4] >= 50:
                a = v/lut_n[k4]; a = np.clip(a,CF,None); a/=a.sum(); lut[k4] = a

        for ig, gt in rounds[ti]:
            cls = build_class_grid(ig)
            sb, nf, co = compute_features(cls, ig)
            H,W = ig.shape

            # LUT prediction
            pred_lut = np.ones((H,W,6))/6
            for y in range(H):
                for x in range(W):
                    ic = int(cls[y,x])
                    if ic == 5: pred_lut[y,x] = [0,0,0,0,0,1]; continue
                    k4 = (ic,int(sb[y,x]),int(nf[y,x]),int(co[y,x]))
                    if k4 in lut: pred_lut[y,x] = lut[k4]
                    else:
                        k3 = k4[:3]
                        pred_lut[y,x] = fb3.get(k3, ca.get(ic, np.ones(6)/6))
            pred_lut = np.clip(pred_lut, CF, None)
            pred_lut /= pred_lut.sum(axis=-1, keepdims=True)

            if alpha == 0:
                pred = pred_lut
            else:
                # Cell model prediction
                pred_cell = predict_cell_distributions(ig, opt_params)
                pred_cell = np.clip(pred_cell, CF, None)
                pred_cell /= pred_cell.sum(axis=-1, keepdims=True)
                
                # Log-space ensemble (geometric mean)
                log_blend = (1-alpha)*np.log(pred_lut) + alpha*np.log(pred_cell)
                pred = np.exp(log_blend)
                pred = np.clip(pred, CF, None)
                pred /= pred.sum(axis=-1, keepdims=True)

            all_scores.append(kl_score(pred, gt))
    
    return np.mean(all_scores)

if __name__ == "__main__":
    rounds = load_rounds()
    print(f"Loaded {len(rounds)} rounds")
    
    opt_vec = np.load("data/cell_model_params.npy")
    opt_params = params_from_vector(opt_vec)
    
    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        cv = run_ensemble_cv(rounds, opt_params, alpha)
        marker = " <-- LUT-only baseline" if alpha == 0 else ""
        print(f"  alpha={alpha:.2f}: CV={cv:.3f}{marker}")
    
    print("Done")
