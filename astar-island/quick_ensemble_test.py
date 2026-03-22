"""
Quick cell+HGB ensemble test.
Uses the cell model params from restart 0 (best=88.69) and tests
whether adding cell predictions as features improves HGB.
Only does 3-fold (groups of 7 rounds) instead of full LOO for speed.
"""
import json, time, numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime

DATA_DIR = Path("data")
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6
LOG = "overnight_log.md"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"| {ts} | {msg} |"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def build_class_grid(ig):
    cg = np.zeros_like(ig)
    for gv, cls in GRID_TO_CLASS.items():
        cg[ig == gv] = cls
    return cg

def compute_features_17(ig):
    H, W = ig.shape
    cls = build_class_grid(ig)
    cls_oh = np.zeros((H, W, 6))
    for c in range(6):
        cls_oh[:, :, c] = (cls == c).astype(float)
    ocean, settle, forest, mountain = (ig == 10), (cls == 1)|(cls == 2), (cls == 4), (cls == 5)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    dist_settle = ndimage.distance_transform_edt(~settle) if settle.any() else np.full((H, W), 40.0)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 40.0)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H, W), 40.0)
    k3, k7, k11 = np.ones((3, 3)), np.ones((7, 7)), np.ones((11, 11))
    n_s3 = ndimage.convolve((cls == 1).astype(float), k3, mode='constant')
    n_s7 = ndimage.convolve(settle.astype(float), k7, mode='constant')
    n_f7 = ndimage.convolve(forest.astype(float), k7, mode='constant')
    n_o7 = ndimage.convolve(ocean.astype(float), k7, mode='constant')
    n_e7 = ndimage.convolve((cls == 0).astype(float), k7, mode='constant')
    n_s11 = ndimage.convolve(settle.astype(float), k11, mode='constant')
    is_coast = (dist_ocean <= 1.5).astype(float)
    features = np.concatenate([
        cls_oh, dist_ocean[:,:,None], dist_settle[:,:,None],
        dist_forest[:,:,None], dist_mountain[:,:,None],
        n_s3[:,:,None], n_s7[:,:,None], n_f7[:,:,None],
        n_o7[:,:,None], n_e7[:,:,None], n_s11[:,:,None],
        is_coast[:,:,None],
    ], axis=-1)
    return features.reshape(-1, 17)

def cell_model_predict(ig, params):
    H, W = ig.shape
    cls = build_class_grid(ig)
    settle, forest, ocean = (cls == 1)|(cls == 2), (cls == 4), (ig == 10)
    dist_s = ndimage.distance_transform_edt(~settle) if settle.any() else np.full((H, W), 40.0)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 40.0)
    dist_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    coastal = (dist_o <= 1.5).astype(float)
    sd = ndimage.uniform_filter(settle.astype(float), size=7, mode='constant')
    pred = np.zeros((H, W, 6))
    m5 = (cls == 5); pred[m5, 5] = 1.0
    deep = ocean & (dist_s > 5) & ~m5; pred[deep, 0] = 1.0
    sp = m5 | deep
    sb_e, ss_e, sb_f, ss_f = params[0:4]
    pb, ps, ruin_s, ruin_p = params[4:8]
    sv_b, sv_d, fp_b, fp_p = params[8:12]
    f_e, f_s, port_surv = params[12:15]
    m0 = (cls == 0) & ~sp
    if m0.any():
        ds, df, co = dist_s[m0], dist_f[m0], coastal[m0]
        p = np.stack([np.maximum(0, 1-sb_e*np.exp(-ds/ss_e)-pb*co*np.exp(-ds/ps)-ruin_p-f_e*(1+1/(1+df))),
                      sb_e*np.exp(-ds/ss_e), pb*co*np.exp(-ds/ps), np.full_like(ds,ruin_p),
                      np.maximum(0, f_e*(1+1/(1+df))), np.zeros_like(ds)], -1)
        pred[m0] = p
    m1 = (cls == 1) & ~sp
    if m1.any():
        _sd, co = sd[m1], coastal[m1]
        sv = np.minimum(sv_b+sv_d*_sd, 0.9)
        p = np.stack([np.maximum(0, 1-sv-pb*co*2-ruin_s-f_s*(1-sv*0.5)),
                      sv, pb*co*2, np.full_like(_sd,ruin_s),
                      np.maximum(0, f_s*(1-sv*0.5)), np.zeros_like(_sd)], -1)
        pred[m1] = p
    m2 = (cls == 2) & ~sp
    if m2.any():
        n2 = m2.sum()
        p = np.stack([np.maximum(0, np.full(n2,1-port_surv-0.10-ruin_s-f_s*0.8)),
                      np.full(n2,0.10), np.full(n2,port_surv), np.full(n2,ruin_s),
                      np.maximum(0, np.full(n2,f_s*0.8)), np.zeros(n2)], -1)
        pred[m2] = p
    m3 = (cls == 3) & ~sp
    if m3.any():
        ds = dist_s[m3]
        p = np.stack([np.maximum(0,1-sb_e*np.exp(-ds/ss_e)*1.2-0.3-0.05),
                      sb_e*np.exp(-ds/ss_e)*1.2, np.zeros_like(ds), np.full_like(ds,0.05),
                      np.maximum(0,np.full_like(ds,0.3)), np.zeros_like(ds)], -1)
        pred[m3] = p
    m4 = (cls == 4) & ~sp
    if m4.any():
        ds, co, _sd = dist_s[m4], coastal[m4], sd[m4]
        fs = np.clip(fp_b-fp_p*_sd, 0.3, 0.99)
        p = np.stack([np.maximum(0,1-fs-sb_f*np.exp(-ds/ss_f)-pb*co*np.exp(-ds/ps)*0.5-ruin_p),
                      sb_f*np.exp(-ds/ss_f), pb*co*np.exp(-ds/ps)*0.5, np.full_like(ds,ruin_p),
                      fs, np.zeros_like(ds)], -1)
        pred[m4] = p
    active = m0|m1|m2|m3|m4
    if active.any():
        pred[active] = np.clip(pred[active], 1e-6, None)
        pred[active] /= pred[active].sum(axis=-1, keepdims=True)
    return pred

def kl_score(pred, gt):
    pred = np.clip(pred, 1e-8, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    kl = np.where(gt > 0, gt * np.log(np.clip(gt, 1e-15, None) / pred), 0).sum(axis=-1)
    kl = np.where(np.isfinite(kl), kl, 0)
    return 100.0 * np.exp(-kl.mean())

def main():
    log("QUICK-TEST ensemble start")
    
    # Load GT
    entries = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        rid = gf.stem.replace("ground_truth_", "")
        with open(gf) as f:
            data = json.load(f)
        for si in sorted(data.keys()):
            e = data[si]
            entries.append((rid, int(si), np.array(e["initial_grid"]), np.array(e["ground_truth"])))
    
    rounds = {}
    for rid, si, ig, gt in entries:
        rounds.setdefault(rid, []).append((si, ig, gt))
    round_ids = sorted(rounds.keys())
    log(f"QUICK-TEST {len(entries)} seeds, {len(round_ids)} rounds")

    # Use calibrated cell model params (from fast_sim restart 0: best=88.69)
    # These are approximate values from L-BFGS-B optimization
    # Try loading saved params, else use defaults
    params_file = DATA_DIR / "cell_model_params_fast.json"
    if params_file.exists():
        cell_params = np.array(json.load(open(params_file)))
        log(f"QUICK-TEST loaded cell params from file")
    else:
        # Use default params (will calibrate inline)
        cell_params = np.array([0.35, 3.0, 0.30, 2.5, 0.05, 2.0, 0.025, 0.01,
                                 0.35, 0.5, 0.80, 0.25, 0.03, 0.20, 0.20])
        log(f"QUICK-TEST using default cell params (no saved file)")
        
        # Quick calibration: just L-BFGS-B from default, ~30s
        from scipy.optimize import minimize
        cal_data = [(e[2], e[3]) for e in entries[::5]]
        def neg_score(x):
            return -np.mean([kl_score(cell_model_predict(ig, x), gt) for ig, gt in cal_data])
        bounds = [(0.01,0.8),(1,10),(0.01,0.8),(1,10),(0.001,0.2),(1,10),
                  (0.001,0.1),(0.001,0.05),(0.05,0.8),(0,2),(0.5,0.99),(0,1),
                  (0.001,0.1),(0.05,0.5),(0.05,0.5)]
        result = minimize(neg_score, cell_params, method='L-BFGS-B', bounds=bounds,
                          options={"maxiter": 100})
        cell_params = result.x
        log(f"QUICK-TEST calibrated cell model: {-result.fun:.2f}")

    # Cell model standalone score
    cell_scores = [kl_score(cell_model_predict(ig, cell_params), gt) for _, _, ig, gt in entries]
    log(f"QUICK-TEST cell model standalone: {np.mean(cell_scores):.2f}+/-{np.std(cell_scores):.2f}")

    # 3-fold CV (groups of 7 rounds)
    hgb_params = dict(max_iter=200, max_depth=6, learning_rate=0.05, min_samples_leaf=50, random_state=42)
    np.random.seed(42)
    fold_ids = np.array_split(np.random.permutation(round_ids), 3)

    for config_name, use_cell in [("HGB 17feat", False), ("HGB+cell 23feat", True)]:
        t0 = time.time()
        all_scores = []
        for fi, fold in enumerate(fold_ids):
            hold_set = set(fold)
            X_train, Y_train = [], []
            for rid in round_ids:
                if rid in hold_set:
                    continue
                for si, ig, gt in rounds[rid]:
                    base = compute_features_17(ig)
                    if use_cell:
                        cp = cell_model_predict(ig, cell_params).reshape(-1, 6)
                        base = np.concatenate([base, cp], axis=1)
                    X_train.append(base)
                    Y_train.append(gt.reshape(-1, 6))
            X_train, Y_train = np.vstack(X_train), np.vstack(Y_train)
            
            models = []
            for c in range(6):
                m = HistGradientBoostingRegressor(**hgb_params)
                m.fit(X_train, Y_train[:, c])
                models.append(m)
            
            for rid in fold:
                for si, ig, gt in rounds[rid]:
                    base = compute_features_17(ig)
                    if use_cell:
                        cp = cell_model_predict(ig, cell_params).reshape(-1, 6)
                        base = np.concatenate([base, cp], axis=1)
                    pred = np.zeros((1600, 6))
                    for c, m in enumerate(models):
                        pred[:, c] = m.predict(base)
                    pred = np.clip(pred, 1e-6, None)
                    pred /= pred.sum(axis=1, keepdims=True)
                    all_scores.append(kl_score(pred.reshape(40, 40, 6), gt))
        
        elapsed = time.time() - t0
        mean_s = np.mean(all_scores)
        std_s = np.std(all_scores)
        log(f"QUICK-TEST {config_name}: 3fold={mean_s:.2f}+/-{std_s:.2f} ({elapsed:.0f}s)")

    # Test blending too
    t0 = time.time()
    blend_scores = []
    for fi, fold in enumerate(fold_ids):
        hold_set = set(fold)
        X_train, Y_train = [], []
        for rid in round_ids:
            if rid in hold_set:
                continue
            for si, ig, gt in rounds[rid]:
                base = compute_features_17(ig)
                cp = cell_model_predict(ig, cell_params).reshape(-1, 6)
                X_train.append(np.concatenate([base, cp], axis=1))
                Y_train.append(gt.reshape(-1, 6))
        X_train, Y_train = np.vstack(X_train), np.vstack(Y_train)
        models = []
        for c in range(6):
            m = HistGradientBoostingRegressor(**hgb_params)
            m.fit(X_train, Y_train[:, c])
            models.append(m)
        
        for rid in fold:
            for si, ig, gt in rounds[rid]:
                base = compute_features_17(ig)
                cp_raw = cell_model_predict(ig, cell_params)
                cp = cp_raw.reshape(-1, 6)
                feat = np.concatenate([base, cp], axis=1)
                hgb_pred = np.zeros((1600, 6))
                for c, m in enumerate(models):
                    hgb_pred[:, c] = m.predict(feat)
                hgb_pred = np.clip(hgb_pred, 1e-6, None)
                hgb_pred /= hgb_pred.sum(axis=1, keepdims=True)
                
                for alpha in [0.0, 0.05, 0.10, 0.15, 0.20]:
                    blended = (1-alpha)*hgb_pred.reshape(40,40,6) + alpha*cp_raw
                    blended = np.clip(blended, 1e-6, None)
                    blended /= blended.sum(axis=-1, keepdims=True)
                    s = kl_score(blended, gt)
                    blend_scores.append((alpha, s))
    
    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20]:
        scores_a = [s for a, s in blend_scores if a == alpha]
        log(f"QUICK-TEST blend alpha={alpha:.2f}: 3fold={np.mean(scores_a):.2f}+/-{np.std(scores_a):.2f}")

    log("QUICK-TEST done")


if __name__ == "__main__":
    main()
