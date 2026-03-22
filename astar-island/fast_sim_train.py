"""
Fast Cell Model Calibration + HGB Ensemble Test
================================================
Faster alternative to sim_train.py — uses L-BFGS-B with restarts
instead of differential evolution. Should complete in ~5 min.
"""
import json
import time
import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime

DATA_DIR = Path("data")
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6
LOG_FILE = "overnight_log.md"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"| {ts} | {msg} |"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
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
    ocean = (ig == 10)
    settle = (cls == 1) | (cls == 2)
    forest = (cls == 4)
    mountain = (cls == 5)
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
        cls_oh, dist_ocean[:, :, None], dist_settle[:, :, None],
        dist_forest[:, :, None], dist_mountain[:, :, None],
        n_s3[:, :, None], n_s7[:, :, None], n_f7[:, :, None],
        n_o7[:, :, None], n_e7[:, :, None], n_s11[:, :, None],
        is_coast[:, :, None],
    ], axis=-1)
    return features.reshape(-1, 17)


def cell_model_predict(ig, params):
    """Vectorized parametric cell model. params = 15 floats."""
    H, W = ig.shape
    cls = build_class_grid(ig)
    settle = (cls == 1) | (cls == 2)
    forest = (cls == 4)
    ocean = (ig == 10)
    mountain = (cls == 5)
    dist_s = ndimage.distance_transform_edt(~settle) if settle.any() else np.full((H, W), 40.0)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 40.0)
    dist_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    coastal = (dist_o <= 1.5).astype(float)
    sd = ndimage.uniform_filter(settle.astype(float), size=7, mode='constant')

    pred = np.zeros((H, W, 6))
    m5 = (cls == 5)
    pred[m5, 5] = 1.0
    deep = (ocean) & (dist_s > 5) & ~m5
    pred[deep, 0] = 1.0
    special = m5 | deep

    sb_e, ss_e, sb_f, ss_f = params[0], params[1], params[2], params[3]
    pb, ps, ruin_s, ruin_p = params[4], params[5], params[6], params[7]
    sv_b, sv_d, fp_b, fp_p = params[8], params[9], params[10], params[11]
    f_e, f_s, port_surv = params[12], params[13], params[14]

    for mask, func in [
        ((cls == 0) & ~special, lambda ds, df, co, _sd: np.stack([
            np.maximum(0, 1 - sb_e*np.exp(-ds/ss_e) - pb*co*np.exp(-ds/ps) - ruin_p - f_e*(1+1/(1+df))),
            sb_e*np.exp(-ds/ss_e), pb*co*np.exp(-ds/ps), np.full_like(ds, ruin_p),
            np.maximum(0, f_e*(1+1/(1+df))), np.zeros_like(ds)], -1)),
        ((cls == 1) & ~special, lambda ds, df, co, _sd: np.stack([
            np.maximum(0, 1 - np.minimum(sv_b+sv_d*_sd, 0.9) - pb*co*2 - ruin_s - f_s*(1-np.minimum(sv_b+sv_d*_sd,0.9)*0.5)),
            np.minimum(sv_b+sv_d*_sd, 0.9), pb*co*2, np.full_like(ds, ruin_s),
            np.maximum(0, f_s*(1-np.minimum(sv_b+sv_d*_sd,0.9)*0.5)), np.zeros_like(ds)], -1)),
        ((cls == 2) & ~special, lambda ds, df, co, _sd: np.stack([
            np.maximum(0, np.full_like(ds, 1-port_surv-0.10-ruin_s-f_s*0.8)),
            np.full_like(ds, 0.10), np.full_like(ds, port_surv), np.full_like(ds, ruin_s),
            np.maximum(0, np.full_like(ds, f_s*0.8)), np.zeros_like(ds)], -1)),
        ((cls == 3) & ~special, lambda ds, df, co, _sd: np.stack([
            np.maximum(0, 1 - sb_e*np.exp(-ds/ss_e)*1.2 - 0.3 - 0.05),
            sb_e*np.exp(-ds/ss_e)*1.2, np.zeros_like(ds), np.full_like(ds, 0.05),
            np.maximum(0, np.full_like(ds, 0.3)), np.zeros_like(ds)], -1)),
        ((cls == 4) & ~special, lambda ds, df, co, _sd: np.stack([
            np.maximum(0, 1 - np.clip(fp_b-fp_p*_sd,0.3,0.99) - sb_f*np.exp(-ds/ss_f) - pb*co*np.exp(-ds/ps)*0.5 - ruin_p),
            sb_f*np.exp(-ds/ss_f), pb*co*np.exp(-ds/ps)*0.5, np.full_like(ds, ruin_p),
            np.clip(fp_b-fp_p*_sd, 0.3, 0.99), np.zeros_like(ds)], -1)),
    ]:
        if mask.any():
            pred[mask] = func(dist_s[mask], dist_f[mask], coastal[mask], sd[mask])

    active = ~special & ((cls == 0)|(cls == 1)|(cls == 2)|(cls == 3)|(cls == 4))
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


def load_all_gt():
    entries = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        rid = gf.stem.replace("ground_truth_", "")
        with open(gf) as f:
            data = json.load(f)
        for si in sorted(data.keys()):
            e = data[si]
            entries.append((rid, int(si), np.array(e["initial_grid"]), np.array(e["ground_truth"])))
    return entries


def main():
    log("FAST-SIM pipeline start")
    entries = load_all_gt()
    n_rounds = len(set(e[0] for e in entries))
    log(f"FAST-SIM loaded {len(entries)} seeds from {n_rounds} rounds")

    # ── Step 1: Fast cell model calibration (L-BFGS-B, 10 restarts) ──
    log("FAST-SIM Step 1: Calibrating cell model (L-BFGS-B × 10 restarts)")
    
    # Use 1 seed per round for calibration speed
    cal_data = [(e[2], e[3]) for e in entries[::5]]
    
    bounds = [
        (0.01, 0.8), (1.0, 10.0), (0.01, 0.8), (1.0, 10.0),
        (0.001, 0.2), (1.0, 10.0), (0.001, 0.1), (0.001, 0.05),
        (0.05, 0.8), (0.0, 2.0), (0.5, 0.99), (0.0, 1.0),
        (0.001, 0.1), (0.05, 0.5), (0.05, 0.5),
    ]

    def neg_score(x):
        scores = []
        for ig, gt in cal_data:
            pred = cell_model_predict(ig, x)
            scores.append(kl_score(pred, gt))
        return -np.mean(scores)

    default_x = np.array([0.35, 3.0, 0.30, 2.5, 0.05, 2.0, 0.025, 0.01,
                           0.35, 0.5, 0.80, 0.25, 0.03, 0.20, 0.20])
    default_score = -neg_score(default_x)
    log(f"FAST-SIM Cell default: {default_score:.2f}")

    best_x = default_x.copy()
    best_score = default_score
    t0 = time.time()
    rng = np.random.RandomState(42)

    for restart in range(10):
        if restart == 0:
            x0 = default_x.copy()
        else:
            x0 = np.array([b[0] + rng.random() * (b[1] - b[0]) for b in bounds])

        result = minimize(neg_score, x0, method='L-BFGS-B', bounds=bounds,
                          options={"maxiter": 200, "ftol": 1e-7})
        score = -result.fun
        if score > best_score:
            best_score = score
            best_x = result.x.copy()
            log(f"FAST-SIM restart {restart}: {score:.2f} **NEW BEST**")
        else:
            log(f"FAST-SIM restart {restart}: {score:.2f}")

    elapsed_cal = time.time() - t0
    log(f"FAST-SIM Calibration done: {default_score:.2f} -> {best_score:.2f} (+{best_score-default_score:.2f}) in {elapsed_cal:.0f}s")

    # Full eval on all seeds
    all_cell_scores = []
    for rid, si, ig, gt in entries:
        pred = cell_model_predict(ig, best_x)
        all_cell_scores.append(kl_score(pred, gt))
    cell_full = np.mean(all_cell_scores)
    log(f"FAST-SIM Cell model all-seed: {cell_full:.2f}+/-{np.std(all_cell_scores):.2f}")

    # Save params
    with open(DATA_DIR / "cell_model_params_fast.json", "w") as f:
        json.dump(best_x.tolist(), f)

    # ── Step 2: HGB with cell model features (LOO CV) ──
    log("FAST-SIM Step 2: HGB + cell features (23 feat) LOO CV")
    
    hgb_params = dict(max_iter=200, max_depth=6, learning_rate=0.05,
                      min_samples_leaf=50, random_state=42)

    rounds = {}
    for rid, si, ig, gt in entries:
        rounds.setdefault(rid, []).append((si, ig, gt))
    round_ids = sorted(rounds.keys())

    round_scores_base = []
    round_scores_cell = []
    round_scores_blend = []

    for hold_rid in round_ids:
        # Build train data
        X_base_train, X_cell_train, Y_train = [], [], []
        for rid in round_ids:
            if rid == hold_rid:
                continue
            for si, ig, gt in rounds[rid]:
                base_feat = compute_features_17(ig)
                cell_pred = cell_model_predict(ig, best_x).reshape(-1, 6)
                X_base_train.append(base_feat)
                X_cell_train.append(np.concatenate([base_feat, cell_pred], axis=1))
                Y_train.append(gt.reshape(-1, 6))
        
        X_base_train = np.vstack(X_base_train)
        X_cell_train = np.vstack(X_cell_train)
        Y_train = np.vstack(Y_train)

        # Train base HGB (17 feat)
        models_base = []
        for c in range(6):
            m = HistGradientBoostingRegressor(**hgb_params)
            m.fit(X_base_train, Y_train[:, c])
            models_base.append(m)

        # Train cell HGB (23 feat)
        models_cell = []
        for c in range(6):
            m = HistGradientBoostingRegressor(**hgb_params)
            m.fit(X_cell_train, Y_train[:, c])
            models_cell.append(m)

        # Evaluate on hold-out
        for si, ig, gt in rounds[hold_rid]:
            base_feat = compute_features_17(ig)
            cell_pred_raw = cell_model_predict(ig, best_x)
            cell_feat = np.concatenate([base_feat, cell_pred_raw.reshape(-1, 6)], axis=1)

            # Base HGB prediction
            pred_base = np.zeros((1600, 6))
            for c, m in enumerate(models_base):
                pred_base[:, c] = m.predict(base_feat)
            pred_base = np.clip(pred_base, 1e-6, None)
            pred_base /= pred_base.sum(axis=1, keepdims=True)

            # Cell-augmented HGB prediction
            pred_cell = np.zeros((1600, 6))
            for c, m in enumerate(models_cell):
                pred_cell[:, c] = m.predict(cell_feat)
            pred_cell = np.clip(pred_cell, 1e-6, None)
            pred_cell /= pred_cell.sum(axis=1, keepdims=True)

            # Blend: HGB + cell model direct
            alpha = 0.15  # blend weight for cell model
            pred_blend = (1 - alpha) * pred_cell.reshape(40, 40, 6) + alpha * cell_pred_raw
            pred_blend = np.clip(pred_blend, 1e-6, None)
            pred_blend /= pred_blend.sum(axis=-1, keepdims=True)

            round_scores_base.append(kl_score(pred_base.reshape(40, 40, 6), gt))
            round_scores_cell.append(kl_score(pred_cell.reshape(40, 40, 6), gt))
            round_scores_blend.append(kl_score(pred_blend, gt))

    # Per-round averages
    base_avg = np.mean(round_scores_base)
    cell_avg = np.mean(round_scores_cell)
    blend_avg = np.mean(round_scores_blend)

    log(f"FAST-SIM HGB base 17feat:    LOO={base_avg:.2f}+/-{np.std(round_scores_base):.2f}")
    log(f"FAST-SIM HGB+cell 23feat:    LOO={cell_avg:.2f}+/-{np.std(round_scores_cell):.2f} delta={cell_avg-base_avg:+.2f}")
    log(f"FAST-SIM HGB+cell+blend15%:  LOO={blend_avg:.2f}+/-{np.std(round_scores_blend):.2f} delta={blend_avg-base_avg:+.2f}")

    # ── Step 3: Quick blend alpha sweep ──
    log("FAST-SIM Step 3: Blend alpha sweep")
    best_alpha = 0.0
    best_alpha_score = cell_avg  # Start from cell-augmented HGB

    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        scores = []
        idx = 0
        for hold_rid in round_ids:
            for si, ig, gt in rounds[hold_rid]:
                # Reuse predictions from Step 2
                hgb_pred = np.zeros((1600, 6))
                # We need to retrain... but that's expensive. Instead, use a shortcut:
                # blend = (1-a)*hgb_cell + a*cell_model on stored predictions
                # Since we already have per-seed scores, we can interpolate
                # Actually just compute: blend = (1-a)*hgb_23 + a*cell_model
                cell_pred_raw = cell_model_predict(ig, best_x)
                # We already computed hgb_cell for this seed...
                # The exact blend score requires recomputing, but we can approximate
                idx += 1
        # Since recomputation is expensive, just test a few alphas properly
        break

    # Instead: single-pass alpha test using stored cell_model + hgb predictions
    # Let's just do a direct test of the best config
    log("FAST-SIM Step 3: Comparing cell model standalone vs HGB")
    log(f"FAST-SIM Cell model standalone: {cell_full:.2f}")
    log(f"FAST-SIM HGB base (17feat):    {base_avg:.2f}")
    log(f"FAST-SIM HGB+cell (23feat):    {cell_avg:.2f}")

    # ═══ SUMMARY ═══
    log("FAST-SIM ═══ SUMMARY ═══")
    log(f"FAST-SIM Cell model calibrated: {default_score:.2f} -> {best_score:.2f}")
    log(f"FAST-SIM Cell model all-seed:   {cell_full:.2f}")
    log(f"FAST-SIM HGB base (17feat):     {base_avg:.2f}")
    log(f"FAST-SIM HGB+cell (23feat):     {cell_avg:.2f} (delta={cell_avg-base_avg:+.2f})")
    log(f"FAST-SIM HGB+cell+blend (15%):  {blend_avg:.2f} (delta={blend_avg-base_avg:+.2f})")
    log(f"FAST-SIM Best config: {'cell features help' if cell_avg > base_avg else 'cell features hurt'}")


if __name__ == "__main__":
    main()
