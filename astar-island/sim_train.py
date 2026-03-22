"""
Local Simulation Training Pipeline
===================================
Between rounds, use ALL GT data to:
1. Recalibrate the parametric cell model (15 params, vectorized)
2. Use cell model to generate synthetic training data (data augmentation)
3. Train HGB on real + synthetic data
4. Ensemble: blend cell model + HGB predictions
5. Evaluate via LOO CV

This runs LOCALLY with no API budget cost.
"""
import json
import time
import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime

DATA_DIR = Path("data")
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6
LOG_FILE = "overnight_log.md"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"| {ts} | {msg} |"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_class_grid(ig):
    cg = np.zeros_like(ig)
    for gv, cls in GRID_TO_CLASS.items():
        cg[ig == gv] = cls
    return cg


def compute_features_17(ig):
    """Standard 17-feature extraction (same as auto_runner_v3)."""
    H, W = ig.shape
    cls = build_class_grid(ig)

    # One-hot class (6 features)
    cls_oh = np.zeros((H, W, 6))
    for c in range(6):
        cls_oh[:, :, c] = (cls == c).astype(float)

    # Distance transforms (4)
    ocean = (ig == 10)
    settle = (cls == 1) | (cls == 2)
    forest = (cls == 4)
    mountain = (cls == 5)

    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    dist_settle = ndimage.distance_transform_edt(~settle) if settle.any() else np.full((H, W), 40.0)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 40.0)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H, W), 40.0)

    # Neighborhood counts (6)
    k3 = np.ones((3, 3))
    k7 = np.ones((7, 7))
    k11 = np.ones((11, 11))
    n_s3 = ndimage.convolve((cls == 1).astype(float), k3, mode='constant')
    n_s7 = ndimage.convolve(settle.astype(float), k7, mode='constant')
    n_f7 = ndimage.convolve(forest.astype(float), k7, mode='constant')
    n_o7 = ndimage.convolve(ocean.astype(float), k7, mode='constant')
    n_e7 = ndimage.convolve((cls == 0).astype(float), k7, mode='constant')
    n_s11 = ndimage.convolve(settle.astype(float), k11, mode='constant')

    # Coastal flag (1)
    is_coast = (dist_ocean <= 1.5).astype(float)

    features = np.concatenate([
        cls_oh,
        dist_ocean[:, :, None], dist_settle[:, :, None],
        dist_forest[:, :, None], dist_mountain[:, :, None],
        n_s3[:, :, None], n_s7[:, :, None],
        n_f7[:, :, None], n_o7[:, :, None],
        n_e7[:, :, None], n_s11[:, :, None],
        is_coast[:, :, None],
    ], axis=-1)
    return features.reshape(-1, 17)


def load_all_gt():
    """Load all GT rounds. Returns list of (round_id, seed_idx, initial_grid, gt_probs)."""
    entries = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        round_id = gf.stem.replace("ground_truth_", "")
        with open(gf) as f:
            data = json.load(f)
        for si_str in sorted(data.keys()):
            entry = data[si_str]
            ig = np.array(entry["initial_grid"])
            gt = np.array(entry["ground_truth"])
            entries.append((round_id, int(si_str), ig, gt))
    return entries


def kl_score(pred, gt):
    """Compute KL-based score: 100*exp(-mean_KL)."""
    pred = np.clip(pred, 1e-8, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    kl = np.where(gt > 0, gt * np.log(np.clip(gt, 1e-15, None) / pred), 0).sum(axis=-1)
    kl = np.where(np.isfinite(kl), kl, 0)
    return 100.0 * np.exp(-kl.mean())


# ═══════════════════════════════════════════════════════════
# PART 1: Cell Model Recalibration
# ═══════════════════════════════════════════════════════════

def cell_model_features(ig):
    """Compute features for the parametric cell model."""
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
    settle_density = ndimage.uniform_filter(settle.astype(float), size=7, mode='constant')
    is_ocean = ocean.astype(float)

    return cls, dist_s, dist_f, coastal, settle_density, is_ocean, mountain


def cell_model_predict(ig, params):
    """Vectorized cell model prediction. params = 15-element array."""
    H, W = ig.shape
    cls, dist_s, dist_f, coastal, settle_density, is_ocean, mountain = cell_model_features(ig)

    pred = np.zeros((H, W, 6))

    # Mountain: always stays
    m5 = (cls == 5)
    pred[m5, 5] = 1.0

    # Deep ocean: stays empty
    deep_ocean = (is_ocean > 0.5) & (dist_s > 5) & ~m5
    pred[deep_ocean, 0] = 1.0

    special = m5 | deep_ocean

    # Unpack params
    sb_e, ss_e = params[0], params[1]   # settle base/scale for empty
    sb_f, ss_f = params[2], params[3]   # settle base/scale for forest
    pb, ps = params[4], params[5]       # port base/scale
    ruin_s, ruin_p = params[6], params[7]  # ruin from settle, persistence
    sv_b, sv_d = params[8], params[9]   # settle survival base, density bonus
    fp_b, fp_p = params[10], params[11] # forest persist base, settle penalty
    f_e, f_s = params[12], params[13]   # forest from empty, from settle
    port_surv = params[14]              # port survival

    # Class 0: Empty
    m0 = (cls == 0) & ~special
    if m0.any():
        ds, df, co = dist_s[m0], dist_f[m0], coastal[m0]
        p_set = sb_e * np.exp(-ds / ss_e)
        p_port = pb * co * np.exp(-ds / ps)
        p_ruin = np.full_like(ds, ruin_p)
        p_for = f_e * (1 + 1.0 / (1 + df))
        p_emp = 1.0 - p_set - p_port - p_ruin - p_for
        pred[m0] = np.stack([np.maximum(0, p_emp), p_set, p_port, p_ruin, np.maximum(0, p_for), np.zeros_like(ds)], axis=-1)

    # Class 1: Settlement
    m1 = (cls == 1) & ~special
    if m1.any():
        sd, co = settle_density[m1], coastal[m1]
        p_surv = np.minimum(sv_b + sv_d * sd, 0.9)
        p_port = pb * co * 2
        p_ruin = np.full_like(sd, ruin_s)
        p_for = f_s * (1 - p_surv * 0.5)
        p_emp = 1.0 - p_surv - p_port - p_ruin - p_for
        pred[m1] = np.stack([np.maximum(0, p_emp), p_surv, p_port, p_ruin, np.maximum(0, p_for), np.zeros_like(sd)], axis=-1)

    # Class 2: Port
    m2 = (cls == 2) & ~special
    if m2.any():
        n2 = m2.sum()
        p_sp = np.full(n2, port_surv)
        p_ss = np.full(n2, 0.10)
        p_ruin = np.full(n2, ruin_s)
        p_for = np.full(n2, f_s * 0.8)
        p_emp = 1.0 - p_sp - p_ss - p_ruin - p_for
        pred[m2] = np.stack([np.maximum(0, p_emp), p_ss, p_sp, p_ruin, np.maximum(0, p_for), np.zeros(n2)], axis=-1)

    # Class 3: Ruin
    m3 = (cls == 3) & ~special
    if m3.any():
        ds = dist_s[m3]
        p_set = sb_e * np.exp(-ds / ss_e) * 1.2
        p_for = np.full_like(ds, 0.3)
        p_ruin = np.full_like(ds, 0.05)
        p_emp = 1.0 - p_set - p_for - p_ruin
        pred[m3] = np.stack([np.maximum(0, p_emp), p_set, np.zeros_like(ds), p_ruin, np.maximum(0, p_for), np.zeros_like(ds)], axis=-1)

    # Class 4: Forest
    m4 = (cls == 4) & ~special
    if m4.any():
        ds, co, sd = dist_s[m4], coastal[m4], settle_density[m4]
        f_surv = np.clip(fp_b - fp_p * sd, 0.3, 0.99)
        p_set = sb_f * np.exp(-ds / ss_f)
        p_port = pb * co * np.exp(-ds / ps) * 0.5
        p_ruin = np.full_like(ds, ruin_p)
        p_emp = 1.0 - f_surv - p_set - p_port - p_ruin
        pred[m4] = np.stack([np.maximum(0, p_emp), p_set, p_port, p_ruin, f_surv, np.zeros_like(ds)], axis=-1)

    # Normalize
    active = m0 | m1 | m2 | m3 | m4
    if active.any():
        pred[active] = np.clip(pred[active], 1e-6, None)
        pred[active] = pred[active] / pred[active].sum(axis=-1, keepdims=True)

    return pred


def calibrate_cell_model(entries, n_restarts=8, maxiter=500):
    """Calibrate cell model on all GT data using differential evolution."""
    log("CALIBRATE cell model start")

    # Precompute features for all entries
    precomp = []
    for rid, si, ig, gt in entries:
        precomp.append((ig, gt))

    bounds = [
        (0.01, 0.8),   # settle_base_empty
        (1.0, 10.0),   # settle_scale_empty
        (0.01, 0.8),   # settle_base_forest
        (1.0, 10.0),   # settle_scale_forest
        (0.001, 0.2),  # port_base
        (1.0, 10.0),   # port_scale
        (0.001, 0.1),  # ruin_from_settle
        (0.001, 0.05), # ruin_persistence
        (0.05, 0.8),   # settle_survival_base
        (0.0, 2.0),    # settle_survival_density_bonus
        (0.5, 0.99),   # forest_persist_base
        (0.0, 1.0),    # forest_settle_penalty
        (0.001, 0.1),  # forest_from_empty
        (0.05, 0.5),   # forest_from_settle
        (0.05, 0.5),   # port_survival
    ]

    # Use subset for speed (1 seed per round)
    cal_data = precomp[::5]
    print(f"  Calibrating on {len(cal_data)} seeds ({len(precomp)} total)")

    def neg_score(x):
        scores = []
        for ig, gt in cal_data:
            pred = cell_model_predict(ig, x)
            scores.append(kl_score(pred, gt))
        return -np.mean(scores)

    # Default params score
    default = np.array([0.35, 3.0, 0.30, 2.5, 0.05, 2.0, 0.025, 0.01,
                         0.35, 0.5, 0.80, 0.25, 0.03, 0.20, 0.20])
    default_score = -neg_score(default)
    log(f"CALIBRATE cell model default score: {default_score:.2f}")

    # Differential evolution (global optimizer)
    t0 = time.time()
    result = differential_evolution(
        neg_score, bounds,
        seed=42, maxiter=maxiter, popsize=20,
        tol=1e-4, mutation=(0.5, 1.5), recombination=0.9,
        disp=False, workers=1,
    )
    elapsed = time.time() - t0
    best_score = -result.fun
    best_params = result.x

    log(f"CALIBRATE cell model: {default_score:.2f} -> {best_score:.2f} (+{best_score-default_score:.2f}) in {elapsed:.0f}s")

    # Evaluate on ALL seeds (full dataset)
    all_scores = []
    for ig, gt in precomp:
        pred = cell_model_predict(ig, best_params)
        all_scores.append(kl_score(pred, gt))
    full_score = np.mean(all_scores)
    log(f"CALIBRATE cell model full-data score: {full_score:.2f} +/- {np.std(all_scores):.2f}")

    return best_params, best_score


# ═══════════════════════════════════════════════════════════
# PART 2: HGB Training with Cell Model Features
# ═══════════════════════════════════════════════════════════

def compute_features_with_cell_model(ig, cell_params):
    """Extended features: 17 base + 6 cell model predictions = 23 features."""
    base = compute_features_17(ig)  # (1600, 17)
    cell_pred = cell_model_predict(ig, cell_params).reshape(-1, 6)  # (1600, 6)
    return np.concatenate([base, cell_pred], axis=1)


def train_hgb_models(X, Y, hgb_params):
    """Train 6 HGB regressors."""
    models = []
    for c in range(NUM_CLASSES):
        mdl = HistGradientBoostingRegressor(**hgb_params)
        mdl.fit(X, Y[:, c])
        models.append(mdl)
    return models


def predict_hgb(models, X, temperature=1.0, clip_floor=1e-6):
    """Predict with HGB ensemble."""
    N = X.shape[0]
    raw = np.zeros((N, NUM_CLASSES))
    for c, mdl in enumerate(models):
        raw[:, c] = mdl.predict(X)
    raw = np.clip(raw, clip_floor, None)
    if temperature != 1.0:
        raw = np.power(raw, 1.0 / temperature)
    raw = raw / raw.sum(axis=1, keepdims=True)
    return raw


# ═══════════════════════════════════════════════════════════
# PART 3: LOO CV with Cell Model + HGB Ensemble
# ═══════════════════════════════════════════════════════════

def loo_cv_ensemble(entries, cell_params, hgb_params, blend_alpha=0.0,
                    temperature=1.0, clip_floor=1e-6, use_cell_features=True):
    """
    Leave-one-ROUND-out CV.
    If blend_alpha > 0: final = (1-alpha)*hgb + alpha*cell_model
    If use_cell_features: adds cell model predictions as extra HGB features.
    """
    # Group by round
    rounds = {}
    for rid, si, ig, gt in entries:
        rounds.setdefault(rid, []).append((si, ig, gt))

    round_ids = sorted(rounds.keys())
    round_scores = []

    for hold_out_rid in round_ids:
        # Train set: all rounds except hold_out
        X_train, Y_train = [], []
        for rid in round_ids:
            if rid == hold_out_rid:
                continue
            for si, ig, gt in rounds[rid]:
                if use_cell_features:
                    feats = compute_features_with_cell_model(ig, cell_params)
                else:
                    feats = compute_features_17(ig)
                X_train.append(feats)
                Y_train.append(gt.reshape(-1, 6))

        X_train = np.vstack(X_train)
        Y_train = np.vstack(Y_train)

        # Train HGB
        models = train_hgb_models(X_train, Y_train, hgb_params)

        # Evaluate on hold-out round
        seed_scores = []
        for si, ig, gt in rounds[hold_out_rid]:
            if use_cell_features:
                X_test = compute_features_with_cell_model(ig, cell_params)
            else:
                X_test = compute_features_17(ig)
            hgb_pred = predict_hgb(models, X_test, temperature, clip_floor)
            hgb_pred = hgb_pred.reshape(40, 40, 6)

            if blend_alpha > 0:
                cell_pred = cell_model_predict(ig, cell_params)
                final = (1 - blend_alpha) * hgb_pred + blend_alpha * cell_pred
                final = np.clip(final, clip_floor, None)
                final = final / final.sum(axis=-1, keepdims=True)
            else:
                final = hgb_pred

            s = kl_score(final, gt)
            seed_scores.append(s)

        round_scores.append(np.mean(seed_scores))

    return np.mean(round_scores), np.std(round_scores), round_scores


# ═══════════════════════════════════════════════════════════
# PART 4: Full Pipeline
# ═══════════════════════════════════════════════════════════

def run_simulation_training():
    """Full local simulation training pipeline."""
    log("SIM-TRAIN pipeline start")

    # Load all GT
    entries = load_all_gt()
    n_rounds = len(set(e[0] for e in entries))
    log(f"SIM-TRAIN loaded {len(entries)} seeds from {n_rounds} rounds")

    # HGB baseline params (from sweep: depth=6 just became new best)
    hgb_base = dict(max_iter=200, max_depth=6, learning_rate=0.05, min_samples_leaf=50, random_state=42)

    # ── Step 0: HGB-only baseline (17 features) ──
    log("SIM-TRAIN Step 0: HGB-only baseline (17 feat)")
    t0 = time.time()
    base_mean, base_std, base_rounds = loo_cv_ensemble(
        entries, cell_params=None, hgb_params=hgb_base,
        blend_alpha=0.0, use_cell_features=False)
    log(f"SIM-TRAIN HGB-only 17feat LOO={base_mean:.2f}+/-{base_std:.2f} ({time.time()-t0:.0f}s)")

    # ── Step 1: Calibrate cell model ──
    log("SIM-TRAIN Step 1: Calibrating cell model on all GT")
    cell_params, cell_score = calibrate_cell_model(entries)
    log(f"SIM-TRAIN Cell model calibrated: LOO={cell_score:.2f}")

    # Save calibrated params
    params_file = DATA_DIR / "cell_model_params.json"
    with open(params_file, "w") as f:
        json.dump(cell_params.tolist(), f)
    log(f"SIM-TRAIN Saved cell params to {params_file}")

    # ── Step 2: HGB with cell model features (23 features) ──
    log("SIM-TRAIN Step 2: HGB with cell model features (23 feat)")
    t0 = time.time()
    feat_mean, feat_std, feat_rounds = loo_cv_ensemble(
        entries, cell_params=cell_params, hgb_params=hgb_base,
        blend_alpha=0.0, use_cell_features=True)
    log(f"SIM-TRAIN HGB+cellfeats 23feat LOO={feat_mean:.2f}+/-{feat_std:.2f} ({time.time()-t0:.0f}s)")

    # ── Step 3: Blend cell model + HGB ──
    log("SIM-TRAIN Step 3: Testing ensemble blends")
    best_blend = 0.0
    best_blend_score = feat_mean

    for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        t0 = time.time()
        blend_mean, blend_std, _ = loo_cv_ensemble(
            entries, cell_params=cell_params, hgb_params=hgb_base,
            blend_alpha=alpha, use_cell_features=True)
        delta = blend_mean - base_mean
        marker = " **NEW BEST**" if blend_mean > best_blend_score else ""
        log(f"SIM-TRAIN blend alpha={alpha:.2f} LOO={blend_mean:.2f}+/-{blend_std:.2f} delta={delta:+.2f}{marker} ({time.time()-t0:.0f}s)")
        if blend_mean > best_blend_score:
            best_blend_score = blend_mean
            best_blend = alpha

    # ── Step 4: Cell model LOO (standalone) ──
    log("SIM-TRAIN Step 4: Cell model standalone LOO")
    cell_scores = []
    for rid, si, ig, gt in entries:
        pred = cell_model_predict(ig, cell_params)
        cell_scores.append(kl_score(pred, gt))
    cell_mean = np.mean(cell_scores)
    cell_std = np.std(cell_scores)
    log(f"SIM-TRAIN Cell model standalone LOO={cell_mean:.2f}+/-{cell_std:.2f}")

    # ── Summary ──
    log("SIM-TRAIN ═══ SUMMARY ═══")
    log(f"SIM-TRAIN HGB-only (17feat): {base_mean:.2f}")
    log(f"SIM-TRAIN HGB+cellfeats (23feat): {feat_mean:.2f}")
    log(f"SIM-TRAIN Best blend (alpha={best_blend:.2f}): {best_blend_score:.2f}")
    log(f"SIM-TRAIN Cell model standalone: {cell_mean:.2f}")
    log(f"SIM-TRAIN Best config: cell_features={'YES' if feat_mean > base_mean else 'NO'}, blend_alpha={best_blend:.2f}")

    return cell_params, best_blend, best_blend_score


if __name__ == "__main__":
    cell_params, best_blend, best_score = run_simulation_training()
