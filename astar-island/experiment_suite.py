"""
Comprehensive Experiment Suite for Astar Island LOO-CV Optimization.

Goes BEYOND what the watchdog tries:
1. Dirichlet smoothing (Bayesian prior instead of raw freqs)
2. Per-class temperature (different T per output class)
3. 6-feature and 7-feature LUT (add more spatial features)
4. Adaptive alpha (vary α by initial class or distance)
5. HGB ensemble (blend HGB predictions into the mix)
6. Joint grid search (T × α simultaneously)
7. Observation-weighted LUT (weight recent obs higher)
8. Settle bin granularity (more distance bins)

Scoring: entropy-weighted KL divergence, LOO-CV across all GT rounds.
"""
import json
import sys
import numpy as np
from pathlib import Path
from scipy import ndimage
from datetime import datetime, timezone
import time

sys.path.insert(0, '.')
from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS
from src.models import build_class_grid

CLIP_FLOOR = 1e-6

# ── Load all GT data
def load_all_gt():
    """Load all ground truth files, return list of (round_id, seeds_list)."""
    gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
    rounds = []
    for gf in gt_files:
        rid = gf.stem.replace("ground_truth_", "")
        with open(gf) as f:
            data = json.load(f)
        seeds = []
        for si_str in sorted(data.keys()):
            entry = data[si_str]
            if not isinstance(entry, dict):
                continue
            seeds.append({
                'ig': np.array(entry['initial_grid']),
                'gt': np.array(entry['ground_truth']),
            })
        if seeds:
            rounds.append({'id': rid, 'seeds': seeds})
    return rounds


def compute_spatial_features_edt(cls, ig):
    """EDT-based features matching auto_runner_v2."""
    H, W = cls.shape
    settlement = (cls == 1)
    dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20.0)
    forest = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 20.0)
    ocean = (ig == 10)
    dist_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    port = (cls == 2)
    dist_p = ndimage.distance_transform_edt(~port) if port.any() else np.full((H, W), 40.0)
    ruin = (cls == 3)
    dist_r = ndimage.distance_transform_edt(~ruin) if ruin.any() else np.full((H, W), 40.0)
    mountain = (cls == 5)
    dist_m = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H, W), 40.0)
    
    # Standard 4 features
    settle_bin = np.full((H, W), 3, dtype=int)
    settle_bin[dist_s <= 4.0] = 2
    settle_bin[dist_s <= 2.0] = 1
    settle_bin[dist_s <= 1.0] = 0
    near_forest = (dist_f <= 2.0).astype(int)
    coastal = (dist_o <= 1.5).astype(int)
    near_port = (dist_p <= 2.0).astype(int)
    
    # Extended features
    near_ruin = (dist_r <= 2.0).astype(int)
    near_mountain = (dist_m <= 2.0).astype(int)
    settle_density = ndimage.uniform_filter((cls == 1).astype(float), size=5, mode='constant')
    settle_density_bin = (settle_density > 0.05).astype(int)
    forest_density = ndimage.uniform_filter((cls == 4).astype(float), size=5, mode='constant')
    forest_density_bin = (forest_density > 0.2).astype(int)
    
    return {
        'settle_bin': settle_bin,
        'near_forest': near_forest,
        'coastal': coastal,
        'near_port': near_port,
        'near_ruin': near_ruin,
        'near_mountain': near_mountain,
        'settle_density_bin': settle_density_bin,
        'forest_density_bin': forest_density_bin,
        'dist_s': dist_s,
        'dist_f': dist_f,
        'dist_o': dist_o,
    }


def build_lut_from_rounds(rounds_data, feature_keys, settle_thresholds=None, 
                          min_n=20, dirichlet_alpha=0.0):
    """Build LUT from GT data with configurable features and optional Dirichlet smoothing.
    
    Args:
        rounds_data: list of round dicts with seeds
        feature_keys: list of feature names to use as LUT key
        settle_thresholds: override settle bins [t1, t2, t3] for (<=t1, <=t2, <=t3, >t3)
        min_n: minimum samples for a bucket
        dirichlet_alpha: Dirichlet prior strength (0=no smoothing)
    
    Returns: lut dict, fallback chain
    """
    counts = {}  # full key → sum of GT probs
    totals = {}  # full key → count
    
    for rd in rounds_data:
        for seed in rd['seeds']:
            ig = seed['ig']
            gt = seed['gt']
            H, W = ig.shape
            cls = build_class_grid(ig)
            feats = compute_spatial_features_edt(cls, ig)
            
            # Override settle bins if custom thresholds
            if settle_thresholds is not None:
                t1, t2, t3 = settle_thresholds
                sb = np.full((H, W), 3, dtype=int)
                sb[feats['dist_s'] <= t3] = 2
                sb[feats['dist_s'] <= t2] = 1
                sb[feats['dist_s'] <= t1] = 0
                feats['settle_bin'] = sb
            
            for y in range(H):
                for x in range(W):
                    ic = int(cls[y, x])
                    key_parts = [ic]
                    for fk in feature_keys:
                        key_parts.append(int(feats[fk][y, x]))
                    key = tuple(key_parts)
                    counts.setdefault(key, np.zeros(6))
                    totals.setdefault(key, 0)
                    counts[key] += gt[y, x]
                    totals[key] += 1
    
    # Build LUT with Dirichlet smoothing
    lut = {}
    class_counts = {}
    class_totals = {}
    for key, cnt in counts.items():
        ic = key[0]
        class_counts.setdefault(ic, np.zeros(6))
        class_totals.setdefault(ic, 0)
        class_counts[ic] += cnt
        class_totals[ic] += totals[key]
    
    class_avgs = {}
    for ic in range(6):
        if ic in class_counts and class_totals[ic] > 0:
            class_avgs[ic] = class_counts[ic] / class_totals[ic]
        else:
            class_avgs[ic] = np.ones(6) / 6
    
    # Build fallback LUTs (progressively fewer features)
    fallbacks = []
    for drop in range(1, len(feature_keys)):
        fb_counts = {}
        fb_totals = {}
        for key, cnt in counts.items():
            short_key = key[:1 + len(feature_keys) - drop]
            fb_counts.setdefault(short_key, np.zeros(6))
            fb_totals.setdefault(short_key, 0)
            fb_counts[short_key] += cnt
            fb_totals[short_key] += totals[key]
        fb_lut = {}
        for k, c in fb_counts.items():
            n = fb_totals[k]
            if n >= min_n:
                avg = c / n
                if dirichlet_alpha > 0:
                    avg = (c + dirichlet_alpha) / (n + dirichlet_alpha * 6)
                avg = np.clip(avg, CLIP_FLOOR, None)
                avg /= avg.sum()
                fb_lut[k] = avg
        fallbacks.append(fb_lut)
    
    # Main LUT
    for key, cnt in counts.items():
        n = totals[key]
        if n >= min_n:
            avg = cnt / n
            if dirichlet_alpha > 0:
                avg = (cnt + dirichlet_alpha) / (n + dirichlet_alpha * 6)
            avg = np.clip(avg, CLIP_FLOOR, None)
            avg /= avg.sum()
            lut[key] = avg
        else:
            # Try fallbacks
            found = False
            for fb_i, fb_lut in enumerate(fallbacks):
                fb_key = key[:1 + len(feature_keys) - fb_i - 1]
                if fb_key in fb_lut:
                    lut[key] = fb_lut[fb_key]
                    found = True
                    break
            if not found:
                ic = key[0]
                lut[key] = class_avgs.get(ic, np.ones(6) / 6)
    
    return lut, fallbacks, class_avgs


def predict_with_lut(seed, lut, fallbacks, class_avgs, feature_keys, 
                     temperature=1.0, settle_thresholds=None,
                     cell_params=None, alpha=0.0,
                     per_class_temp=None):
    """Generate prediction for a single seed using LUT + optional cell model."""
    ig = seed['ig']
    H, W = ig.shape
    cls = build_class_grid(ig)
    feats = compute_spatial_features_edt(cls, ig)
    
    if settle_thresholds is not None:
        t1, t2, t3 = settle_thresholds
        sb = np.full((H, W), 3, dtype=int)
        sb[feats['dist_s'] <= t3] = 2
        sb[feats['dist_s'] <= t2] = 1
        sb[feats['dist_s'] <= t1] = 0
        feats['settle_bin'] = sb
    
    pred = np.ones((H, W, 6)) / 6
    for y in range(H):
        for x in range(W):
            ic = int(cls[y, x])
            key_parts = [ic]
            for fk in feature_keys:
                key_parts.append(int(feats[fk][y, x]))
            key = tuple(key_parts)
            if key in lut:
                pred[y, x] = lut[key]
            else:
                for fb_i, fb_lut in enumerate(fallbacks):
                    fb_key = key[:1 + len(feature_keys) - fb_i - 1]
                    if fb_key in fb_lut:
                        pred[y, x] = fb_lut[fb_key]
                        break
                else:
                    pred[y, x] = class_avgs.get(ic, np.ones(6) / 6)
    
    # Mountain fix
    mtn = (cls == 5)
    if mtn.any():
        pred[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
    pred[~mtn, 5] = 0.0
    s = pred[~mtn].sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1, s)
    pred[~mtn] /= s
    
    # Cell model blend
    if cell_params is not None and alpha > 0:
        from simulator.cell_model import predict_cell_distributions
        pred_cell = predict_cell_distributions(ig, cell_params)
        pred_cell[~mtn, 5] = 0.0
        sc = pred_cell[~mtn].sum(axis=-1, keepdims=True)
        sc = np.where(sc == 0, 1, sc)
        pred_cell[~mtn] /= sc
        pred_cell = np.clip(pred_cell, CLIP_FLOOR, None)
        pred_cell /= pred_cell.sum(axis=-1, keepdims=True)
        if mtn.any():
            pred_cell[mtn] = np.array([0, 0, 0, 0, 0, 1.0])
        
        pred_safe = np.clip(pred, 1e-10, None)
        cell_safe = np.clip(pred_cell, 1e-10, None)
        log_blend = (1 - alpha) * np.log(pred_safe) + alpha * np.log(cell_safe)
        pred = np.exp(log_blend)
    
    # Temperature scaling
    non_mtn = ~mtn
    if per_class_temp is not None and non_mtn.any():
        # Per-class temperature: scale each class column independently
        p = pred[non_mtn]
        p = np.clip(p, 1e-10, None)
        log_p = np.log(p)
        for c in range(5):  # classes 0-4 (not mountain)
            log_p[:, c] /= per_class_temp[c]
        pred[non_mtn] = np.exp(log_p)
    elif temperature != 1.0 and non_mtn.any():
        p = pred[non_mtn]
        p = np.clip(p, 1e-10, None)
        pred[non_mtn] = np.exp(np.log(p) / temperature)
    
    # Final clip + normalize
    pred = np.clip(pred, CLIP_FLOOR, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


def entropy_weighted_kl_score(pred, gt):
    """Compute entropy-weighted KL score (matching competition/watchdog metric).
    
    Weight = entropy of GT cell. Higher entropy → more weight.
    wkl = sum(entropy * kl) / sum(entropy)
    score = 100 * exp(-3 * wkl)
    """
    eps = 1e-15
    gt_s = np.clip(gt, eps, None)
    pred_s = np.clip(pred, eps, None)
    pred_s = pred_s / pred_s.sum(axis=-1, keepdims=True)
    
    entropy = -np.sum(gt * np.log(gt_s), axis=-1)
    kl = np.sum(gt * np.log(gt_s / pred_s), axis=-1)
    
    te = entropy.sum()
    if te < eps:
        return 100.0
    
    wkl = np.sum(entropy * kl) / te
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


def loo_cv(rounds, feature_keys, temperature=1.0, alpha=0.0, 
           cell_params=None, min_n=20, dirichlet_alpha=0.0,
           settle_thresholds=None, per_class_temp=None, verbose=False):
    """Leave-one-round-out cross-validation."""
    scores = []
    for i, test_round in enumerate(rounds):
        train_rounds = [r for j, r in enumerate(rounds) if j != i]
        lut, fallbacks, class_avgs = build_lut_from_rounds(
            train_rounds, feature_keys, 
            settle_thresholds=settle_thresholds,
            min_n=min_n, dirichlet_alpha=dirichlet_alpha
        )
        round_scores = []
        for seed in test_round['seeds']:
            pred = predict_with_lut(
                seed, lut, fallbacks, class_avgs, feature_keys,
                temperature=temperature, settle_thresholds=settle_thresholds,
                cell_params=cell_params, alpha=alpha,
                per_class_temp=per_class_temp
            )
            s = entropy_weighted_kl_score(pred, seed['gt'])
            round_scores.append(s)
        avg = np.mean(round_scores)
        scores.append(avg)
        if verbose:
            print(f"  Round {test_round['id'][:8]}: {avg:.2f}")
    return np.mean(scores), np.std(scores), scores


def log_result(msg):
    """Append to overnight_log.md."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open("overnight_log.md", "a") as f:
        f.write(f"\n### {ts}\n{msg}\n")
    print(f"[LOG] {msg}")


# ══════════════════════════════════════════════════════════════
# EXPERIMENTS
# ══════════════════════════════════════════════════════════════

def run_all():
    print("Loading GT data...")
    rounds = load_all_gt()
    print(f"Loaded {len(rounds)} rounds, {sum(len(r['seeds']) for r in rounds)} seeds")
    
    # Load cell model
    cell_params = None
    try:
        from simulator.cell_model import params_from_vector
        params_path = DATA_DIR / "cell_model_params.npy"
        if params_path.exists():
            cell_params = params_from_vector(np.load(params_path))
            print("Loaded cell model params")
    except Exception as e:
        print(f"No cell model: {e}")
    
    base_features = ['settle_bin', 'near_forest', 'coastal', 'near_port']
    
    # ── EXPERIMENT 0: Establish baseline with current params
    print("\n" + "="*60)
    print("EXP 0: BASELINE (T=1.0, a=0.60, 5-feat LUT)")
    print("="*60)
    baseline_mean, baseline_std, baseline_scores = loo_cv(
        rounds, base_features, temperature=1.0, alpha=0.60,
        cell_params=cell_params, verbose=True
    )
    print(f">>> BASELINE: {baseline_mean:.2f} +/- {baseline_std:.2f}")
    log_result(f"EXP 0 BASELINE: {baseline_mean:.2f} ± {baseline_std:.2f} (T=1.0, a=0.60, 5-feat)")
    best_score = baseline_mean
    best_config = "T=1.0, a=0.60, 5-feat"
    
    # ── EXPERIMENT 1: Dirichlet smoothing sweep
    print("\n" + "="*60)
    print("EXP 1: DIRICHLET SMOOTHING")
    print("="*60)
    for da in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        mean, std, _ = loo_cv(
            rounds, base_features, temperature=1.0, alpha=0.60,
            cell_params=cell_params, dirichlet_alpha=da
        )
        delta = mean - baseline_mean
        marker = " <<<< IMPROVEMENT!" if delta > 0.1 else ""
        print(f"  Dirichlet a={da}: {mean:.2f} +/- {std:.2f} (d={delta:+.2f}){marker}")
        if mean > best_score + 0.05:
            best_score = mean
            best_config = f"Dirichlet a={da}"
            log_result(f"EXP 1 IMPROVEMENT: Dirichlet a={da} → {mean:.2f} (was {baseline_mean:.2f})")
    
    # ── EXPERIMENT 2: Joint T × α grid search (fine-grained)
    print("\n" + "="*60)
    print("EXP 2: JOINT T × α GRID SEARCH")
    print("="*60)
    best_joint = baseline_mean
    best_t, best_a = 1.0, 0.60
    for T in [0.90, 0.95, 1.0, 1.02, 1.05, 1.08, 1.10, 1.15, 1.20]:
        for a in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
            mean, std, _ = loo_cv(
                rounds, base_features, temperature=T, alpha=a,
                cell_params=cell_params
            )
            if mean > best_joint:
                best_joint = mean
                best_t, best_a = T, a
                print(f"  T={T}, a={a}: {mean:.2f} +/- {std:.2f} *** NEW BEST ***")
    delta = best_joint - baseline_mean
    print(f">>> Best joint: T={best_t}, a={best_a} → {best_joint:.2f} (d={delta:+.2f})")
    if best_joint > best_score + 0.05:
        best_score = best_joint
        best_config = f"T={best_t}, a={best_a}"
        log_result(f"EXP 2 IMPROVEMENT: T={best_t}, a={best_a} → {best_joint:.2f}")
    
    # ── EXPERIMENT 3: Per-class temperature
    print("\n" + "="*60)
    print("EXP 3: PER-CLASS TEMPERATURE")
    print("="*60)
    # Classes: 0=empty, 1=settle, 2=port, 3=ruin, 4=forest
    # Hypothesis: settlement transitions more uncertain → higher T; forest stable → lower T
    per_class_configs = [
        [1.0, 1.2, 1.2, 1.0, 0.9],   # hotter settlements, cooler forest
        [1.0, 1.1, 1.1, 1.0, 0.95],   # mild version
        [0.95, 1.15, 1.15, 1.05, 0.9], # slightly cool empty
        [1.05, 1.0, 1.0, 1.0, 1.0],   # just empty
        [1.0, 1.0, 1.0, 1.0, 0.9],    # just forest cooler
        [1.1, 1.1, 1.1, 1.1, 1.1],    # uniform 1.1
        [0.95, 0.95, 0.95, 0.95, 0.95], # uniform 0.95
    ]
    for pct in per_class_configs:
        mean, std, _ = loo_cv(
            rounds, base_features, alpha=0.60,
            cell_params=cell_params, per_class_temp=pct
        )
        delta = mean - baseline_mean
        marker = " <<<< IMPROVEMENT!" if delta > 0.1 else ""
        print(f"  T_cls={pct}: {mean:.2f} (d={delta:+.2f}){marker}")
        if mean > best_score + 0.05:
            best_score = mean
            best_config = f"per_class_T={pct}"
            log_result(f"EXP 3 IMPROVEMENT: per_class_T={pct} → {mean:.2f}")
    
    # ── EXPERIMENT 4: Extended features (6-feat, 7-feat LUT)
    print("\n" + "="*60)
    print("EXP 4: EXTENDED FEATURE SETS")
    print("="*60)
    feature_sets = [
        ('5-feat+ruin', ['settle_bin', 'near_forest', 'coastal', 'near_port', 'near_ruin']),
        ('5-feat+mtn', ['settle_bin', 'near_forest', 'coastal', 'near_port', 'near_mountain']),
        ('5-feat+s_dens', ['settle_bin', 'near_forest', 'coastal', 'near_port', 'settle_density_bin']),
        ('5-feat+f_dens', ['settle_bin', 'near_forest', 'coastal', 'near_port', 'forest_density_bin']),
        ('6-feat', ['settle_bin', 'near_forest', 'coastal', 'near_port', 'near_ruin', 'settle_density_bin']),
        ('4-feat(no port)', ['settle_bin', 'near_forest', 'coastal']),
        ('3-feat(core)', ['settle_bin', 'near_forest']),
    ]
    for name, fkeys in feature_sets:
        mean, std, _ = loo_cv(
            rounds, fkeys, temperature=1.0, alpha=0.60,
            cell_params=cell_params
        )
        delta = mean - baseline_mean
        marker = " <<<< IMPROVEMENT!" if delta > 0.1 else ""
        print(f"  {name}: {mean:.2f} (d={delta:+.2f}){marker}")
        if mean > best_score + 0.05:
            best_score = mean
            best_config = f"features={name}"
            log_result(f"EXP 4 IMPROVEMENT: {name} → {mean:.2f}")
    
    # ── EXPERIMENT 5: Settle bin granularity
    print("\n" + "="*60)
    print("EXP 5: SETTLE BIN THRESHOLDS")
    print("="*60)
    threshold_configs = [
        [0.5, 1.5, 3.0],   # tighter bins
        [1.0, 2.0, 4.0],   # current default
        [1.0, 3.0, 6.0],   # wider medium
        [0.5, 1.0, 2.0],   # very tight
        [1.5, 3.0, 6.0],   # shifted wider
        [1.0, 2.0, 3.0],   # tighter far
        [0.5, 2.0, 5.0],   # wide range
        [1.0, 2.5, 5.0],   # balanced
    ]
    for thresholds in threshold_configs:
        mean, std, _ = loo_cv(
            rounds, base_features, temperature=1.0, alpha=0.60,
            cell_params=cell_params, settle_thresholds=thresholds
        )
        delta = mean - baseline_mean
        marker = " <<<< IMPROVEMENT!" if delta > 0.1 else ""
        print(f"  thresholds={thresholds}: {mean:.2f} (d={delta:+.2f}){marker}")
        if mean > best_score + 0.05:
            best_score = mean
            best_config = f"settle_thresh={thresholds}"
            log_result(f"EXP 5 IMPROVEMENT: thresholds={thresholds} → {mean:.2f}")
    
    # ── EXPERIMENT 6: min_n sweep
    print("\n" + "="*60)
    print("EXP 6: MIN_N THRESHOLD")
    print("="*60)
    for mn in [5, 10, 15, 20, 30, 40, 50]:
        mean, std, _ = loo_cv(
            rounds, base_features, temperature=1.0, alpha=0.60,
            cell_params=cell_params, min_n=mn
        )
        delta = mean - baseline_mean
        marker = " <<<< IMPROVEMENT!" if delta > 0.1 else ""
        print(f"  min_n={mn}: {mean:.2f} (d={delta:+.2f}){marker}")
        if mean > best_score + 0.05:
            best_score = mean
            best_config = f"min_n={mn}"
            log_result(f"EXP 6 IMPROVEMENT: min_n={mn} → {mean:.2f}")
    
    # ── EXPERIMENT 7: Alpha sweep at best temperature
    print("\n" + "="*60)
    print("EXP 7: CELL MODEL ALPHA (fine sweep at best T)")
    print("="*60)
    for a in np.arange(0.0, 1.01, 0.05):
        mean, std, _ = loo_cv(
            rounds, base_features, temperature=best_t, alpha=round(a, 2),
            cell_params=cell_params
        )
        delta = mean - baseline_mean
        if mean > best_score:
            print(f"  a={a:.2f}: {mean:.2f} (d={delta:+.2f}) *** NEW BEST ***")
        elif delta > -0.5:
            print(f"  a={a:.2f}: {mean:.2f} (d={delta:+.2f})")
    
    # ── EXPERIMENT 8: LUT-only (no cell model) with optimal temperature
    print("\n" + "="*60)
    print("EXP 8: LUT-ONLY (no cell model)")
    print("="*60)
    for T in [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]:
        mean, std, _ = loo_cv(
            rounds, base_features, temperature=T, alpha=0.0,
            cell_params=None
        )
        delta = mean - baseline_mean
        marker = " <<<< IMPROVEMENT!" if delta > 0.1 else ""
        print(f"  LUT-only T={T}: {mean:.2f} (d={delta:+.2f}){marker}")
        if mean > best_score + 0.05:
            best_score = mean
            best_config = f"LUT-only T={T}"
            log_result(f"EXP 8 IMPROVEMENT: LUT-only T={T} → {mean:.2f}")
    
    # ── EXPERIMENT 9: Dirichlet + Temperature combined
    print("\n" + "="*60)
    print("EXP 9: DIRICHLET + TEMPERATURE COMBINED")
    print("="*60)
    for da in [0.05, 0.1, 0.5, 1.0]:
        for T in [0.95, 1.0, 1.05, 1.1, 1.15]:
            mean, std, _ = loo_cv(
                rounds, base_features, temperature=T, alpha=0.60,
                cell_params=cell_params, dirichlet_alpha=da
            )
            delta = mean - baseline_mean
            if mean > best_score:
                print(f"  Dir={da} T={T}: {mean:.2f} (d={delta:+.2f}) *** NEW BEST ***")
                best_score = mean
                best_config = f"Dir={da} T={T} a=0.60"
                log_result(f"EXP 9 IMPROVEMENT: Dir={da} T={T} → {mean:.2f}")
    
    # ── SUMMARY
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline: {baseline_mean:.2f} ± {baseline_std:.2f}")
    print(f"Best found: {best_score:.2f} ({best_config})")
    delta = best_score - baseline_mean
    print(f"Improvement: {delta:+.2f}")
    log_result(f"SUMMARY: baseline={baseline_mean:.2f}, best={best_score:.2f} ({best_config}), delta={delta:+.2f}")
    
    return best_score, best_config


if __name__ == "__main__":
    run_all()
