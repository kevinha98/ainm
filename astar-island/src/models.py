"""
Astar Island — Prediction Models
=================================
Each model takes initial state and returns H×W×6 probability predictions.
Models are designed to capture different aspects of the simulation dynamics.
"""
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from src.settings import NUM_CLASSES, MAP_H, MAP_W, GRID_TO_CLASS, CLASS_NAMES


def build_class_grid(grid_np):
    """Convert raw grid values to prediction classes (0-5)."""
    cg = np.zeros_like(grid_np)
    for gv, cls in GRID_TO_CLASS.items():
        cg[grid_np == gv] = cls
    return cg


# ── Spatial feature extraction ────────────────────────────────

def extract_features(grid_np, settle_set, port_set):
    """Extract per-cell features for ML models."""
    H, W = grid_np.shape
    feats = []

    # One-hot terrain type
    for v in [0, 1, 2, 3, 4, 5, 10, 11]:
        feats.append((grid_np == v).astype(float).ravel())

    cls_grid = build_class_grid(grid_np)
    feats.append(cls_grid.ravel().astype(float))

    # Spatial position
    yy, xx = np.meshgrid(np.arange(H) / H, np.arange(W) / W, indexing='ij')
    feats.append(yy.ravel())
    feats.append(xx.ravel())
    feats.append((np.minimum(yy, 1 - yy) * 2).ravel())  # distance from edge
    feats.append((np.minimum(xx, 1 - xx) * 2).ravel())

    # Neighborhood composition at multiple radii
    for v in [0, 1, 2, 3, 4, 5, 10, 11]:
        mask = (grid_np == v).astype(float)
        for radius in [1, 2, 3, 5]:
            size = 2 * radius + 1
            kernel = np.ones((size, size))
            kernel[radius, radius] = 0
            count = ndimage.convolve(mask, kernel, mode='constant', cval=0)
            total = ndimage.convolve(np.ones_like(mask), kernel, mode='constant', cval=0)
            feats.append((count / np.maximum(total, 1)).ravel())

    # Distance transforms
    settle_mask = np.zeros((H, W), dtype=bool)
    for sy, sx in settle_set:
        if 0 <= sy < H and 0 <= sx < W:
            settle_mask[sy, sx] = True
    dist_s = ndimage.distance_transform_cdt(~settle_mask, metric='taxicab').astype(float) if settle_mask.any() else np.full((H, W), 80.0)
    feats.append((dist_s / 80).ravel())
    feats.append(np.exp(-dist_s / 5).ravel())

    port_mask = np.zeros((H, W), dtype=bool)
    for sy, sx in port_set:
        if 0 <= sy < H and 0 <= sx < W:
            port_mask[sy, sx] = True
    dist_p = ndimage.distance_transform_cdt(~port_mask, metric='taxicab').astype(float) if port_mask.any() else np.full((H, W), 80.0)
    feats.append((dist_p / 80).ravel())
    feats.append(np.exp(-dist_p / 8).ravel())

    ocean_mask = (grid_np == 10)
    dist_o = ndimage.distance_transform_cdt(~ocean_mask, metric='taxicab').astype(float) if ocean_mask.any() else np.full((H, W), 80.0)
    feats.append((dist_o / 80).ravel())
    feats.append((dist_o <= 2).astype(float).ravel())  # coastal

    forest_mask = (grid_np == 4)
    dist_f = ndimage.distance_transform_cdt(~forest_mask, metric='taxicab').astype(float) if forest_mask.any() else np.full((H, W), 80.0)
    feats.append((dist_f / 80).ravel())

    # Density features
    feats.append(ndimage.uniform_filter(settle_mask.astype(float), size=5, mode='constant').ravel())
    feats.append(ndimage.uniform_filter(settle_mask.astype(float), size=9, mode='constant').ravel())
    feats.append(ndimage.uniform_filter(ocean_mask.astype(float), size=5, mode='constant').ravel())
    feats.append(ndimage.uniform_filter(forest_mask.astype(float), size=5, mode='constant').ravel())
    feats.append(ndimage.uniform_filter(forest_mask.astype(float), size=9, mode='constant').ravel())
    feats.append(ndimage.uniform_filter((grid_np == 5).astype(float), size=5, mode='constant').ravel())

    # Edge features
    ocean_f = ocean_mask.astype(float)
    feats.append(((ndimage.maximum_filter(ocean_f, size=3) - ocean_f) * (~ocean_mask).astype(float)).ravel())
    forest_f = forest_mask.astype(float)
    feats.append(((ndimage.maximum_filter(forest_f, size=3) - forest_f) * (~forest_mask).astype(float)).ravel())

    return np.column_stack(feats)


# ═══════════════════════════════════════════════════════════════
# MODEL 1: Data-Driven Markov (uses learned transitions if available)
# ═══════════════════════════════════════════════════════════════

def model_markov(grid_np, settle_set, port_set, learned_T=None):
    """Context-aware Markov chain model.
    If learned_T is provided, uses real transition data as base."""
    H, W = grid_np.shape
    pred = np.zeros((H, W, NUM_CLASSES))
    ocean_mask = (grid_np == 10)
    cls_grid = build_class_grid(grid_np)

    # Settlement neighborhood
    s_mask = np.zeros((H, W), dtype=float)
    for sy, sx in settle_set:
        if 0 <= sy < H and 0 <= sx < W:
            s_mask[sy, sx] = 1.0
    sn = ndimage.uniform_filter(s_mask, size=7, mode='constant') * 49
    sn1 = ndimage.uniform_filter(s_mask, size=3, mode='constant') * 9

    # Port proximity
    p_mask = np.zeros((H, W), dtype=float)
    for sy, sx in port_set:
        if 0 <= sy < H and 0 <= sx < W:
            p_mask[sy, sx] = 1.0
    port_near = ndimage.maximum_filter(p_mask, size=7) > 0

    # Forest neighborhood
    fn = ndimage.uniform_filter((grid_np == 4).astype(float), size=5, mode='constant') * 25

    # Coastal adjacency
    coastal = (ndimage.maximum_filter(ocean_mask.astype(float), size=3) > 0) & (~ocean_mask)

    # Base transition matrix (learned or heuristic)
    if learned_T is not None:
        BASE_T = learned_T.copy()
    else:
        # Calibrated from Round 1 ground truth (8000 cells, 5 seeds)
        BASE_T = np.array([
            # Empty    Settl   Port    Ruin    Forest  Mount
            [0.995, 0.005, 0.000, 0.000, 0.000, 0.000],  # From Empty/Plains
            [0.413, 0.572, 0.015, 0.000, 0.000, 0.000],  # From Settlement
            [0.857, 0.000, 0.143, 0.000, 0.000, 0.000],  # From Port
            [0.500, 0.000, 0.000, 0.500, 0.000, 0.000],  # From Ruin (no data, guess)
            [0.000, 0.012, 0.000, 0.000, 0.988, 0.000],  # From Forest
            [0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # From Mountain
        ])

    for y in range(H):
        for x in range(W):
            if ocean_mask[y, x]:
                pred[y, x] = [0.998, 0.0004, 0.0002, 0.0002, 0.0006, 0.0006]
                continue

            cls = cls_grid[y, x]
            p = BASE_T[cls].copy()
            ns = sn[y, x]
            is_coast = coastal[y, x]
            near_port = port_near[y, x]
            nf = fn[y, x]

            if cls == 0:  # Plains — 99.5% stay same
                if ns > 0:
                    growth = min(ns * 0.003, 0.02)
                    p[1] += growth; p[0] -= growth
                if is_coast and ns > 0:
                    p[2] += 0.002; p[0] -= 0.002

            elif cls == 1:  # Settlement — 57% stay, 41% → Empty, 1.5% → Port
                if sn1[y, x] > 1:
                    stab = min(sn1[y, x] * 0.02, 0.10)
                    p[1] += stab; p[0] -= stab
                if is_coast:
                    p[2] += 0.03; p[0] -= 0.03
                if sn1[y, x] <= 1 and not near_port:
                    p[0] += 0.05; p[1] -= 0.05

            elif cls == 2:  # Port — 85.7% → Empty, 14.3% stay
                if ns > 2:
                    p[2] += 0.05; p[0] -= 0.05

            elif cls == 3:  # Ruin — no data, assume decays to empty
                pass  # Use base rates

            elif cls == 4:  # Forest — 98.8% stay, 1.2% → Settlement
                if ns > 0:
                    clear = min(ns * 0.003, 0.02)
                    p[1] += clear; p[4] -= clear

            p = np.clip(p, 1e-6, None)
            p /= p.sum()
            pred[y, x] = p

    return pred


# ═══════════════════════════════════════════════════════════════
# MODEL 2: Monte Carlo Simulation (vectorized, dynamic neighborhoods)
# ═══════════════════════════════════════════════════════════════

def _compute_mc_context(cls_state, ocean_mask, immutable):
    """Fully vectorized: compute per-cell stay probs and transition CDFs."""
    H, W = cls_state.shape
    s_mask = (cls_state == 1).astype(float)
    s_count = ndimage.uniform_filter(s_mask, size=5, mode='constant') * 25
    coastal = (ndimage.maximum_filter(ocean_mask.astype(float), size=3) > 0) & (~ocean_mask)
    f_mask = (cls_state == 4).astype(float)
    f_count = ndimage.uniform_filter(f_mask, size=5, mode='constant') * 25
    settle_near = ndimage.maximum_filter(s_mask, size=7) > 0

    stay = np.full((H, W), 0.9996)  # Reality: 99.5% stay after 50 steps → ~0.9999/step
    stay[ocean_mask] = 1.0
    stay[immutable & ~ocean_mask] = 1.0  # Mountains: 100% stay
    stay[(cls_state == 4) & ~settle_near] = 0.99998  # Forest: 98.8% stay after 50 → very stable
    stay[(cls_state == 4) & settle_near] = 0.9997
    stay[(cls_state == 1) & (s_count > 2)] = 0.989  # Settlement with neighbors: ~57% survive 50 steps
    stay[(cls_state == 1) & (s_count <= 2)] = 0.982  # Isolated settlement: more likely to die
    stay[cls_state == 2] = 0.962  # Port: only 14.3% survive → high change rate
    stay[cls_state == 3] = 0.990

    # Vectorized transition distributions per class (when change happens)
    # Calibrated from Round 1 ground truth
    base = {
        0: np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),     # Empty → Settlement (only observed transition)
        1: np.array([0.96, 0.0, 0.04, 0.0, 0.0, 0.0]),    # Settlement → mostly Empty, some Port
        2: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),      # Port → Empty
        3: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),      # Ruin → Empty (guess)
        4: np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),      # Forest → Settlement (only observed)
        5: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),      # Mountain → Mountain (never changes)
    }

    trans = np.zeros((H, W, NUM_CLASSES))
    for cls_val, bd in base.items():
        mask = (cls_state == cls_val) & ~immutable
        if not mask.any():
            continue
        trans[mask] = bd

    # Context-dependent adjustments (vectorized, SMALL — reality is very stable)
    ns = s_count
    has_s = ns > 0

    # Class 0 (Empty/Plains): rare transition to settlement near existing ones
    m0 = (cls_state == 0) & ~immutable
    s_growth = np.clip(ns * 0.005, 0, 0.03)
    trans[m0, 1] += (s_growth * has_s)[m0]
    trans[m0, 0] -= (s_growth * has_s)[m0]

    # Class 1 (Settlement): coastal ones more likely to become port
    m1 = (cls_state == 1) & ~immutable
    trans[m1 & coastal, 2] += 0.10
    trans[m1 & coastal, 0] -= 0.10

    # Class 4 (Forest): near settlements, tiny chance → settlement
    m4 = (cls_state == 4) & ~immutable
    trans[m4 & has_s, 1] += np.clip(ns * 0.005, 0, 0.03)[m4 & has_s]
    trans[m4 & has_s, 0] -= np.clip(ns * 0.005, 0, 0.03)[m4 & has_s]

    # Normalize
    active = ~immutable
    trans[active] = np.clip(trans[active], 1e-6, None)
    row_sums = trans[active].sum(axis=-1, keepdims=True)
    trans[active] /= row_sums

    cdf = np.cumsum(trans, axis=-1)
    return stay, cdf


def model_monte_carlo(grid_np, settle_set, port_set, n_sims=2000, n_steps=50,
                       recompute_every=10):
    """Vectorized MC with DYNAMIC neighborhood recomputation.

    Every `recompute_every` steps, re-evaluates settlement/forest neighborhoods
    from the current simulated state. Captures cascading growth/decline effects.
    """
    H, W = grid_np.shape
    ocean_mask = (grid_np == 10)
    mtn_mask = (grid_np == 5)
    immutable = ocean_mask | mtn_mask
    cls_grid = build_class_grid(grid_np)

    # Initial context from the starting state
    stay, cdf = _compute_mc_context(cls_grid, ocean_mask, immutable)

    counts = np.zeros((H, W, NUM_CLASSES))
    rng = np.random.default_rng(42)

    for sim in range(n_sims):
        state = cls_grid.copy()
        cur_stay, cur_cdf = stay, cdf

        for step in range(n_steps):
            # Re-evaluate neighborhood context periodically
            if step > 0 and step % recompute_every == 0:
                cur_stay, cur_cdf = _compute_mc_context(state, ocean_mask, immutable)

            r = rng.random((H, W))
            changes = (r > cur_stay) & (~immutable)
            if not changes.any():
                continue
            cy, cx = np.where(changes)
            r2 = rng.random(len(cy))
            cell_cdfs = cur_cdf[cy, cx]
            new_classes = (r2[:, None] > cell_cdfs).sum(axis=1)
            new_classes = np.clip(new_classes, 0, NUM_CLASSES - 1)
            state[cy, cx] = new_classes

        for c in range(NUM_CLASSES):
            counts[:, :, c] += (state == c)

    pred = counts / n_sims
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


# ═══════════════════════════════════════════════════════════════
# MODEL 3: HistGradientBoosting (trained on all seeds)
# ═══════════════════════════════════════════════════════════════

def model_hgb(grids, all_s, all_p, obs_grids=None, obs_masks=None):
    """Gradient boosting classifier. If observation data available, trains on real transitions."""
    Xs, ys = [], []

    if obs_grids is not None and obs_masks is not None:
        # Train on REAL transitions: initial features → observed final class
        for si in range(len(grids)):
            if si < len(obs_grids) and obs_masks[si].any():
                X = extract_features(grids[si], all_s[si], all_p[si])
                y_true = build_class_grid(obs_grids[si]).ravel()
                mask = obs_masks[si].ravel()
                Xs.append(X[mask])
                ys.append(y_true[mask])
        if Xs:
            print(f"    Using {sum(len(x) for x in Xs)} observed cells for training")

    if not Xs:
        # Fallback: train on initial state (predicts no change as baseline)
        for si in range(len(grids)):
            Xs.append(extract_features(grids[si], all_s[si], all_p[si]))
            ys.append(build_class_grid(grids[si]).ravel())

    X, y = np.vstack(Xs), np.concatenate(ys)
    print(f"    {X.shape[0]} x {X.shape[1]} features")

    clf = HistGradientBoostingClassifier(
        max_iter=400, max_depth=8, min_samples_leaf=5,
        learning_rate=0.08, l2_regularization=0.1,
        class_weight='balanced', random_state=42
    )
    clf.fit(X, y)
    print(f"    Train acc: {clf.score(X, y):.4f}")

    preds = {}
    n = MAP_H * MAP_W
    all_X = np.vstack([extract_features(grids[si], all_s[si], all_p[si]) for si in range(len(grids))])
    for si in range(len(grids)):
        probs = clf.predict_proba(all_X[si*n:(si+1)*n])
        pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES))
        for ci, c in enumerate(clf.classes_):
            pred[:, :, int(c)] = probs[:, ci].reshape(MAP_H, MAP_W)
        drift = 0.08
        pred = (1 - drift) * pred + drift * (np.ones(NUM_CLASSES) / NUM_CLASSES)
        pred = np.clip(pred, 1e-6, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        preds[si] = pred

    return preds, clf


# ═══════════════════════════════════════════════════════════════
# MODEL 4: MRF Spatial Smoothing
# ═══════════════════════════════════════════════════════════════

def model_mrf(base, grid_np, iters=20, temp=0.20):
    """Markov Random Field smoothing — enforces spatial coherence."""
    pred = base.copy()
    om = (grid_np == 10)
    mm = (grid_np == 5)
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float) / 4
    for _ in range(iters):
        nb = np.stack([ndimage.convolve(pred[:, :, c], k, mode='nearest') for c in range(NUM_CLASSES)], axis=-1)
        new = (1 - temp) * pred + temp * nb
        new[om] = pred[om]
        new[mm] = pred[mm]
        new = np.clip(new, 1e-6, None)
        new /= new.sum(axis=-1, keepdims=True)
        pred = new
    return pred


# ═══════════════════════════════════════════════════════════════
# MODEL 5: Settlement Cellular Automata
# ═══════════════════════════════════════════════════════════════

def model_settlement_ca(grid_np, settle_set, port_set, n_steps=50, n_runs=200):
    """Vectorized CA calibrated to Round 1 reality: 57% settlements survive,
    41% die to Empty, 1.5% become Port. No Ruins observed."""
    H, W = grid_np.shape
    ocean_mask = (grid_np == 10)
    mtn_mask = (grid_np == 5)
    immutable = ocean_mask | mtn_mask
    cls_grid = build_class_grid(grid_np)
    coastal = (ndimage.maximum_filter(ocean_mask.astype(float), size=3) > 0) & (~ocean_mask)

    init_settle = np.zeros((H, W), dtype=bool)
    for sy, sx in settle_set:
        if 0 <= sy < H and 0 <= sx < W:
            init_settle[sy, sx] = True

    counts = np.zeros((H, W, NUM_CLASSES))
    rng = np.random.default_rng(123)
    k3 = np.ones((3, 3)); k3[1, 1] = 0

    for run in range(n_runs):
        state = cls_grid.copy()
        alive = init_settle.copy()

        for step in range(n_steps):
            s_neighbors = ndimage.convolve(alive.astype(float), k3, mode='constant', cval=0)

            # Settlement survival: ~57% survive 50 steps → per-step ~98.9%
            surv_prob = np.full((H, W), 0.989)
            surv_prob += np.clip(s_neighbors * 0.002, 0, 0.008)  # neighbors help slightly
            surv_prob[coastal] += 0.002  # coastal slightly more stable
            surv_prob = np.clip(surv_prob, 0, 0.998)

            r = rng.random((H, W))
            deaths = alive & (r > surv_prob)
            state[deaths] = 0  # Die to EMPTY (not Ruin — reality shows 0 ruins)
            alive = alive & ~deaths

            # Expansion: very rare — only 0.5% of plains become settlement
            expandable = (~immutable) & (~alive) & (state == 0)
            expand_chance = np.clip(s_neighbors * 0.0003, 0, 0.002)
            r = rng.random((H, W))
            new_settle = expandable & (r < expand_chance)
            state[new_settle] = 1
            alive = alive | new_settle

            # Port formation: very rare (1.5% of settlements → port over 50 steps)
            r = rng.random((H, W))
            new_port = alive & coastal & (state == 1) & (r < 0.0003)
            state[new_port] = 2

        for c in range(NUM_CLASSES):
            counts[:, :, c] += (state == c)

    pred = counts / n_runs
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


# ═══════════════════════════════════════════════════════════════
# MODEL 6: Observation-Informed Direct (when we have ground truth viewports)
# ═══════════════════════════════════════════════════════════════

def model_observation_direct(obs_grid, obs_mask, grid_np):
    """Directly use observation data where available, fall back to prior elsewhere."""
    H, W = grid_np.shape
    pred = np.zeros((H, W, NUM_CLASSES))
    cls_grid = build_class_grid(grid_np)
    ocean_mask = (grid_np == 10)
    mtn_mask = (grid_np == 5)

    for y in range(H):
        for x in range(W):
            if obs_mask[y, x]:
                # Observed cell: this IS the simulated future — very high confidence
                obs_cls = GRID_TO_CLASS.get(obs_grid[y, x], 0)
                pred[y, x] = 0.0002
                pred[y, x, obs_cls] = 0.999
            elif ocean_mask[y, x]:
                pred[y, x] = [0.998, 0.0004, 0.0002, 0.0002, 0.0006, 0.0006]
            elif mtn_mask[y, x]:
                pred[y, x] = [0.001, 0.001, 0.001, 0.001, 0.001, 0.995]
            else:
                # Unobserved: use learned transition base
                init_cls = cls_grid[y, x]
                pred[y, x] = 0.002
                pred[y, x, init_cls] = 0.99

    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    # Skip MRF smoothing — observations are ground truth, don't blur them
    return pred


# ═══════════════════════════════════════════════════════════════
# Ensemble & Calibration
# ═══════════════════════════════════════════════════════════════

def ensemble(preds, weights):
    """Weighted ensemble of model predictions."""
    c = np.zeros_like(next(iter(preds.values())))
    tw = sum(weights.values())
    for name, pred in preds.items():
        if name in weights:
            c += (weights[name] / tw) * pred
    c = np.clip(c, 1e-6, None)
    c /= c.sum(axis=-1, keepdims=True)
    return c


def calibrate(pred, grid_np):
    """Terrain-aware calibration: push certainty based on cell dynamics."""
    H, W = grid_np.shape
    ocean = (grid_np == 10)
    mountain = (grid_np == 5)
    cls_grid = build_class_grid(grid_np)

    # Ocean: virtually immutable
    pred[ocean] = 0.0004
    pred[ocean, 0] = 0.998

    # Mountain: virtually immutable
    pred[mountain] = 0.001
    pred[mountain, 5] = 0.995

    # Isolated forests (no settlements within radius 5): very stable
    s_mask = (cls_grid == 1).astype(float)
    s_near = ndimage.maximum_filter(s_mask, size=11) > 0
    isolated_forest = (cls_grid == 4) & ~s_near
    if isolated_forest.any():
        boost = 0.15
        pred[isolated_forest, 4] += boost
        for c in [0, 1, 2, 3, 5]:
            pred[isolated_forest, c] *= (1 - boost / 5)

    # Plains far from everything: also quite stable
    f_mask = (cls_grid == 4).astype(float)
    f_near = ndimage.maximum_filter(f_mask, size=7) > 0
    isolated_plains = (cls_grid == 0) & ~s_near & ~f_near & ~ocean & ~mountain
    if isolated_plains.any():
        boost = 0.10
        pred[isolated_plains, 0] += boost
        for c in [1, 2, 3, 4, 5]:
            pred[isolated_plains, c] *= (1 - boost / 5)

    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


def sharpen(pred, temperature=0.85, grid_np=None):
    """Per-cell temperature sharpening. Stable cells get sharper predictions."""
    if grid_np is not None:
        # Adaptive temperature: stable cells → sharper, dynamic cells → softer
        cls_grid = build_class_grid(grid_np)
        ocean = (grid_np == 10)
        mountain = (grid_np == 5)
        s_mask = (cls_grid == 1).astype(float)
        s_near = ndimage.maximum_filter(s_mask, size=9) > 0

        temp_map = np.full(pred.shape[:2], temperature)
        temp_map[ocean] = 0.3       # Very sharp — near certain
        temp_map[mountain] = 0.3    # Very sharp
        temp_map[s_near & ~ocean & ~mountain] = 0.95  # Softer near settlements (uncertain)

        log_p = np.log(pred + 1e-15) / temp_map[..., None]
    else:
        log_p = np.log(pred + 1e-15) / temperature

    log_p -= log_p.max(axis=-1, keepdims=True)
    p = np.exp(log_p)
    p /= p.sum(axis=-1, keepdims=True)
    return p


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def log_loss(pred, grid_np):
    cg = build_class_grid(grid_np)
    H, W = grid_np.shape
    p = pred[np.arange(H)[:, None], np.arange(W)[None, :], cg]
    return -np.log(np.clip(p, 1e-15, None)).mean()


def entropy(pred):
    return -np.sum(pred * np.log(pred + 1e-15), axis=-1).mean()


def compute_stats(pred, grid_np):
    am = np.argmax(pred, axis=-1)
    return {
        "dist": {CLASS_NAMES[c]: int((am == c).sum()) for c in range(NUM_CLASSES)},
        "conf": float(pred.max(axis=-1).mean()),
        "ent": float(entropy(pred)),
        "ll": float(log_loss(pred, grid_np)),
    }
