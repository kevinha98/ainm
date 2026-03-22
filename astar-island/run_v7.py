"""
Round 2 submission: Distribution-based prediction.
Key insight: Ground truth is a PROBABILITY DISTRIBUTION, not one-hot.
The simulation is stochastic — we must predict the true distribution, not a single outcome.

Strategy:
1. Compute average GT distribution per initial class from Round 1
2. Use observations as Bayesian evidence to shift predictions
3. Use spatial features (distance to ocean) for edge cells
"""
import json
import time
import sys
import numpy as np
from pathlib import Path

from src.settings import DATA_DIR, NUM_CLASSES, MAP_H, MAP_W, GRID_TO_CLASS, CLASS_NAMES
from src.api import AstarAPI
from src.observer import observe_round, build_observed_grid
from src.models import build_class_grid, extract_features, compute_stats

from sklearn.ensemble import HistGradientBoostingRegressor


def load_r1_distributions():
    """Load Round 1 ground truth and compute average distributions."""
    gt_file = DATA_DIR / "ground_truth_71451d74.json"
    ri_file = DATA_DIR / "round_info.json"
    
    if not gt_file.exists():
        print("  WARNING: No Round 1 ground truth available!")
        return None, None
    
    with open(gt_file) as f:
        gt_data = json.load(f)
    
    # Find the round_info file that has the R1 initial states
    # We need to check if current round_info is R1 or R2
    # For R1 analysis, we stored the initial states in the gt_data itself
    r1_initial_grids = {}
    for si in range(5):
        seed_data = gt_data[str(si)]
        if 'initial_grid' in seed_data:
            r1_initial_grids[si] = np.array(seed_data['initial_grid'])
    
    if not r1_initial_grids:
        # Fallback: try round_info (might be overwritten with R2)
        with open(ri_file) as f:
            rd = json.load(f)
        if rd.get('id', '').startswith('71451d74'):
            for si, st in enumerate(rd['initial_states']):
                r1_initial_grids[si] = np.array(st['grid'])
    
    if not r1_initial_grids:
        print("  WARNING: Cannot find R1 initial grids!")
        return None, None
    
    # Collect training data
    X_all, Y_all = [], []
    for si in range(5):
        gt = np.array(gt_data[str(si)]['ground_truth'])
        ig = r1_initial_grids[si]
        cls = build_class_grid(ig)
        
        settle_set = set(map(tuple, np.argwhere(cls == 1).tolist()))
        port_set = set(map(tuple, np.argwhere(cls == 2).tolist()))
        
        X = extract_features(ig, settle_set, port_set)
        Y = gt.reshape(-1, 6)
        X_all.append(X)
        Y_all.append(Y)
    
    return np.vstack(X_all), np.vstack(Y_all)


def train_dist_models(X, Y):
    """Train one regressor per class."""
    models = []
    for c in range(NUM_CLASSES):
        m = HistGradientBoostingRegressor(
            max_iter=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=30, random_state=42
        )
        m.fit(X, Y[:, c])
        pred_c = m.predict(X)
        mse = np.mean((pred_c - Y[:, c]) ** 2)
        print(f"    Class {c} ({CLASS_NAMES[c]}): MSE={mse:.6f}")
        models.append(m)
    return models


def predict_dist(models, X):
    """Predict 6-class distribution."""
    n = X.shape[0]
    pred = np.zeros((n, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        pred[:, c] = models[c].predict(X)
    pred = np.clip(pred, 0.002, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


def bayesian_update(prior, obs_class, obs_strength=2.0):
    """Update distribution with observation evidence.
    
    Observations show ONE realization of the stochastic sim.
    They're like drawing one sample from the GT distribution.
    Use as weak Bayesian evidence — shift prior toward observed class.
    
    obs_strength: how much to weight observation. 
    1.0 = add one pseudo-count to observed class
    2.0 = add two pseudo-counts (moderate update)
    """
    updated = prior.copy()
    # Treat prior as Dirichlet with concentration = obs_strength
    # Add evidence for observed class
    pseudo_total = prior.sum() * (obs_strength + 1)  # effective total
    updated[obs_class] += obs_strength / (obs_strength + 1)
    updated = np.clip(updated, 0.002, None)
    updated /= updated.sum()
    return updated


def main():
    flags = set(sys.argv[1:])
    no_submit = "--no-submit" in flags
    
    t_total = time.time()
    print("=" * 65)
    print("  ASTAR ISLAND -- Distribution Predictor v7")
    print("=" * 65)
    
    api = AstarAPI()
    
    # Step 1: Get round
    print("\n[1] ROUND DETECTION")
    rd = api.get_active_round()
    if rd is None:
        print("  No active round!")
        return
    
    round_id = rd['id']
    n_seeds = rd.get('seeds_count', len(rd.get('initial_states', [])))
    print(f"  Round {rd['round_number']}: {rd['status']} ({rd['map_width']}x{rd['map_height']}, {n_seeds} seeds)")
    print(f"  ID: {round_id[:12]}...")
    
    # Step 2: Load observations
    print("\n[2] OBSERVATIONS")
    obs_file = DATA_DIR / f"observations_{round_id[:8]}.json"
    observations = None
    if obs_file.exists():
        with open(obs_file) as f:
            observations = json.load(f)
        n_obs = sum(len(v) for v in observations.values())
        print(f"  Loaded {n_obs} cached observations")
    else:
        # Try to observe
        budget = api.get_budget()
        remaining = budget["queries_max"] - budget["queries_used"]
        if remaining > 0:
            print(f"  {remaining} queries available -- observing...")
            observations = observe_round(api, rd, steps=50)
        else:
            print("  No observations available")
    
    # Step 3: Compute per-class average distributions from R1 ground truth
    print("\n[3] LOADING R1 GROUND TRUTH DISTRIBUTIONS")
    gt_file = DATA_DIR / "ground_truth_71451d74.json"
    if gt_file.exists():
        with open(gt_file) as f:
            gt_data = json.load(f)
        
        # Compute per-initial-class average GT distributions
        # This way ocean stays ocean, but settlements get their own average
        all_gt_by_class = {c: [] for c in range(NUM_CLASSES)}
        # Also need R1 initial grids — they're stored in gt_data[seed]['initial_grid']
        for si in range(5):
            gt = np.array(gt_data[str(si)]['ground_truth'])
            ig = np.array(gt_data[str(si)]['initial_grid'])
            ic = build_class_grid(ig)
            for y in range(40):
                for x in range(40):
                    c = ic[y, x]
                    all_gt_by_class[c].append(gt[y, x])
        
        class_avg_dist = {}
        for c in range(NUM_CLASSES):
            if all_gt_by_class[c]:
                class_avg_dist[c] = np.mean(all_gt_by_class[c], axis=0)
            else:
                class_avg_dist[c] = np.array([0.63, 0.14, 0.012, 0.011, 0.186, 0.021])
            print(f"  {CLASS_NAMES[c]:>20s} (n={len(all_gt_by_class[c]):5d}): {[round(x,3) for x in class_avg_dist[c].tolist()]}")
        
        # Also per-ocean-distance: cells near ocean boundary are special
        all_gt_ocean_edge = []
        all_gt_ocean_deep = []
        for si in range(5):
            gt = np.array(gt_data[str(si)]['ground_truth'])
            ig = np.array(gt_data[str(si)]['initial_grid'])
            ocean = (ig == 10)
            from scipy import ndimage as ndi
            land = ~ocean
            dist_ocean = ndi.distance_transform_cdt(land, metric='taxicab') if land.any() else np.zeros_like(ig)
            # Ocean cells on the border (adjacent to land)
            for y in range(40):
                for x in range(40):
                    if ocean[y, x]:
                        is_edge = False
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = y+dy, x+dx
                                if 0 <= ny < 40 and 0 <= nx < 40 and not ocean[ny, nx]:
                                    is_edge = True
                        if is_edge:
                            all_gt_ocean_edge.append(gt[y, x])
                        else:
                            all_gt_ocean_deep.append(gt[y, x])
        
        if all_gt_ocean_deep:
            ocean_deep_avg = np.mean(all_gt_ocean_deep, axis=0)
            print(f"  {'Ocean (deep)':>20s} (n={len(all_gt_ocean_deep):5d}): {[round(x,3) for x in ocean_deep_avg.tolist()]}")
        if all_gt_ocean_edge:
            ocean_edge_avg = np.mean(all_gt_ocean_edge, axis=0)
            print(f"  {'Ocean (edge)':>20s} (n={len(all_gt_ocean_edge):5d}): {[round(x,3) for x in ocean_edge_avg.tolist()]}")
    else:
        class_avg_dist = {c: np.array([0.63, 0.14, 0.012, 0.011, 0.186, 0.021]) for c in range(NUM_CLASSES)}
        class_avg_dist[0] = np.array([0.93, 0.03, 0.01, 0.001, 0.02, 0.005])  # ocean
        print("  Using hardcoded avg dist")
    
    # Step 4: Parse seeds and predict
    print("\n[4] PREDICTION")
    grids, all_s, all_p = [], [], []
    for st in rd["initial_states"]:
        g = np.array(st["grid"])
        grids.append(g)
        ss, ps = set(), set()
        for s in st.get("settlements", []):
            if s.get("alive", True):
                ss.add((s["y"], s["x"]))
                if s.get("has_port"):
                    ps.add((s["y"], s["x"]))
        all_s.append(ss)
        all_p.append(ps)
    
    # Build observed grids
    obs_grids, obs_masks = [], []
    if observations:
        for si in range(n_seeds):
            si_str = str(si)
            if si_str in observations and observations[si_str]:
                og, om = build_observed_grid(observations[si_str])
                obs_grids.append(og)
                obs_masks.append(om)
            else:
                obs_grids.append(np.zeros((MAP_H, MAP_W), dtype=int))
                obs_masks.append(np.zeros((MAP_H, MAP_W), dtype=bool))
    
    final = {}
    for si in range(n_seeds):
        g = grids[si]
        cls_grid = build_class_grid(g)
        
        # Base prediction: per-initial-class average distribution
        pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES))
        for y in range(MAP_H):
            for x in range(MAP_W):
                c = cls_grid[y, x]
                pred[y, x] = class_avg_dist.get(c, class_avg_dist[0])
        
        # Bayesian update with observations — SKIP if per-class avg works well
        # CV shows observations HURT score (shift away from stable averages)
        has_obs = len(obs_grids) > si and obs_masks[si].any()
        if has_obs:
            coverage = obs_masks[si].sum() / obs_masks[si].size * 100
            print(f"  Seed {si}: {coverage:.0f}% observed ({len(all_s[si])} settlements)")
            print(f"    SKIPPING observation updates (per-class avg is better alone)")
        else:
            print(f"  Seed {si}: no observations ({len(all_s[si])} settlements)")
        
        # Ensure valid
        pred = np.clip(pred, 0.002, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        
        se = compute_stats(pred, g)
        print(f"    ent={se['ent']:.3f} conf={se['conf']:.3f}")
        
        final[si] = pred
    
    # Step 5: Save
    print(f"\n[5] SAVING")
    improved = []
    for si in range(n_seeds):
        p = final[si]
        improved.append({
            "seed_index": si,
            "probabilities": p.tolist(),
            "model": "distribution_v7",
        })
    with open(DATA_DIR / "improved_predictions.json", "w") as f:
        json.dump(improved, f)
    print("  Saved improved_predictions.json")
    
    # Step 6: Submit
    if not no_submit and rd["status"] == "active":
        print(f"\n[6] SUBMITTING")
        for si in range(n_seeds):
            p = final[si]
            p = np.clip(p, 1e-6, None)
            p /= p.sum(axis=-1, keepdims=True)
            ok, text = api.submit_prediction(round_id, si, p.tolist())
            status = "OK" if ok else f"FAIL: {text[:80]}"
            print(f"  Seed {si}: {status}")
            time.sleep(0.5)
        print("  All seeds submitted!")
    else:
        print(f"\n[6] SKIPPED SUBMISSION")
    
    tt = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  Total time: {tt:.0f}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
