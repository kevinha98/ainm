"""
Dry-run test: simulate auto_runner's observe_and_calibrate flow
using a completed round (R4) to verify it works end-to-end.
Does NOT make API calls — uses cached GT to simulate observations.
"""
import json, numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("data")
NC = 6
GRID_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:0, 11:0}
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
CLIP = 0.0001

def build_class_grid(ig):
    cls = np.zeros_like(ig)
    for raw, c in GRID_TO_CLASS.items(): cls[ig == raw] = c
    return cls

def extract_features(ig):
    cls = build_class_grid(ig); H, W = ig.shape
    ocean = (ig == 10); mountain = (ig == 5)
    settlement = (cls == 1); forest = (cls == 4); empty = (cls == 0)
    is_coast = ~ocean & (ndimage.maximum_filter(ocean.astype(float), size=3) > 0)
    dist_ocean = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H,W), 20)
    dist_settle = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H,W), 20)
    dist_forest = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H,W), 20)
    dist_mountain = ndimage.distance_transform_edt(~mountain) if mountain.any() else np.full((H,W), 20)
    k3, k7, k11 = np.ones((3,3)), np.ones((7,7)), np.ones((11,11))
    features = np.concatenate([
        np.eye(NC)[cls],
        dist_ocean[:,:,None], dist_settle[:,:,None], dist_forest[:,:,None], dist_mountain[:,:,None],
        ndimage.convolve(settlement.astype(float), k3, mode='constant')[:,:,None],
        ndimage.convolve(settlement.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(forest.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(ocean.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(empty.astype(float), k7, mode='constant')[:,:,None],
        ndimage.convolve(settlement.astype(float), k11, mode='constant')[:,:,None],
        is_coast[:,:,None].astype(float),
    ], axis=-1)
    return features.reshape(-1, features.shape[-1])

def kl_score(gt, pred):
    g = np.clip(gt.reshape(-1, NC), 1e-10, None)
    p = np.clip(pred.reshape(-1, NC), 1e-10, None)
    p /= p.sum(axis=-1, keepdims=True); g /= g.sum(axis=-1, keepdims=True)
    return 100 * np.exp(-np.mean(np.sum(g * np.log(g / p), axis=-1)))

def load_all():
    rounds = []
    for gf in sorted(DATA_DIR.glob("ground_truth_*.json")):
        rid = gf.stem.replace("ground_truth_", "")
        with open(gf) as f: data = json.load(f)
        seeds = []
        for si in sorted(data.keys()):
            gt = np.array(data[si].get('ground_truth', []))
            ig = np.array(data[si].get('initial_grid', []))
            if gt.size > 0 and ig.size > 0: seeds.append((ig, gt))
        rounds.append((rid, seeds))
    return rounds

def sample_obs(gt, rng):
    flat = np.clip(gt.reshape(-1, NC), 1e-10, None)
    flat /= flat.sum(axis=-1, keepdims=True)
    cumsum = np.cumsum(flat, axis=-1)
    u = rng.random(len(flat))
    return (u[:, None] < cumsum).argmax(axis=-1).reshape(gt.shape[:2])


def simulate_auto_runner(rounds, hold_idx, rng):
    """Simulate what auto_runner would do for a new round.
    Train on all OTHER rounds, predict on hold-out, simulate observation calibration."""
    
    _, test_seeds = rounds[hold_idx]
    train_data = [s for i,(_, seeds) in enumerate(rounds) if i!=hold_idx for s in seeds]
    
    # Train HGB (same as auto_runner)
    X, Y = [], []
    for ig, gt in train_data:
        X.append(extract_features(ig)); Y.append(gt.reshape(-1,NC))
    X, Y = np.vstack(X), np.vstack(Y)
    models = [HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05,
              min_samples_leaf=50, random_state=42).fit(X, Y[:,c]) for c in range(NC)]
    
    # HGB predictions (same as auto_runner)
    grids = [ig for ig,_ in test_seeds]
    gts = [gt for _,gt in test_seeds]
    n_seeds = len(grids)
    H, W = grids[0].shape
    
    preds = {}
    TEMPERATURE = 1.15
    for si in range(n_seeds):
        Xt = extract_features(grids[si])
        p = np.column_stack([m.predict(Xt) for m in models])
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        # Temperature scaling
        log_p = np.log(p)
        scaled = log_p / TEMPERATURE
        scaled -= scaled.max(axis=-1, keepdims=True)
        p = np.exp(scaled)
        p = np.clip(p, CLIP, None); p /= p.sum(axis=-1, keepdims=True)
        preds[si] = p.reshape(H, W, NC)
    
    # Score before calibration
    before_scores = [kl_score(gts[si], preds[si]) for si in range(n_seeds)]
    before_avg = np.mean(before_scores)
    
    # Simulate observation calibration (same logic as auto_runner)
    vp_size = 15
    def grid_positions(dim):
        n = max(1, -(-dim // vp_size))  # ceil division
        if n == 1:
            return [0]
        step = (dim - vp_size) / (n - 1)
        return [round(i * step) for i in range(n)]
    
    row_starts = grid_positions(H)
    col_starts = grid_positions(W)
    viewport_positions = [(r, c) for r in row_starts for c in col_starts]
    
    per_cls_obs = np.zeros((NC, NC))
    per_cls_pred = np.zeros((NC, NC))
    per_cls_n = np.zeros(NC)
    
    # Settlement proximity split (dist_settle <= 2.0)
    SETTLE_DIST_THRESH = 2.0
    settle_obs = {True: np.zeros((NC, NC)), False: np.zeros((NC, NC))}
    settle_pred = {True: np.zeros((NC, NC)), False: np.zeros((NC, NC))}
    settle_n = {True: np.zeros(NC), False: np.zeros(NC)}
    
    settle_masks = {}
    for si in range(n_seeds):
        cls_g = build_class_grid(grids[si])
        settlement = (cls_g == 1)
        dist_s = ndimage.distance_transform_edt(~settlement) if settlement.any() else np.full((H, W), 20)
        settle_masks[si] = dist_s <= SETTLE_DIST_THRESH
    
    obs_used = 0
    for si in range(n_seeds):
        for row, col in viewport_positions:
            if obs_used >= 45: break
            
            # Simulate API observation: sample from GT
            obs = sample_obs(gts[si], rng)
            
            cls = build_class_grid(grids[si])
            near_settle = settle_masks[si]
            for vy in range(min(vp_size, H - row)):
                for vx in range(min(vp_size, W - col)):
                    gy, gx = row + vy, col + vx
                    if gy >= H or gx >= W: continue
                    ic = cls[gy, gx]
                    oc = obs[gy, gx]
                    ns = bool(near_settle[gy, gx])
                    
                    per_cls_obs[ic, oc] += 1
                    per_cls_pred[ic] += preds[si][gy, gx]
                    per_cls_n[ic] += 1
                    
                    settle_obs[ns][ic, oc] += 1
                    settle_pred[ns][ic] += preds[si][gy, gx]
                    settle_n[ns][ic] += 1
            obs_used += 1
        if obs_used >= 45: break
    
    # Apply settlement-proximity calibration
    calibrated = {}
    for si in range(n_seeds):
        pred = preds[si].copy().reshape(-1, NC)
        cls = build_class_grid(grids[si]).ravel()
        near_settle_flat = settle_masks[si].ravel()
        
        for ns in [True, False]:
            for ic in range(NC):
                n = settle_n[ns][ic]
                if n < 10:
                    n = per_cls_n[ic]
                    if n < 10: continue
                    obs_freq = per_cls_obs[ic] / n
                    pred_avg = per_cls_pred[ic] / n
                else:
                    obs_freq = settle_obs[ns][ic] / n
                    pred_avg = settle_pred[ns][ic] / n
                ratio = np.where(pred_avg > 0.01, np.clip(obs_freq / pred_avg, 0.01, 100.0), 1.0)
                mask = (cls == ic) & (near_settle_flat == ns)
                pred[mask] *= ratio
        pred = np.clip(pred, CLIP, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        calibrated[si] = pred.reshape(H, W, NC)
    
    # Score after calibration
    after_scores = [kl_score(gts[si], calibrated[si]) for si in range(n_seeds)]
    after_avg = np.mean(after_scores)
    
    return before_avg, after_avg, before_scores, after_scores, obs_used, viewport_positions


if __name__ == "__main__":
    print("=" * 70)
    print("  AUTO_RUNNER DRY RUN — Simulating on known rounds")
    print("=" * 70)
    
    rounds = load_all()
    rng = np.random.default_rng(42)
    
    total_before, total_after = [], []
    for hold in range(len(rounds)):
        rid = rounds[hold][0][:8]
        before, after, bs, acs, obs, vps = simulate_auto_runner(rounds, hold, rng)
        total_before.append(before); total_after.append(after)
        seed_str = " ".join(f"{s:.1f}" for s in acs)
        print(f"\n  Round {hold+1} ({rid}):")
        print(f"    Before obs: {before:.2f}")
        print(f"    After obs:  {after:.2f} (delta: {after-before:+.2f})")
        print(f"    Per-seed:   [{seed_str}]")
        print(f"    Viewports:  {len(vps)} positions, {obs} total obs used")
    
    print(f"\n{'='*70}")
    print(f"  SUMMARY:")
    print(f"    Before obs avg: {np.mean(total_before):.2f}")
    print(f"    After obs avg:  {np.mean(total_after):.2f}")
    print(f"    Delta:          {np.mean(total_after)-np.mean(total_before):+.2f}")
    print(f"{'='*70}")
