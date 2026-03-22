"""Test an enhanced cell model with more features for the ensemble."""
import numpy as np
import json
import time
from pathlib import Path
from scipy import ndimage
from scipy.optimize import differential_evolution
from dataclasses import dataclass

DATA_DIR = Path("data")
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


def build_class_grid(grid_np):
    cg = np.zeros_like(grid_np)
    for gv, cls in GRID_TO_CLASS.items():
        cg[grid_np == gv] = cls
    return cg


def compute_features_v2(ig):
    """Enhanced features: adds port distance, ruin distance, settlement density."""
    H, W = ig.shape
    cls = build_class_grid(ig)
    
    settle = (cls == 1) | (cls == 2)
    dist_s = ndimage.distance_transform_edt(~settle) if settle.any() else np.full((H, W), 40.0)
    
    forest = (cls == 4)
    dist_f = ndimage.distance_transform_edt(~forest) if forest.any() else np.full((H, W), 40.0)
    
    ocean = (ig == 10)
    dist_o = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full((H, W), 40.0)
    coastal = (dist_o <= 1.5).astype(float)
    
    port = (cls == 2)
    dist_p = ndimage.distance_transform_edt(~port) if port.any() else np.full((H, W), 40.0)
    
    settle_density = ndimage.uniform_filter(settle.astype(float), size=7, mode='constant')
    forest_density = ndimage.uniform_filter(forest.astype(float), size=7, mode='constant')
    
    is_ocean = ocean.astype(float)
    
    return cls, dist_s, dist_f, dist_p, coastal, settle_density, forest_density, is_ocean


@dataclass
class CellParamsV2:
    """Enhanced cell model parameters (20 params)."""
    # Empty → Settlement
    settle_base_empty: float = 0.35
    settle_scale_empty: float = 3.0
    # Forest → Settlement  
    settle_base_forest: float = 0.30
    settle_scale_forest: float = 2.5
    # Port formation
    port_base: float = 0.05
    port_scale: float = 2.0
    port_dist_scale: float = 3.0  # NEW: near existing ports more likely
    # Ruin
    ruin_from_settle: float = 0.025
    ruin_persistence: float = 0.01
    # Settlement survival
    settle_survival_base: float = 0.35
    settle_survival_density: float = 0.5
    # Forest dynamics
    forest_persist_base: float = 0.80
    forest_settle_penalty: float = 0.25
    forest_from_empty: float = 0.03
    forest_from_settle: float = 0.20
    forest_density_bonus: float = 0.1  # NEW: nearby forests boost persistence
    # Port survival
    port_survival: float = 0.20
    # Coastal effects  
    coastal_settle_bonus: float = 0.0  # NEW: coastal cells more likely to become settlements
    coastal_forest_penalty: float = 0.0  # NEW: coastal cells less likely to become forest


def predict_v2(ig, params: CellParamsV2):
    H, W = ig.shape
    cls, dist_s, dist_f, dist_p, coastal, settle_density, forest_density, is_ocean = compute_features_v2(ig)
    
    pred = np.zeros((H, W, 6))
    
    m5 = (cls == 5)
    pred[m5, 5] = 1.0
    
    deep_ocean = (is_ocean > 0.5) & (dist_s > 5) & ~m5
    pred[deep_ocean, 0] = 1.0
    
    special = m5 | deep_ocean
    m0 = (cls == 0) & ~special
    m1 = (cls == 1) & ~special
    m2 = (cls == 2) & ~special
    m3 = (cls == 3) & ~special
    m4 = (cls == 4) & ~special
    
    if m0.any():
        ds = dist_s[m0]
        df = dist_f[m0]
        dp = dist_p[m0]
        co = coastal[m0]
        p_settle = params.settle_base_empty * np.exp(-ds / params.settle_scale_empty) * (1 + params.coastal_settle_bonus * co)
        p_port = params.port_base * co * np.exp(-ds / params.port_scale) * np.exp(-dp / params.port_dist_scale)
        p_ruin = np.full_like(ds, params.ruin_persistence)
        p_forest = params.forest_from_empty * (1 + 1.0 / (1 + df)) * (1 - params.coastal_forest_penalty * co)
        p_empty = 1.0 - p_settle - p_port - p_ruin - p_forest
        p = np.stack([np.maximum(0, p_empty), p_settle, p_port, p_ruin, np.maximum(0, p_forest), np.zeros_like(ds)], axis=-1)
        pred[m0] = p
    
    if m1.any():
        sd = settle_density[m1]
        co = coastal[m1]
        dp = dist_p[m1]
        p_survive = np.minimum(params.settle_survival_base + params.settle_survival_density * sd, 0.9)
        p_port = params.port_base * co * 2 * np.exp(-dp / params.port_dist_scale)
        p_ruin = np.full_like(sd, params.ruin_from_settle)
        p_forest = params.forest_from_settle * (1 - p_survive * 0.5)
        p_empty = 1.0 - p_survive - p_port - p_ruin - p_forest
        p = np.stack([np.maximum(0, p_empty), p_survive, p_port, p_ruin, np.maximum(0, p_forest), np.zeros_like(sd)], axis=-1)
        pred[m1] = p
    
    if m2.any():
        n2 = m2.sum()
        p_survive_port = np.full(n2, params.port_survival)
        p_survive_settle = np.full(n2, 0.10)
        p_ruin = np.full(n2, params.ruin_from_settle)
        p_forest = np.full(n2, params.forest_from_settle * 0.8)
        p_empty = 1.0 - p_survive_port - p_survive_settle - p_ruin - p_forest
        p = np.stack([np.maximum(0, p_empty), p_survive_settle, p_survive_port, p_ruin, np.maximum(0, p_forest), np.zeros(n2)], axis=-1)
        pred[m2] = p
    
    if m3.any():
        ds = dist_s[m3]
        p_settle = params.settle_base_empty * np.exp(-ds / params.settle_scale_empty) * 1.2
        p_forest = np.full_like(ds, 0.3)
        p_ruin = np.full_like(ds, 0.05)
        p_empty = 1.0 - p_settle - p_forest - p_ruin
        p = np.stack([np.maximum(0, p_empty), p_settle, np.zeros_like(ds), p_ruin, np.maximum(0, p_forest), np.zeros_like(ds)], axis=-1)
        pred[m3] = p
    
    if m4.any():
        ds = dist_s[m4]
        co = coastal[m4]
        sd = settle_density[m4]
        fd = forest_density[m4]
        forest_survive = np.clip(params.forest_persist_base - params.forest_settle_penalty * sd + params.forest_density_bonus * fd, 0.3, 0.99)
        p_settle = params.settle_base_forest * np.exp(-ds / params.settle_scale_forest) * (1 + params.coastal_settle_bonus * co)
        p_port = params.port_base * co * np.exp(-ds / params.port_scale) * 0.5
        p_ruin = np.full_like(ds, params.ruin_persistence)
        p_empty = 1.0 - forest_survive - p_settle - p_port - p_ruin
        p = np.stack([np.maximum(0, p_empty), p_settle, p_port, p_ruin, forest_survive, np.zeros_like(ds)], axis=-1)
        pred[m4] = p
    
    active = m0 | m1 | m2 | m3 | m4
    if active.any():
        pred[active] = np.clip(pred[active], 1e-6, None)
        pred[active] = pred[active] / pred[active].sum(axis=-1, keepdims=True)
    
    return pred


def kl_score(pred, gt):
    pred = np.clip(pred, 1e-8, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    kl = np.where(gt > 0, gt * np.log(np.clip(gt, 1e-15, None) / pred), 0).sum(axis=-1)
    return 100 - np.mean(np.where(np.isfinite(kl), kl, 0)) * 100


def params_from_vector(x):
    return CellParamsV2(
        settle_base_empty=x[0], settle_scale_empty=x[1],
        settle_base_forest=x[2], settle_scale_forest=x[3],
        port_base=x[4], port_scale=x[5], port_dist_scale=x[6],
        ruin_from_settle=x[7], ruin_persistence=x[8],
        settle_survival_base=x[9], settle_survival_density=x[10],
        forest_persist_base=x[11], forest_settle_penalty=x[12],
        forest_from_empty=x[13], forest_from_settle=x[14],
        forest_density_bonus=x[15],
        port_survival=x[16],
        coastal_settle_bonus=x[17], coastal_forest_penalty=x[18],
    )


# Load data
entries = []
gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
for gf in gt_files:
    with open(gf) as f:
        data = json.load(f)
    for si_str in sorted(data.keys()):
        entry = data[si_str]
        entries.append((np.array(entry['initial_grid']), np.array(entry['ground_truth'])))

print(f"Loaded {len(entries)} entries")

# Test default v2 params
params = CellParamsV2()
scores = [kl_score(predict_v2(ig, params), gt) for ig, gt in entries]
print(f"Default V2 params: {np.mean(scores):.4f}")

# Compare with current v1
from simulator.cell_model import predict_cell_distributions as predict_v1, params_from_vector as pv1
opt_vec = np.load('data/cell_model_params.npy')
v1_params = pv1(opt_vec)
scores_v1 = [kl_score(predict_v1(ig, v1_params), gt) for ig, gt in entries]
print(f"Optimized V1 params: {np.mean(scores_v1):.4f}")

# Optimize V2
bounds = [
    (0.01, 0.8),   # settle_base_empty
    (0.5, 15.0),   # settle_scale_empty
    (0.01, 0.8),   # settle_base_forest
    (0.5, 15.0),   # settle_scale_forest
    (0.001, 0.3),  # port_base
    (0.5, 15.0),   # port_scale
    (0.5, 15.0),   # port_dist_scale (NEW)
    (0.001, 0.15), # ruin_from_settle
    (0.001, 0.08), # ruin_persistence
    (0.05, 0.9),   # settle_survival_base
    (0.0, 3.0),    # settle_survival_density
    (0.4, 0.99),   # forest_persist_base
    (0.0, 5.0),    # forest_settle_penalty
    (0.001, 0.15), # forest_from_empty
    (0.05, 0.6),   # forest_from_settle
    (0.0, 1.0),    # forest_density_bonus (NEW)
    (0.02, 0.6),   # port_survival
    (0.0, 1.0),    # coastal_settle_bonus (NEW)
    (0.0, 1.0),    # coastal_forest_penalty (NEW)
]

def objective(x):
    params = params_from_vector(x)
    scores = [kl_score(predict_v2(ig, params), gt) for ig, gt in entries]
    return -np.mean(scores)

print(f"\nOptimizing V2 cell model ({len(bounds)} params)...")
t0 = time.time()
result = differential_evolution(
    objective, bounds, seed=42, maxiter=150, popsize=15,
    mutation=(0.5, 1.5), recombination=0.9, tol=0.0001, polish=True
)
t1 = time.time()
print(f"V2 result: {-result.fun:.4f} in {t1-t0:.0f}s")

# LOO-CV of V2
n_rounds = len(entries) // 5
round_scores = []
for r in range(n_rounds):
    seeds = entries[r*5:(r+1)*5]
    rs = [kl_score(predict_v2(ig, params_from_vector(result.x)), gt) for ig, gt in seeds]
    round_scores.append(np.mean(rs))
print(f"V2 per-round mean: {np.mean(round_scores):.4f}")

# Compare V1 vs V2 per round
print("\nV1 vs V2 per-round:")
for r in range(n_rounds):
    seeds = entries[r*5:(r+1)*5]
    s1 = np.mean([kl_score(predict_v1(ig, v1_params), gt) for ig, gt in seeds])
    s2 = np.mean([kl_score(predict_v2(ig, params_from_vector(result.x)), gt) for ig, gt in seeds])
    d = s2 - s1
    print(f"  R{r+1}: V1={s1:.4f} V2={s2:.4f} diff={d:+.4f}")

# Save if significantly better
if -result.fun > np.mean(scores_v1) + 0.05:
    np.save('data/cell_model_v2_params.npy', result.x)
    print(f"\nSaved V2 params")
else:
    print(f"\nV2 not significantly better than V1, not saving")
