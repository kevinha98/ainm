"""Re-optimize cell model with vectorized predictions and full data."""
import numpy as np
import time
from scipy.optimize import differential_evolution
from simulator.cell_model import (
    predict_cell_distributions, params_from_vector, vector_from_params,
    kl_score, load_all_gt, CellParams
)

entries = load_all_gt()
print(f"Loaded {len(entries)} GT entries (all seeds)")

# LOO-CV objective: leave-one-round-out
def loo_cv_score(x):
    """Negative mean LOO-CV score."""
    params = params_from_vector(x)
    n_rounds = len(entries) // 5
    round_scores = []
    for leave_out in range(n_rounds):
        # Predict on left-out round
        seeds = entries[leave_out*5:(leave_out+1)*5]
        scores = []
        for ig, gt in seeds:
            pred = predict_cell_distributions(ig, params)
            scores.append(kl_score(pred, gt))
        round_scores.append(np.mean(scores))
    return -np.mean(round_scores)

# Train on all data (for deployment) 
def full_objective(x):
    params = params_from_vector(x)
    scores = []
    for ig, gt in entries:
        pred = predict_cell_distributions(ig, params)
        scores.append(kl_score(pred, gt))
    return -np.mean(scores)

# Current best
current = np.load('data/cell_model_params.npy')
print(f"Current params score: {-full_objective(current):.4f}")
print(f"Current LOO-CV: {-loo_cv_score(current):.4f}")

bounds = [
    (0.01, 0.8),   # settle_base_empty
    (0.5, 15.0),   # settle_scale_empty
    (0.01, 0.8),   # settle_base_forest
    (0.5, 15.0),   # settle_scale_forest
    (0.001, 0.3),  # port_base
    (0.5, 15.0),   # port_scale
    (0.001, 0.15), # ruin_from_settle
    (0.001, 0.08), # ruin_persistence
    (0.05, 0.9),   # settle_survival_base
    (0.0, 3.0),    # settle_survival_density_bonus
    (0.4, 0.99),   # forest_persist_base
    (0.0, 5.0),    # forest_settle_penalty
    (0.001, 0.15), # forest_from_empty
    (0.05, 0.6),   # forest_from_settle
    (0.02, 0.6),   # port_survival
]

print("\nOptimizing with differential_evolution on ALL 70 seeds...")
t0 = time.time()
result = differential_evolution(
    full_objective, bounds,
    seed=42, maxiter=200, popsize=20,
    mutation=(0.5, 1.5), recombination=0.9,
    tol=0.0001, polish=True,
    x0=current,  # seed with current best
    callback=lambda xk, convergence: print(f"  iter: score={-full_objective(xk):.4f} conv={convergence:.6f}") if np.random.random() < 0.1 else None
)
t1 = time.time()

print(f"\nDE result: {-result.fun:.4f} in {t1-t0:.0f}s ({result.nit} iterations)")
new_params = result.x
print(f"New LOO-CV: {-loo_cv_score(new_params):.4f}")

# Per-round comparison
print("\nPer-round comparison (new vs old):")
params_new = params_from_vector(new_params)
params_old = params_from_vector(current)
n_rounds = len(entries) // 5
for r in range(n_rounds):
    seeds = entries[r*5:(r+1)*5]
    s_new = np.mean([kl_score(predict_cell_distributions(ig, params_new), gt) for ig, gt in seeds])
    s_old = np.mean([kl_score(predict_cell_distributions(ig, params_old), gt) for ig, gt in seeds])
    diff = s_new - s_old
    print(f"  Round {r+1}: new={s_new:.4f} old={s_old:.4f} {'+'if diff>=0 else ''}{diff:.4f}")

# Save if better
if -result.fun > -full_objective(current) + 0.001:
    np.save('data/cell_model_params.npy', new_params)
    print(f"\nSaved new params to data/cell_model_params.npy")
else:
    print(f"\nNo significant improvement, keeping old params")

# Print param values for reference
pnew = params_from_vector(new_params)
print(f"\nOptimized params:")
for field in CellParams.__dataclass_fields__:
    print(f"  {field}: {getattr(pnew, field):.6f}")
