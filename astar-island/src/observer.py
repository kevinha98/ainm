"""
Astar Island — Strategic Observer
Decides WHERE to observe and saves ALL results to disk.
"""
import json
import numpy as np
from pathlib import Path
from src.settings import MAP_H, MAP_W, VIEWPORT_SIZE, MAX_QUERIES, DATA_DIR, GRID_TO_CLASS
from src.api import AstarAPI


def compute_viewport_plan(grid_np, settle_set, budget_remaining, n_seeds):
    """Decide which viewports to observe per seed.
    Strategy: prioritize settlement-dense regions first (most dynamic),
    then fill in remaining coverage."""
    H, W = grid_np.shape
    VP = VIEWPORT_SIZE  # 15
    from scipy import ndimage

    # 3x3 tiling positions for full coverage
    rows = [0, 13, 25]
    cols = [0, 13, 25]
    full_grid = [(r, c) for r in rows for c in cols]

    full_coverage_per_seed = len(full_grid)  # 9
    total_for_all = full_coverage_per_seed * n_seeds

    if budget_remaining >= total_for_all:
        # Full coverage for all seeds
        return {si: full_grid for si in range(n_seeds)}, "full_coverage"

    # Limited budget: score viewports by settlement density (information value)
    # Compute settlement heatmap
    s_mask = np.zeros((H, W), dtype=float)
    for sy, sx in settle_set:
        if 0 <= sy < H and 0 <= sx < W:
            s_mask[sy, sx] = 1.0

    # Rank viewports by number of settlements + nearby dynamic cells
    s_density = ndimage.uniform_filter(s_mask, size=VP, mode='constant')

    scored_vps = []
    for r, c in full_grid:
        region_density = s_density[min(r + VP//2, H-1), min(c + VP//2, W-1)]
        # Count actual terrain diversity in this viewport
        vp_grid = grid_np[r:r+VP, c:c+VP]
        diversity = len(set(vp_grid.ravel()))
        score = region_density * 10 + diversity * 0.1
        scored_vps.append((score, r, c))

    scored_vps.sort(reverse=True)  # highest score first
    ranked_vps = [(r, c) for _, r, c in scored_vps]

    queries_per_seed = budget_remaining // n_seeds
    n_vp = min(queries_per_seed, len(ranked_vps))

    if n_vp == 0:
        return {si: [] for si in range(n_seeds)}, "no_budget"

    return {si: ranked_vps[:n_vp] for si in range(n_seeds)}, f"smart_{n_vp}"


def observe_round(api, round_data, steps=50):
    """Execute observation plan and save ALL results to disk."""
    round_id = round_data["id"]
    n_seeds = round_data.get("seeds_count", len(round_data.get("initial_states", [])))
    obs_file = DATA_DIR / f"observations_{round_id[:8]}.json"

    # Load existing observations (resume support)
    observations = {}
    if obs_file.exists():
        with open(obs_file) as f:
            observations = json.load(f)
        print(f"  Resumed {sum(len(v) for v in observations.values())} existing observations")

    # Check budget
    budget = api.get_budget()
    remaining = budget["queries_max"] - budget["queries_used"]
    if remaining <= 0:
        print(f"  Budget exhausted ({budget['queries_used']}/{budget['queries_max']})")
        return observations

    # Get initial grids for planning
    grids = []
    settle_sets = []
    for st in round_data["initial_states"]:
        g = np.array(st["grid"])
        grids.append(g)
        ss = set()
        for s in st.get("settlements", []):
            if s.get("alive", True):
                ss.add((s["y"], s["x"]))
        settle_sets.append(ss)

    # Plan viewports
    plan, strategy = compute_viewport_plan(grids[0], settle_sets[0], remaining, n_seeds)
    total_planned = sum(len(v) for v in plan.items())
    print(f"  Strategy: {strategy} ({total_planned} viewports, {remaining} budget)")

    # Execute observations
    for si in range(n_seeds):
        seed_key = str(si)
        if seed_key not in observations:
            observations[seed_key] = []

        existing_positions = {(o["row"], o["col"]) for o in observations[seed_key]}

        for row, col in plan.get(si, []):
            if (row, col) in existing_positions:
                continue  # skip already observed

            result = api.simulate(round_id, si, row, col, steps)

            if "error" in result:
                print(f"  Seed {si} ({row},{col}): {result['error']}")
                if result["error"] == "budget_exhausted":
                    _save_observations(obs_file, observations)
                    return observations
                continue

            obs_entry = {
                "row": row,
                "col": col,
                "steps": steps,
                "grid": result.get("grid", []),
                "queries_used": result.get("queries_used"),
            }
            observations[seed_key].append(obs_entry)

            # Save after every observation (crash-safe)
            _save_observations(obs_file, observations)
            print(f"  Seed {si} ({row:2},{col:2}): OK ({result.get('queries_used')}/{budget['queries_max']})")

    print(f"  Total observations: {sum(len(v) for v in observations.values())}")
    return observations


def _save_observations(path, observations):
    with open(path, "w") as f:
        json.dump(observations, f)


def build_observed_grid(observations_seed, map_h=MAP_H, map_w=MAP_W):
    """Reconstruct a full observed grid from viewport observations.
    Returns grid and confidence mask (observed=True, inferred=False)."""
    grid = np.full((map_h, map_w), -1, dtype=int)  # -1 = not observed
    observed = np.zeros((map_h, map_w), dtype=bool)

    for obs in observations_seed:
        r, c = obs["row"], obs["col"]
        vp = np.array(obs["grid"])
        vh, vw = vp.shape
        grid[r:r+vh, c:c+vw] = vp
        observed[r:r+vh, c:c+vw] = True

    return grid, observed


def compute_transition_matrix(initial_grids, observations, n_classes=6):
    """Learn transition probabilities from initial→observed data.
    Returns NxN matrix where T[i,j] = P(class j after 50 years | class i initially)."""
    counts = np.zeros((n_classes, n_classes))

    for si_str, obs_list in observations.items():
        si = int(si_str)
        if si >= len(initial_grids):
            continue
        init_grid = initial_grids[si]

        for obs in obs_list:
            r, c = obs["row"], obs["col"]
            vp = np.array(obs["grid"])
            vh, vw = vp.shape

            for dy in range(vh):
                for dx in range(vw):
                    gy, gx = r + dy, c + dx
                    if gy < MAP_H and gx < MAP_W:
                        init_class = GRID_TO_CLASS.get(init_grid[gy, gx], 0)
                        final_class = GRID_TO_CLASS.get(vp[dy, dx], 0)
                        counts[init_class, final_class] += 1

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    T = np.where(row_sums > 0, counts / row_sums, 1.0 / n_classes)
    return T, counts
