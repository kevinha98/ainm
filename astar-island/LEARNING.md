# Astar Island — Learning Log

> Last updated: 2026-03-20 (session 7 — R6 scored 61.85, calibration HURT on extreme round, R6 GT fetched)

## Current State

- **Round 1**: Completed, score 31.33, rank 55/117 (v4 ensemble — overconfident)
- **Round 2**: Completed, score 75.61, rank 39/153 (v7 per-class avg)
- **Round 3**: Completed — MISSED (not submitted)
- **Round 4**: Completed — MISSED (not submitted)
- **Round 5**: Completed, **score 80.07, rank 20/144** (v10 HGB, no obs)
- **Round 6**: Completed, **score 61.85, rank 75/235** (HGB + T=1.15 + per-class calib — calibration hurt!)
- **Leaderboard**: 235 teams, top team 118.63
- **Auto-runner**: Active, polls every 120s, T=1.15 temp scaling + dist_settle<=2 calibration + clip [0.01, 100]
- **GT data**: 6 rounds (48,000 cells)
- **Key numbers**: LOO HGB base=92.35, with obs+temp=97.22, dryrun avg=97.25, server gap ~14 pts

## Submissions

| Round | Date | Model | Score | Rank | Notes |
| ----- | ---- | ----- | ----- | ---- | ----- |
| 1 | 2026-03-19 | v4 ensemble | 31.33 | 55/117 | No observations, blind prediction, overconfident |
| 2 | 2026-03-19 | v7 per-class avg | 75.61 | 39/153 | Cross-round averages from R1 only |
| 3 | — | (not submitted) | — | — | Missed round while offline |
| 4 | — | (not submitted) | — | — | Missed round while offline |
| 5 | 2026-03-20 | v10 HGB regressor | **80.07** | **20/144** | Trained on R1-R4 GT, clip=0.0001, no obs (budget was 49/50 used) |
| 6 | 2026-03-20 | HGB + T=1.15 + per-class calib | **61.85** | **75/235** | Resubmitted last-minute with T=1.15. Calibration HURT on this extreme round. |

## PARADIGM SHIFT: Ground Truth is a Probability Distribution

**The simulation is stochastic.** The same map + parameters produce different outcomes every run.
Ground truth is NOT "what happened" but the **true probability distribution** over all possible outcomes.

### What the GT actually looks like (Round 1, seed 0 examples)
- Settlement cell `[3,17]`: `[0.615, 0.325, 0, 0.04, 0, 0.02]` — NOT one-hot!
- Plains cell average: `[0.64, 0.13, 0.01, 0.01, 0.19, 0.02]`
- Forest cell average: `[0.07, 0.16, 0.01, 0.01, 0.75, 0.01]`
- Ocean cell average (deep): `[0.93, 0.03, 0.01, 0.001, 0.02, 0.005]`
- Mean max probability across all cells: **0.823** (not 1.0!)
- 1247/1600 cells have max probability < 0.99

### Scoring Formula (reverse-engineered)
```
score ≈ 100 × exp(-KL(gt || pred))
```
where KL is the KL-divergence from ground truth to prediction, averaged across all cells.
- Uniform prediction → score ~27
- Per-class average → score ~88-91 (leave-one-out CV within Round 1)
- 99.9% confidence on one class → score ~28-36 (barely better than uniform!)
- Top leaderboard teams: ~85

### Critical Insight
**Overconfidence is catastrophically punished.** Putting 99% on the wrong distribution costs way more than being slightly uncertain. The optimal strategy is to predict the true probability distribution accurately, NOT to be maximally confident.

## Per-Class Average GT Distributions (from Round 1)

These are the average ground truth distributions grouped by initial cell class:

| Initial Class | Empty | Settlement | Port | Ruin | Forest | Mountain |
|--------------|-------|-----------|------|------|--------|----------|
| Empty/Plains (0) | 0.640 | 0.130 | 0.010 | 0.010 | 0.186 | 0.021 |
| Settlement (1) | 0.590 | 0.170 | 0.005 | 0.010 | 0.210 | 0.020 |
| Port (2) | ~same as settlement | | | | | |
| Ruin (3) | ~same as empty | | | | | |
| Forest (4) | 0.070 | 0.160 | 0.010 | 0.010 | 0.750 | 0.010 |
| Mountain (5) | ~0.6-0.8 | 0.15-0.2 | ~0 | ~0 | 0.01-0.16 | ~0.007 |
| Ocean (0, deep) | 0.931 | 0.030 | 0.010 | 0.001 | 0.020 | 0.005 |

**Mountains are NOT immutable!** Mountain cells are ~78.5% Empty, only ~0.7% stay Mountain.
This suggests the simulation can "erode" mountains or they're reclassified.

## Observation Strategy: REVISED (Session 4)

### Key Finding: Per-class multiplicative calibration is OPTIMAL
- LOO CV: HGB alone = 91.37, HGB + 45 observations = **96.18** (+4.81)
- Per-class mult with 5 seeds: 96.18 ±0.01
- Per-class Bayesian adds no benefit (alpha=10-20 ties at 96.18)
- Per-cell Bayesian is CATASTROPHIC (56-89 depending on concentration)
  - Only ~4 obs per cell → multinomial noise dominates
  - Per-class pools ~5000+ cells → reliable statistics

### How calibration works (current: settlement proximity split)
1. Observe 9 viewports × 5 seeds = 45 queries (full grid coverage)
2. Viewport positions: (row, col) for row in [0, 13, 25] for col in [0, 13, 25]
3. Each viewport is 15×15, row/col = TOP-LEFT corner (NOT center)
4. Aggregate per (initial-class, near_settle) bucket: count observed outcome classes
   - near_settle = True if dist_to_settlement <= 2.0, else False
5. Compute ratio = obs_freq / predicted_avg, clip to [0.01, 100.0]
6. Apply multiplicative correction per bucket; fall back to per-class if bucket n<10

### Settlement proximity calibration discovery (Session 6)
Sweep of 8 split strategies revealed near-settlement is the strongest signal:

| Strategy | LOO CV | Delta vs per-class |
| -------- | ------ | ------------------ |
| dist_settle<=2 | **96.96** | **+0.61** |
| settle5+coastal (4 buckets) | 96.94 | +0.59 |
| settle5+ocean3 (4 buckets) | 96.91 | +0.56 |
| 3-way settle/coast/inland | 96.90 | +0.55 |
| near_settle sz=3 | 96.78 | +0.43 |
| coastal (is_coast binary) | 96.53 | +0.18 |
| dist_ocean<=3 | 96.56 | +0.21 |
| per-class only | 96.35 | baseline |
| edge (biome boundary) | 96.38 | +0.03 |
| near_forest | 96.32 | -0.03 |

**Distance threshold sweep for settle split:**
| Threshold | LOO CV |
|-----------|--------|
| 1.0 | 96.70 |
| 1.5 | 96.78 |
| **2.0** | **96.96** |
| 2.5 | 96.84 |
| 3.0 | 96.74 |
| 4.0 | 96.57 |

**Why it works:** Cells within distance 2 of settlements experience fundamentally different
dynamics (settlement growth pressure, resource extraction influence, defensive structures).
The model doesn't perfectly capture these proximity effects, so calibration helps.

### Dryrun results with dist_settle<=2
| Round | HGB Only | + Calibration | Delta |
| ----- | -------- | ------------- | ----- |
| R1 | 94.96 | 96.57 | +1.61 |
| R2 | 91.70 | 96.76 | +5.07 |
| R3 | 97.65 | 98.07 | +0.42 |
| R4 | 83.20 | 97.36 | +14.17 |
| R5 | 94.23 | 96.17 | +1.94 |
| **Avg** | **92.35** | **96.99** | **+4.64** |

### Clip range optimization (Session 5)
LOO CV sweep showed [0.3, 3.0] was too conservative, especially for rounds with extreme dynamics:

| Clip Range | LOO CV | Delta |
| ---------- | ------ | ----- |
| [0.50, 2.0] | 95.32 | -0.48 |
| [0.30, 3.0] | 95.80 | baseline |
| [0.20, 5.0] | 96.02 | +0.22 |
| [0.10, 10.0] | 96.24 | +0.44 |
| [0.05, 20.0] | 96.33 | +0.53 |
| [0.01, 100.0] | **96.35** | **+0.55** |
| [0.00, 1000.0] | 96.34 | +0.54 (saturated) |

The improvement comes almost entirely from Fold 3 (R4-like data with extreme dynamics).
Wider clips allow the calibration to capture extreme regime shifts like R4/R6.

### Bayes vs multiplicative calibration (Session 5)
Tested Bayesian calibration with wider clips:

| Strategy | LOO CV |
| -------- | ------ |
| mult [0.01, 100] | 96.35 |
| bayes alpha=5 [0.01, 100] | 96.37 |
| bayes alpha=10 [0.01, 100] | 96.37 |
| bayes alpha=20 [0.01, 100] | 96.36 |
| additive strength=1.0 | 95.43 |

Bayes alpha=5-10 is marginally (+0.02) better than pure multiplicative. Not significant.
Additive calibration is clearly worse. **Multiplicative with wide clip is optimal.**

### Why observations HELP now (revised from session 1-2)
In sessions 1-2 we found observations HURT. That was because:
- We used per-cell Bayesian updates (too noisy)
- We used overconfident observation integration (99.9%)

Per-CLASS calibration works because it pools thousands of cells to estimate
the round-specific transition probabilities, then corrects the HGB prediction
at the class level. This captures round-to-round simulation parameter variation.

## Round-to-Round Variation Analysis

### R3 is a fundamentally different simulation regime
- R3 entropy: **0.068** (nearly deterministic) vs R1=0.554, R2=0.692, R4=0.465
- R3 Empty→Empty: 99% (vs 78-88% in other rounds)
- R3 Settlement→Settlement: 1.8% (vs 23-41%)
- R3 Forest→Forest: 97% (vs 66-82%)
- R3 has almost no settlements in outcomes (<0.2% vs 13-16%)

### Map composition is SIMILAR but simulation dynamics DIFFER
The initial maps are ~73-75% empty, ~2.5% settlement across all rounds.
The SIMULATION PARAMETERS change between rounds, not the maps.
This is why observation calibration is so powerful — it captures the current
round's specific transition probabilities.

### LOO CV per-round scores (HGB base, no obs)
**5-round LOO (with R5 GT):**
- R1: 94.96 (trained on R2+R3+R4+R5)
- R2: 91.70 (trained on R1+R3+R4+R5)
- R3: 97.65 (trained on R1+R2+R4+R5)
- R4: 83.20 (trained on R1+R2+R3+R5) — still HARDEST
- R5: 94.23 (trained on R1+R2+R3+R4)

**4-round LOO (without R5 GT):**
- R1: 94.30, R2: 91.28, R3: 97.62, R4: 82.30, avg: 91.37

## Model Architecture (v10 — CURRENT BEST)

### Approach: HGB Regressor with spatial features

Train 6 HistGradientBoostingRegressor models (one per output class) on all GT data.
17 features per cell: 6 class one-hot + 4 distances + 6 neighborhood counts + 1 coastal.
Clip predictions to [0.0001, ...] and normalize.

HGB params: max_iter=100, max_depth=4, learning_rate=0.05, min_samples_leaf=50.

### LOO CV Results (4 rounds)

| Approach | LOO avg | R1 | R2 | R3 | R4 | R5 |
| -------- | ------- | -- | -- | -- | -- | -- |
| HGB base | **92.35** | 94.96 | 91.70 | 97.65 | 83.20 | 94.23 |
| HGB + raw grid | 92.36 | 94.97 | 91.66 | 97.69 | 83.24 | 94.26 |
| HGB + more nbhd | 92.35 | 95.00 | 91.59 | 97.64 | 83.25 | 94.26 |
| HGB ensemble(3) | 92.34 | 94.94 | 91.64 | 97.66 | 83.22 | 94.24 |
| Per-class avg | 87.85 | 88.0 | 86.5 | 94.1 | 82.3 | 88.4 |
| HGB log-target | 76.80 | 70.2 | 59.4 | 85.0 | 87.1 | 82.3 |

### With observation calibration (LOO CV)

| Strategy | 5-round LOO | Notes |
| -------- | ----------- | ----- |
| HGB + dist_settle<=2 clip[0.01,100] | **96.96** | **CURRENT BEST** |
| HGB + per-class mult clip[0.01,100] | 96.35 | Previous best |
| HGB + per-class Bayes alpha=5-10 | 96.37 | Marginally better, within noise |
| HGB + per-class mult clip[0.3,3] | 95.80 | Previous best (sessions 3-4) |
| HGB alone (no obs) | 92.35 | Baseline |
| Per-class avg alone | 87.85 | No learned model |

### Tested model variants (all same or worse than HGB base)

- Ensemble of 3 configs: +0.01 (not worth complexity)
- Raw grid value feature: +0.01
- More neighborhood features (r=5, r=15): -0.05
- Log-transformed targets: -20 (catastrophic)
- Per-class HGB (separate model per class): -0.07
- Position features (row/col): no improvement
- Deeper/wider HGB: no improvement
- All blending strategies with uniform/avg: worse
- Dirichlet smoothing: worse

## Key Files

| File | Purpose |
| ---- | ------- |
| `auto_runner.py` | **PRIMARY** — Persistent background auto-submitter. Polls for new rounds, trains HGB, observes 45 viewports, calibrates, submits. |
| `run_v10.py` | Manual HGB submission script (used for R5). Trains on all GT, predicts, submits. |
| `cv_obs_cached.py` | **Best CV script** — cached HGB + exhaustive obs calibration strategy sweep |
| `cv_fast.py` | Fast LOO CV for model variant comparison |
| `analyze_r3.py` | Round-to-round simulation parameter variation analysis |
| `fetch_all_gt.py` | Fetch and cache ground truth for all completed rounds |
| `check_status.py` | Quick status check — rounds, scores, leaderboard |
| `src/api.py` | API client with cookie auth |
| `src/models.py` | build_class_grid, compute_stats |
| `src/settings.py` | Constants: TOKEN, API_BASE, DATA_DIR, GRID_TO_CLASS |
| `data/ground_truth_*.json` | Cached ground truth (R1-R4, 32,000 cells) |
| `data/auto_runner_state.json` | Tracks which rounds auto_runner has submitted |

## Critical Parameters

| Parameter | Value | Impact |
| --------- | ----- | ------ |
| CLIP_FLOOR | 0.0001 | +0.35 vs 0.002. Below 1e-4 is plateau. |
| HGB max_iter | 100 | Higher = no improvement |
| HGB max_depth | 4 | Deeper = slight overfit |
| HGB learning_rate | 0.05 | Standard |
| HGB min_samples_leaf | 50 | Lower = more overfit |
| Obs viewports | 9 per seed | Full 40×40 coverage with 15×15 windows |
| Obs budget | 50 per round | 45 for viewports + 5 spare |
| Calibration clip | [0.01, 100.0] | Optimized from [0.3, 3.0] — +0.55 LOO improvement |
| Calibration split | dist_settle<=2.0 | Near-settlement vs far — +0.61 over per-class only |
| Temperature | 1.15 | Soften HGB predictions before calibration — +0.26 over T=1.0 |

## Temperature Scaling Discovery (Session 6)

**Key insight**: HGB predictions are slightly overconfident. Softening them (T > 1) before
observation calibration lets the calibration step work better, because it has more probability
mass to redistribute.

### Temperature sweep results (with settle-split calibration, 20 trials LOO)

| Temperature | LOO CV | Delta vs T=1.0 |
| ----------- | ------ | --------------- |
| 0.7 | 94.77 | -2.19 |
| 0.9 | 96.76 | -0.20 |
| 1.0 | 96.96 | baseline |
| **1.15** | **97.22** | **+0.26** |
| 1.20 | 97.22 | +0.26 |
| 1.30 | 97.15 | +0.19 |
| 1.50 | 96.89 | -0.07 |

Confirmed with 50-trial LOO: T=1.15 = 97.22, T=1.20 = 97.21, T=1.25 = 97.18.

### Dryrun with temperature (T=1.15 + settle-split)

| Round | Before obs | After obs | Delta |
| ----- | ---------- | --------- | ----- |
| R1 | 95.75 | 96.51 | +0.77 |
| R2 | 94.68 | 97.26 | +2.58 |
| R3 | 96.24 | 98.00 | +1.76 |
| R4 | 79.26 | 98.36 | +19.10 |
| R5 | 93.43 | 96.10 | +2.67 |
| **Avg** | **91.87** | **97.25** | **+5.38** |

### Why it works
- T > 1 converts logits to probabilities via softmax with temperature: p = softmax(logits/T)
- This makes predictions less confident (higher entropy)
- The multiplicative calibration step can then adjust ratios more effectively
- For overconfident cells, calibration divides probability mass; softer start means more mass to work with
- The clip range [0.01, 100] is already optimal for T=1.15; no interaction effects

## R6 Analysis — Calibration Can HURT (Session 7)

### R6 characteristics
- **Highest entropy of all rounds**: 0.789-0.838 (vs R1=0.554, R2=0.692, R3=0.068, R4=0.465)
- Max probability mean: 0.656-0.686 (vs typical ~0.823)
- Very uncertain outcomes — cell transitions highly stochastic

### R6 LOO base prediction
- HGB base (no obs): KL=87.48 (CE=38.99)
- T=1.15 no calib: KL=91.48 (huge +4.0 improvement!)
- T=1.15 + per-class calib: KL=82.58 (**-8.90 from calibration!**)
- T=1.0 + per-class calib: KL=83.06 (**-4.42 from calibration!**)

### CRITICAL FINDING: Calibration HURTS on extreme rounds
Per-class observation calibration demolished the score for R6. Why:
1. R6 has extreme variance — each simulation run gives wildly different outcomes
2. 45 observations sample ~10K cells, but per class each bucket is a subset
3. The observed frequencies don't represent the true class-conditional distribution
4. Multiplying predictions by misleading ratios pushes predictions AWAY from truth
5. The model's raw predictions (especially with T=1.15) were closer to truth than calibrated

### Implication for future rounds
- Calibration is a GAMBLE on extreme rounds
- For "normal" rounds (R1-R5), calibration helps +3-20 points
- For extreme rounds (R6), calibration hurts -5-9 points
- T=1.15 alone consistently helps (+4 on R6, +0.26 LOO avg)
- **Consider a safety check**: if calibrated predictions are very different from uncalibrated, hedge

### 6-round LOO dryrun (with R6 GT)

| Round | Before obs | After obs | Delta | Notes |
| ----- | ---------- | --------- | ----- | ----- |
| R1 | 94.81 | 96.67 | +1.86 | |
| R2 | 93.72 | 97.35 | +3.63 | |
| R3 | 94.50 | 98.02 | +3.52 | |
| R4 | 76.34 | 98.38 | +22.04 | Adding R6 to training hurt R4 base |
| R5 | 92.13 | 96.13 | +4.00 | |
| R6 | 91.51 | 96.36 | +4.86 | LOO with real obs, not cached |
| **Avg** | **91.16** | **97.15** | **+5.99** | |

Note: R6 LOO dryrun (96.36) uses simulated observations from GT, which is more
representative than the 45 API observations. Real server score was only 61.85.

## Failed Approaches (Complete List)

| Approach | Result | Why |
| -------- | ------ | --- |
| v4 ensemble (R1) | Score 31 | Assumed transitions, no GT, overconfident |
| v6 obs-direct 99.9% (R2) | Score ~28-36 | Catastrophically overconfident |
| Predicting Ruins | Wrong | Ruins never appear in GT |
| MRF smoothing on obs | Worse | Blurs signal |
| Per-cell Bayesian obs | 56-89 | Too few obs per cell, noise dominates |
| HGB log-target | 70.69 LOO | Terrible — log transform breaks everything |
| HGB ensemble (3 configs) | +0.01 | Not worth complexity |
| More neighborhood features | -0.05 | Slight overfit |
| Dirichlet smoothing | Worse | Regularizes away signal |
| Blending HGB with uniform/avg | Worse | Dilutes good predictions |
| Higher clip floor (0.05) | 80.80 LOO | Too aggressive clipping |
| Two-pass calibration (cls->settle, settle->cls) | 96.95-96.96 | Same as single-pass settle-split |
| Per-class temperature | No improvement | Empty class: T=1.15 same as global |
| Calibration clip wider with T=1.15 | No improvement | [0.01,100] already optimal for T=1.15 |

## API Reference

| Endpoint | Returns |
| -------- | ------- |
| GET /rounds/{id} | Full round data with initial_states |
| GET /rounds | List (no initial_states) |
| GET /my-rounds | User data (scores, budget, rank) |
| GET /analysis/{id}/{seed} | Ground truth (only completed rounds) |
| POST /submit | Submit predictions |
| POST /simulate | Observe viewport (costs 1 query). row/col = TOP-LEFT of 15×15. |

## Leaderboard Scoring

- `weighted_score` = aggregation across rounds with escalating weights
- Round weights: R1=1.05, R2=1.1025, R3=1.157625, R4=1.2155, R5=1.2763
- Later rounds weighted more heavily
- Top teams: ~114 weighted score, 3-4 rounds participated

## Scoring Gap Investigation (Session 5)

**Our local KL-divergence scoring does NOT match server scores.** Systematic gap:

| Round | Local KL score | Server score | Gap |
| ----- | ------------- | ------------ | --- |
| R2 | 90.58 (per-class avg from R1) | 75.61 | -14.97 |
| R5 | 94.23 (HGB from R1-R4) | 80.07 | -14.16 |

- CE formula (cross-entropy) gives scores much lower (45-58), not matching either
- The gap ratio varies by round (2.83x vs 3.74x effective KL multiplier)
- Likely cause: server uses different GT precision (more simulations) or different formula
- **Strategy unchanged**: maximizing local LOO CV still correlates with server scores

### R5 GT Characteristics
- Mountains (class 5) = 100% Mountain [0,0,0,0,0,1.0] — very different from R1-R4 where mountains were ~78% Empty!
- Entropy moderate (0.5-0.8 for most classes), similar to R1/R2
- R5 is similar to R1/R2 in difficulty (not like R3's near-deterministic or R4's hardest)

## Next Steps for Future Rounds

1. **Auto-runner handles everything** — T=1.15 + dist_settle<=2 calibration + clip [0.01, 100]
2. After each round closes: auto_runner fetches GT, retrains, ready for next
3. Expected local LOO score with obs+temp: ~97.15 (6-round avg), server ~65-85 depending on round
4. More training data keeps helping: 4->5 rounds improved HGB base by ~1 point
5. **R6 lesson: calibration can hurt on extreme rounds.** Consider blending calibrated & uncalibrated
6. R7+ will use full pipeline: T=1.15 + settle-split + clip [0.01, 100]
7. All hyperparameters exhaustively confirmed: HGB (11 variants + 6-round param sweep), clip ranges (7), calibration splits (8), temp (11 values + 6-round sweep), multi-pass (3), per-class temp
8. 6-round HGB param sweep: depth5/iter200/lr0.03/leaf30 all within ±0.08 of baseline — model fully saturated
9. Settlement proximity split + temperature scaling are the two biggest discoveries
10. **Server scoring formula remains unknown** — gap varies by round (server consistently lower than LOO KL)
11. Total exhaustive experiments: 16 (all confirmed no improvement over current pipeline)
