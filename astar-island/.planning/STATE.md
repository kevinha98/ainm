# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-19)

**Core value:** Maximize prediction accuracy across all rounds
**Current focus:** Phase 2 — Model Optimization + Learning Loop

## Current Position

Phase: 2 of 4 (Model Optimization)
Status: v5 submitted, awaiting Round 1 results
Last activity: 2026-03-19 — v5 submitted (dynamic MC, adaptive sharpening, terrain-aware calibration)

Progress: [#####.....] 50%

## Codebase (cleaned)

```
C:\ainm\astar-island\
├── src/
│   ├── settings.py   — Config, constants, paths
│   ├── api.py         — AstarAPI class (all HTTP)
│   ├── observer.py    — Viewport planning + crash-safe observation
│   └── models.py      — 6 models + ensemble/calibrate/sharpen
├── run.py             — Auto-runner: detect → observe → predict → submit (35s)
├── analyze_round.py   — Post-round: ground truth → learn transitions → evaluate
├── .planning/         — GSD project docs
├── data/              — Predictions, logs, round data, visualizations
├── LEARNING.md        — Quick reference
├── README.md
└── requirements.txt
```

## Performance Metrics

**Velocity:**
- Total plans completed: 4 (Phase 1 + modular rewrite)
- Total execution time: ~2 hours across sessions

**Prediction versions:**

| Version | Models | Entropy | Confidence | Log-loss | Notes |
|---------|--------|---------|------------|----------|-------|
| v2 | 4 | ~0.85 | ~68% | ~0.450 | Basic ensemble |
| v3 | 5 | 0.745 | 75.5% | 0.299 | +CalibrationCA |
| v4 | 6 | 0.742 | 75.7% | 0.297 | +ObsDirect, modular |
| v5 | 6 | 0.821 | 73.0% | 0.339 | Dynamic MC, adaptive sharpening, terrain calibration |

## Accumulated Context

### Decisions

- Lost 50 observations in Round 1 (not saved to disk) → fixed with crash-safe persistence
- v4 ensemble (6 models) is best so far: entropy 0.742, conf 75.7%
- Ocean/mountain calibration gives near-free accuracy on ~13% of cells
- `/rounds/{id}` is the correct endpoint for initial_states (not `/rounds` list)
- Moved project from OneDrive to C:\ainm for speed

### Key Data

- Round 1: ID 71451d74..., active, 50/50 budget spent
- Grid: 40×40, 5 seeds, 6 prediction classes, 50-year simulation
- Round weight: 1.05
- Python 3.14, numpy, scipy, scikit-learn, requests

### Blockers/Concerns

- No ground truth until Round 1 completes
- Exact scoring function unknown (likely log-loss or cross-entropy)
- Can't observe more this round (budget exhausted)

## Next Actions

1. **When Round 1 completes**: `python analyze_round.py` → learn real transition probabilities
2. **When new round appears**: `python run.py` → auto-detect, observe (save budget wisely), predict, submit
3. **Model improvements**: Use learned_transitions.json from analyze_round.py to improve Markov model
4. **Observation strategy**: Focus viewport queries on dynamic regions (settlements, forest-settlement boundaries)

## Session Continuity

Last session: 2026-03-19
Stopped at: v4 submitted, codebase cleaned, analysis pipeline ready
Resume with: Check round status → run analyze_round.py if completed → run.py for new rounds
