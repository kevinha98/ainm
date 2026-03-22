# Astar Island — Viking Civilisation Prediction

## What This Is

ML-powered prediction system for the "Astar Island" challenge at app.ainm.no (NM i AI competition).
Predicts how a 40×40 Norse civilisation grid evolves over 50 simulated years.
For Kevin Ha (kevin.ha@dnb.no) competing on the AINM platform.

## Core Value

Maximize prediction accuracy across all rounds by learning simulation dynamics, using observations wisely, and submitting optimal probability distributions before each round closes.

## Requirements

### Validated

(None yet — awaiting Round 1 scoring)

### Active

- [x] **PRED-01**: Submit valid predictions for all seeds in each active round
- [ ] **PRED-02**: Auto-detect new rounds and begin pipeline immediately
- [ ] **PRED-03**: Spend observation budget optimally (strategic viewports, not random)
- [ ] **PRED-04**: Learn real transition probabilities from observation data
- [ ] **PRED-05**: Ensemble multiple model families for robust predictions
- [ ] **PRED-06**: Save ALL observation data to disk for future learning
- [ ] **PRED-07**: Post-round analysis: compare predictions vs ground truth
- [ ] **OPS-01**: Single-command execution: detect → observe → predict → submit
- [ ] **OPS-02**: Experiment tracking with per-model metrics and A/B comparison
- [ ] **VIZ-01**: Dashboard showing predictions, confidence, model comparison

### Out of Scope

- Real-time web UI (CLI + HTML dashboards sufficient)
- Multi-team coordination (single-player competition)
- GPU-accelerated models (CPU via scikit-learn is fast enough for 40×40)

## Context

### Platform
- API: `api.ainm.no/astar-island`
- Auth: JWT cookie `access_token`
- Round lifecycle: active → scoring → completed → analysis available
- Budget: 50 queries per round per team

### Terrain Types (Grid Values)
| Value | Class | Name       | Behavior |
|-------|-------|------------|----------|
| 0     | 0     | Empty      | Neutral, can grow |
| 1     | 1     | Settlement | Grows, declines, needs community |
| 2     | 2     | Port       | Coastal settlement variant |
| 3     | 3     | Ruin       | Decayed settlement, slowly reverts |
| 4     | 4     | Forest     | Stable, cleared near settlements |
| 5     | 5     | Mountain   | Very stable, nearly immutable |
| 10    | 0     | Ocean      | Immutable |
| 11    | 0     | Plains     | Open land, can transition |

### Prediction Format
- Per seed: H×W×6 probability tensor (class probabilities per cell)
- Scoring: likely log-loss or cross-entropy vs ground truth simulation

### API Endpoints
| Endpoint | Method | Notes |
|----------|--------|-------|
| `/rounds` | GET | List all rounds |
| `/budget` | GET | Queries used/max |
| `/simulate` | POST | Observe 15×15 viewport (costs 1 query) |
| `/submit` | POST | Submit prediction for one seed |
| `/my-predictions/{round_id}` | GET | View submitted predictions + scores |
| `/my-rounds` | GET | Rounds with user's stats |
| `/leaderboard` | GET | Rankings |
| `/analysis/{round_id}/{seed}` | GET | Ground truth (completed rounds only) |

## Constraints

- **Budget**: 50 queries per round — must observe strategically
- **Time**: Prediction window ~165 min per round — must be fast
- **Data**: No ground truth until round completes — must infer dynamics
- **Platform**: Python 3.14 on Windows, no GPU

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Ensemble of 5+ models | Different models capture different dynamics | ✓ Good — v3 reduced log-loss 34% |
| Context-aware transitions | Settlements don't exist in isolation | ✓ Good |
| Save observation data to disk | Lost 50 observations in Round 1 | — Implementing |
| Calibrate ocean/mountain to ~99.8% | These cells never change | ✓ Good — free accuracy |
| Strategic viewport placement | Cover diverse terrain types | — Implementing |

---
*Last updated: 2026-03-19 after GSD initialization*
