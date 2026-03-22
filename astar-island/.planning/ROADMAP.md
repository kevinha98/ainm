# Roadmap: Astar Island Prediction System

## Overview

Build a competition-winning prediction system in 4 phases: first clean up the codebase and data pipeline, then build a smart observation strategy, then optimize the prediction models, and finally automate everything for hands-free round participation.

## Phases

- [x] **Phase 1: Data Pipeline & Observation** — Rock-solid data capture + strategic query spending
- [ ] **Phase 2: Model Optimization** — Find the best prediction model through systematic experimentation
- [ ] **Phase 3: Auto-Runner** — Single-command pipeline: detect round → observe → predict → submit
- [ ] **Phase 4: Learning Loop** — Use completed round analysis to improve models over time

## Phase Details

### Phase 1: Data Pipeline & Observation
**Goal**: Never lose observation data again. Spend queries wisely.
**Depends on**: Nothing
**Requirements**: PRED-03, PRED-04, PRED-06
**Success Criteria**:
  1. All /simulate responses saved to disk in structured format
  2. Strategic viewport selection covers diverse terrain
  3. Observations resume if pipeline interrupted mid-round

Plans:
- [x] 01-01: Clean data pipeline with disk-persisted observations
- [x] 01-02: Strategic viewport selector (maximize information gain)

### Phase 2: Model Optimization
**Goal**: Find the best-performing prediction model through systematic comparison.
**Depends on**: Phase 1
**Requirements**: PRED-04, PRED-05, OPS-02
**Success Criteria**:
  1. Multiple model architectures tested and benchmarked
  2. Ensemble weights optimized via cross-validation
  3. Confidence on non-trivial cells > 60%

Plans:
- [ ] 02-01: Systematic model comparison (current 5 models + new candidates)
- [ ] 02-02: Hyperparameter optimization + ensemble weight tuning
- [ ] 02-03: Transition learning from observation data (when available)

### Phase 3: Auto-Runner
**Goal**: Fully automated round participation.
**Depends on**: Phase 2
**Requirements**: OPS-01, PRED-02
**Success Criteria**:
  1. Single `python run.py` detects active round, observes, predicts, submits
  2. Handles rate limits, auth refresh, errors gracefully
  3. Logs everything for post-mortem

Plans:
- [ ] 03-01: Round detector + orchestrator
- [ ] 03-02: Error handling + retry logic

### Phase 4: Learning Loop
**Goal**: Get smarter after every round.
**Depends on**: Phase 3
**Requirements**: PRED-07
**Success Criteria**:
  1. Post-round analysis auto-downloads ground truth
  2. Transition matrix updated from real data
  3. Model accuracy improves round-over-round

Plans:
- [ ] 04-01: Post-round analysis pipeline
- [ ] 04-02: Adaptive model retraining

---
*Last updated: 2026-03-19*
