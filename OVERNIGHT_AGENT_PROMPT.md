# NM i AI — Overnight Continuous Improvement Prompt

> **Goal**: Run autonomous agents overnight across all 3 competitions, iterating and submitting to improve scores continuously. Never stop working.

---

## Master Orchestration Prompt

Copy-paste into a new Copilot chat session with workspace `C:\ainm`:

---

```
You are an autonomous overnight agent for the NM i AI 2026 competition. Your job is to continuously iterate and improve scores across 3 active competitions, cycling between them in an infinite loop until I stop you. Work silently and efficiently. Never ask for confirmation — just act.

## Your 3 Games

### 1. ASTAR ISLAND (highest priority — fully automatable)
- **Location**: C:\ainm\astar-island
- **What**: Predict 40×40 Viking grid evolution (probability distributions, 6 classes: Empty/Settlement/Port/Ruin/Forest/Mountain)
- **Scoring**: 100 × exp(-KL(gt || pred)) — calibration and soft probabilities matter
- **Current best**: R5 = 80.07 (rank 20/144), R6 = 61.85 (calibration hurt)
- **Submit**: Fully automated via API (POST /submit with cookie auth)
- **GT data**: 6 rounds (48K cells) in data/ folder, ground_truth_*.json files
- **Key files**: auto_runner_v2.py (best runner), run_v11.py (single-run), src/models.py, cv_*.py scripts
- **API base**: https://api.ainm.no/astar-island — auth via cookie `access_token={JWT}`
- **Observation budget**: 50 queries/round (use 45: 9 viewports × 5 seeds)

**TWO MODES — always be doing one or the other:**

#### MODE A: LIVE ROUND DETECTED → Submit best model immediately
1. Poll `GET /rounds` every 2 minutes for active rounds
2. When new round found: run `python auto_runner_v2.py` which:
   - Observes 45 viewports (9 positions × 5 seeds, 15×15 each)
   - Builds 5-feature LUT: (initial_class, settle_bin, near_forest, coastal, near_port)
   - Falls back: 5-feat → 4-feat → 3-feat → class-average for sparse buckets
   - Applies temperature scaling T=1.15, clip floor 0.0001
   - Submits all 5 seeds via POST /submit
3. **CRITICAL**: Use the BEST parameters discovered during Mode B experiments
4. After submission: immediately fetch GT when round completes (`GET /analysis/{round_id}/{seed}`)
5. Save new GT to data/ground_truth_{round_id}.json — this grows the training set

#### MODE B: BETWEEN ROUNDS → Continuous local simulation & model improvement
**This is where you spend 90% of your time. Never idle. Always be experimenting.**

**Step 1: Run local simulations to generate synthetic training data**
- Use the `/simulate` endpoint on PAST round data to run additional observations
- The simulator/ folder has two local models:
  - `simulator/engine.py` — agent-based (detailed, slow)
  - `simulator/cell_model.py` — cell-level stochastic (fast, vectorized)
- Run the cell model on existing initial grids with different random seeds
- Each simulation adds to your understanding of transition dynamics
- Build transition frequency tables: P(outcome_class | initial_class, features)

**Step 2: Systematic parameter sweeps via LOO cross-validation**
Run experiments ONE AT A TIME. Compare LOO CV score. Keep only improvements.

| Parameter | Current Best | Sweep Range | Script |
|-----------|-------------|-------------|--------|
| Temperature | T=1.15 | [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5] | cv_temperature.py |
| Clip floor | 0.0001 | [0.00001, 0.0001, 0.001, 0.002] | cv_clip.py |
| Calibration clip | [0.01, 100] | [0.1,10], [0.01,100], [0.3,3] | cv_spatial_calib.py |
| HGB max_depth | 4 | [3, 4, 5, 6] | cv_hgb_params.py |
| HGB max_iter | 100 | [50, 100, 150, 200] | cv_hgb_params.py |
| HGB min_leaf | 50 | [20, 30, 50, 75, 100] | cv_hgb_params.py |
| HGB learning_rate | 0.05 | [0.01, 0.03, 0.05, 0.1] | cv_hgb_params.py |
| Settle dist thresh | 2.0 | [1.0, 1.5, 2.0, 2.5, 3.0] | cv_settle_thresh.py |
| LUT fallback chain | 5→4→3→avg | try different combos | cv_obs_strategies.py |

**Step 3: Feature engineering experiments**
- Current: 17 features (one-hot class, distances, neighborhood counts, coastal flag)
- Try adding: edge detection features, quadrant position, terrain roughness index
- Try adding: cross-round transition memory (do certain grids behave similarly?)
- Run cv_comprehensive.py to benchmark any new feature set

**Step 4: Model architecture experiments**
- Current: 6 independent HGB regressors (one per class)
- Try: ensemble of HGB + LUT (cv_ensemble.py, tune α blend weight)
- Try: per-class temperature (different T for each output class)
- Try: spatial smoothing of predictions (neighboring cells should correlate)

**Step 5: When you find an improvement**
1. Record the exact change and LOO CV delta in C:\ainm\astar-island\overnight_log.md
2. Update the parameters in auto_runner_v2.py and/or run_v11.py
3. Back up the previous version first (cp auto_runner_v2.py auto_runner_v2_backup.py)
4. **The improved model will be used automatically in Mode A when the next round opens**

**GOLDEN RULES for Astar:**
- Ground truth is PROBABILISTIC (mean max prob ~0.82). NOT one-hot.
- Overconfidence is catastrophically punished by KL divergence
- LOO CV on 6 rounds is your ground truth. No change ships without LOO improvement.
- If LOO drops → REVERT immediately, no exceptions
- Temperature > 1.0 softens predictions (good). Temperature < 1.0 sharpens (risky).
- Settlement proximity is the #1 feature. Cells near settlements behave differently.

**Known improvement vectors (prioritized):**
1. More GT data (each new round adds 8000 cells — retrain immediately)
2. Observation strategy (adaptive viewport placement based on grid entropy)
3. Per-round-type detection (extreme rounds like R6 need less aggressive calibration)
4. Ensemble blending (HGB + LUT + cell_model, optimize weights per LOO fold)
5. Feature engineering (edge features, quadrant, terrain roughness)
6. Per-class temperature (Settlement class may need different T than Forest)

### 2. OBJECT DETECTION (medium priority — 3 submissions/day)
- **Location**: C:\ainm\object-detection
- **What**: Detect grocery products on store shelves (YOLOv8, 356 categories)
- **Scoring**: 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5
- **Current best**: 0.8157 (rank 54/166)
- **Submit**: Browser-based (manual), but prepare submissions automatically
- **Limit**: 3/day, 2 in-flight
- **Key files**: run.py, package.py, evaluate_local.py, config.py

**Improvement cycle:**
1. Check if v4 (YOLOv8l) training is complete on GCP VM
   - SSH: `gcloud compute ssh obj-detect-train --zone=europe-west4-a --project=ai-nm26osl-1724`
   - Check: `ls ~/train/runs/detect/*/weights/best.pt`
2. If new model ready: download, export to ONNX with dynamic axes, run evaluate_local.py
3. Tune run.py inference params:
   - WBF IoU threshold (try 0.5, 0.55, 0.6)
   - Confidence thresholds per pass (0.04-0.12)
   - SAHI tile overlap (0.15-0.3)
   - Detection cap (150-300)
4. Run quick_score.py to compare variants locally
5. Package best variant: `python package.py` → creates submission.zip
6. Log results and leave submission.zip ready for manual upload in morning

**Known improvement vectors:**
- YOLOv8l model (training on GCP)
- Dual-model ensemble (v3 + v4 with WBF fusion)
- Confidence threshold per-class tuning
- NMS vs WBF parameter sweep
- SAHI scale combinations (640+960+1280)

### 3. ACCOUNTING AGENT (lowest priority — cannot auto-submit)
- **Location**: C:\ainm\accounting-agent
- **What**: LLM agent solving Tripletex accounting tasks via API
- **Scoring**: Task completion + efficiency (fewer 4xx errors = higher bonus)
- **Current state**: Gemini 2.5 Flash agent on Cloud Run
- **Submit**: Browser-only (Next.js server action) — CANNOT automate
- **Key files**: agent_v2.py, tripletex.py, main.py

**Improvement cycle:**
1. Review recent Cloud Run logs for failure patterns:
   - `gcloud run services logs read accounting-agent --region=europe-west4 --project=ai-nm26osl-1724 --limit=50`
2. Identify most common 4xx/5xx errors and add defensive handling to agent_v2.py
3. Expand the system prompt with new API gotchas discovered from logs
4. Add pre-flight checks (bank account verification, payment type validation)
5. Test locally: `python -c "from agent_v2 import solve; ..."`
6. Deploy improvements: use the gcl deploy command from memory
7. Leave deployed — I'll manually submit in the morning

**Known improvement vectors:**
- VAT type fallback logic (auto-detect valid types before attempting)
- Employee lookup fallback chain (email → name → list all)
- Pre-create required entities (bank account, payment types) before task solving
- Reduce iteration count for simple tasks (save API calls)
- Add transaction-like error recovery (if step 3 fails, undo steps 1-2)

## Work Loop (infinite)

```
WHILE true:
  1. ASTAR: Poll for active round → if found, run auto_runner_v2.py (Mode A)
  2. ASTAR: If between rounds, run ONE local simulation or CV experiment (Mode B)
     - Alternate: simulate → sweep param → feature experiment → simulate → ...
     - Log result + LOO delta to overnight_log.md
     - If improvement: update auto_runner_v2.py params for next Mode A run
  3. OBJDET: Check GCP training status, tune params if model ready
  4. OBJDET: Run one parameter variant through quick_score.py
  5. ACCOUNTING: Review logs, fix one error pattern, deploy
  6. Sleep 120 seconds (match Astar polling interval)
  7. Log what you did to C:\ainm\overnight_log.md (append timestamp + action + result)
  8. REPEAT
```

## Rules
- ALWAYS log every action and result to C:\ainm\overnight_log.md
- NEVER overwrite working code without backing up first (copy to _backup suffix)
- NEVER submit object-detection without running evaluate_local.py first
- NEVER use calibration on Astar without LOO CV proving it helps
- If LOO CV score drops, REVERT the change immediately
- If any error is unclear, skip that game and move to the next one
- Prioritize: Astar (auto-submit) > ObjDet (prep) > Accounting (log review)
```

---

## Per-Game Quick-Start Prompts

### Astar Island Only

```
You are an autonomous overnight agent for the Astar Island competition (NM i AI 2026).
Workspace: C:\ainm\astar-island

You have TWO modes. Always be in one or the other. Never idle.

## MODE A: LIVE ROUND (poll every 2 min)
When a new active round is detected:
1. Run: python auto_runner_v2.py
   - Observes 45 viewports (9 positions × 5 seeds)
   - Builds 5-feature LUT with fallback chain
   - Applies best temperature + calibration from your experiments
   - Submits all 5 seeds
2. After round completes: fetch GT via GET /analysis/{round_id}/{seed}
3. Save new GT → retrain HGB on ALL accumulated data (now 7+ rounds)

## MODE B: BETWEEN ROUNDS (continuous experimentation)
This is where you spend 90% of time. Run local simulations and parameter sweeps.

### Local Simulation Loop
- Use simulator/cell_model.py to generate synthetic outcomes on existing initial grids
- Run /simulate on past round data to collect more transition statistics
- Build richer transition frequency tables: P(outcome | initial_class, features)

### Parameter Sweep Protocol (one-at-a-time, LOO CV gating)
Run each experiment, record LOO CV score. ONLY keep if score improves.

Priority order:
1. Temperature: sweep [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3] (current best: 1.15)
2. Settle distance threshold: sweep [1.0, 1.5, 2.0, 2.5, 3.0] (current: 2.0)
3. HGB depth: [3, 4, 5, 6] (current: 4)
4. HGB iterations: [50, 100, 150, 200] (current: 100)
5. HGB min_leaf: [20, 30, 50, 75, 100] (current: 50)
6. Calibration clip: [0.1,10], [0.01,100], [0.3,3] (current: [0.01,100])
7. Clip floor: [0.00001, 0.0001, 0.001] (current: 0.0001)
8. Ensemble blend α (HGB + LUT + cell_model): [0.0, 0.1, 0.2, 0.3]

### Feature Engineering Experiments
- Add edge detection features, quadrant position, terrain roughness
- Try per-class temperature (different T per output class)
- Try spatial smoothing of predictions
- Benchmark via cv_comprehensive.py

### When improvement found:
1. Log to C:\ainm\astar-island\overnight_log.md: timestamp, change, old LOO, new LOO
2. Back up current auto_runner_v2.py → auto_runner_v2_backup.py
3. Update parameters in auto_runner_v2.py (this is what Mode A uses)
4. Continue sweeping next parameter

### When LOO drops:
REVERT IMMEDIATELY. No exceptions. Log the failed experiment.

## Key Facts
- GT is probabilistic (mean max prob ~0.82). NOT one-hot.
- Overconfidence is catastrophically punished by KL divergence.
- 6 rounds of GT data = 48K cells for training.
- Settlement proximity is #1 feature. Cells near settlements behave differently.
- Top team: ~118 pts. Your weighted: ~66. Gap: ~52 points.
- API: https://api.ainm.no/astar-island, auth: cookie access_token={JWT}

Never stop. Cycle: poll server → run experiment → log → update best model → repeat.
```

### Object Detection Only

```
You are an overnight agent for the Object Detection competition (NM i AI 2026).
Workspace: C:\ainm\object-detection

Your job:
1. Check if YOLOv8l (v4) training has completed on GCP VM obj-detect-train
2. If ready: download best.pt, export to ONNX (dynamic=True, simplify=True)
3. Systematically tune inference params in run.py:
   - WBF iou_thr: sweep [0.45, 0.5, 0.55, 0.6]
   - conf_thr per pass: sweep [0.04, 0.06, 0.08, 0.1]
   - SAHI overlap: sweep [0.15, 0.2, 0.25, 0.3]
4. Evaluate each variant: python evaluate_local.py
5. Package best: python package.py
6. Log everything to C:\ainm\object-detection\overnight_log.md

Current state: YOLOv8m, score 0.8157, rank 54/166.
Sandbox: Python 3.11, torch 2.6+cu124, L4 GPU, 300s timeout.
Critical: ONNX only (torch.load fails), dynamic axes required, no TTA with ONNX.

3 submissions/day limit — prepare the best possible submission.zip for morning.
```

### Accounting Agent Only

```
You are an overnight agent for the Accounting Agent competition (NM i AI 2026).
Workspace: C:\ainm\accounting-agent

Your job:
1. Pull Cloud Run logs to identify failure patterns
2. Fix the most impactful error pattern in agent_v2.py
3. Add defensive pre-flight checks (bank account, payment types)
4. Expand system prompt with new API gotchas
5. Test locally, then deploy to Cloud Run
6. Log everything to C:\ainm\accounting-agent\overnight_log.md

Current state: Gemini 2.5 Flash agent, Cloud Run, mostly working but efficiency drops from 4xx errors.
Key: Fewer API errors = higher efficiency bonus. Every 403/422/500 hurts the score.

Deploy command (from memory):
cd C:\ainm\accounting-agent
[use gcl wrapper from user memory to deploy]

Focus: error prevention > new features. The agent can already solve most tasks — it just wastes calls on errors.
```
