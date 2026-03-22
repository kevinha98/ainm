# Astar Island — Viking Civilisation Prediction

## Setup

```bash
pip install -r requirements.txt
export AINM_TOKEN="<your-token>"   # from browser cookies on app.ainm.no
```

## Configuration

Set environment variable `AINM_TOKEN` with your auth token from browser dev tools (cookie `access_token` on api.ainm.no requests).

Constants (grid size, classes, etc.) are in `src/settings.py`.

## Usage

```bash
# Full pipeline: observe → predict → submit
python run.py
```

## Strategy

- 50 queries shared across 5 seeds
- Phase 1: 9 viewports per seed (45 queries) for ~full map coverage on each seed
- Phase 2: 5 remaining queries target high-uncertainty regions
- Prediction: 6-model ensemble (Markov, Monte Carlo, HGB, MRF, settlement CA, observation-direct)
- Calibration and sharpening post-processing

## File Structure

```text
run.py             — Main entry point (observe → predict → submit)
src/settings.py    — API config, grid constants, class mappings
src/api.py         — REST API client
src/observer.py    — Viewport placement & observation strategy
src/models.py      — 6 prediction models + ensemble + calibration
simulator/         — Cell-based simulation engine
data/              — Cached observations & predictions
```
