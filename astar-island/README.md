# Astar Island — Viking Civilisation Prediction

## Setup
```bash
pip install -r requirements.txt
```

## Configuration
1. Sign in at `app.ainm.no` with Google
2. Get your auth token (check browser dev tools → Network tab → look for `Authorization` header or cookie on API calls)
3. Set in `config.py`:
   - `AUTH_TOKEN` — your bearer token
   - `TEAM_ID` — your team ID
   - `ROUND_ID` — current round ID (visible on the Astar Island page)

## Usage
```bash
# 1. Run full pipeline (observe → analyze → predict → submit)
python main.py

# 2. Or step by step:
python main.py observe      # Spend queries to observe viewports
python main.py analyze      # Analyze observations, find patterns
python main.py predict      # Generate probability tensors
python main.py submit       # Submit predictions for all 5 seeds

# 3. Visualize current state
python main.py visualize
```

## Strategy
- 50 queries shared across 5 seeds
- Phase 1: 9 viewports per seed (45 queries) for ~full map coverage on each seed
- Phase 2: 5 remaining queries target high-uncertainty regions
- Prediction: spatial kernel smoothing + terrain adjacency model + empirical frequencies

## File Structure
```
config.py          — Auth, team, round settings
api_client.py      — REST API wrapper
observer.py        — Viewport placement & observation strategy
analyzer.py        — Pattern extraction from observations
predictor.py       — Probability tensor generation
submitter.py       — Prediction submission
visualizer.py      — Map & prediction visualization
main.py            — Orchestrator
data/              — Cached observations & predictions
```
