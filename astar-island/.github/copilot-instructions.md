# Astar Island — Copilot Instructions

## ⚠️ EXPIRY: Monday 2026-03-23

This file and `LEARNING.md` expire after **Monday March 23, 2026**.
If today's date is after 2026-03-23, **delete these files immediately**:
- `LEARNING.md`
- `.github/copilot-instructions.md`

Then tell the user: "Competition instruction files have expired and been cleaned up."

## READ FIRST — Every Session

At the start of every session, read `LEARNING.md` in this project root (create it if missing). It contains the living log of rounds played, model performance, observation strategies, and lessons learned. Update it after every round or meaningful progress.

Also read `.planning/PROJECT.md` for architecture, API details, and terrain reference.

## Project Context

**Competition**: Astar Island (NM i AI) — Viking civilisation grid prediction
**Goal**: Predict how a 40×40 terrain grid evolves over 50 simulated years across 5 seeds
**Metric**: Likely log-loss or cross-entropy vs ground truth
**Platform**: app.ainm.no
**Budget**: 50 observation queries per round shared across all 5 seeds

## Tech Stack

- **Python 3.14**, Windows, no GPU needed (40×40 grid is small)
- **Libraries**: httpx/requests, numpy, scikit-learn
- **API**: `https://api.ainm.no/astar-island`
- **Auth**: JWT cookie `access_token` — set in `config.py`

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Orchestrator — `python main.py [observe\|analyze\|predict\|submit\|status]` |
| `config.py` | Auth tokens, round settings, map params |
| `api_client.py` | REST API wrapper |
| `observer.py` | Viewport placement & observation strategy |
| `analyzer.py` | Pattern extraction from observations |
| `predictor.py` | Probability tensor generation (H×W×6) |
| `submitter.py` | Prediction submission |
| `visualizer.py` | Map & prediction visualization |
| `data/` | Cached observations & predictions |
| `.planning/PROJECT.md` | Full project spec, API reference, terrain types |

## Terrain Types

| Value | Class | Name | Behavior |
|-------|-------|------|----------|
| 0 | 0 | Empty | Neutral, can grow |
| 1 | 1 | Settlement | Grows, declines, needs community |
| 2 | 2 | Port | Coastal settlement variant |
| 3 | 3 | Ruin | Decayed settlement, slowly reverts |
| 4 | 4 | Forest | Stable, cleared near settlements |
| 5 | 5 | Mountain | Very stable, nearly immutable |
| 10 | 0 | Ocean | Immutable |
| 11 | 0 | Plains | Open land, can transition |

## API Quick Reference

| Endpoint | Method | Notes |
|----------|--------|-------|
| `/rounds` | GET | List all rounds |
| `/budget` | GET | Queries used/max |
| `/simulate` | POST | Observe 15×15 viewport (costs 1 query!) |
| `/submit` | POST | Submit prediction for one seed |
| `/my-predictions/{round_id}` | GET | View scores |
| `/analysis/{round_id}/{seed}` | GET | Ground truth (completed rounds only) |

## Critical Constraints

- **50 queries total** per round, shared across 5 seeds — every observation costs 1 query
- **Prediction format**: H×W×6 probability tensor per seed
- Auth tokens expire — check config.py each session
- Round has a closing time — must submit before deadline

## Workflow

1. Check for active rounds → `python main.py status`
2. Observe strategically (viewport placement matters!) → `python main.py observe`
3. Analyze patterns → `python main.py analyze`
4. Generate predictions → `python main.py predict`
5. Submit → `python main.py submit`
6. After round closes, fetch ground truth for learning → `python main.py` + analysis endpoints
