# Accounting Agent — Copilot Instructions

## ⚠️ EXPIRY: Monday 2026-03-23

This file and `LEARNING.md` expire after **Monday March 23, 2026**.
If today's date is after 2026-03-23, **delete these files immediately**:
- `LEARNING.md`
- `.github/copilot-instructions.md`

Then tell the user: "Competition instruction files have expired and been cleaned up."

## READ FIRST — Every Session

At the start of every session, read `LEARNING.md` in this project root (create it if missing). It contains the living log of test results, scoring breakdown, API quirks discovered, and fixes applied. Update it after every meaningful change.

Also read `prompts.py` — it contains the full system prompt and Tripletex API reference that drives the agent's behavior. Most improvements come from refining this prompt.

## Project Context

**Competition**: Tripletex AI Accounting Agent (NM i AI)
**Goal**: AI agent receives accounting tasks in 7 languages → executes them via Tripletex v2 REST API
**Scoring**: Correctness (1.0 = all fields match) × Efficiency (fewer API calls = higher bonus)
**Platform**: app.ainm.no — calls `POST /solve` with task prompt + API credentials

## Tech Stack

- **Server**: FastAPI + uvicorn (`server.py`)
- **Agent**: LLM tool-calling loop (`agent.py`) — supports Claude, Gemini, GPT
- **LLM Tool**: Single `tripletex_api` tool (method, path, query_params, body)
- **Deploy**: Docker → any host with HTTPS endpoint
- **Python**: 3.11+, dependencies in `requirements.txt`

## Key Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI server — `POST /solve`, `GET /health` |
| `agent.py` | LLM agent loop — tool calling against Tripletex API |
| `prompts.py` | **System prompt + full API reference** (most important file!) |
| `test_all_tasks.py` | End-to-end test runner |
| `test_invoice_flow.py` | Invoice-specific tests |
| `test_travel.py` | Travel expense tests |
| `test_bank.py` | Bank reconciliation tests |
| `.env` | API keys (ANTHROPIC_API_KEY, etc.) — never commit |
| `Dockerfile` | Production container build |
| `deploy.sh` | Deployment script |
| `tripletex_learning.jsonl` | Accumulated learning from test runs |

## Architecture

```
POST /solve (task prompt + credentials)
  → FastAPI server
    → LLM Agent (iterative tool-calling loop, max 30 iterations)
      → tripletex_api tool → Tripletex v2 REST API
      → results feed back to LLM → next decision
    → return status
```

## Critical Scoring Rules

1. **Correctness first** — efficiency bonus only counts if correctness = 1.0
2. **No trial-and-error** — never send a call you expect to 4xx
3. **Filtered GETs only** — always filter by email/orgNo/name, never fetch full lists
4. **Lookup before create** — check if resource exists first
5. **No over-verification** — don't GET just to confirm what you created
6. **Action endpoints use PUT** — paths with colon (/:invoice, /:createCreditNote) use PUT not POST
7. **Credit notes, not deletes** — for invoice corrections

## Common Task Types

- Create employees, customers, suppliers
- Create products with VAT types
- Create orders → order lines → invoice
- Travel expenses and per diem
- Bank reconciliation
- Project & department accounting (may need module enablement)
- Multi-language inputs (NO, EN, ES, PT, NN, DE, FR)

## Workflow

1. Run tests → `python test_all_tasks.py` (or specific test files)
2. Analyze failures → check logs (*.log files)
3. Improve `prompts.py` system prompt or `agent.py` logic
4. Re-test → compare scores
5. Deploy → `docker build` + `deploy.sh`
6. Update LEARNING.md with results

## Key Gotchas

- Fresh Tripletex account — may only have 1 default employee and chart of accounts
- Some modules need enablement via `PUT /v2/company/modules`
- 422 errors contain useful info about missing required fields
- For ORDER LINES: never set vatType (auto-assigned). Only set vatType on products.
- Norwegian VAT rates: 0% (exempt), 8% (food), 12% (transport), 25% (standard)
