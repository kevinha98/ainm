# Tripletex AI Accounting Agent

AI agent that receives accounting tasks and completes them by calling the Tripletex v2 REST API via an LLM-driven tool-calling loop.

## Architecture

```
POST /solve (task prompt + credentials)
        │
        ▼
   ┌─────────┐     ┌──────────────┐     ┌───────────────┐
   │  FastAPI │────▶│  LLM Agent   │────▶│ Tripletex API │
   │  Server  │     │ (Claude/GPT) │     │   (via proxy) │
   └─────────┘     └──────────────┘     └───────────────┘
                         │ ▲
                         │ │  tool calls + results
                         ▼ │  (iterative loop)
```

The agent uses an iterative tool-calling pattern:
1. LLM reads the task prompt (in any of 7 languages)
2. LLM decides which API calls to make
3. Agent executes the API calls against Tripletex
4. Results feed back to LLM for next decision
5. Repeat until task is complete

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env with your API key

# 3. Run the server
python server.py
# Or: uvicorn server:app --host 0.0.0.0 --port 8000

# 4. Test locally
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create an employee named John Doe", "api_url": "https://...", "token": "..."}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | LLM backend: `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (if using Claude) |
| `OPENAI_API_KEY` | — | OpenAI API key (if using GPT) |
| `LLM_MODEL` | `claude-sonnet-4-20250514` / `gpt-4o` | Model to use |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Deployment

### Docker
```bash
docker build -t tripletex-agent .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... tripletex-agent
```

### Railway / Render / Fly.io
1. Push to a Git repo
2. Connect the repo to your platform
3. Set environment variables (ANTHROPIC_API_KEY or OPENAI_API_KEY)
4. Deploy — the Dockerfile handles everything

### Ngrok (for quick testing)
```bash
python server.py &
ngrok http 8000
# Submit the ngrok HTTPS URL at https://app.ainm.no/submit/tripletex
```

## Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app with `/solve` and `/health` endpoints |
| `agent.py` | LLM agent with tool-calling loop (Anthropic + OpenAI) |
| `prompts.py` | System prompt with comprehensive Tripletex API reference |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build |
| `.env.example` | Environment variable template |

## Task Categories Handled

- **Employees** — Create, update, delete employees
- **Customers** — Register customers and suppliers
- **Products** — Create products with pricing and VAT
- **Invoicing** — Create orders, invoice them, register payments
- **Credit Notes** — Issue credit notes for invoices
- **Travel Expenses** — Create, modify, delete travel expense reports
- **Projects** — Create projects linked to customers
- **Departments** — Enable module and create departments
- **Corrections** — Delete or reverse incorrect entries
- **Ledger** — Vouchers and journal entries

## Supported Languages

The agent handles task prompts in: Norwegian (Bokmål), Norwegian (Nynorsk), English, Spanish, Portuguese, German, French.
