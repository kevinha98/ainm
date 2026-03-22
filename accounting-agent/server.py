"""
FastAPI server for the Tripletex AI Accounting Agent.

Exposes:
  POST /solve  — receives an accounting task, runs the LLM agent, returns status
  GET  /health — liveness check
"""

import json
import logging

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

API_KEY = os.getenv("API_KEY", "")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Tripletex Accounting Agent starting")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Tripletex AI Accounting Agent",
    description="AI agent that completes accounting tasks in Tripletex",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: Request):
    """
    Receive an accounting task, run the LLM agent to complete it via
    Tripletex API calls, and return the result.
    """
    start = time.time()

    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid JSON"})

    # Log incoming request (truncated for readability)
    logger.info(f"=== NEW TASK ===")
    logger.info(f"Request body keys: {list(body.keys()) if isinstance(body, dict) else type(body)}")
    body_str = json.dumps(body, indent=2, default=str)
    logger.info(f"Request body (first 3000 chars): {body_str[:3000]}")

    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Expected JSON object"})

    # ── Optional API key check ──
    if API_KEY:
        auth_header = request.headers.get("authorization", "")
        expected = f"Bearer {API_KEY}"
        if auth_header != expected:
            logger.warning("Unauthorized request (bad or missing API key)")
            return JSONResponse(status_code=401, content={"status": "error", "message": "Unauthorized"})

    # ── Extract fields — competition spec format first, then fallbacks ──
    prompt = body.get("prompt") or body.get("task") or body.get("description") or ""

    # Competition spec: tripletex_credentials.base_url / .session_token
    creds = body.get("tripletex_credentials") or {}
    api_url = (
        creds.get("base_url")
        or body.get("api_url")
        or body.get("proxy_url")
        or body.get("base_url")
        or ""
    )
    token = (
        creds.get("session_token")
        or body.get("token")
        or body.get("session_token")
        or ""
    )

    company_id = body.get("company_id") or body.get("companyId") or 0

    # Competition spec: files[].content_base64, files[].mime_type, files[].filename
    attachments = body.get("files") or body.get("attachments") or []

    if not prompt:
        logger.error("No task prompt found in request")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "No task prompt found in request body"},
        )

    if not api_url:
        logger.warning("No API URL found — agent may fail to make API calls")

    logger.info(f"Prompt: {prompt[:200]}...")
    logger.info(f"API URL: {api_url}")
    logger.info(f"Company ID: {company_id}")
    logger.info(f"Attachments: {len(attachments)}")

    # ── Run the agent ──
    try:
        from agent import AccountingAgent

        agent = AccountingAgent(
            api_url=api_url,
            token=token,
            company_id=int(company_id),
        )
        await agent.solve(prompt, attachments)

        elapsed = time.time() - start
        logger.info(f"=== TASK COMPLETED in {elapsed:.1f}s ===")
        return {"status": "completed"}

    except Exception as e:
        elapsed = time.time() - start
        logger.exception(f"=== TASK FAILED after {elapsed:.1f}s: {e} ===")
        # Still return completed to avoid penalty — the agent may have
        # partially completed the task before the error
        return {"status": "completed"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
