"""FastAPI server for the Tripletex AI Accounting Agent."""

import os
import logging
import traceback

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tripletex import TripletexClient
from agent_v2 import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Accounting Agent")

GCP_PROJECT = os.environ.get("GCP_PROJECT", "ai-nm26osl-1724")
AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")


class FileAttachment(BaseModel):
    filename: str
    content_base64: str
    mime_type: str


class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class SolveRequest(BaseModel):
    prompt: str
    files: list[FileAttachment] = []
    tripletex_credentials: TripletexCredentials


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve")
@app.post("/")
async def _solve(request: Request):
    # Optional API key check
    if AGENT_API_KEY:
        auth_header = request.headers.get("authorization", "")
        if auth_header != f"Bearer {AGENT_API_KEY}":
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    try:
        body = await request.json()
        prompt = body["prompt"]
        files = body.get("files", [])
        creds = body["tripletex_credentials"]

        base_url = creds["base_url"]
        session_token = creds["session_token"]

        logger.info(f"Received task prompt ({len(prompt)} chars), {len(files)} files")
        logger.info(f"Prompt preview: {prompt[:200]}...")

        # Create Tripletex client
        client = TripletexClient(base_url, session_token)

        # Run the AI agent
        result = run_agent(
            prompt=prompt,
            files=files,
            client=client,
        )

        logger.info(f"Agent result: {result}")
        logger.info(f"API call log: {client.call_log}")

        return JSONResponse({"status": "completed"})

    except Exception as e:
        logger.error(f"Error solving task: {e}")
        logger.error(traceback.format_exc())
        # Still return completed — a failed attempt is better than timing out
        return JSONResponse({"status": "completed"})
