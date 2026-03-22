"""LLM Agent using Vertex AI Gemini to interpret accounting tasks and call Tripletex API."""

import json
import re
from typing import Any

from vertexai.generative_models import GenerativeModel, Part, Content, FunctionDeclaration, Tool

from tripletex import TripletexClient

SYSTEM_PROMPT = """You are an expert accounting AI agent. You receive accounting task prompts (in Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French) and execute them using the Tripletex REST API.

## Your Goal
Interpret the task prompt, determine which Tripletex API calls to make, and execute them correctly with MINIMUM API calls and ZERO errors.

## Key Principles
1. PARSE the prompt fully before making ANY API call. Extract all entity names, field values, relationships.
2. PLAN the exact sequence of API calls needed.
3. EXECUTE precisely — avoid trial-and-error. Every 4xx error hurts your efficiency score.
4. If you created something, use the returned ID — don't fetch it again.
5. Some tasks require creating prerequisites first (e.g., customer before invoice).

## Tripletex API Patterns
- Auth: Basic Auth, username "0", password = session_token (handled for you)
- GET responses: {"fullResultSize": N, "values": [...]}
- POST/PUT responses: {"value": {...}}
- DELETE: returns 204 No Content
- Use ?fields=* to see all fields, or specify needed fields
- Date format: YYYY-MM-DD
- References use {"id": <int>} objects

## Common Endpoints
- /employee — GET, POST, PUT — Manage employees
- /customer — GET, POST, PUT — Manage customers  
- /product — GET, POST — Manage products
- /invoice — GET, POST — Create/query invoices
- /order/orderline — POST — Create order lines (needed for invoices)
- /order — GET, POST — Manage orders (needed for invoices)
- /travelExpense — GET, POST, PUT, DELETE — Travel expenses
- /project — GET, POST — Manage projects
- /department — GET, POST — Manage departments
- /ledger/account — GET — Chart of accounts
- /ledger/posting — GET — Ledger postings
- /ledger/voucher — GET, POST, DELETE — Vouchers
- /invoice/payment — POST — Register invoice payments
- /supplier — GET, POST — Manage suppliers
- /contact — GET, POST, PUT — Manage contacts

## Invoice Creation Flow
1. Ensure customer exists (POST /customer or GET /customer)
2. Create order: POST /order with {"customer": {"id": X}, "orderDate": "YYYY-MM-DD", "deliveryDate": "YYYY-MM-DD"}
3. Add order lines: POST /order/orderline with {"order": {"id": X}, "product": {"id": Y}, "count": N} or description/unitPrice
4. Create invoice: POST /invoice with {"invoiceDate": "YYYY-MM-DD", "invoiceDueDate": "YYYY-MM-DD", "orders": [{"id": X}]}

## Employee Creation
POST /employee with: {"firstName": "...", "lastName": "...", "email": "...", "dateOfBirth": "YYYY-MM-DD"}
- For admin role: check what roles/access levels are available

## Product Creation
POST /product with: {"name": "...", "number": N, "priceExcludingVatCurrency": X.XX, "vatType": {"id": N}}

## Payment Registration
POST /invoice/:createCreditNote or use payment endpoints

## Travel Expense
POST /travelExpense with required fields

## CRITICAL RULES
- Always respond with a tool call. Never just describe what you would do.
- After completing all needed API calls, call the 'task_complete' function.
- If unsure about field names, do ONE exploratory GET with ?fields=* to learn the schema.
- Norwegian/Nynorsk names may have æ, ø, å — preserve them exactly.
- "kontoadministrator" or "administrator" means the employee should have admin access/role.
"""


def build_tools():
    """Build Gemini function declarations for the agent tools."""

    tripletex_get = FunctionDeclaration(
        name="tripletex_get",
        description="Make a GET request to the Tripletex API. Use for listing/searching entities.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "API path, e.g. '/employee', '/customer', '/invoice'. Must start with '/'."
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters as key-value pairs, e.g. {\"fields\": \"*\", \"count\": 10, \"name\": \"Ola\"}"
                }
            },
            "required": ["path"]
        }
    )

    tripletex_post = FunctionDeclaration(
        name="tripletex_post",
        description="Make a POST request to the Tripletex API. Use for creating entities.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "API path, e.g. '/employee', '/customer', '/order/orderline'. Must start with '/'."
                },
                "body": {
                    "type": "object",
                    "description": "JSON request body with the entity data to create"
                },
                "params": {
                    "type": "object",
                    "description": "Optional query parameters"
                }
            },
            "required": ["path", "body"]
        }
    )

    tripletex_put = FunctionDeclaration(
        name="tripletex_put",
        description="Make a PUT request to the Tripletex API. Use for updating entities. Requires the entity ID in the path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "API path with ID, e.g. '/employee/123', '/customer/456'"
                },
                "body": {
                    "type": "object",
                    "description": "JSON request body with updated entity data"
                },
                "params": {
                    "type": "object",
                    "description": "Optional query parameters"
                }
            },
            "required": ["path", "body"]
        }
    )

    tripletex_delete = FunctionDeclaration(
        name="tripletex_delete",
        description="Make a DELETE request to the Tripletex API. Use for removing entities.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "API path with ID, e.g. '/travelExpense/123'"
                }
            },
            "required": ["path"]
        }
    )

    task_complete = FunctionDeclaration(
        name="task_complete",
        description="Signal that the accounting task has been completed. Call this ONLY after all necessary API calls have been made.",
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was done"
                }
            },
            "required": ["summary"]
        }
    )

    return Tool(function_declarations=[tripletex_get, tripletex_post, tripletex_put, tripletex_delete, task_complete])


def run_agent(prompt: str, files: list[dict], client: TripletexClient, project_id: str, location: str = "europe-west4") -> str:
    """Run the agent loop: LLM interprets prompt, calls Tripletex API, repeats until done."""

    import vertexai
    vertexai.init(project=project_id, location=location)

    model = GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT,
        tools=[build_tools()],
    )

    # Build initial user message
    user_parts = [Part.from_text(f"## Task Prompt\n{prompt}")]

    if files:
        file_descriptions = []
        for f in files:
            file_descriptions.append(f"- {f['filename']} ({f['mime_type']})")
        user_parts.append(Part.from_text(f"\n## Attached Files\n" + "\n".join(file_descriptions)))
        # For PDFs/images, include the base64 data so the model can read them
        for f in files:
            if f["mime_type"].startswith("image/"):
                import base64
                user_parts.append(Part.from_data(
                    data=base64.b64decode(f["content_base64"]),
                    mime_type=f["mime_type"]
                ))

    user_parts.append(Part.from_text("\nAnalyze this task and execute the required Tripletex API calls. Be precise and efficient."))

    chat = model.start_chat()
    response = chat.send_message(user_parts)

    # Agent loop — max 20 iterations to prevent infinite loops
    for iteration in range(20):
        # Check if model wants to call functions
        function_calls = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.function_call:
                    function_calls.append(part.function_call)

        if not function_calls:
            # Model responded with text only — done or confused
            break

        # Execute each function call
        function_responses = []
        task_done = False

        for fc in function_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}

            if name == "task_complete":
                task_done = True
                function_responses.append(Part.from_function_response(
                    name="task_complete",
                    response={"status": "completed", "summary": args.get("summary", "")}
                ))
                continue

            # Execute Tripletex API call
            if name == "tripletex_get":
                path = args["path"]
                params = args.get("params")
                # Convert params values to strings for requests
                if params:
                    params = {k: str(v) if not isinstance(v, str) else v for k, v in params.items()}
                result = client.get(path, params=params)

            elif name == "tripletex_post":
                path = args["path"]
                body = args.get("body", {})
                params = args.get("params")
                if params:
                    params = {k: str(v) if not isinstance(v, str) else v for k, v in params.items()}
                result = client.post(path, json_body=body, params=params)

            elif name == "tripletex_put":
                path = args["path"]
                body = args.get("body", {})
                params = args.get("params")
                if params:
                    params = {k: str(v) if not isinstance(v, str) else v for k, v in params.items()}
                result = client.put(path, json_body=body, params=params)

            elif name == "tripletex_delete":
                path = args["path"]
                result = client.delete(path)

            else:
                result = {"error": f"Unknown function: {name}"}

            function_responses.append(Part.from_function_response(
                name=name,
                response={"result": result}
            ))

        # Send function results back to the model
        response = chat.send_message(function_responses)

        if task_done:
            break

    return f"Agent completed in {iteration + 1} iterations. API calls made: {len(client.call_log)}"
