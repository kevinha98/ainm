"""Thin wrapper around the Tripletex v2 REST API."""

import requests
from typing import Any


class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self.call_log: list[dict] = []

    def _request(self, method: str, path: str, params: dict | None = None, json_body: dict | None = None) -> dict:
        # Strip /v2 prefix from path if base_url already contains it (prevents /v2/v2/...)
        if path.startswith("/v2/") and self.base_url.endswith("/v2"):
            path = path[3:]  # "/v2/customer" → "/customer"
        url = f"{self.base_url}{path}"
        resp = requests.request(
            method=method.upper(),
            url=url,
            auth=self.auth,
            params=params,
            json=json_body,
            timeout=60,
        )
        log_entry = {
            "method": method.upper(),
            "path": path,
            "status": resp.status_code,
        }
        self.call_log.append(log_entry)

        if resp.status_code == 204:
            return {"status": 204, "message": "No Content"}
        try:
            return resp.json()
        except Exception:
            return {"status": resp.status_code, "text": resp.text[:500]}

    def get(self, path: str, params: dict | None = None) -> dict:
        return self._request("GET", path, params=params)

    def post(self, path: str, json_body: dict | None = None, params: dict | None = None) -> dict:
        return self._request("POST", path, params=params, json_body=json_body)

    def put(self, path: str, json_body: dict | None = None, params: dict | None = None) -> dict:
        return self._request("PUT", path, params=params, json_body=json_body)

    def delete(self, path: str, params: dict | None = None) -> dict:
        return self._request("DELETE", path, params=params)
