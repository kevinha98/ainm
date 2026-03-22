"""
Astar Island — API Client
Handles all communication with api.ainm.no/astar-island.
"""
import json
import time
import requests
from pathlib import Path
from src.settings import API_BASE, TOKEN, DATA_DIR


class AstarAPI:
    def __init__(self):
        self.cookies = {"access_token": TOKEN}
        self.headers = {"Content-Type": "application/json"}

    def _get(self, path, retries=3):
        for attempt in range(retries):
            try:
                r = requests.get(f"{API_BASE}{path}", cookies=self.cookies, headers=self.headers, timeout=30)
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

    def _post(self, path, data, retries=2):
        for attempt in range(retries):
            try:
                r = requests.post(f"{API_BASE}{path}", cookies=self.cookies, headers=self.headers, json=data, timeout=30)
                return r
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

    # ── Round info ──────────────────────────────────────────
    def get_rounds(self):
        return self._get("/rounds")

    def get_active_round(self):
        """Get active round with full data including initial_states.
        /rounds/{id} is the only endpoint with initial_states + seeds."""
        # First find the active round from the list
        rounds = self.get_rounds()
        active_id = None
        for r in rounds:
            if r["status"] == "active":
                active_id = r["id"]
                break
        if active_id is None:
            return None

        # Fetch full round data (includes initial_states with all seeds)
        full = self._get(f"/rounds/{active_id}")

        # Merge in user-specific data from /my-rounds (scores, budget, etc.)
        my_rounds = self._get("/my-rounds")
        for mr in my_rounds:
            if mr["id"] == active_id:
                for k, v in mr.items():
                    if k not in full:
                        full[k] = v
                break

        return full

    def get_budget(self):
        return self._get("/budget")

    def get_my_predictions(self, round_id):
        return self._get(f"/my-predictions/{round_id}")

    def get_leaderboard(self):
        return self._get("/leaderboard")

    def get_analysis(self, round_id, seed_index):
        return self._get(f"/analysis/{round_id}/{seed_index}")

    def get_my_rounds(self):
        return self._get("/my-rounds")

    # ── Observation (costs 1 query) ────────────────────────
    def simulate(self, round_id, seed_index, row, col, steps=50):
        """Observe a 15x15 viewport of the simulated grid.
        Costs 1 query from the budget."""
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "row": row,
            "col": col,
            "steps": steps,
        }
        r = self._post("/simulate", payload)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            return {"error": "budget_exhausted", "detail": r.text}
        else:
            return {"error": f"http_{r.status_code}", "detail": r.text}

    # ── Submission ─────────────────────────────────────────
    def submit_prediction(self, round_id, seed_index, prediction):
        """Submit H×W×6 probability prediction for one seed.
        prediction: list of lists of lists (H×W×6)."""
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        r = self._post("/submit", payload)
        return r.status_code == 200, r.text

    # ── Save round data ───────────────────────────────────
    def fetch_and_save_round(self, round_data=None):
        """Fetch full round info and save to disk."""
        if round_data is None:
            round_data = self.get_active_round()
        if round_data is None:
            return None
        path = DATA_DIR / f"round_{round_data['id'][:8]}.json"
        with open(path, "w") as f:
            json.dump(round_data, f, indent=2)
        return round_data
