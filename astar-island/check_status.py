"""Quick status check — rounds, scores, leaderboard."""
import requests
from src.api import AstarAPI
from src.settings import API_BASE

api = AstarAPI()

# All rounds
r = requests.get(f'{API_BASE}/rounds', cookies=api.cookies)
rounds = r.json()
print(f'Total rounds: {len(rounds)}')
for rd in rounds:
    print(f'  {rd["id"][:12]}... status={rd["status"]} weight={rd.get("weight","?")}')

# My rounds
r2 = requests.get(f'{API_BASE}/my-rounds', cookies=api.cookies)
my = r2.json()
print(f'\nMy rounds: {len(my)}')
for m in my:
    rid = m["id"][:12]
    score = m.get("round_score")
    budget = f'{m.get("queries_used",0)}/{m.get("queries_max",0)}'
    seeds = m.get("seeds_submitted", 0)
    closes = m.get("closes_at", "?")
    print(f'  {rid}... score={score} budget={budget} seeds_submitted={seeds} status={m.get("status")} closes={closes}')

# Leaderboard
r3 = requests.get(f'{API_BASE}/leaderboard', cookies=api.cookies)
lb = r3.json()
print(f'\nLeaderboard ({len(lb)} entries):')
for i, entry in enumerate(lb[:10]):
    email = entry.get("email", "?")[:25]
    score = entry.get("score", "?")
    print(f'  #{i+1}: {email} score={score}')
