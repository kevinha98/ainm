"""Quick R5 score check."""
import datetime
import sys

print("Time:", datetime.datetime.now().strftime("%H:%M:%S"))
sys.path.insert(0, "C:/ainm/astar-island")
from src.api import AstarAPI

api = AstarAPI()
my = api.get_my_rounds()
for m in sorted(my, key=lambda x: x.get("round_number", 0)):
    rn = m.get("round_number")
    status = m.get("status")
    score = m.get("round_score")
    rank = m.get("rank")
    total = m.get("total_teams")
    ws = m.get("weighted_score")
    budget = f"{m.get('queries_used', 0)}/{m.get('queries_max', 50)}"
    print(f"R{rn}: status={status} score={score} rank={rank}/{total} ws={ws} budget={budget}")
    ss = m.get("seed_scores", [])
    if ss:
        print(f"  seeds={[round(x, 2) for x in ss]}")

rounds = api.get_rounds()
print()
for r in rounds:
    print(f"Round {r.get('round_number')}: {r.get('status')}")
