"""Wait for R5 closure and check scores."""
import time
import datetime
import sys

sys.path.insert(0, "C:/ainm/astar-island")
from src.api import AstarAPI

# Wait until 09:50 to ensure R5 has closed
while True:
    now = datetime.datetime.now()
    if now.hour > 9 or (now.hour == 9 and now.minute >= 50):
        break
    remaining = (9 * 60 + 50 - now.hour * 60 - now.minute) * 60 - now.second
    ts = now.strftime("%H:%M:%S")
    print(f"  Waiting... {remaining}s until 09:50 (now {ts})")
    time.sleep(30)

print("\n=== R5 should be closed now ===")
api = AstarAPI()

my = api.get_my_rounds()
for m in my:
    rn = m.get("round_number", "?")
    score = m.get("round_score")
    status = m.get("status")
    rank = m.get("rank")
    total = m.get("total_teams")
    ws = m.get("weighted_score")
    seeds = m.get("seeds_submitted", 0)
    ss = m.get("seed_scores", [])
    budget_used = m.get("queries_used", 0)
    budget_max = m.get("queries_max", 50)
    print(
        f"R{rn}: status={status} score={score} weighted={ws} "
        f"rank={rank}/{total} seeds={seeds} budget={budget_used}/{budget_max}"
    )
    if ss:
        print(f"  seed_scores={[round(x, 2) for x in ss]}")
print()

# Also check round statuses
rounds = api.get_rounds()
for r in rounds:
    rid = r["id"][:8]
    print(f"Round {r.get('round_number')}: {r.get('status')} id={rid}")
