"""Check scores and auto_runner log after R5 closure."""
import time
import datetime
import sys
import json

sys.path.insert(0, "C:/ainm/astar-island")
from src.api import AstarAPI

# Wait until 09:51
while True:
    now = datetime.datetime.now()
    target_min = 51
    if now.hour > 9 or (now.hour == 9 and now.minute >= target_min):
        break
    remaining = (9 * 60 + target_min - now.hour * 60 - now.minute) * 60 - now.second
    ts = now.strftime("%H:%M:%S")
    print(f"  Waiting... {remaining}s until 09:{target_min} (now {ts})", flush=True)
    time.sleep(15)

print(f"\n=== Time: {datetime.datetime.now().strftime('%H:%M:%S')} ===", flush=True)

# Check scores
api = AstarAPI()
my = api.get_my_rounds()
total_weighted = 0.0

for m in sorted(my, key=lambda x: x.get("round_number", 0)):
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

    if ws:
        total_weighted += ws

    print(
        f"R{rn}: status={status} score={score} weighted={ws} "
        f"rank={rank}/{total} seeds={seeds} budget={budget_used}/{budget_max}",
        flush=True,
    )
    if ss:
        print(f"  seed_scores={[round(x, 2) for x in ss]}", flush=True)

print(f"\nTotal weighted score: {total_weighted:.2f}", flush=True)

# Check round statuses
print("\n--- Round statuses ---", flush=True)
rounds = api.get_rounds()
for r in rounds:
    rid = r["id"][:8]
    print(f"  Round {r.get('round_number')}: {r.get('status')} id={rid}", flush=True)

# Check auto_runner log tail
print("\n--- Auto_runner log tail ---", flush=True)
with open("C:/ainm/astar-island/logs/auto_runner.log") as f:
    lines = f.readlines()
for line in lines[-10:]:
    print(f"  {line.rstrip()}", flush=True)

# Check GT files
print("\n--- Ground truth files ---", flush=True)
from pathlib import Path
for gf in sorted(Path("C:/ainm/astar-island/data").glob("ground_truth_*.json")):
    with open(gf) as f:
        data = json.load(f)
    n_seeds = len(data)
    print(f"  {gf.name}: {n_seeds} seeds", flush=True)
