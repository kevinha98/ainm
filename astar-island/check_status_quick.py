"""Quick status check: rounds, scores, active status."""
from src.api import AstarAPI
import json

api = AstarAPI()
rounds = api.get_rounds()
for r in rounds:
    rid = r["id"][:8]
    print(f"R{r['round_number']}: {r['status']} id={rid}")

print()
my = api._get("/my-rounds")
for m in my:
    rn = m.get("round_number", "?")
    s = m.get("score", "N/A")
    st = m.get("status", "?")
    print(f"My R{rn}: score={s} status={st}")

# Check leaderboard
try:
    lb = api._get("/leaderboard")
    print("\n--- Leaderboard Top 10 ---")
    for i, entry in enumerate(lb[:10]):
        name = entry.get("team_name", entry.get("username", "?"))
        ws = entry.get("weighted_score", entry.get("score", "?"))
        print(f"  {i+1}. {name}: {ws}")
except Exception as e:
    print(f"Leaderboard error: {e}")
