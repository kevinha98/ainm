"""Check leaderboard and our position."""
import sys
sys.path.insert(0, "C:/ainm/astar-island")
from src.api import AstarAPI

api = AstarAPI()

# Try leaderboard endpoint
try:
    lb = api._get("/leaderboard")
    if isinstance(lb, list):
        print(f"=== LEADERBOARD ({len(lb)} teams) ===\n")
        print(f"{'Rank':>4s} {'Team':>20s} {'Weighted':>10s} {'Rounds':>6s}")
        print("-" * 50)
        for i, entry in enumerate(lb[:30]):
            team = entry.get("team_name", entry.get("email", "?"))[:20]
            ws = entry.get("weighted_score", 0)
            nr = entry.get("rounds_participated", "?")
            rank = entry.get("rank", i + 1)
            print(f"{rank:>4d} {team:>20s} {ws:>10.2f} {nr:>6}")
        
        # Find us
        print("\n--- Our position ---")
        for entry in lb:
            email = entry.get("email", "")
            if "kevin" in email.lower() or "dnb" in email.lower() or "ha" in email.lower():
                print(f"Found: {entry}")
    elif isinstance(lb, dict):
        for k, v in lb.items():
            print(f"{k}: {v}")
    else:
        print("Unexpected response:", type(lb), str(lb)[:200])
except Exception as e:
    print(f"Leaderboard error: {e}")

# Also check our weighted score from my-rounds
print("\n--- Our rounds ---")
my = api.get_my_rounds()
total_ws = 0
for m in sorted(my, key=lambda x: x.get("round_number", 0)):
    rn = m.get("round_number")
    score = m.get("round_score")
    ws = m.get("weighted_score")
    rw = m.get("round_weight")
    if ws:
        total_ws += ws
    elif score and rw:
        ws_calc = score * rw
        total_ws += ws_calc
        ws = f"~{ws_calc:.2f}"
    print(f"  R{rn}: score={score} ws={ws} weight={rw}")
print(f"\n  Total weighted score: {total_ws:.2f}")
