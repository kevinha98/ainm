"""Find our team on the leaderboard."""
import sys
sys.path.insert(0, "C:/ainm/astar-island")
from src.api import AstarAPI

api = AstarAPI()
lb = api._get("/leaderboard")

# Print all fields in first entry to understand structure
if lb:
    print("Fields in leaderboard entry:", list(lb[0].keys()))
    print("Sample:", lb[0])
    print()

# Search for our team
for entry in lb:
    for val in entry.values():
        if isinstance(val, str) and ("kevin" in val.lower() or "dnb" in val.lower()):
            print("FOUND:", entry)

# Print ranks 15-50
print("\n--- Ranks 15-50 ---")
for entry in lb:
    r = entry.get("rank", 999)
    if 15 <= r <= 50:
        name = str(entry.get("team_name", entry.get("email", "?")))[:25]
        ws = entry.get("weighted_score", 0)
        nr = entry.get("rounds_participated", "?")
        print(f"  #{r:>3d}: {name:25s} ws={ws:8.2f} rounds={nr}")
