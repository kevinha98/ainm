"""Find Proptonomy on leaderboard + sweep status."""
import sys
sys.path.insert(0, ".")
from src.api import AstarAPI

api = AstarAPI()
lb = api.get_leaderboard()

for e in lb:
    if "propto" in e["team_name"].lower() or "propto" in e["team_slug"].lower():
        r = e["rank"]
        ws = e["weighted_score"]
        hs = e["hot_streak_score"]
        rp = e["rounds_participated"]
        print(f"Proptonomy: #{r}  weighted={ws:.2f}  hot_streak={hs:.4f}  rounds={rp}")
        
        # Show nearby teams
        for e2 in lb:
            if abs(e2["rank"] - r) <= 5:
                marker = " <<< US" if e2["rank"] == r else ""
                print(f"  #{e2['rank']:3d}: {e2['team_name']:<25s}  ws={e2['weighted_score']:8.2f}  hs={e2['hot_streak_score']:8.4f}{marker}")
        break
else:
    print("Proptonomy NOT FOUND. Searching partial matches...")
    for e in lb:
        n = e["team_name"].lower()
        if any(x in n for x in ["prop", "tono", "kevin", "dnb"]):
            print(f"  #{e['rank']}: {e['team_name']}")
