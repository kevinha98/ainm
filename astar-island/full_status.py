"""Full status: leaderboard position, sweep progress, R22 countdown."""
import sys, json
from datetime import datetime, timezone
sys.path.insert(0, ".")
from src.api import AstarAPI

api = AstarAPI()

# === Our team position ===
lb = api.get_leaderboard()
print(f"=== Leaderboard ({len(lb)} teams) ===")
print(f"{'Rank':>4s}  {'Team':<25s}  {'Score':>8s}  {'Rounds':>6s}  {'HotStreak':>10s}")
for e in lb[:10]:
    print(f"{e['rank']:4d}  {e['team_name']:<25s}  {e['weighted_score']:8.2f}  {e['rounds_participated']:6d}  {e['hot_streak_score']:10.4f}")

# Find our team
our = [e for e in lb if "kevin" in e["team_name"].lower() or "ha" in e["team_slug"].lower() or "dnb" in e["team_name"].lower()]
if not our:
    # Try by team_id from my-rounds
    my = api.get_my_rounds()
    if my:
        our = [e for e in lb if e["rank"] > 200]  # broad search
print("\n--- Searching for our team ---")
for e in lb:
    if e["rank"] >= 200:
        print(f"  #{e['rank']:3d}: {e['team_name']:<25s}  score={e['weighted_score']:.2f}  streak={e['hot_streak_score']:.4f}")
        if e["rank"] > 220:
            break

# === R22 status ===
my = api.get_my_rounds()
active = [m for m in my if m["status"] == "active"]
if active:
    a = active[0]
    closes = a.get("closes_at", "?")
    print(f"\n=== R22 (active) ===")
    print(f"  Score: {a.get('round_score')}")
    print(f"  Budget: {a.get('queries_used')}/{a.get('queries_max')}")
    print(f"  Seeds submitted: {a.get('seeds_submitted')}")
    print(f"  Closes at: {closes}")
    
    # Countdown
    if closes != "?":
        from dateutil import parser as dp
        close_time = dp.parse(closes)
        now = datetime.now(timezone.utc)
        delta = close_time - now
        mins = delta.total_seconds() / 60
        print(f"  Time remaining: {mins:.0f} min")
