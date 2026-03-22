"""Poll until R5 becomes completed, then show scores."""
import time
import datetime
import sys

sys.path.insert(0, "C:/ainm/astar-island")
from src.api import AstarAPI

api = AstarAPI()

for attempt in range(60):  # max 60 attempts x 15s = 15 min
    now = datetime.datetime.now().strftime("%H:%M:%S")
    rounds = api.get_rounds()
    r5 = None
    for r in rounds:
        if r.get("round_number") == 5:
            r5 = r
            break

    if r5 and r5.get("status") == "completed":
        print(f"\n=== R5 COMPLETED at {now} ===")
        my = api.get_my_rounds()
        for m in sorted(my, key=lambda x: x.get("round_number", 0)):
            rn = m.get("round_number")
            status = m.get("status")
            score = m.get("round_score")
            rank = m.get("rank")
            total = m.get("total_teams")
            ws = m.get("weighted_score")
            budget = f"{m.get('queries_used', 0)}/{m.get('queries_max', 50)}"
            print(
                f"R{rn}: status={status} score={score} "
                f"rank={rank}/{total} ws={ws} budget={budget}"
            )
            ss = m.get("seed_scores", [])
            if ss:
                print(f"  seeds={[round(x, 2) for x in ss]}")

        # Check for R6
        print()
        for r in rounds:
            rnum = r.get("round_number")
            rst = r.get("status")
            print(f"Round {rnum}: {rst}")
        break
    else:
        status = r5.get("status", "?") if r5 else "not found"
        print(f"  [{now}] R5 status: {status} (waiting...)", flush=True)
        time.sleep(15)
else:
    print("Timed out waiting for R5 to complete!")
