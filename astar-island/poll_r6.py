"""Poll until R6 closes and show scores."""
import time
import datetime
import sys

sys.path.insert(0, "C:/ainm/astar-island")
from src.api import AstarAPI

api = AstarAPI()

for attempt in range(300):  # max 300 × 15s = 75 min
    now = datetime.datetime.now().strftime("%H:%M:%S")
    try:
        rounds = api.get_rounds()
        r6 = None
        for r in rounds:
            if r.get("round_number") == 6:
                r6 = r
                break

        if r6 and r6.get("status") == "completed":
            print(f"\n=== R6 COMPLETED at {now} ===")
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
            break
        else:
            st = r6.get("status", "?") if r6 else "not found"
            print(f"  [{now}] R6: {st}", flush=True)
            time.sleep(15)
    except Exception as e:
        print(f"  [{now}] Error: {e}", flush=True)
        time.sleep(30)
else:
    print("Timed out!")
