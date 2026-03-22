"""Quick status check."""
from src.api import AstarAPI
api = AstarAPI()
rounds = api.get_rounds()
for r in rounds:
    rn = r.get('round_number', 0)
    if rn >= 20:
        print(f"R{rn}: {r['status']}")
