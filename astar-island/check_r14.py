from src.api import AstarAPI
import json
api = AstarAPI()
rounds = api.get_rounds()
print(f"Total: {len(rounds)} rounds")
for r in rounds[-5:]:
    print(f"  {r['id'][:8]} round={r.get('round_number','?')} status={r['status']}")

# Check budget
b = api.get_budget()
print(f"\nBudget: {json.dumps(b)}")

# Check my predictions for R14
try:
    p = api.get_my_predictions('d0a2c894-2162-4d49-86cf-435b9013f3b8')
    print(f"\nR14 predictions: {len(p)} seeds submitted")
except Exception as e:
    print(f"\nR14 predictions error: {e}")
