"""Check submission status for the active round."""
from src.api import AstarAPI
api = AstarAPI()
rd = api.get_active_round()
rid = rd['id']
print(f"Round: {rid[:12]}  status={rd['status']}")

preds = api.get_my_predictions(rid)
print(f"Predictions: {len(preds)} entries")
for p in preds:
    si = p.get('seed_index', '?')
    ts = p.get('submitted_at', p.get('timestamp', '?'))
    model = p.get('model', '?')
    print(f"  Seed {si}: {ts}  model={model}")
