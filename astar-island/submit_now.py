"""Submit per-class average predictions, with retry on rate limit."""
import json, time, numpy as np
from src.api import AstarAPI
from src.settings import DATA_DIR

api = AstarAPI()

# Get round with retry
for attempt in range(5):
    try:
        rd = api.get_active_round()
        break
    except Exception as e:
        print(f"  Retry {attempt+1}: {e}")
        time.sleep(5)

round_id = rd['id']
print(f"Round: {round_id[:12]}  status={rd['status']}")

with open(DATA_DIR / 'improved_predictions.json') as f:
    preds = json.load(f)

for entry in preds:
    si = entry['seed_index']
    p = np.array(entry['probabilities'])
    p = np.clip(p, 1e-6, None)
    p /= p.sum(axis=-1, keepdims=True)
    
    for attempt in range(5):
        print(f"  Seed {si} (attempt {attempt+1})...", end=" ")
        ok, text = api.submit_prediction(round_id, si, p.tolist())
        if ok:
            print("OK")
            break
        elif "Rate" in text or "429" in text or "rate" in text.lower():
            print(f"rate limited, waiting 5s...")
            time.sleep(5)
        else:
            print(f"FAIL: {text[:80]}")
            break
    time.sleep(2)

print("Done!")
