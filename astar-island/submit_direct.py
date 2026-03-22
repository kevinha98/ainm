"""Submit per-class avg predictions using known round ID (avoids get_active_round rate limit)."""
import json, time, numpy as np, requests
from src.settings import DATA_DIR, TOKEN, API_BASE

cookies = {"access_token": TOKEN}
headers = {"Content-Type": "application/json"}

# Known Round 2 ID
round_id = "76909e29-f664-4b2f-b16b-61b7507277e9"

with open(DATA_DIR / 'improved_predictions.json') as f:
    preds = json.load(f)

print(f"Submitting to round {round_id[:12]}...")

for entry in preds:
    si = entry['seed_index']
    p = np.array(entry['probabilities'])
    p = np.clip(p, 1e-6, None)
    p /= p.sum(axis=-1, keepdims=True)
    
    payload = {
        "round_id": round_id,
        "seed_index": si,
        "prediction": p.tolist(),
    }
    
    for attempt in range(8):
        print(f"  Seed {si} attempt {attempt+1}...", end=" ", flush=True)
        try:
            r = requests.post(f"{API_BASE}/submit", cookies=cookies, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                print("OK")
                break
            elif r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"HTTP {r.status_code}: {r.text[:100]}")
                break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
    
    time.sleep(3)

print("\nAll done!")
