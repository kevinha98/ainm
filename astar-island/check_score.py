import json
from src.api import AstarAPI

api = AstarAPI()

# Find our team identity from my_rounds
my = api.get_my_rounds()
print(f"My rounds response: {len(my)} entries")
if my:
    r0 = my[0]
    print("Keys:", list(r0.keys()))
    print(json.dumps(r0, indent=2, default=str)[:500])

# Check R14 score (completed)
print("\n=== R14 SCORE ===")
preds14 = api.get_my_predictions('d0a2c894-2162-4d49-86cf-435b9013f3b8')
for p in preds14:
    print(f"  seed={p.get('seed_index')}, score={p.get('score')}")
if preds14:
    scores = [p['score'] for p in preds14 if p.get('score') is not None]
    if scores:
        print(f"  R14 avg: {sum(scores)/len(scores):.2f}")

# Check R13 too
print("\n=== R13 SCORE ===")
preds13 = api.get_my_predictions('7b4bda99-6165-4221-97cc-27880f5e6d95')
for p in preds13:
    print(f"  seed={p.get('seed_index')}, score={p.get('score')}")
if preds13:
    scores = [p['score'] for p in preds13 if p.get('score') is not None]
    if scores:
        print(f"  R13 avg: {sum(scores)/len(scores):.2f}")

# R15 - still active
print("\n=== R15 (still active - no score yet) ===")
preds15 = api.get_my_predictions('cc5442dd-bc5d-418b-911b-7eb960cb0390')
for p in preds15:
    print(f"  seed={p.get('seed_index')}, score={p.get('score')}")
