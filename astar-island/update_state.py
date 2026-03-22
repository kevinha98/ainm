"""Update state and redeploy runner."""
import json

state_file = "data/auto_runner_v2_state.json"
r14_id = "d0a2c894-2162-4d49-86cf-435b9013f3b8"

with open(state_file) as f:
    state = json.load(f)

if r14_id not in state["submitted_rounds"]:
    state["submitted_rounds"].append(r14_id)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    print("Added R14 to submitted_rounds")
else:
    print("R14 already in submitted_rounds")

print(f"Submitted rounds: {len(state['submitted_rounds'])}")
for rid in state["submitted_rounds"]:
    print(f"  {rid[:8]}")
