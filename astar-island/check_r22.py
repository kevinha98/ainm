"""Quick R22 check."""
from src.api import AstarAPI
api = AstarAPI()
my = api.get_my_rounds()
r22 = my[0]
status = r22.get("status")
score = r22.get("round_score")
closes = r22.get("closes_at")
print(f"R22: status={status}  score={score}  closes={closes}")

rounds = api.get_rounds()
r0_status = rounds[0].get("status")
print(f"Latest round: {r0_status}")
