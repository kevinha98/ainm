"""Check detailed round scores and ranks."""
import requests, json
from src.api import AstarAPI
from src.settings import API_BASE
api = AstarAPI()

r2 = requests.get(f'{API_BASE}/my-rounds', cookies=api.cookies)
my = r2.json()
for m in my:
    rid = m['id'][:12]
    rn = m.get('round_number', '?')
    score = m.get('round_score')
    rank = m.get('rank')
    total = m.get('total_teams')
    status = m.get('status')
    seeds = m.get('seeds_submitted')
    seed_scores = m.get('seed_scores')
    qu = m.get('queries_used', 0)
    qm = m.get('queries_max', 0)
    print(f'Round {rn} [{rid}] status={status} score={score} rank={rank}/{total} seeds={seeds} budget={qu}/{qm}')
    if seed_scores:
        print(f'  seed_scores={seed_scores}')

print()
for m in my:
    print(f'Round {m.get("round_number")}: weight={m.get("round_weight")}')

# Check leaderboard scoring formula
r = requests.get(f'{API_BASE}/leaderboard', cookies=api.cookies)
lb = r.json()
# Show teams that participated in exactly 2 rounds to understand scoring
print('\nTeams with 2 rounds (to understand weighted_score):')
for e in lb:
    if e['rounds_participated'] == 2:
        print(f'  {e["team_name"][:25]:25s} weighted={e["weighted_score"]:.2f} hot_streak={e["hot_streak_score"]:.2f}')
        if len([x for x in lb if x['rounds_participated']==2]) > 10:
            break
