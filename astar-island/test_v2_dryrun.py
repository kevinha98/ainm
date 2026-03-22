"""Dry-run test of auto_runner_v2's observe_and_predict using historical GT data.
Simulates the full flow: observation → LUT → prediction → score."""
import json, numpy as np
from scipy import ndimage
from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES
from src.models import build_class_grid

CLIP_FLOOR = 0.0001
SETTLE_DIST_THRESH = 2.0
TEMPERATURE = 1.0

def score_pred(pred, gt):
    pred = np.clip(pred, 1e-12, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt_safe = np.clip(gt, 1e-15, None)
    kl = np.sum(gt_safe * np.log(gt_safe / pred), axis=-1).mean()
    return 100 * np.exp(-kl)

class FakeAPI:
    """Simulate the API using GT data (single-sim sampling)."""
    def __init__(self, gt_data, seed_offset=42):
        self.gt_data = gt_data
        self.seed_offset = seed_offset
        self.queries_used = 0
        self.queries_max = 50
    
    def get_my_rounds(self):
        return [{"id": "test", "queries_used": self.queries_used, "queries_max": self.queries_max}]
    
    def simulate(self, round_id, seed_index, row, col, steps=50):
        if self.queries_used >= self.queries_max:
            return {"error": "budget_exhausted"}
        self.queries_used += 1
        
        entry = self.gt_data[str(seed_index)]
        gt = np.array(entry['ground_truth'])
        H, W = gt.shape[:2]
        
        rng = np.random.RandomState(self.seed_offset + seed_index * 1000 + row * 100 + col)
        
        r_end = min(row + 15, H)
        c_end = min(col + 15, W)
        viewport = np.zeros((r_end - row, c_end - col), dtype=int)
        
        for vy in range(viewport.shape[0]):
            for vx in range(viewport.shape[1]):
                gy, gx = row + vy, col + vx
                gt_dist = np.clip(gt[gy, gx], 0, None)
                s = gt_dist.sum()
                if s > 0:
                    gt_dist = gt_dist / s
                else:
                    gt_dist = np.ones(6) / 6
                viewport[vy, vx] = rng.choice(6, p=gt_dist)
        
        return {"grid": viewport.tolist()}

# Import the actual auto_runner_v2 observe_and_predict function
import importlib.util
spec = importlib.util.spec_from_file_location("auto_runner_v2", "auto_runner_v2.py")
m = importlib.util.module_from_spec(spec)
# Suppress logging during import
import logging
logging.disable(logging.CRITICAL)
spec.loader.exec_module(m)
logging.disable(logging.NOTSET)

# Patch out time.sleep in the module to speed up test
import time as _time
m.time = type('FakeTime', (), {'sleep': lambda self, x: None, 'time': _time.time})()


ROUND_MAP = {
    '71451d74': 'R1', '76909e29': 'R2', 'f1dac9a9': 'R3', '8e839974': 'R4',
    'fd3c92ff': 'R5', 'ae78003a': 'R6', '36e581f1': 'R7', 'c5cdf100': 'R8',
    '2a341ace': 'R9', '75e625c3': 'R10', '324fde07': 'R11', '795bfb1f': 'R12'
}

# Test with a few GT rounds
gt_files = sorted(DATA_DIR.glob('ground_truth_*.json'))
all_gt = {}
for gf in gt_files:
    rid = gf.stem.replace('ground_truth_', '')
    with open(gf) as f:
        all_gt[rid] = json.load(f)

print(f"Testing auto_runner_v2 observe_and_predict with {len(all_gt)} GT rounds\n")

test_rids = list(all_gt.keys())[:3]  # Test 3 rounds for speed
for rid in test_rids:
    rname = ROUND_MAP.get(rid[:8], rid[:8])
    gt_data = all_gt[rid]
    
    # Build grids like the real runner would
    grids = []
    for si_str in sorted(gt_data.keys()):
        entry = gt_data[si_str]
        if isinstance(entry, dict):
            grids.append(np.array(entry['initial_grid']))
    
    # Create fake API
    fake_api = FakeAPI(gt_data)
    
    # Run the actual observe_and_predict function
    predictions = m.observe_and_predict(fake_api, "test", grids)
    
    print(f"\n{rname}: queries used={fake_api.queries_used}")
    
    # Score against GT
    scores = []
    for si in range(len(grids)):
        gt = np.array(gt_data[str(si)]['ground_truth'])
        pred = predictions[si]
        s = score_pred(pred, gt)
        scores.append(s)
        print(f"  Seed {si}: {s:.2f}")
    print(f"  MEAN: {np.mean(scores):.2f}")

print("\n--- Dry run complete ---")
