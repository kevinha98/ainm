"""Debug class mapping in ground truth."""
import json
import numpy as np
from src.settings import GRID_TO_CLASS, CLASS_NAMES, NUM_CLASSES

print('GRID_TO_CLASS:', GRID_TO_CLASS)
print('CLASS_NAMES:', CLASS_NAMES)
print('NUM_CLASSES:', NUM_CLASSES)

with open('data/round_info.json') as f:
    rd = json.load(f)
with open('data/ground_truth_71451d74.json') as f:
    data = json.load(f)

ig = np.array(rd['initial_states'][0]['grid'])
gt = np.array(data['0']['ground_truth'])

unique_vals = sorted(set(ig.ravel().tolist()))
print(f'\nUnique grid values: {unique_vals}')
for v in unique_vals:
    count = (ig == v).sum()
    cls = GRID_TO_CLASS.get(v, -1)
    name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else '???'
    print(f'  Grid {v:2d} -> class {cls} ({name}): {count} cells')
    # Show avg GT for these cells
    mask = (ig == v)
    avg_gt = gt[mask].mean(axis=0)
    print(f'           avg GT: {[round(x,3) for x in avg_gt.tolist()]}')

# The GT has 6 columns. Let's verify: what do the columns represent?
# If column 0 = class 0 = Empty/Ocean/Plains, then ocean cells should have col 0 = 1.0
# Let's check
print('\n--- Ocean cells (grid=10) GT ---')
ocean = gt[ig == 10]
print(f'  First 3: {ocean[:3].tolist()}')
print(f'  avg: {ocean.mean(axis=0).round(4).tolist()}')

# The API documentation might tell us what the 6 classes are
# Let's also check: do ANY cells have significant mass in columns 2, 3?
print(f'\n--- Column analysis ---')
for col in range(6):
    col_vals = gt[:, :, col]
    nonzero = (col_vals > 0.01).sum()
    mean_when_present = col_vals[col_vals > 0.01].mean() if nonzero > 0 else 0
    print(f'  Col {col} ({CLASS_NAMES[col] if col < len(CLASS_NAMES) else "?"}): nonzero={nonzero}, max={col_vals.max():.3f}, avg_when_present={mean_when_present:.3f}')

# KEY QUESTION: Is the GT class ordering the same as our CLASS_NAMES?
# Maybe the GT uses a DIFFERENT ordering!
# Let's check: find a cell that's clearly ocean (grid=10, border)
# It should have GT = [1, 0, 0, 0, 0, 0] if class 0 = ocean/empty
# OR maybe ocean has its own index?
print('\n--- Testing class ordering ---')
# Find clear ocean cell (row 0 or 39)
print(f'  [0,0] grid={ig[0,0]} gt={gt[0,0].tolist()}')
print(f'  [39,0] grid={ig[39,0]} gt={gt[39,0].tolist()}')

# Find clear mountain cell
mtn = np.argwhere(ig == 5)
if len(mtn) > 0:
    y, x = mtn[0]
    print(f'  [{y},{x}] grid=5(mtn) gt={gt[y,x].tolist()}')

# Find clear settlement
settle = np.argwhere(ig == 1)
if len(settle) > 0:
    y, x = settle[0]
    print(f'  [{y},{x}] grid=1(settle) gt={gt[y,x].tolist()}')

# Find clear forest
forest = np.argwhere(ig == 4)
if len(forest) > 0:
    y, x = forest[0]
    print(f'  [{y},{x}] grid=4(forest) gt={gt[y,x].tolist()}')

# Find clear plains
plains = np.argwhere(ig == 11)
if len(plains) > 0:
    y, x = plains[0]
    print(f'  [{y},{x}] grid=11(plains) gt={gt[y,x].tolist()}')
