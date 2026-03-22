"""Quick test: AST parse + flip bbox logic."""
import ast, copy

# 1. Verify run.py parses
with open('run.py') as f:
    ast.parse(f.read())
print('AST parse OK')

# 2. Test flip logic
def _unflip_dets(dets, img_w):
    for d in dets:
        x, y, w, h = d['bbox']
        d['bbox'][0] = round(img_w - x - w, 1)
    return dets

# Box at x=100, w=50 in 1920px -> x=1770
dets = [{'category_id': 5, 'bbox': [100.0, 200.0, 50.0, 80.0], 'score': 0.9}]
result = _unflip_dets(dets, 1920)
assert result[0]['bbox'] == [1770.0, 200.0, 50.0, 80.0], f"Got {result[0]['bbox']}"
print('Flip test 1 OK: x=100,w=50 -> x=1770')

# Box at x=0, w=100 -> x=1820
dets2 = [{'category_id': 0, 'bbox': [0.0, 0.0, 100.0, 100.0], 'score': 0.5}]
result2 = _unflip_dets(dets2, 1920)
assert result2[0]['bbox'] == [1820.0, 0.0, 100.0, 100.0], f"Got {result2[0]['bbox']}"
print('Flip test 2 OK: x=0,w=100 -> x=1820')

# Double flip = identity
dets3 = [{'category_id': 3, 'bbox': [500.0, 300.0, 75.0, 120.0], 'score': 0.8}]
orig = copy.deepcopy(dets3)
flipped = _unflip_dets(dets3, 1920)
back = _unflip_dets(flipped, 1920)
assert back[0]['bbox'] == orig[0]['bbox'], f"Double flip failed"
print('Flip test 3 OK: double flip = identity')

# Edge: box at right edge, x=1870, w=50 -> x=0
dets4 = [{'category_id': 1, 'bbox': [1870.0, 100.0, 50.0, 50.0], 'score': 0.7}]
result4 = _unflip_dets(dets4, 1920)
assert result4[0]['bbox'] == [0.0, 100.0, 50.0, 50.0], f"Got {result4[0]['bbox']}"
print('Flip test 4 OK: right edge -> left edge')

# 3. Check banned imports
with open('run.py') as f:
    src = f.read()
for banned in ['cv2', 'tensorflow', 'keras', 'sklearn', 'scipy']:
    assert f'import {banned}' not in src, f"Banned import: {banned}"
print('No banned imports')

# 4. Check key params
assert "conf_type='box_and_model_avg'" in src, "conf_type not found"
assert 'sigma=5.0' in src, "sigma=5.0 not found"
assert '_infer_full_tta' in src, "_infer_full_tta not found"
assert 'FULL_TTA' in src, "FULL_TTA mode not found"
assert 'ImageOps.mirror' in src, "flip operation not found"
print('All key params verified')

print('\nAll tests passed!')
