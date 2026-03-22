"""Update run.py with optimized parameters from sweep v5b."""
import sys

with open('run.py', 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 1. WBF function signature: add conf_type parameter
old = "def wbf_fuse(passes_dets, img_w, img_h, iou_thresh=0.50, skip_box_thresh=0.02):"
new = "def wbf_fuse(passes_dets, img_w, img_h, iou_thresh=0.50, skip_box_thresh=0.02, conf_type='box_and_model_avg'):"
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("1. Updated wbf_fuse signature with conf_type param")
else:
    print("WARNING: Could not find wbf_fuse signature to update")

# 2. WBF call: use conf_type parameter instead of hardcoded 'max'
old2 = "conf_type='max',"
new2 = "conf_type=conf_type,"
if old2 in content:
    content = content.replace(old2, new2)
    changes += 1
    print("2. Updated WBF call to use conf_type parameter")

# 3. Sigma: change all 1.5 to 5.0
sigma_count = content.count('sigma=1.5')
if sigma_count > 0:
    content = content.replace('sigma=1.5', 'sigma=5.0')
    changes += 1
    print(f"3. Updated {sigma_count} sigma=1.5 -> sigma=5.0")

# 4. Update docstring
old4 = 'Key insight: soft-NMS boosts cls_mAP by +0.04 over hard NMS'
new4 = 'Sweep v5b (29 imgs): conf_type=bma + sigma=5.0 -> local 0.9232'
if old4 in content:
    content = content.replace(old4, new4)
    changes += 1

# Verify
assert "conf_type='box_and_model_avg'" in content, "conf_type not set"
assert 'sigma=5.0' in content, "sigma not updated"
assert 'sigma=1.5' not in content, "sigma=1.5 still present!"
assert 'conf_type=conf_type,' in content, "conf_type param not passed"

with open('run.py', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print(f"\nDone: {changes} changes applied")
print(f"  sigma=5.0 count: {content.count('sigma=5.0')}")
print(f"  sigma=1.5 count: {content.count('sigma=1.5')}")
print(f"  box_and_model_avg count: {content.count('box_and_model_avg')}")
