"""Create variant submission without flip TTA but with sigma=0.85 + conf=0.01."""
import zipfile

z = zipfile.ZipFile('C:/ainm/object-detection/submissions/submission.zip')
code = z.read('run.py').decode()

# Remove flip TTA block
old = """    # Flip TTA: run v4 and v6 on horizontally flipped image (highest-impact models)
    if model_v4:
        flip_v4 = _run_flip_tta(model_v4, img_path, device, detection_only, imgsz=1280)
        all_passes.append(flip_v4)
        all_weights.append(2)
    if model_v6:
        flip_v6_1280 = _run_flip_tta(model_v6, img_path, device, detection_only, imgsz=1280)
        flip_v6_1408 = _run_flip_tta(model_v6, img_path, device, detection_only, imgsz=1408)
        all_passes.extend([flip_v6_1280, flip_v6_1408])
        all_weights.extend([3, 1])

    # WBF fusion"""
new = """    # WBF fusion"""

code = code.replace(old, new)

z2 = zipfile.ZipFile('C:/ainm/object-detection/submissions/submission_sigma85_noflip.zip', 'w', zipfile.ZIP_DEFLATED)
z2.writestr('run.py', code)
for w in ['best.onnx', 'best_v4.onnx', 'best_v6.onnx']:
    z2.writestr(w, z.read(w))
z2.close()

print('Created submission_sigma85_noflip.zip')
print('Has flip TTA calls:', 'flip_v4 = _run_flip' in code)
print('Has sigma=0.85:', 'sigma=0.85' in code)
print('Has conf=0.01:', 'conf=0.01' in code)

import os
size = os.path.getsize('C:/ainm/object-detection/submissions/submission_sigma85_noflip.zip') / 1024 / 1024
print(f'Size: {size:.1f} MB')
