"""Quick sigma exploration — find optimal soft-NMS sigma."""
import numpy as np
from pathlib import Path
from evaluate_local import evaluate_mAP, load_coco_ground_truth
from sweep_fast import YOLO_VAL_DIR, ANN_PATH
from sweep_v4 import (
    load_model, cache_model_passes, evaluate_config,
    WEIGHTS_V3, WEIGHTS_V4
)

np.random.seed(42)
all_imgs = sorted(YOLO_VAL_DIR.glob("*.jpg"))
indices = np.random.choice(len(all_imgs), 10, replace=False)
img_paths = [all_imgs[i] for i in sorted(indices)]
gt = load_coco_ground_truth(str(ANN_PATH))

model_v3, device = load_model(WEIGHTS_V3)
model_v3(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
cache_v3 = cache_model_passes(img_paths, model_v3, device, [1280, 1408, 1536], "v3")
del model_v3

model_v4, _ = load_model(WEIGHTS_V4)
model_v4(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
cache_v4 = cache_model_passes(img_paths, model_v4, device, [1280, 1536], "v4")
del model_v4

v3p = ["full_1280", "full_1408", "full_1536"]
v4p = ["full_1280", "full_1536"]
print("\nSigma exploration:")
for sigma in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]:
    cfg = {"v3_passes": v3p, "v4_passes": v4p, "conf": 0.03, "wbf_iou": 0.35,
           "nms_type": "soft", "nms_sigma": sigma}
    res = evaluate_config(cache_v3, cache_v4, gt, cfg, f"s={sigma}")
    print(f"  sigma={sigma:6.1f} | Score={res['combined']:.4f} | "
          f"Det={res['det_mAP']:.4f} | Cls={res['cls_mAP']:.4f} | N={res['n_preds']}")
