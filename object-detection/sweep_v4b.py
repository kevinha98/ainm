"""Sweep V4b — Fine-tune around v4 winner: v3trio+v4duo with soft-NMS.

Winner from v4: v3trio+v4duo_wbf=0.35_soft → 0.9382 (Δ=+0.0138 vs baseline)
  - v3@1280 + v3@1408 + v3@1536 + v4@1280 + v4@1536 (5 passes)
  - WBF=0.35, conf=0.03, soft-NMS sigma=0.5

This sweep fine-tunes around the winner:
  1. Soft-NMS sigma: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0
  2. WBF IoU around 0.35 in tight steps
  3. Confidence threshold tuning with soft-NMS
  4. Score threshold for soft-NMS pruning

Reuses cached data from sweep_v4 (same random seed → same images).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from evaluate_local import evaluate_mAP, load_coco_ground_truth
from sweep_fast import (
    extract_dets, wbf_fuse, hard_nms_class_agnostic,
    soft_nms_class_agnostic, _iou, YOLO_VAL_DIR, ANN_PATH
)
from sweep_v4 import (
    load_model, cache_model_passes, apply_pipeline, run_config, evaluate_config,
    WEIGHTS_V3, WEIGHTS_V4
)


def get_finetune_configs():
    """Fine-grained sweep around the winning configuration."""
    configs = []

    # Base config: v3trio + v4duo
    base_v3 = ["full_1280", "full_1408", "full_1536"]
    base_v4 = ["full_1280", "full_1536"]

    # ── A. Soft-NMS sigma sweep (the most important param)
    for sigma in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5]:
        configs.append((f"5pass_s={sigma}", {
            "v3_passes": base_v3, "v4_passes": base_v4,
            "conf": 0.03, "wbf_iou": 0.35,
            "nms_type": "soft", "nms_sigma": sigma
        }))

    # ── B. WBF fine-tune with best sigma values
    for sigma in [0.5, 0.7]:
        for wbf in [0.30, 0.33, 0.35, 0.37, 0.40, 0.45]:
            configs.append((f"5pass_wbf={wbf}_s={sigma}", {
                "v3_passes": base_v3, "v4_passes": base_v4,
                "conf": 0.03, "wbf_iou": wbf,
                "nms_type": "soft", "nms_sigma": sigma
            }))

    # ── C. Confidence threshold with soft-NMS
    for conf in [0.01, 0.02, 0.025, 0.03, 0.04, 0.05]:
        configs.append((f"5pass_c={conf}_s=0.5", {
            "v3_passes": base_v3, "v4_passes": base_v4,
            "conf": conf, "wbf_iou": 0.35,
            "nms_type": "soft", "nms_sigma": 0.5
        }))

    # ── D. Without 1408 (4 passes — current runtime is feasible)
    base_v3_no1408 = ["full_1280", "full_1536"]
    for sigma in [0.5, 0.7]:
        configs.append((f"4pass_no1408_s={sigma}", {
            "v3_passes": base_v3_no1408, "v4_passes": base_v4,
            "conf": 0.03, "wbf_iou": 0.35,
            "nms_type": "soft", "nms_sigma": sigma
        }))

    # ── E. conftype variations with soft-NMS
    for ct in ["avg", "box_and_model_avg"]:
        configs.append((f"5pass_{ct}_s=0.5", {
            "v3_passes": base_v3, "v4_passes": base_v4,
            "conf": 0.03, "wbf_iou": 0.35,
            "nms_type": "soft", "nms_sigma": 0.5, "conf_type": ct
        }))
        configs.append((f"5pass_{ct}_s=0.7", {
            "v3_passes": base_v3, "v4_passes": base_v4,
            "conf": 0.03, "wbf_iou": 0.35,
            "nms_type": "soft", "nms_sigma": 0.7, "conf_type": ct
        }))

    # Reference: hard NMS at 5 passes (to confirm soft-NMS advantage)
    configs.append(("5pass_hard_ref", {
        "v3_passes": base_v3, "v4_passes": base_v4,
        "conf": 0.03, "wbf_iou": 0.35, "nms_type": "hard"
    }))

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=10)
    args = parser.parse_args()

    all_imgs = sorted(YOLO_VAL_DIR.glob("*.jpg"))
    n = min(args.n_images, len(all_imgs))
    np.random.seed(42)
    indices = np.random.choice(len(all_imgs), n, replace=False)
    img_paths = [all_imgs[i] for i in sorted(indices)]
    print(f"Using {n}/{len(all_imgs)} val images")

    gt = load_coco_ground_truth(str(ANN_PATH))

    # Cache same as sweep_v4
    model_v3, device = load_model(WEIGHTS_V3)
    model_v3(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
    print(f"\nCaching v3 @ [1280, 1408, 1536]...")
    cache_v3 = cache_model_passes(img_paths, model_v3, device, [1280, 1408, 1536], "v3")
    del model_v3

    model_v4, _ = load_model(WEIGHTS_V4)
    model_v4(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
    print(f"Caching v4 @ [1280, 1536]...")
    cache_v4 = cache_model_passes(img_paths, model_v4, device, [1280, 1536], "v4")
    del model_v4

    configs = get_finetune_configs()
    print(f"\nSweeping {len(configs)} fine-tune configurations...")
    print("=" * 110)

    results = []
    for label, config in configs:
        res = evaluate_config(cache_v3, cache_v4, gt, config, label)
        results.append(res)
        print(f"  {label:45s} | Score={res['combined']:.4f} | "
              f"Det={res['det_mAP']:.4f} | Cls={res['cls_mAP']:.4f} | N={res['n_preds']:5d}")

    results.sort(key=lambda x: x["combined"], reverse=True)
    ref = next((r for r in results if r["label"] == "5pass_hard_ref"), results[-1])

    print("\n" + "=" * 110)
    print(f"TOP 15 (hard NMS ref = {ref['combined']:.4f})")
    print("=" * 110)
    for i, r in enumerate(results[:15]):
        delta = r["combined"] - ref["combined"]
        print(f"  #{i+1:2d} {r['label']:45s} | {r['combined']:.4f} ({delta:+.4f})"
              f" | Det={r['det_mAP']:.4f} | Cls={r['cls_mAP']:.4f}")

    out_path = Path("sweep_v4b_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    winner = results[0]
    print(f"\nWINNER: {winner['label']} → {winner['combined']:.4f}")
    print(f"  Det={winner['det_mAP']:.4f} | Cls={winner['cls_mAP']:.4f}")


if __name__ == "__main__":
    main()
