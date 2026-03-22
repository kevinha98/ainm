"""Quick targeted eval: test conf_type and combinations with v3+v4 ensemble (29 images)."""
import pickle, json
from pathlib import Path
import numpy as np
from evaluate_local import evaluate_mAP, load_coco_ground_truth
from sweep_v5b import wbf_fuse, soft_nms, hard_nms, run_eval

ANN_PATH = Path("data/coco/train/annotations.json")

def main():
    with open("cache_v3_29.pkl", "rb") as f:
        cache_v3 = pickle.load(f)
    with open("cache_v4_29.pkl", "rb") as f:
        cache_v4 = pickle.load(f)
    gt = load_coco_ground_truth(str(ANN_PATH))
    print(f"Loaded v3 ({len(cache_v3)} imgs) + v4 ({len(cache_v4)} imgs)")

    trio3 = ["full_1280", "full_1408", "full_1536"]
    v4d = ["full_1280", "full_1536"]

    configs = [
        # conf_type sweep (THE key parameter)
        ("ct=max_s1.5", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5, "conf_type": "max"}),
        ("ct=avg_s1.5", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5, "conf_type": "avg"}),
        ("ct=bma_s1.5", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5, "conf_type": "box_and_model_avg"}),

        # conf_type + sigma=3.0 combos
        ("ct=max_s3.0", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 3.0, "conf_type": "max"}),
        ("ct=avg_s3.0", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 3.0, "conf_type": "avg"}),
        ("ct=bma_s3.0", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 3.0, "conf_type": "box_and_model_avg"}),

        # conf_type + sigma=1.0 (conservative)
        ("ct=bma_s1.0", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.0, "conf_type": "box_and_model_avg"}),

        # conf_type + different conf thresholds
        ("ct=bma_c0.01", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5, "conf_type": "box_and_model_avg", "conf": 0.01}),
        ("ct=bma_c0.02", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5, "conf_type": "box_and_model_avg", "conf": 0.02}),

        # Best combo: conf_type + sigma + conf
        ("BEST_bma_s3_c01", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 3.0, "conf_type": "box_and_model_avg", "conf": 0.01}),
        ("BEST_bma_s3_c02", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 3.0, "conf_type": "box_and_model_avg", "conf": 0.02}),
        ("BEST_avg_s3_c01", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 3.0, "conf_type": "avg", "conf": 0.01}),

        # Even higher sigma combos
        ("ct=bma_s5.0", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 5.0, "conf_type": "box_and_model_avg"}),
        ("ct=bma_s10", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 10.0, "conf_type": "box_and_model_avg"}),
    ]

    print(f"\nEvaluating {len(configs)} configs...")
    print("=" * 100)
    results = []
    for label, config in configs:
        comb, det, cls, n = run_eval(cache_v3, cache_v4, gt, config)
        results.append((label, comb, det, cls, n))
        print(f"  {label:30s} | {comb:.4f} | Det={det:.4f} | Cls={cls:.4f} | N={n:5d}")

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 100)
    print("RANKED RESULTS:")
    for i, (label, comb, det, cls, n) in enumerate(results):
        print(f"  #{i+1:2d} {label:30s} | {comb:.4f} | Det={det:.4f} | Cls={cls:.4f}")


if __name__ == "__main__":
    main()
