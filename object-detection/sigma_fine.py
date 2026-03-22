"""Fine-tune sigma with conf_type='box_and_model_avg' to find the true optimum."""
import pickle
from pathlib import Path
from evaluate_local import evaluate_mAP, load_coco_ground_truth
from sweep_v5b import run_eval

ANN_PATH = Path("data/coco/train/annotations.json")

def main():
    with open("cache_v3_29.pkl", "rb") as f:
        cache_v3 = pickle.load(f)
    with open("cache_v4_29.pkl", "rb") as f:
        cache_v4 = pickle.load(f)
    gt = load_coco_ground_truth(str(ANN_PATH))

    trio3 = ["full_1280", "full_1408", "full_1536"]
    v4d = ["full_1280", "full_1536"]

    # Fine sweep sigma 2.0 to 20.0 with conf_type='box_and_model_avg'
    sigmas = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 50.0]
    results = []
    for s in sigmas:
        config = {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": s, "conf_type": "box_and_model_avg"}
        comb, det, cls, n = run_eval(cache_v3, cache_v4, gt, config)
        results.append((s, comb, det, cls, n))
        print(f"  sigma={s:5.1f} | {comb:.4f} | Det={det:.4f} | Cls={cls:.4f} | N={n:5d}")

    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\nBest: sigma={results[0][0]} -> {results[0][1]:.4f}")


if __name__ == "__main__":
    main()
