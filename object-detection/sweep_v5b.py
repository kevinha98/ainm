"""Sweep V5b - Fast validation on ALL 29 images, v3+v4 separately cached.

Strategy: Cache v3 detections, save to pickle, release memory, then cache v4.
This avoids the memory issue of loading both models simultaneously.
"""
import argparse, json, time, pickle
from pathlib import Path

import numpy as np
from PIL import Image
from evaluate_local import evaluate_mAP, load_coco_ground_truth

YOLO_VAL_DIR = Path("data/yolo/images/val")
ANN_PATH = Path("data/coco/train/annotations.json")
WEIGHTS_V3 = Path("best.onnx")
WEIGHTS_V4 = Path("best_v4.onnx")
CACHE_FILE_V3 = Path("cache_v3_29.pkl")
CACHE_FILE_V4 = Path("cache_v4_29.pkl")


def _iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2); ya = max(y1, y2)
    xb = min(x1+w1, x2+w2); yb = min(y1+h1, y2+h2)
    inter = max(0, xb-xa) * max(0, yb-ya)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0


def extract_dets(results):
    dets = []
    for r in results:
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            w, h = x2-x1, y2-y1
            if w < 3 or h < 3:
                continue
            dets.append({
                "category_id": int(r.boxes.cls[i].item()),
                "bbox": [round(x1,1), round(y1,1), round(w,1), round(h,1)],
                "score": round(float(r.boxes.conf[i].item()), 4),
            })
    return dets


def wbf_fuse(passes_dets, img_w, img_h, iou_thresh=0.35, skip_box_thresh=0.005, conf_type='max'):
    from ensemble_boxes import weighted_boxes_fusion
    boxes_list, scores_list, labels_list = [], [], []
    for pass_dets in passes_dets:
        boxes, scores, labels = [], [], []
        for d in pass_dets:
            x, y, w, h = d["bbox"]
            x1 = max(0.0, x / img_w); y1 = max(0.0, y / img_h)
            x2 = min(1.0, (x + w) / img_w); y2 = min(1.0, (y + h) / img_h)
            if x2 <= x1 or y2 <= y1: continue
            boxes.append([x1, y1, x2, y2])
            scores.append(d["score"])
            labels.append(d["category_id"])
        if boxes:
            boxes_list.append(np.array(boxes, dtype=np.float32))
            scores_list.append(np.array(scores, dtype=np.float32))
            labels_list.append(np.array(labels, dtype=np.int32))
    if not boxes_list:
        return []
    fb, fs, fl = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_thresh, skip_box_thr=skip_box_thresh, conf_type=conf_type)
    result = []
    for box, score, label in zip(fb, fs, fl):
        x1, y1, x2, y2 = box
        result.append({
            "category_id": int(label),
            "bbox": [round(x1*img_w,1), round(y1*img_h,1), round((x2-x1)*img_w,1), round((y2-y1)*img_h,1)],
            "score": round(float(score), 4),
        })
    return result


def soft_nms(dets, iou_thresh=0.45, sigma=0.5, score_thresh=0.01):
    if not dets: return dets
    dets = [d.copy() for d in dets]
    dets.sort(key=lambda x: x["score"], reverse=True)
    kept, absorbed = [], []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        cur_abs = [best.copy()]
        remaining = []
        for d in dets:
            iou = _iou(best["bbox"], d["bbox"])
            if iou >= iou_thresh:
                cur_abs.append(d.copy())
            d["score"] *= np.exp(-(iou ** 2) / sigma)
            if d["score"] >= score_thresh:
                remaining.append(d)
        absorbed.append(cur_abs)
        dets = sorted(remaining, key=lambda x: x["score"], reverse=True)
    for ki, k in enumerate(kept):
        if len(absorbed[ki]) > 1:
            cat_scores = {}
            for ab in absorbed[ki]:
                cat_scores[ab["category_id"]] = cat_scores.get(ab["category_id"], 0) + ab["score"]
            kept[ki]["category_id"] = max(cat_scores, key=cat_scores.get)
    return kept


def hard_nms(dets, iou_thresh=0.45):
    if not dets: return dets
    dets_sorted = sorted(dets, key=lambda x: x["score"], reverse=True)
    kept, absorbed = [], []
    for d in dets_sorted:
        merged = False
        for ki, k in enumerate(kept):
            if _iou(d["bbox"], k["bbox"]) >= iou_thresh:
                absorbed[ki].append(d)
                merged = True
                break
        if not merged:
            kept.append(d.copy())
            absorbed.append([d])
    for ki, k in enumerate(kept):
        if len(absorbed[ki]) > 1:
            cat_scores = {}
            for ab in absorbed[ki]:
                cat_scores[ab["category_id"]] = cat_scores.get(ab["category_id"], 0) + ab["score"]
            kept[ki]["category_id"] = max(cat_scores, key=cat_scores.get)
    return kept


def cache_model(img_paths, weights_path, scales, name):
    """Cache inference at multiple scales, return cache dict."""
    import torch
    _ol = torch.load
    def _pl(*a, **k):
        if 'weights_only' not in k: k['weights_only'] = False
        return _ol(*a, **k)
    torch.load = _pl
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {weights_path}...")
    model = YOLO(str(weights_path), task="detect")
    model(str(img_paths[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
    print(f"  -> {device}, warming up done")

    cache = {}
    t0 = time.time()
    for idx, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        img_w, img_h = img.size
        image_id = int(img_path.stem.split("_")[-1])
        img.close()

        entry = {"img_w": img_w, "img_h": img_h}
        ti = time.time()
        for s in scales:
            results = model(str(img_path), device=device, verbose=False,
                            conf=0.01, iou=0.7, max_det=600, imgsz=s, augment=False)
            entry[f"full_{s}"] = extract_dets(results)
        cache[image_id] = entry
        if (idx + 1) % 5 == 0 or idx == 0:
            n = sum(len(entry.get(f"full_{s}", [])) for s in scales)
            print(f"  {name} [{idx+1}/{len(img_paths)}] img_{image_id:05d}: {n} dets | {time.time()-ti:.1f}s")
    print(f"{name} done in {time.time()-t0:.0f}s")
    del model
    return cache


def run_eval(cache_v3, cache_v4, gt, config):
    v3p = config.get("v3_passes", ["full_1280"])
    v4p = config.get("v4_passes", [])
    all_preds = []

    for image_id in cache_v3:
        v3_data = cache_v3[image_id]
        img_w, img_h = v3_data["img_w"], v3_data["img_h"]
        conf = config.get("conf", 0.03)

        pass_list = []
        for key in v3p:
            if key in v3_data:
                pass_list.append([d for d in v3_data[key] if d["score"] >= conf])
        if cache_v4 and image_id in cache_v4:
            for key in v4p:
                if key in cache_v4[image_id]:
                    pass_list.append([d for d in cache_v4[image_id][key] if d["score"] >= conf])
        pass_list = [p for p in pass_list if p]
        if not pass_list:
            continue

        if len(pass_list) == 1:
            dets = pass_list[0]
        else:
            dets = wbf_fuse(pass_list, img_w, img_h,
                            iou_thresh=config.get("wbf_iou", 0.35),
                            skip_box_thresh=config.get("skip_box_thresh", 0.005),
                            conf_type=config.get("conf_type", "max"))

        nms_type = config.get("nms_type", "soft")
        if nms_type == "soft":
            dets = soft_nms(dets, iou_thresh=config.get("nms_iou", 0.45),
                            sigma=config.get("nms_sigma", 1.5))
        else:
            dets = hard_nms(dets, iou_thresh=config.get("nms_iou", 0.45))

        max_dets = config.get("max_dets", 300)
        if len(dets) > max_dets:
            dets.sort(key=lambda x: x["score"], reverse=True)
            dets = dets[:max_dets]

        for d in dets:
            d["image_id"] = image_id
            all_preds.append(d)

    pred_ids = set(cache_v3.keys())
    gt_f = [g for g in gt if g["image_id"] in pred_ids]
    det = evaluate_mAP(all_preds, gt_f, iou_threshold=0.5, ignore_category=True)
    cls = evaluate_mAP(all_preds, gt_f, iou_threshold=0.5, ignore_category=False)
    combined = 0.7 * det["mAP"] + 0.3 * cls["mAP"]
    return combined, det["mAP"], cls["mAP"], len(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-v4", action="store_true", help="Skip v4 caching")
    args = parser.parse_args()

    all_imgs = sorted(YOLO_VAL_DIR.glob("*.jpg"))
    print(f"Using ALL {len(all_imgs)} val images")

    gt = load_coco_ground_truth(str(ANN_PATH))

    # Cache v3 (or load from disk)
    if CACHE_FILE_V3.exists():
        print(f"Loading v3 cache from {CACHE_FILE_V3}")
        with open(CACHE_FILE_V3, "rb") as f:
            cache_v3 = pickle.load(f)
    else:
        cache_v3 = cache_model(all_imgs, WEIGHTS_V3, [1280, 1408, 1536], "v3")
        with open(CACHE_FILE_V3, "wb") as f:
            pickle.dump(cache_v3, f)
        print(f"Saved v3 cache to {CACHE_FILE_V3}")

    # Cache v4 (or load from disk)
    cache_v4 = None
    if not args.skip_v4:
        if CACHE_FILE_V4.exists():
            print(f"Loading v4 cache from {CACHE_FILE_V4}")
            with open(CACHE_FILE_V4, "rb") as f:
                cache_v4 = pickle.load(f)
        else:
            cache_v4 = cache_model(all_imgs, WEIGHTS_V4, [1280, 1536], "v4")
            with open(CACHE_FILE_V4, "wb") as f:
                pickle.dump(cache_v4, f)
            print(f"Saved v4 cache to {CACHE_FILE_V4}")

    # Define configs
    trio3 = ["full_1280", "full_1408", "full_1536"]
    duo3 = ["full_1280", "full_1536"]
    v4d = ["full_1280", "full_1536"]
    v4s = ["full_1280"]

    configs = []

    # References
    configs.append(("REF_hard_3p", {"v3_passes": duo3, "v4_passes": v4s, "nms_type": "hard"}))
    configs.append(("REF_soft_5p_s1.5", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5}))

    # A. Sigma sweep on 5-pass (the key variable)
    for s in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
        configs.append((f"5p_s={s}", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": s}))

    # B. Sigma sweep on 3-pass
    for s in [0.5, 0.7, 1.0, 1.5, 2.0]:
        configs.append((f"3p_s={s}", {"v3_passes": duo3, "v4_passes": v4s, "nms_sigma": s}))

    # C. WBF IoU sweep
    for w in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        configs.append((f"5p_wbf={w}", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5, "wbf_iou": w}))

    # D. Conf threshold sweep
    for c in [0.01, 0.02, 0.03, 0.05]:
        configs.append((f"5p_c={c}", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5, "conf": c}))

    # E. conf_type sweep
    for ct in ["max", "avg", "box_and_model_avg"]:
        configs.append((f"5p_ct={ct}", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5, "conf_type": ct}))

    # F. NMS IoU sweep
    for ni in [0.35, 0.40, 0.45, 0.50, 0.55]:
        configs.append((f"5p_niou={ni}", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5, "nms_iou": ni}))

    # G. skip_box_thresh sweep
    for sbt in [0.001, 0.005, 0.01, 0.02]:
        configs.append((f"5p_sbt={sbt}", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5, "skip_box_thresh": sbt}))

    # H. max_dets sweep
    for md in [200, 300, 400, 500]:
        configs.append((f"5p_md={md}", {"v3_passes": trio3, "v4_passes": v4d, "nms_sigma": 1.5, "max_dets": md}))

    # I. Combined: different sigma + wbf + conf
    for s in [0.7, 1.0, 1.5]:
        for w in [0.30, 0.35, 0.40]:
            for c in [0.02, 0.03]:
                configs.append((f"combo_s{s}_w{w}_c{c}", {
                    "v3_passes": trio3, "v4_passes": v4d,
                    "nms_sigma": s, "wbf_iou": w, "conf": c
                }))

    # J. V3-only configs (no v4 dependency)
    for s in [0.5, 0.7, 1.0, 1.5]:
        configs.append((f"v3only_3s_s={s}", {"v3_passes": trio3, "nms_sigma": s}))
    configs.append(("v3only_1280", {"v3_passes": ["full_1280"], "nms_type": "hard"}))

    print(f"\nSweeping {len(configs)} configs...")
    print("=" * 110)

    results = []
    for label, config in configs:
        if cache_v4 is None and config.get("v4_passes"):
            # Skip configs needing v4 if not loaded
            pass
        comb, det, cls, n = run_eval(cache_v3, cache_v4, gt, config)
        results.append((label, comb, det, cls, n))
        print(f"  {label:40s} | {comb:.4f} | Det={det:.4f} | Cls={cls:.4f} | N={n:5d}")

    results.sort(key=lambda x: x[1], reverse=True)
    ref_soft = next((r for r in results if "REF_soft" in r[0]), None)
    ref_hard = next((r for r in results if "REF_hard" in r[0]), None)

    print("\n" + "=" * 110)
    print(f"TOP 25 (submitted_soft={ref_soft[1]:.4f}, prev_hard={ref_hard[1]:.4f})")
    print("=" * 110)
    for i, (label, comb, det, cls, n) in enumerate(results[:25]):
        delta = comb - ref_soft[1]
        print(f"  #{i+1:2d} {label:40s} | {comb:.4f} ({delta:+.4f}) | Det={det:.4f} | Cls={cls:.4f}")

    with open("sweep_v5b_results.json", "w") as f:
        json.dump([{"label": l, "combined": c, "det_mAP": d, "cls_mAP": cl, "n_preds": n}
                   for l, c, d, cl, n in results], f, indent=2)

    print(f"\nWINNER: {results[0][0]} -> {results[0][1]:.4f}")


if __name__ == "__main__":
    main()
