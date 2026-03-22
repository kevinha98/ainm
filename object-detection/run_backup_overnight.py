"""Inference entry point for the NorgesGruppen object detection competition.

Sandbox contract:
    python run.py --input /data/images --output /output/predictions.json

Strategy - multi-scale + triple-model ensemble with WBF + soft-NMS:
  v3 (YOLOv8m 104MB) + v4 (YOLOv8l 176MB) + v6 (YOLO11m 78MB)
  Sweep v8 3-model: 8 passes, WBF iou=0.4989, weights=[1,2,1,2,3,4,1,2] -> local 0.95145
  Soft-NMS: sigma=1.06, iou=0.309, score=9e-6, max_dets=400

  Adaptive time budget:
    FULL:   v3@1280+1408+1536 + v4@1280+1536 + v6@1280+1408+1536 -> WBF -> soft-NMS (8 passes)
    MEDIUM: v3@1280 + v4@1280+1536 + v6@1280+1536 -> WBF -> soft-NMS (5 passes)
    FAST:   v3@1280 only -> soft-NMS (1 pass)
"""
import argparse
import json
import time
import random
from pathlib import Path

# -- Constants --
SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_WEIGHTS = SCRIPT_DIR / "best.onnx"       # v3 YOLOv8m (primary)
ONNX_WEIGHTS_V4 = SCRIPT_DIR / "best_v4.onnx"  # v4 YOLOv8l (ensemble partner)
ONNX_WEIGHTS_V6 = SCRIPT_DIR / "best_v6.onnx"  # v6 YOLO11m (third model)
ONNX_WEIGHTS_2 = SCRIPT_DIR / "best2.onnx"    # Legacy secondary model
FINETUNED_WEIGHTS = SCRIPT_DIR / "best.pt"     # Fallback
PRETRAINED_WEIGHTS = SCRIPT_DIR / "yolov8m.pt"  # Detection-only fallback
TOTAL_BUDGET_SEC = 270  # 300s sandbox timeout minus 30s safety margin
MAX_DETS_PER_IMAGE = 450  # Cap detections per image (optimized: 450 > 400 for cls_mAP)


def get_image_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def list_images(input_dir: Path) -> list:
    valid = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in valid)


# -- Random Baseline --
def predict_random(input_dir: Path) -> list[dict]:
    predictions = []
    for img in list_images(input_dir):
        image_id = get_image_id(img)
        for _ in range(random.randint(5, 20)):
            predictions.append({
                "image_id": image_id,
                "category_id": random.randint(0, 355),
                "bbox": [
                    round(random.uniform(0, 1500), 1),
                    round(random.uniform(0, 800), 1),
                    round(random.uniform(20, 200), 1),
                    round(random.uniform(20, 200), 1),
                ],
                "score": round(random.uniform(0.01, 1.0), 3),
            })
    return predictions


# -- WBF (Weighted Boxes Fusion) --
def wbf_fuse(passes_dets, img_w, img_h, iou_thresh=0.50, skip_box_thresh=0.02, conf_type='box_and_model_avg', weights=None):
    from ensemble_boxes import weighted_boxes_fusion
    import numpy as np

    boxes_list = []
    scores_list = []
    labels_list = []

    for pass_dets in passes_dets:
        boxes = []
        scores = []
        labels = []
        for d in pass_dets:
            x, y, w, h = d["bbox"]
            x1 = max(0.0, x / img_w)
            y1 = max(0.0, y / img_h)
            x2 = min(1.0, (x + w) / img_w)
            y2 = min(1.0, (y + h) / img_h)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            scores.append(d["score"])
            labels.append(d["category_id"])
        if boxes:
            boxes_list.append(np.array(boxes, dtype=np.float32))
            scores_list.append(np.array(scores, dtype=np.float32))
            labels_list.append(np.array(labels, dtype=np.int32))

    if not boxes_list:
        return []

    kwargs = dict(iou_thr=iou_thresh, skip_box_thr=skip_box_thresh, conf_type=conf_type)
    if weights is not None:
        kwargs['weights'] = weights
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, **kwargs
    )

    result = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x1, y1, x2, y2 = box
        result.append({
            "category_id": int(label),
            "bbox": [
                round(x1 * img_w, 1),
                round(y1 * img_h, 1),
                round((x2 - x1) * img_w, 1),
                round((y2 - y1) * img_h, 1),
            ],
            "score": round(float(score), 3),
        })
    return result


# -- NMS helpers --
def _iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix = max(0, min(ax2, bx2) - max(ax, bx))
    iy = max(0, min(ay2, by2) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def nms_per_image(dets, iou_thresh=0.5):
    by_class = {}
    for d in dets:
        by_class.setdefault(d["category_id"], []).append(d)
    kept = []
    for cls_dets in by_class.values():
        cls_dets.sort(key=lambda x: x["score"], reverse=True)
        selected = []
        for d in cls_dets:
            if all(_iou(d["bbox"], s["bbox"]) < iou_thresh for s in selected):
                selected.append(d)
        kept.extend(selected)
    return kept


def _soft_nms_class_agnostic(dets, iou_thresh=0.45, sigma=0.5, score_thresh=0.01):
    """Soft-NMS: decay overlapping box scores with Gaussian penalty.
    Keeps more boxes than hard NMS, significantly improving cls_mAP.
    Uses category voting for overlapping boxes with different labels."""
    import numpy as np
    if not dets:
        return dets
    dets = [d.copy() for d in dets]
    dets.sort(key=lambda x: x["score"], reverse=True)
    kept = []
    absorbed = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        cur_abs = [best.copy()]
        remaining = []
        for d in dets:
            iou = _iou(best["bbox"], d["bbox"])
            if iou >= iou_thresh:
                cur_abs.append(d.copy())
            # Gaussian decay
            d["score"] *= np.exp(-(iou ** 2) / sigma)
            if d["score"] >= score_thresh:
                remaining.append(d)
        absorbed.append(cur_abs)
        dets = sorted(remaining, key=lambda x: x["score"], reverse=True)
    # Category voting
    for ki, k in enumerate(kept):
        if len(absorbed[ki]) > 1:
            cat_scores = {}
            for ab in absorbed[ki]:
                cat_scores[ab["category_id"]] = cat_scores.get(ab["category_id"], 0) + ab["score"]
            kept[ki]["category_id"] = max(cat_scores, key=cat_scores.get)
    return kept


# -- Detection helpers --
def _extract_dets(results, detection_only, offset_x=0, offset_y=0, clip_w=None, clip_h=None):
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y
            if clip_w is not None:
                x1 = max(0, min(x1, clip_w))
                x2 = max(0, min(x2, clip_w))
            if clip_h is not None:
                y1 = max(0, min(y1, clip_h))
                y2 = max(0, min(y2, clip_h))
            w, h = x2 - x1, y2 - y1
            if w < 3 or h < 3:
                continue
            cat_id = 0 if detection_only else int(r.boxes.cls[i].item())
            dets.append({
                "category_id": cat_id,
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "score": round(float(r.boxes.conf[i].item()), 3),
            })
    return dets


# -- Single-image inference at three quality levels --
def _infer_full(model_v3, model_v4, img_path, device, detection_only, model_v6=None):
    """FULL: v3@1280+1408+1536 + v4@1280+1536 + v6@1280+1408+1536, WBF + soft-NMS (8 passes).
    3-model sweep winner: local 0.9490."""
    from PIL import Image
    img = Image.open(img_path)
    img_w, img_h = img.size
    img.close()

    # v3 at 3 scales
    r1 = model_v3(str(img_path), device=device, verbose=False,
                  conf=0.03, iou=0.5, max_det=400, imgsz=1280, augment=False)
    pass_v3_1280 = _extract_dets(r1, detection_only)

    r2 = model_v3(str(img_path), device=device, verbose=False,
                  conf=0.03, iou=0.5, max_det=400, imgsz=1408, augment=False)
    pass_v3_1408 = _extract_dets(r2, detection_only)

    r3 = model_v3(str(img_path), device=device, verbose=False,
                  conf=0.03, iou=0.5, max_det=400, imgsz=1536, augment=False)
    pass_v3_1536 = _extract_dets(r3, detection_only)

    # v4 at 2 scales
    r4 = model_v4(str(img_path), device=device, verbose=False,
                  conf=0.03, iou=0.5, max_det=400, imgsz=1280, augment=False)
    pass_v4_1280 = _extract_dets(r4, detection_only)

    r5 = model_v4(str(img_path), device=device, verbose=False,
                  conf=0.03, iou=0.5, max_det=400, imgsz=1536, augment=False)
    pass_v4_1536 = _extract_dets(r5, detection_only)

    all_passes = [pass_v3_1280, pass_v3_1408, pass_v3_1536, pass_v4_1280, pass_v4_1536]
    all_weights = [1, 2, 1, 2, 3]

    # v6 at 3 scales (if available)
    if model_v6:
        r6 = model_v6(str(img_path), device=device, verbose=False,
                      conf=0.03, iou=0.5, max_det=400, imgsz=1280, augment=False)
        r7 = model_v6(str(img_path), device=device, verbose=False,
                      conf=0.03, iou=0.5, max_det=400, imgsz=1408, augment=False)
        r8 = model_v6(str(img_path), device=device, verbose=False,
                      conf=0.03, iou=0.5, max_det=400, imgsz=1536, augment=False)
        all_passes.extend([_extract_dets(r6, detection_only), _extract_dets(r7, detection_only), _extract_dets(r8, detection_only)])
        all_weights.extend([4, 1, 2])

    # WBF fusion
    try:
        fused = wbf_fuse(all_passes, img_w, img_h, iou_thresh=0.4989, skip_box_thresh=0.005,
                         weights=all_weights)
        return _soft_nms_class_agnostic(fused, iou_thresh=0.309, sigma=1.06, score_thresh=9e-6)
    except Exception:
        all_dets = [d for p in all_passes for d in p]
        return _soft_nms_class_agnostic(nms_per_image(all_dets, iou_thresh=0.5), iou_thresh=0.45, score_thresh=0.001)


def _infer_medium(model_v3, model_v4, img_path, device, detection_only, model_v6=None):
    """MEDIUM: v3@1280 + v4@1280+1536 + v6@1280+1536, WBF + soft-NMS (5 passes)."""
    from PIL import Image
    img = Image.open(img_path)
    img_w, img_h = img.size
    img.close()

    r1 = model_v3(str(img_path), device=device, verbose=False,
                  conf=0.03, iou=0.5, max_det=400, imgsz=1280, augment=False)
    pass_v3 = _extract_dets(r1, detection_only)

    r2 = model_v4(str(img_path), device=device, verbose=False,
                  conf=0.03, iou=0.5, max_det=400, imgsz=1280, augment=False)
    pass_v4_1280 = _extract_dets(r2, detection_only)

    r3 = model_v4(str(img_path), device=device, verbose=False,
                  conf=0.03, iou=0.5, max_det=400, imgsz=1536, augment=False)
    pass_v4_1536 = _extract_dets(r3, detection_only)

    all_passes = [pass_v3, pass_v4_1280, pass_v4_1536]
    all_weights = [1, 2, 3]

    if model_v6:
        r4 = model_v6(str(img_path), device=device, verbose=False,
                      conf=0.03, iou=0.5, max_det=400, imgsz=1280, augment=False)
        r5 = model_v6(str(img_path), device=device, verbose=False,
                      conf=0.03, iou=0.5, max_det=400, imgsz=1536, augment=False)
        all_passes.extend([_extract_dets(r4, detection_only), _extract_dets(r5, detection_only)])
        all_weights.extend([4, 2])

    try:
        fused = wbf_fuse(all_passes, img_w, img_h,
                         iou_thresh=0.4989, skip_box_thresh=0.005,
                         weights=all_weights)
        return _soft_nms_class_agnostic(fused, iou_thresh=0.309, sigma=1.06, score_thresh=9e-6)
    except Exception:
        all_dets = [d for p in all_passes for d in p]
        return _soft_nms_class_agnostic(
            nms_per_image(all_dets, iou_thresh=0.5),
            iou_thresh=0.309, sigma=1.06, score_thresh=9e-6)


def _infer_fast(model_v3, img_path, device, detection_only):
    """FAST: v3@1280 only + soft-NMS (single pass fallback)."""
    results = model_v3(str(img_path), device=device, verbose=False,
                       conf=0.03, iou=0.5, max_det=400, imgsz=1280, augment=False)
    dets = _extract_dets(results, detection_only)
    return _soft_nms_class_agnostic(dets, iou_thresh=0.309, sigma=1.06, score_thresh=9e-6)


# -- Torch compatibility patch --
def _patch_torch_load():
    import torch
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load


# -- Time-budgeted inference --
def predict_yolo(input_dir: Path, weights: Path, detection_only: bool = False) -> list[dict]:
    """Adaptive ensemble inference: v3 multi-scale + v4, time-budgeted."""
    import torch
    _patch_torch_load()
    from ultralytics import YOLO

    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load primary model (v3)
    model_v3 = YOLO(str(weights), task="detect")

    # Load secondary model (v4) for ensemble
    model_v4 = None
    if ONNX_WEIGHTS_V4.exists():
        model_v4 = YOLO(str(ONNX_WEIGHTS_V4), task="detect")
        print(f"Loaded v4: {ONNX_WEIGHTS_V4.name}")

    # Load third model (v6) for triple ensemble
    model_v6 = None
    if ONNX_WEIGHTS_V6.exists():
        model_v6 = YOLO(str(ONNX_WEIGHTS_V6), task="detect")
        print(f"Loaded v6: {ONNX_WEIGHTS_V6.name}")

    print(f"Ensemble: v3={weights.name}, v4={'yes' if model_v4 else 'no'}, v6={'yes' if model_v6 else 'no'}")

    images = list_images(input_dir)
    n_images = len(images)
    predictions = []

    print(f"Device: {device} | Models: v3+{'v4+' if model_v4 else ''}{'v6' if model_v6 else ''} | Images: {n_images} | Budget: {TOTAL_BUDGET_SEC}s")

    # Warm-up all models
    if n_images > 0:
        model_v3(str(images[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
        if model_v4:
            model_v4(str(images[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
        if model_v6:
            model_v6(str(images[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
        print(f"Warm-up done in {time.time() - t_start:.1f}s")

    mode = "FULL"

    for idx, img_path in enumerate(images):
        elapsed = time.time() - t_start
        remaining = TOTAL_BUDGET_SEC - elapsed
        images_left = n_images - idx

        if images_left > 0:
            time_per_image = remaining / images_left
            if time_per_image < 1.0:
                mode = "FAST"
            elif time_per_image < 2.5:
                mode = "MEDIUM"

        t_img = time.time()

        if mode == "FULL" and model_v4:
            dets = _infer_full(model_v3, model_v4, img_path, device, detection_only, model_v6=model_v6)
        elif mode == "MEDIUM" and model_v4:
            dets = _infer_medium(model_v3, model_v4, img_path, device, detection_only, model_v6=model_v6)
        else:
            dets = _infer_fast(model_v3, img_path, device, detection_only)

        # Cap detections per image to reduce false positives
        if len(dets) > MAX_DETS_PER_IMAGE:
            dets.sort(key=lambda x: x["score"], reverse=True)
            dets = dets[:MAX_DETS_PER_IMAGE]

        image_id = get_image_id(img_path)
        for d in dets:
            d["image_id"] = image_id
            predictions.append(d)

        img_time = time.time() - t_img
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  [{idx+1}/{n_images}] {mode} | {img_time:.1f}s | "
                  f"{len(dets)} dets | {remaining:.0f}s left")

    elapsed = time.time() - t_start
    print(f"Done: {len(predictions)} predictions in {elapsed:.1f}s")
    return predictions


# -- Main --
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    predictions = []

    try:
        if ONNX_WEIGHTS.exists():
            print(f"Using ONNX model: {ONNX_WEIGHTS.name} + ensemble")
            predictions = predict_yolo(input_dir, ONNX_WEIGHTS, detection_only=False)
        elif FINETUNED_WEIGHTS.exists():
            print(f"Using fine-tuned .pt model: {FINETUNED_WEIGHTS.name}")
            predictions = predict_yolo(input_dir, FINETUNED_WEIGHTS, detection_only=False)
        elif PRETRAINED_WEIGHTS.exists():
            print(f"Using pretrained model (detection-only): {PRETRAINED_WEIGHTS.name}")
            predictions = predict_yolo(input_dir, PRETRAINED_WEIGHTS, detection_only=True)
        else:
            print("No model weights found - using random baseline")
            predictions = predict_random(input_dir)
    except Exception as e:
        print(f"ERROR during inference: {e}")
        print("Falling back to random baseline")
        predictions = predict_random(input_dir)

    print(f"Total predictions: {len(predictions)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
