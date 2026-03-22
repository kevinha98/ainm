"""Inference entry point for the NorgesGruppen object detection competition.

Sandbox contract:
    python run.py --input /data/images --output /output/predictions.json

Strategy — adaptive time-budgeted inference with WBF fusion:
  - FULL mode:   Multi-scale SAHI (640+960) + full-image, WBF fusion across passes
  - MEDIUM mode: SAHI (640) + full-image, WBF fusion
  - FAST mode:   Full-image pass only

Uses Weighted Boxes Fusion (ensemble-boxes) to merge overlapping detections
from multiple inference passes, treating each pass as a separate "model".
This produces more accurate boxes and higher recall than simple NMS.
"""
import argparse
import json
import time
import random
from pathlib import Path

# ── Constants ───────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_WEIGHTS = SCRIPT_DIR / "best.onnx"       # Primary: no torch.load needed
FINETUNED_WEIGHTS = SCRIPT_DIR / "best.pt"     # Fallback
PRETRAINED_WEIGHTS = SCRIPT_DIR / "yolov8m.pt"  # Detection-only fallback
TOTAL_BUDGET_SEC = 270  # 300s sandbox timeout minus 30s safety margin


def get_image_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def list_images(input_dir: Path) -> list:
    valid = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in valid)


# ── Random Baseline ─────────────────────────────────────────────────────
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


# ── WBF (Weighted Boxes Fusion) ─────────────────────────────────────────
def wbf_fuse(passes_dets, img_w, img_h, iou_thresh=0.55, skip_box_thresh=0.01):
    """Apply WBF across multiple inference passes.

    Each element in passes_dets is a list of detections from one pass
    (full-image, SAHI-640, SAHI-960). WBF treats each pass as a separate
    "model" and fuses overlapping boxes by averaging coordinates weighted
    by confidence. This is superior to NMS which just picks one box.

    Args:
        passes_dets: List of [dets_pass1, dets_pass2, ...] where each is
                     a list of {"category_id", "bbox": [x,y,w,h], "score"}
        img_w, img_h: Image dimensions for normalization
        iou_thresh: IoU threshold for fusion (0.55 works well for products)
        skip_box_thresh: Minimum score to keep
    """
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

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_thresh,
        skip_box_thr=skip_box_thresh,
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


# ── NMS fallback ────────────────────────────────────────────────────────
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


# ── Detection helpers ───────────────────────────────────────────────────
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
            if w < 2 or h < 2:
                continue
            cat_id = 0 if detection_only else int(r.boxes.cls[i].item())
            dets.append({
                "category_id": cat_id,
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "score": round(float(r.boxes.conf[i].item()), 3),
            })
    return dets


def _get_slices(img_w, img_h, slice_size=640, overlap=0.2):
    step = int(slice_size * (1 - overlap))
    slices = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            sw = min(slice_size, img_w - x)
            sh = min(slice_size, img_h - y)
            if sw < slice_size * 0.3 or sh < slice_size * 0.3:
                continue
            slices.append((x, y, sw, sh))
    return slices


def _run_sahi_pass(model, img_path, img_w, img_h, device, detection_only, slice_size, overlap=0.2):
    """Run a single SAHI sliced inference pass and return detections."""
    import numpy as np
    from PIL import Image

    dets = []
    slices = _get_slices(img_w, img_h, slice_size=slice_size, overlap=overlap)
    for cx, cy, cw, ch in slices:
        crop_img = Image.open(img_path).crop((cx, cy, cx + cw, cy + ch))
        crop_arr = np.array(crop_img)
        results = model(crop_arr, device=device, verbose=False,
                        conf=0.05, iou=0.7, max_det=300, augment=False)
        dets.extend(_extract_dets(results, detection_only,
                                  offset_x=cx, offset_y=cy,
                                  clip_w=img_w, clip_h=img_h))
    return dets


# ── Single-image inference at three quality levels ──────────────────────
def _infer_full(model, img_path, device, detection_only, use_tta=False):
    """FULL: full-image + multi-scale SAHI (640+960) with WBF fusion across passes."""
    from PIL import Image

    img = Image.open(img_path)
    img_w, img_h = img.size
    img.close()

    # Pass 1: Full-image (best for large/medium objects)
    full_results = model(str(img_path), device=device, verbose=False,
                         conf=0.05, iou=0.7, max_det=600, augment=use_tta)
    pass_full = _extract_dets(full_results, detection_only)

    # Pass 2: SAHI at 640 (best for small products)
    pass_640 = _run_sahi_pass(model, img_path, img_w, img_h, device, detection_only, slice_size=640)

    # Pass 3: SAHI at 960 (good for medium products, better context)
    pass_960 = _run_sahi_pass(model, img_path, img_w, img_h, device, detection_only, slice_size=960)

    # Fuse all three passes with WBF
    try:
        return wbf_fuse([pass_full, pass_640, pass_960], img_w, img_h, iou_thresh=0.55)
    except Exception:
        all_dets = pass_full + pass_640 + pass_960
        return nms_per_image(all_dets, iou_thresh=0.5)


def _infer_medium(model, img_path, device, detection_only, use_tta=False):
    """MEDIUM: full-image + SAHI (640) with WBF fusion."""
    from PIL import Image

    img = Image.open(img_path)
    img_w, img_h = img.size
    img.close()

    full_results = model(str(img_path), device=device, verbose=False,
                         conf=0.05, iou=0.7, max_det=600, augment=use_tta)
    pass_full = _extract_dets(full_results, detection_only)

    pass_640 = _run_sahi_pass(model, img_path, img_w, img_h, device, detection_only, slice_size=640)

    try:
        return wbf_fuse([pass_full, pass_640], img_w, img_h, iou_thresh=0.55)
    except Exception:
        return nms_per_image(pass_full + pass_640, iou_thresh=0.5)


def _infer_fast(model, img_path, device, detection_only):
    """FAST: full-image pass only."""
    results = model(str(img_path), device=device, verbose=False,
                    conf=0.10, iou=0.6, max_det=300, augment=False)
    return _extract_dets(results, detection_only)


# ── Torch compatibility patch ───────────────────────────────────────────
def _patch_torch_load():
    import torch
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load


# ── Time-budgeted inference ─────────────────────────────────────────────
def predict_yolo(input_dir: Path, weights: Path, detection_only: bool = False) -> list[dict]:
    """Adaptive inference: starts with best quality, downgrades if time is tight."""
    import torch
    _patch_torch_load()
    from ultralytics import YOLO

    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_onnx = str(weights).endswith(".onnx")
    if is_onnx:
        model = YOLO(str(weights), task="detect")
    else:
        model = YOLO(str(weights))

    images = list_images(input_dir)
    n_images = len(images)
    predictions = []

    use_tta = (device == "cuda") and (not is_onnx)

    fmt = "ONNX" if is_onnx else "PyTorch"
    print(f"Device: {device} | Format: {fmt} | TTA: {use_tta} | Images: {n_images} | Budget: {TOTAL_BUDGET_SEC}s")

    # Warm-up
    if n_images > 0:
        model(str(images[0]), device=device, verbose=False, conf=0.5, max_det=1)
        print(f"Warm-up done in {time.time() - t_start:.1f}s")

    mode = "FULL"

    for idx, img_path in enumerate(images):
        elapsed = time.time() - t_start
        remaining = TOTAL_BUDGET_SEC - elapsed
        images_left = n_images - idx

        if images_left > 0:
            time_per_image = remaining / images_left
            if time_per_image < 1.5:
                mode = "FAST"
            elif time_per_image < 5.0:
                mode = "MEDIUM"

        t_img = time.time()

        if mode == "FULL":
            dets = _infer_full(model, img_path, device, detection_only, use_tta=use_tta)
        elif mode == "MEDIUM":
            dets = _infer_medium(model, img_path, device, detection_only, use_tta=use_tta)
        else:
            dets = _infer_fast(model, img_path, device, detection_only)

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


# ── Main ────────────────────────────────────────────────────────────────
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
            print(f"Using ONNX model: {ONNX_WEIGHTS.name}")
            predictions = predict_yolo(input_dir, ONNX_WEIGHTS, detection_only=False)
        elif FINETUNED_WEIGHTS.exists():
            print(f"Using fine-tuned .pt model: {FINETUNED_WEIGHTS.name}")
            predictions = predict_yolo(input_dir, FINETUNED_WEIGHTS, detection_only=False)
        elif PRETRAINED_WEIGHTS.exists():
            print(f"Using pretrained model (detection-only): {PRETRAINED_WEIGHTS.name}")
            predictions = predict_yolo(input_dir, PRETRAINED_WEIGHTS, detection_only=True)
        else:
            print("No model weights found — using random baseline")
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
