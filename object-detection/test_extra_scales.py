"""Test extra scales: compare 5-pass vs 7-pass using cached + freshly computed detections."""
import pickle, json, time
import numpy as np
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion

# Load caches
print("Loading caches...")
c3 = pickle.load(open('cache_v3_29.pkl', 'rb'))
c4 = pickle.load(open('cache_v4_29.pkl', 'rb'))

image_ids = sorted(c3.keys())
print(f"Images: {len(image_ids)}")

# Load GT
with open("data/coco/train/annotations.json") as f:
    gt = json.load(f)
gt_by_image = {}
for ann in gt["annotations"]:
    gt_by_image.setdefault(ann["image_id"], []).append(ann)

# Generate v3@1152 and v3@1664 detections
print("Loading v3 model for new scales...")
import torch
_orig = torch.load
def _p(*a, **k):
    k.setdefault("weights_only", False)
    return _orig(*a, **k)
torch.load = _p

from ultralytics import YOLO
model_v3 = YOLO("best.onnx", task="detect")
device = "cpu"

# Find image paths from GT
img_dir = Path("data/coco/train/images")
id_to_path = {}
for img_info in gt["images"]:
    p = img_dir / img_info["file_name"]
    if p.exists():
        id_to_path[img_info["id"]] = p

def extract_dets(results):
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            w, h = x2 - x1, y2 - y1
            if w < 3 or h < 3:
                continue
            dets.append({
                "category_id": int(r.boxes.cls[i].item()),
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "score": round(float(r.boxes.conf[i].item()), 3),
            })
    return dets

extra = {1152: {}, 1664: {}}
for scale in [1152, 1664]:
    t0 = time.time()
    for img_id in image_ids:
        if img_id not in id_to_path:
            continue
        r = model_v3(str(id_to_path[img_id]), device=device, verbose=False,
                     conf=0.03, iou=0.5, max_det=400, imgsz=scale, augment=False)
        extra[scale][img_id] = extract_dets(r)
    print(f"  v3@{scale}: {time.time()-t0:.1f}s")

# NMS + WBF helpers
def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix = max(0, min(ax2, bx2) - max(ax, bx))
    iy = max(0, min(ay2, by2) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0

def soft_nms(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.01):
    if not dets:
        return dets
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
            cs = {}
            for ab in absorbed[ki]:
                cs[ab["category_id"]] = cs.get(ab["category_id"], 0) + ab["score"]
            kept[ki]["category_id"] = max(cs, key=cs.get)
    return kept

def wbf(passes, img_w, img_h, iou_thresh=0.35, skip=0.005, ct='box_and_model_avg'):
    bl, sl, ll = [], [], []
    for p in passes:
        b, s, l = [], [], []
        for d in p:
            x, y, w, h = d["bbox"]
            x1, y1 = max(0, x/img_w), max(0, y/img_h)
            x2, y2 = min(1, (x+w)/img_w), min(1, (y+h)/img_h)
            if x2 <= x1 or y2 <= y1:
                continue
            b.append([x1,y1,x2,y2]); s.append(d["score"]); l.append(d["category_id"])
        if b:
            bl.append(np.array(b,dtype=np.float32))
            sl.append(np.array(s,dtype=np.float32))
            ll.append(np.array(l,dtype=np.int32))
    if not bl:
        return []
    fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thresh, skip_box_thr=skip, conf_type=ct)
    return [{"category_id": int(la), "bbox": [round(b[0]*img_w,1),round(b[1]*img_h,1),
             round((b[2]-b[0])*img_w,1),round((b[3]-b[1])*img_h,1)], "score": round(float(sc),3)}
            for b, sc, la in zip(fb, fs, fl)]

def compute_ap(preds, gts, iou_thresh=0.5):
    if not gts:
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    matched = [False] * len(gts)
    tp, fp = [], []
    for p in preds:
        best_iou, best_j = 0, -1
        for j, g in enumerate(gts):
            iou = _iou(p["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and not matched[best_j]:
            tp.append(1); fp.append(0); matched[best_j] = True
        else:
            tp.append(0); fp.append(1)
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    rec = tp_c / len(gts); prec = tp_c / (tp_c + fp_c)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])

def evaluate(name, get_passes_fn):
    det_aps, cls_aps = [], []
    for img_id in image_ids:
        if img_id not in id_to_path:
            continue
        img_w = c3[img_id]['img_w']
        img_h = c3[img_id]['img_h']
        passes = get_passes_fn(img_id)
        fused = wbf(passes, img_w, img_h)
        dets = soft_nms(fused)
        
        gts_raw = gt_by_image.get(img_id, [])
        gts = [{"bbox": g["bbox"], "category_id": g["category_id"]} for g in gts_raw]
        
        det_p = [{"bbox": d["bbox"], "score": d["score"], "category_id": 0} for d in dets]
        det_g = [{"bbox": g["bbox"], "category_id": 0} for g in gts]
        det_aps.append(compute_ap(det_p, det_g))
        cls_aps.append(compute_ap(dets, gts))
    
    det = np.mean(det_aps); cls = np.mean(cls_aps)
    comb = 0.7 * det + 0.3 * cls
    print(f"  {name:35s} -> {comb:.4f}  (Det={det:.4f}  Cls={cls:.4f})")
    return comb

print("\n=== Comparing ensemble configs ===\n")

# Current 5-pass
evaluate("5-pass (current)", lambda iid: [
    c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
    c4[iid]['full_1280'], c4[iid]['full_1536'],
])

# 7-pass (+1152, +1664)
evaluate("7-pass (+v3@1152, +v3@1664)", lambda iid: [
    c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
    c4[iid]['full_1280'], c4[iid]['full_1536'],
    extra[1152].get(iid, []), extra[1664].get(iid, []),
])

# 6-pass (+1664 only)
evaluate("6-pass (+v3@1664)", lambda iid: [
    c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
    c4[iid]['full_1280'], c4[iid]['full_1536'],
    extra[1664].get(iid, []),
])

# 6-pass (+1152 only)
evaluate("6-pass (+v3@1152)", lambda iid: [
    c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
    c4[iid]['full_1280'], c4[iid]['full_1536'],
    extra[1152].get(iid, []),
])

print("\nDone!")
