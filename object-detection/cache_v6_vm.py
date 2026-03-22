"""Cache v6 (YOLO11m) detections on GCP VM using GPU. Upload val images first."""
import time, pickle
from pathlib import Path
from PIL import Image

YOLO_VAL_DIR = Path("data/yolo/images/val")
WEIGHTS_V6 = Path("runs/detect/runs/detect/train_v6/weights/best.pt")
CACHE_FILE = Path("cache_v6_29.pkl")


def extract_dets(results):
    dets = []
    for r in results:
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            w, h = x2 - x1, y2 - y1
            if w < 3 or h < 3:
                continue
            dets.append({
                "category_id": int(r.boxes.cls[i].item()),
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "score": round(float(r.boxes.conf[i].item()), 4),
            })
    return dets


def main():
    import torch
    _ol = torch.load
    def _pl(*a, **k):
        if 'weights_only' not in k:
            k['weights_only'] = False
        return _ol(*a, **k)
    torch.load = _pl
    from ultralytics import YOLO

    all_imgs = sorted(YOLO_VAL_DIR.glob("*.jpg"))
    print(f"Caching v6 @ [1280, 1408, 1536] on {len(all_imgs)} images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {WEIGHTS_V6}...")
    model = YOLO(str(WEIGHTS_V6), task="detect")
    model(str(all_imgs[0]), device=device, verbose=False, conf=0.5, max_det=1, imgsz=1280)
    print(f"  -> {device}, warmup done")

    cache = {}
    t0 = time.time()
    scales = [1280, 1408, 1536]
    for idx, img_path in enumerate(all_imgs):
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
        n = sum(len(entry.get(f"full_{s}", [])) for s in scales)
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(all_imgs)}] img_{image_id:05d}: {n} dets | {time.time()-ti:.1f}s")

    print(f"v6 done in {time.time()-t0:.0f}s")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved to {CACHE_FILE}")
    total_dets = sum(sum(len(v.get(f"full_{s}", [])) for s in scales) for v in cache.values())
    print(f"Total detections cached: {total_dets}")


if __name__ == "__main__":
    main()
