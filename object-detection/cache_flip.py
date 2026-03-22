"""Cache horizontal flip TTA detections for all 3 models at multiple scales.

Run on GCP VM (with GPU):
  python cache_flip.py

After caching, run sweep_flip.py to find optimal params for the extended ensemble.
"""
import pickle, time, json
from pathlib import Path
import numpy as np
from PIL import Image

YOLO_VAL_DIR = Path("data/yolo/images/val")
CACHE_FLIP_V3 = Path("cache_flip_v3_29.pkl")
CACHE_FLIP_V4 = Path("cache_flip_v4_29.pkl")
CACHE_FLIP_V6 = Path("cache_flip_v6_29.pkl")


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


def cache_flip_model(img_paths, weights_path, scales, name):
    """Cache inference on horizontally flipped images."""
    import torch
    _ol = torch.load
    def _pl(*a, **k):
        if 'weights_only' not in k:
            k['weights_only'] = False
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
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.array(img_flip)
        img.close()

        entry = {"img_w": img_w, "img_h": img_h}
        ti = time.time()
        for s in scales:
            results = model(arr, device=device, verbose=False,
                          conf=0.01, iou=0.7, max_det=600, imgsz=s, augment=False)
            dets = extract_dets(results)
            # Flip bboxes back: x -> img_w - x - w
            for d in dets:
                x, y, w, h = d["bbox"]
                d["bbox"] = [round(img_w - x - w, 1), y, w, h]
            entry[f"flip_{s}"] = dets
        cache[image_id] = entry
        if (idx + 1) % 5 == 0 or idx == 0:
            n = sum(len(entry.get(f"flip_{s}", [])) for s in scales)
            print(f"  {name} [{idx+1}/{len(img_paths)}] img_{image_id:05d}: {n} flip dets | {time.time()-ti:.1f}s")
    print(f"{name} done in {time.time()-t0:.0f}s")
    del model
    return cache


if __name__ == '__main__':
    all_imgs = sorted(YOLO_VAL_DIR.glob("*.jpg"))
    print(f"Caching flip TTA for {len(all_imgs)} val images")

    # v3
    if not CACHE_FLIP_V3.exists():
        cv3 = cache_flip_model(all_imgs, Path("best.onnx"), [1280, 1408, 1536], "v3_flip")
        with open(CACHE_FLIP_V3, "wb") as f:
            pickle.dump(cv3, f)
        print(f"Saved {CACHE_FLIP_V3}")
    else:
        print(f"Skipping v3 (cache exists)")

    # v4
    if not CACHE_FLIP_V4.exists():
        cv4 = cache_flip_model(all_imgs, Path("best_v4.onnx"), [1280, 1536], "v4_flip")
        with open(CACHE_FLIP_V4, "wb") as f:
            pickle.dump(cv4, f)
        print(f"Saved {CACHE_FLIP_V4}")
    else:
        print(f"Skipping v4 (cache exists)")

    # v6
    if not CACHE_FLIP_V6.exists():
        cv6 = cache_flip_model(all_imgs, Path("best_v6.onnx"), [1280, 1408, 1536], "v6_flip")
        with open(CACHE_FLIP_V6, "wb") as f:
            pickle.dump(cv6, f)
        print(f"Saved {CACHE_FLIP_V6}")
    else:
        print(f"Skipping v6 (cache exists)")

    print("\nDone! Now run sweep_flip.py to find optimal params with flip TTA.")
