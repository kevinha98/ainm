"""Prepare data for full-dataset training (ALL 248 images in train).

Uses the same 248 images for both train and val since we evaluate via competition submissions.
This gives ~15% more training data compared to the default 85/15 split.
"""
import json, shutil
from pathlib import Path
from collections import defaultdict

COCO_ANN = Path("data/coco/train/annotations.json")
COCO_IMG = Path("data/coco/train/images")
YOLO_DIR = Path("data/yolo_full")  # Separate directory to not overwrite original


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    xc = max(0.0, min(1.0, (x + w / 2) / img_w))
    yc = max(0.0, min(1.0, (y + h / 2) / img_h))
    wn = max(0.0, min(1.0, w / img_w))
    hn = max(0.0, min(1.0, h / img_h))
    return xc, yc, wn, hn


def main():
    coco = json.load(open(COCO_ANN, encoding="utf-8"))
    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    print(f"Dataset: {len(images)} images, {len(coco['annotations'])} annotations, {len(categories)} categories")

    # Create directories
    for split in ["train", "val"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_ids = sorted(images.keys())
    total_anns = 0

    # ALL images go into train AND val (same set)
    for split in ["train", "val"]:
        for img_id in all_ids:
            img_info = images[img_id]
            filename = img_info["file_name"]
            stem = Path(filename).stem

            # Copy image
            dst = YOLO_DIR / "images" / split / filename
            if not dst.exists():
                src = COCO_IMG / filename
                shutil.copy2(src, dst)

            # Write YOLO labels
            anns = anns_by_image.get(img_id, [])
            lines = []
            for ann in anns:
                xc, yc, wn, hn = coco_to_yolo_bbox(ann["bbox"], img_info["width"], img_info["height"])
                if wn < 1e-6 or hn < 1e-6:
                    continue
                lines.append(f"{ann['category_id']} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            (YOLO_DIR / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
            )
            if split == "train":
                total_anns += len(lines)

    # Write dataset.yaml
    names_lines = []
    for cat_id in sorted(categories.keys()):
        name = categories[cat_id].replace('"', '\\"')
        names_lines.append(f'  {cat_id}: "{name}"')

    yaml_content = f"""# Full dataset (ALL 248 images for training)
path: {YOLO_DIR.resolve().as_posix()}
train: images/train
val: images/val
nc: {len(categories)}
names:
{chr(10).join(names_lines)}
"""
    (YOLO_DIR / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"Written: {len(all_ids)} images to train+val, {total_anns} annotations")
    print(f"YOLO dir: {YOLO_DIR.resolve()}")
    print(f"Dataset YAML: {YOLO_DIR.resolve() / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
