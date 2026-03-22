"""Convert COCO-format annotations to YOLO-format for ultralytics training.

Reads:  data/coco/annotations.json + data/coco/images/
Writes: data/yolo/images/{train,val}/, data/yolo/labels/{train,val}/, data/yolo/dataset.yaml

COCO bbox: [x, y, width, height] in pixels (top-left origin)
YOLO bbox: [class_id, x_center, y_center, width, height] normalized to [0,1]

Usage:
    python prepare_data.py                        # Default 85/15 split
    python prepare_data.py --val-split 0.2        # 80/20 split
    python prepare_data.py --copy-images           # Copy images (default: symlink)
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from config import (ANNOTATIONS_FILE, COCO_DIR, IMAGES_DIR, NUM_CATEGORIES,
                    RANDOM_SEED, VAL_SPLIT, YOLO_DIR)


def load_coco_annotations(annotations_path: Path) -> dict:
    """Load and index COCO annotations file."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Index images by id
    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    print(f"Loaded: {len(images)} images, {len(coco['annotations'])} annotations, {len(categories)} categories")
    return images, anns_by_image, categories


def coco_to_yolo_bbox(bbox: list, img_w: int, img_h: int) -> tuple:
    """Convert COCO [x, y, w, h] pixels → YOLO [x_center, y_center, w, h] normalized."""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    return x_center, y_center, w_norm, h_norm


def split_dataset(image_ids: list, val_ratio: float, seed: int) -> tuple:
    """Random train/val split."""
    ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    split_idx = int(len(ids) * (1 - val_ratio))
    return ids[:split_idx], ids[split_idx:]


def write_yolo_labels(
    image_ids: list,
    images: dict,
    anns_by_image: dict,
    split_name: str,
    yolo_dir: Path,
    source_images_dir: Path,
    copy_images: bool,
) -> dict:
    """Write YOLO-format label files and link/copy images for a split."""
    labels_dir = yolo_dir / "labels" / split_name
    images_out_dir = yolo_dir / "images" / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "annotations": 0, "skipped_empty": 0}

    for img_id in image_ids:
        img_info = images[img_id]
        img_w, img_h = img_info["width"], img_info["height"]
        filename = img_info["file_name"]
        stem = Path(filename).stem

        # Source image
        src_img = source_images_dir / filename
        if not src_img.exists():
            # Try nested paths
            for candidate in source_images_dir.rglob(filename):
                src_img = candidate
                break

        dst_img = images_out_dir / filename
        if not dst_img.exists():
            if copy_images:
                shutil.copy2(src_img, dst_img)
            else:
                # Use relative path for symlink portability
                try:
                    dst_img.symlink_to(src_img)
                except (PermissionError, NotImplementedError):
                    shutil.copy2(src_img, dst_img)

        # Write labels
        anns = anns_by_image.get(img_id, [])
        if not anns:
            stats["skipped_empty"] += 1
            # Still create empty label file (YOLO convention for background images)
            (labels_dir / f"{stem}.txt").touch()
            stats["images"] += 1
            continue

        lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            xc, yc, wn, hn = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
            # Skip degenerate boxes
            if wn < 1e-6 or hn < 1e-6:
                continue
            lines.append(f"{cat_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            stats["annotations"] += 1

        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["images"] += 1

    return stats


def write_dataset_yaml(yolo_dir: Path, categories: dict) -> Path:
    """Write dataset.yaml for ultralytics training."""
    yaml_path = yolo_dir / "dataset.yaml"

    # Build names dict — use double quotes (names contain apostrophes like KELLOGG'S)
    names_lines = []
    for cat_id in sorted(categories.keys()):
        name = categories[cat_id].replace('"', '\\"')
        names_lines.append(f'  {cat_id}: "{name}"')

    content = f"""# NorgesGruppen Object Detection Dataset
# Auto-generated by prepare_data.py

path: {yolo_dir.as_posix()}
train: images/train
val: images/val

nc: {NUM_CATEGORIES}

names:
{chr(10).join(names_lines)}
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO format")
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT,
                        help=f"Validation split ratio (default: {VAL_SPLIT})")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of symlinking (uses more disk space)")
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_FILE,
                        help="Path to COCO annotations.json")
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR,
                        help="Path to COCO images directory")
    parser.add_argument("--output-dir", type=Path, default=YOLO_DIR,
                        help="Output directory for YOLO dataset")
    args = parser.parse_args()

    if not args.annotations.exists():
        print(f"ERROR: Annotations file not found: {args.annotations}")
        print("Download NM_NGD_coco_dataset.zip and extract to data/coco/")
        return

    print("Loading COCO annotations...")
    images, anns_by_image, categories = load_coco_annotations(args.annotations)

    print(f"Splitting dataset (val_ratio={args.val_split}, seed={RANDOM_SEED})...")
    train_ids, val_ids = split_dataset(list(images.keys()), args.val_split, RANDOM_SEED)
    print(f"  Train: {len(train_ids)} images, Val: {len(val_ids)} images")

    # Clean output dir
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    print("Writing train split...")
    train_stats = write_yolo_labels(
        train_ids, images, anns_by_image, "train",
        args.output_dir, args.images_dir, args.copy_images
    )
    print(f"  {train_stats['images']} images, {train_stats['annotations']} annotations")

    print("Writing val split...")
    val_stats = write_yolo_labels(
        val_ids, images, anns_by_image, "val",
        args.output_dir, args.images_dir, args.copy_images
    )
    print(f"  {val_stats['images']} images, {val_stats['annotations']} annotations")

    print("Writing dataset.yaml...")
    yaml_path = write_dataset_yaml(args.output_dir, categories)
    print(f"  {yaml_path}")

    print("\nDone! To train:")
    print(f"  yolo detect train data={yaml_path} model=yolov8m.pt imgsz=640 epochs=80")


if __name__ == "__main__":
    main()
