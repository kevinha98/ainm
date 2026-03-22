"""Centralized configuration for the object detection competition."""
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
COCO_DIR = DATA_DIR / "coco"
PRODUCT_IMG_DIR = DATA_DIR / "products"
YOLO_DIR = DATA_DIR / "yolo"          # Converted YOLO-format dataset
RUNS_DIR = PROJECT_ROOT / "runs"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

COCO_TRAIN_DIR = COCO_DIR / "train"
ANNOTATIONS_FILE = COCO_TRAIN_DIR / "annotations.json"
IMAGES_DIR = COCO_TRAIN_DIR / "images"

# ── Dataset ─────────────────────────────────────────────────────────────
NUM_CATEGORIES = 356          # 0-355 (355 products + unknown_product)
UNKNOWN_CATEGORY_ID = 355
VAL_SPLIT = 0.15              # 15% validation split
RANDOM_SEED = 42

# ── Training Hyperparameters ────────────────────────────────────────────
MODEL_BASE = "yolov8m.pt"     # Medium variant — good balance for L4 GPU
IMGSZ = 640                   # Training image size (increase to 1280 in Phase 4)
EPOCHS = 80
BATCH_SIZE = 16               # Adjust for available GPU memory
PATIENCE = 15                 # Early stopping patience

# ── Inference ───────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.25         # Minimum confidence for predictions
IOU_THRESHOLD = 0.45          # NMS IoU threshold
MAX_DET = 300                 # Max detections per image

# ── Submission Limits ───────────────────────────────────────────────────
MAX_ZIP_SIZE_MB = 420
MAX_FILES = 1000
MAX_PYTHON_FILES = 10
MAX_WEIGHT_FILES = 3
MAX_WEIGHT_SIZE_MB = 420
ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".cfg",
                      ".pt", ".pth", ".onnx", ".safetensors", ".npy"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
BANNED_IMPORTS = {"os", "subprocess", "socket", "ctypes", "builtins"}
BANNED_CALLS = {"eval(", "exec(", "compile(", "__import__("}

# ── Sandbox Environment (reference) ────────────────────────────────────
SANDBOX = {
    "python": "3.11",
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "ultralytics": "8.1.0",
    "onnxruntime-gpu": "1.20.0",
    "cuda": "12.4",
    "gpu": "NVIDIA L4 (24 GB VRAM)",
    "timeout_seconds": 300,
    "memory_gb": 8,
}
