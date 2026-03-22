"""Run pretrained YOLOv8 detection on sample shelf images and generate an interactive HTML viewer.

Usage:
    python visualize_detections.py
    python visualize_detections.py --num-images 5  
    python visualize_detections.py --weights best.pt  # Use fine-tuned model
"""
import argparse
import base64
import json
from pathlib import Path

def encode_image_base64(path: Path) -> str:
    """Read image and return base64-encoded data URI."""
    data = path.read_bytes()
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"

def run_detection_pretrained(image_paths: list, weights: str = "yolov8n.pt"):
    """Run YOLOv8 detection and return results per image."""
    import torch
    from ultralytics import YOLO

    # Fix for PyTorch 2.6+ which defaults weights_only=True.
    # ultralytics 8.1.0 .pt files need pickle (trusted source: ultralytics hub).
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {weights}")
    model = YOLO(weights)
    
    all_results = []
    for i, img_path in enumerate(image_paths):
        print(f"  Detecting [{i+1}/{len(image_paths)}]: {img_path.name}")
        results = model(str(img_path), device=device, verbose=False, conf=0.25, iou=0.45, max_det=300)
        
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            img_h, img_w = r.orig_shape
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
                cls_id = int(r.boxes.cls[j].item())
                conf = float(r.boxes.conf[j].item())
                # Get class name from model
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # x,y,w,h
                    "xyxy": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                })
        
        all_results.append({
            "filename": img_path.name,
            "image_id": int(img_path.stem.split("_")[-1]),
            "detections": detections,
            "num_detections": len(detections),
        })
    
    return all_results

def load_ground_truth(annotations_path: Path) -> dict:
    """Load COCO ground truth annotations indexed by image_id."""
    with open(annotations_path, "r") as f:
        coco = json.load(f)
    
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}
    
    gt_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        x, y, w, h = ann["bbox"]
        gt_by_image[img_id].append({
            "bbox": [x, y, w, h],
            "xyxy": [x, y, x + w, y + h],
            "class_id": ann["category_id"],
            "class_name": categories.get(ann["category_id"], "unknown"),
            "product_code": ann.get("product_code", ""),
        })
    
    return gt_by_image, images, categories

def generate_html(image_paths: list, det_results: list, gt_by_image: dict, images_meta: dict) -> str:
    """Generate interactive HTML with detection overlays."""
    
    # Build image data with base64 encoding
    images_data = []
    for img_path, det in zip(image_paths, det_results):
        img_b64 = encode_image_base64(img_path)
        meta = None
        for m in images_meta.values():
            if m["file_name"] == img_path.name:
                meta = m
                break
        
        img_w = meta["width"] if meta else 2000
        img_h = meta["height"] if meta else 1500
        image_id = det["image_id"]
        gt = gt_by_image.get(image_id, [])
        
        images_data.append({
            "src": img_b64,
            "filename": img_path.name,
            "image_id": image_id,
            "width": img_w,
            "height": img_h,
            "detections": det["detections"],
            "ground_truth": gt,
            "num_det": det["num_detections"],
            "num_gt": len(gt),
        })
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NorgesGruppen Object Detection Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }}
.header {{ background: #1e293b; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; }}
.header h1 {{ font-size: 20px; font-weight: 600; }}
.header .stats {{ font-size: 14px; color: #94a3b8; }}
.controls {{ background: #1e293b; padding: 12px 24px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid #334155; }}
.controls label {{ font-size: 13px; color: #94a3b8; }}
.controls select, .controls input {{ background: #334155; color: #e2e8f0; border: 1px solid #475569; border-radius: 6px; padding: 6px 10px; font-size: 13px; }}
.controls button {{ background: #3b82f6; color: white; border: none; border-radius: 6px; padding: 6px 14px; font-size: 13px; cursor: pointer; }}
.controls button:hover {{ background: #2563eb; }}
.controls button.active {{ background: #10b981; }}
.toggle-group {{ display: flex; gap: 4px; }}
.toggle-group button {{ background: #475569; }}
.toggle-group button.active {{ background: #3b82f6; }}
.viewer {{ position: relative; margin: 16px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
.canvas-wrapper {{ position: relative; overflow: auto; max-height: calc(100vh - 180px); }}
canvas {{ display: block; }}
.sidebar {{ position: fixed; right: 0; top: 0; width: 340px; height: 100vh; background: #1e293b; border-left: 1px solid #334155; overflow-y: auto; padding: 16px; z-index: 100; display: none; }}
.sidebar.open {{ display: block; }}
.sidebar h3 {{ font-size: 15px; margin-bottom: 12px; }}
.det-item {{ background: #334155; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px; font-size: 12px; cursor: pointer; }}
.det-item:hover {{ background: #475569; }}
.det-item .cls {{ font-weight: 600; color: #60a5fa; }}
.det-item .conf {{ color: #fbbf24; }}
.det-item .gt {{ color: #34d399; }}
.nav-arrows {{ display: flex; gap: 8px; }}
.nav-arrows button {{ font-size: 18px; width: 36px; height: 36px; }}
.badge {{ display: inline-block; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600; }}
.badge-det {{ background: #3b82f620; color: #60a5fa; border: 1px solid #3b82f640; }}
.badge-gt {{ background: #10b98120; color: #34d399; border: 1px solid #10b98140; }}
.conf-slider {{ width: 120px; }}
</style>
</head>
<body>

<div class="header">
    <h1>🔍 NorgesGruppen Shelf Detection</h1>
    <div class="stats" id="stats"></div>
</div>

<div class="controls">
    <div class="nav-arrows">
        <button onclick="prevImage()">◀</button>
        <button onclick="nextImage()">▶</button>
    </div>
    <select id="imageSelect" onchange="loadImage(this.value)"></select>
    
    <div class="toggle-group">
        <button id="btnDet" class="active" onclick="toggleDetections()">Detections</button>
        <button id="btnGT" class="active" onclick="toggleGT()">Ground Truth</button>
    </div>
    
    <label>Confidence: <span id="confVal">0.25</span></label>
    <input type="range" class="conf-slider" id="confSlider" min="0" max="100" value="25" oninput="updateConf(this.value)">
    
    <button onclick="toggleSidebar()">📋 Details</button>
</div>

<div class="viewer">
    <div class="canvas-wrapper">
        <canvas id="canvas"></canvas>
    </div>
</div>

<div class="sidebar" id="sidebar">
    <h3>Detection Details</h3>
    <div id="detList"></div>
</div>

<script>
const IMAGES = {json.dumps(images_data, separators=(',', ':'))};

let currentIdx = 0;
let showDetections = true;
let showGT = true;
let confThreshold = 0.25;
let currentImage = null;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Populate image selector
const sel = document.getElementById('imageSelect');
IMAGES.forEach((img, i) => {{
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `${{img.filename}} (${{img.num_det}} det / ${{img.num_gt}} gt)`;
    sel.appendChild(opt);
}});

function loadImage(idx) {{
    currentIdx = parseInt(idx);
    sel.value = currentIdx;
    const data = IMAGES[currentIdx];
    
    const img = new Image();
    img.onload = () => {{
        currentImage = img;
        // Scale to fit
        const maxW = window.innerWidth - 32;
        const maxH = window.innerHeight - 180;
        let scale = Math.min(maxW / data.width, maxH / data.height, 1);
        canvas.width = data.width * scale;
        canvas.height = data.height * scale;
        ctx.scale(scale, scale);
        draw();
    }};
    img.src = data.src;
    
    updateStats();
    updateDetailsList();
}}

function draw() {{
    if (!currentImage) return;
    const data = IMAGES[currentIdx];
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const scale = canvas.width / data.width;
    ctx.scale(scale, scale);
    
    ctx.drawImage(currentImage, 0, 0, data.width, data.height);
    
    // Draw ground truth (green)
    if (showGT) {{
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 2 / scale;
        ctx.setLineDash([6 / scale, 4 / scale]);
        data.ground_truth.forEach(gt => {{
            const [x, y, w, h] = gt.bbox;
            ctx.strokeRect(x, y, w, h);
        }});
        ctx.setLineDash([]);
    }}
    
    // Draw detections (blue/red based on confidence)
    if (showDetections) {{
        data.detections.forEach(det => {{
            if (det.confidence < confThreshold) return;
            const [x1, y1, x2, y2] = det.xyxy;
            const w = x2 - x1, h = y2 - y1;
            
            // Color by confidence
            const hue = det.confidence > 0.5 ? 210 : (det.confidence > 0.3 ? 45 : 0);
            const sat = 80, light = 55;
            ctx.strokeStyle = `hsl(${{hue}}, ${{sat}}%, ${{light}}%)`;
            ctx.lineWidth = 2.5 / scale;
            ctx.strokeRect(x1, y1, w, h);
            
            // Label
            const label = `${{det.class_name.substring(0, 20)}} ${{(det.confidence * 100).toFixed(0)}}%`;
            ctx.font = `${{12 / scale}}px sans-serif`;
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = `hsla(${{hue}}, ${{sat}}%, 20%, 0.85)`;
            ctx.fillRect(x1, y1 - 16 / scale, tw + 6 / scale, 16 / scale);
            ctx.fillStyle = '#fff';
            ctx.fillText(label, x1 + 3 / scale, y1 - 4 / scale);
        }});
    }}
}}

function updateStats() {{
    const data = IMAGES[currentIdx];
    const filteredDet = data.detections.filter(d => d.confidence >= confThreshold);
    document.getElementById('stats').textContent = 
        `Image ${{currentIdx + 1}}/${{IMAGES.length}} | ${{filteredDet.length}} detections (≥${{(confThreshold * 100).toFixed(0)}}%) | ${{data.num_gt}} ground truth`;
}}

function updateDetailsList() {{
    const data = IMAGES[currentIdx];
    const list = document.getElementById('detList');
    const filtered = data.detections
        .filter(d => d.confidence >= confThreshold)
        .sort((a, b) => b.confidence - a.confidence);
    
    let html = `<p style="margin-bottom:8px;font-size:12px;color:#94a3b8">
        <span class="badge badge-det">${{filtered.length}} detections</span>
        <span class="badge badge-gt">${{data.num_gt}} ground truth</span>
    </p>`;
    
    filtered.forEach(det => {{
        html += `<div class="det-item">
            <span class="cls">${{det.class_name}}</span>
            <span class="conf">${{(det.confidence * 100).toFixed(1)}}%</span>
            <br><span style="color:#94a3b8">bbox: [${{det.bbox.map(v => v.toFixed(0)).join(', ')}}]</span>
        </div>`;
    }});
    
    if (data.ground_truth.length > 0) {{
        html += `<h3 style="margin-top:16px;margin-bottom:8px">Ground Truth</h3>`;
        data.ground_truth.slice(0, 50).forEach(gt => {{
            html += `<div class="det-item">
                <span class="gt">${{gt.class_name}}</span>
                <br><span style="color:#94a3b8">bbox: [${{gt.bbox.map(v => v.toFixed(0)).join(', ')}}]</span>
            </div>`;
        }});
        if (data.ground_truth.length > 50) {{
            html += `<p style="color:#94a3b8;font-size:12px;margin-top:8px">...and ${{data.ground_truth.length - 50}} more</p>`;
        }}
    }}
    list.innerHTML = html;
}}

function prevImage() {{ loadImage((currentIdx - 1 + IMAGES.length) % IMAGES.length); }}
function nextImage() {{ loadImage((currentIdx + 1) % IMAGES.length); }}

function toggleDetections() {{
    showDetections = !showDetections;
    document.getElementById('btnDet').classList.toggle('active');
    draw();
}}

function toggleGT() {{
    showGT = !showGT;
    document.getElementById('btnGT').classList.toggle('active');
    draw();
}}

function updateConf(val) {{
    confThreshold = val / 100;
    document.getElementById('confVal').textContent = confThreshold.toFixed(2);
    draw();
    updateStats();
    updateDetailsList();
}}

function toggleSidebar() {{
    document.getElementById('sidebar').classList.toggle('open');
}}

// Keyboard navigation
document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowLeft') prevImage();
    if (e.key === 'ArrowRight') nextImage();
    if (e.key === 'd') toggleDetections();
    if (e.key === 'g') toggleGT();
}});

// Load first image
loadImage(0);
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to process")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Model weights")
    parser.add_argument("--output", type=str, default="detection_viewer.html", help="Output HTML file")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    images_dir = project_root / "data" / "coco" / "train" / "images"
    annotations_path = project_root / "data" / "coco" / "train" / "annotations.json"

    if not images_dir.exists():
        print(f"ERROR: Images not found at {images_dir}")
        return

    # Select images
    all_images = sorted(images_dir.glob("*.jpg"))
    selected = all_images[:args.num_images]
    print(f"Processing {len(selected)} images with {args.weights}...")

    # Run detection
    det_results = run_detection_pretrained(selected, args.weights)

    # Load ground truth
    gt_by_image, images_meta, categories = load_ground_truth(annotations_path)
    print(f"Ground truth: {len(gt_by_image)} images, {len(categories)} categories")

    # Generate HTML
    print("Generating HTML viewer...")
    html = generate_html(selected, det_results, gt_by_image, images_meta)
    
    output_path = project_root / args.output
    output_path.write_text(html, encoding="utf-8")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Viewer saved: {output_path} ({size_mb:.1f} MB)")
    print(f"Open in browser to view detections!")

if __name__ == "__main__":
    main()
