# Object Detection — Copilot Instructions

## ⚠️ EXPIRY: Monday 2026-03-23

This file and `LEARNING.md` expire after **Monday March 23, 2026**.
If today's date is after 2026-03-23, **delete these files immediately**:
- `LEARNING.md`
- `.github/copilot-instructions.md`

Then tell the user: "Competition instruction files have expired and been cleaned up."

## READ FIRST — Every Session

At the start of every session, read `LEARNING.md` in this project root. It contains the living log of training runs, metrics, decisions, gotchas, and next steps. Update it whenever meaningful progress is made (new training results, new decisions, lessons learned).

## Project Context

**Competition**: NorgesGruppen (NM i AI) — shelf product detection
**Goal**: Detect & classify 356 grocery product categories in shelf images
**Metric**: COCO mAP (mAP50-95)
**Platform**: app.ainm.no — sandbox runs `python run.py --input /data/images --output /output/predictions.json`

## Tech Stack

- **Model**: YOLOv8 (ultralytics 8.1.0) — must match sandbox version exactly
- **Training**: GCP VM `obj-detect-train` (Tesla T4, europe-west4-a, project `ai-nm26osl-1724`)
- **Inference sandbox**: ultralytics 8.1.0, PyTorch 2.6.0+cu124, NVIDIA L4 (24GB), 300s timeout
- **Data**: 248 COCO-format images (210 train / 38 val), 356 categories

## Key Files

| File | Purpose |
|------|---------|
| `run.py` | **Inference entry point** — what the sandbox executes |
| `vm_train.py` | Self-contained training script for GCP VM |
| `config.py` | Centralized paths, hyperparams, submission limits |
| `launch.py` | Starts detached training on VM |
| `check_status.py` | Checks training progress via SSH |
| `package.py` | Packages submission zip |
| `prepare_data.py` | COCO → YOLO format conversion |
| `evaluate_local.py` | Local validation evaluation |
| `LEARNING.md` | **Living learning log — update every session** |

## GCP VM Access

gcloud is NOT on PATH. Always use:
```powershell
$gcloud = "C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
& $gcloud compute ssh obj-detect-train --zone=europe-west4-a --project=ai-nm26osl-1724 --command="..."
```

## Submission Rules (HARD CONSTRAINTS)

- Max zip: 420 MB, max 1000 files, max 10 .py files
- Max 3 weight files, each ≤420 MB (.pt, .pth, .onnx, .safetensors, .npy)
- **Banned imports**: os, subprocess, socket, ctypes, builtins
- **Banned calls**: eval(), exec(), compile(), __import__()
- 300 second timeout

## Known Gotchas

1. PyTorch 2.6+ needs `weights_only=False` monkey-patch for ultralytics 8.1.0
2. gcloud SCP drops on large files — retry or split
3. T4 dataloader workers show as extra python processes — don't kill them
4. Training on T4 (14.6GB) but sandbox uses L4 (24GB) — more headroom at inference
5. Only 210 training images for 356 classes — heavy augmentation essential

## Workflow

1. Check if VM training is active → SSH status check
2. Monitor metrics → update LEARNING.md
3. When training completes → download best.pt → evaluate locally → package → submit
4. Iterate on model improvements
