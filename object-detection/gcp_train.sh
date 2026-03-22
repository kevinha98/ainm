#!/bin/bash
# ============================================================================
# GCP VM Training Setup for NorgesGruppen Object Detection
# ============================================================================
# 
# Run this from Google Cloud Shell (https://shell.cloud.google.com)
# or any machine with gcloud CLI authenticated.
#
# This script:
#   1. Creates a GPU VM (T4) in your GCP project
#   2. Uploads training code
#   3. Downloads training data (you provide the URLs)
#   4. Runs COCO→YOLO conversion + training
#   5. Downloads best.pt weights when done
#
# Usage:
#   chmod +x gcp_train.sh
#   ./gcp_train.sh create     # Create VM
#   ./gcp_train.sh upload     # Upload code + data to VM
#   ./gcp_train.sh train      # Run training on VM
#   ./gcp_train.sh download   # Download trained weights
#   ./gcp_train.sh destroy    # Delete VM when done
#   ./gcp_train.sh all        # Do everything in sequence
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_ID="ai-nm26osl-1724"
ZONE="europe-west4-a"              # Has T4 GPUs, close to Norway
VM_NAME="obj-detect-train"
MACHINE_TYPE="n1-standard-8"       # 8 vCPU, 30 GB RAM
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="pytorch-latest-gpu"  # Deep Learning VM with PyTorch + CUDA
IMAGE_PROJECT="deeplearning-platform-release"

# Local paths (adjust if needed)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_CODE_DIR="$SCRIPT_DIR"
LOCAL_DATA_DIR="$SCRIPT_DIR/data"

# Remote paths
REMOTE_DIR="/home/$USER/object-detection"

# ── Functions ──────────────────────────────────────────────────────────────

configure_project() {
    echo "Configuring GCP project: $PROJECT_ID"
    gcloud config set project "$PROJECT_ID"
    gcloud config set compute/zone "$ZONE"
}

create_vm() {
    echo "Creating GPU VM: $VM_NAME"
    echo "  Machine: $MACHINE_TYPE + $GPU_COUNT x $GPU_TYPE"
    echo "  Zone: $ZONE"
    echo "  Disk: $BOOT_DISK_SIZE"
    
    gcloud compute instances create "$VM_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --maintenance-policy=TERMINATE \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --metadata="install-nvidia-driver=True" \
        --scopes="default,storage-rw"
    
    echo ""
    echo "VM created. Waiting for startup (GPU drivers install ~2-3 min)..."
    sleep 30
    
    # Wait until SSH is ready
    for i in $(seq 1 20); do
        if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="echo ready" 2>/dev/null; then
            echo "VM is ready!"
            return 0
        fi
        echo "  Waiting for SSH... ($i/20)"
        sleep 15
    done
    echo "WARNING: Could not verify SSH — VM may still be booting"
}

setup_vm() {
    echo "Installing training dependencies on VM..."
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command='
        # Install exact sandbox versions
        pip install ultralytics==8.1.0 pycocotools pyyaml
        
        # Verify GPU
        python3 -c "import torch; print(f\"GPU: {torch.cuda.get_device_name(0)}\"); print(f\"CUDA: {torch.version.cuda}\")"
        python3 -c "from ultralytics import YOLO; print(f\"ultralytics: {YOLO.__module__}\")"
        
        echo "Setup complete!"
    '
}

upload_code() {
    echo "Uploading training code to VM..."
    
    # Create remote directory
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="mkdir -p $REMOTE_DIR"
    
    # Upload Python files
    for f in run.py train.py prepare_data.py evaluate_local.py config.py package.py; do
        if [ -f "$LOCAL_CODE_DIR/$f" ]; then
            gcloud compute scp "$LOCAL_CODE_DIR/$f" "$VM_NAME:$REMOTE_DIR/$f" --zone="$ZONE"
            echo "  Uploaded $f"
        fi
    done
    
    echo "Code uploaded."
}

upload_data() {
    echo "Uploading training data to VM..."
    echo "Looking for data in: $LOCAL_DATA_DIR"
    
    COCO_ZIP="$LOCAL_DATA_DIR/NM_NGD_coco_dataset.zip"
    PRODUCT_ZIP="$LOCAL_DATA_DIR/NM_NGD_product_images.zip"
    
    if [ ! -f "$COCO_ZIP" ]; then
        echo "ERROR: $COCO_ZIP not found"
        echo "Download it from the competition website first."
        echo "Place it in: $LOCAL_DATA_DIR/"
        return 1
    fi
    
    # Upload COCO dataset
    echo "Uploading COCO dataset (~864 MB, this may take a while)..."
    gcloud compute scp "$COCO_ZIP" "$VM_NAME:$REMOTE_DIR/NM_NGD_coco_dataset.zip" --zone="$ZONE"
    
    # Upload product images if available
    if [ -f "$PRODUCT_ZIP" ]; then
        echo "Uploading product images (~60 MB)..."
        gcloud compute scp "$PRODUCT_ZIP" "$VM_NAME:$REMOTE_DIR/NM_NGD_product_images.zip" --zone="$ZONE"
    fi
    
    # Extract on VM
    echo "Extracting data on VM..."
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        cd $REMOTE_DIR
        mkdir -p data/coco
        unzip -o NM_NGD_coco_dataset.zip -d data/coco/
        echo 'COCO dataset extracted'
        ls -la data/coco/
        
        if [ -f NM_NGD_product_images.zip ]; then
            mkdir -p data/products
            unzip -o NM_NGD_product_images.zip -d data/products/
            echo 'Product images extracted'
        fi
        
        # Clean up zips to free disk space
        rm -f NM_NGD_coco_dataset.zip NM_NGD_product_images.zip
    "
    
    echo "Data uploaded and extracted."
}

prepare_data() {
    echo "Converting COCO → YOLO format on VM..."
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        cd $REMOTE_DIR
        python3 prepare_data.py --copy-images
    "
}

run_training() {
    echo "Starting training on VM (this will take a while)..."
    echo "You can disconnect — training runs in tmux session."
    
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        cd $REMOTE_DIR
        
        # Run in tmux so it survives SSH disconnection
        tmux new-session -d -s train '
            cd $REMOTE_DIR
            python3 train.py --data data/yolo/dataset.yaml --epochs 80 --batch 16 2>&1 | tee training.log
            echo \"Training complete! Check training.log for results.\"
        '
        
        echo 'Training started in tmux session.'
        echo 'To monitor: gcloud compute ssh $VM_NAME -- tmux attach -t train'
        echo 'To check GPU: gcloud compute ssh $VM_NAME -- nvidia-smi'
    "
}

check_training() {
    echo "Checking training status..."
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        cd $REMOTE_DIR
        
        # Check if tmux session exists
        if tmux has-session -t train 2>/dev/null; then
            echo 'Training is RUNNING'
            echo ''
            echo 'Last 20 lines of training.log:'
            tail -20 training.log 2>/dev/null || echo '(no log yet)'
        else
            echo 'Training session ended'
            echo ''
            echo 'Last 20 lines of training.log:'
            tail -20 training.log 2>/dev/null
        fi
        
        echo ''
        nvidia-smi 2>/dev/null || echo '(nvidia-smi not available)'
        
        echo ''
        echo 'Best weights:'
        ls -la runs/detect/train/weights/best.pt 2>/dev/null || echo '(not yet available)'
    "
}

download_weights() {
    echo "Downloading trained weights from VM..."
    
    # Download best.pt
    gcloud compute scp "$VM_NAME:$REMOTE_DIR/runs/detect/train/weights/best.pt" \
        "$LOCAL_CODE_DIR/best.pt" --zone="$ZONE"
    
    if [ -f "$LOCAL_CODE_DIR/best.pt" ]; then
        SIZE=$(du -h "$LOCAL_CODE_DIR/best.pt" | cut -f1)
        echo "Downloaded best.pt ($SIZE)"
        echo ""
        echo "Next steps:"
        echo "  python evaluate_local.py --weights best.pt"
        echo "  python package.py --weights best.pt"
    else
        echo "ERROR: Failed to download best.pt"
    fi
}

destroy_vm() {
    echo "Deleting VM: $VM_NAME"
    read -p "Are you sure? (y/N) " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
        echo "VM deleted."
    else
        echo "Cancelled."
    fi
}

ssh_vm() {
    echo "SSHing into VM..."
    gcloud compute ssh "$VM_NAME" --zone="$ZONE"
}

# ── Main ───────────────────────────────────────────────────────────────────

case "${1:-help}" in
    create)
        configure_project
        create_vm
        setup_vm
        ;;
    upload)
        upload_code
        upload_data
        prepare_data
        ;;
    train)
        run_training
        ;;
    status)
        check_training
        ;;
    download)
        download_weights
        ;;
    destroy)
        destroy_vm
        ;;
    ssh)
        ssh_vm
        ;;
    all)
        configure_project
        create_vm
        setup_vm
        upload_code
        upload_data
        prepare_data
        run_training
        echo ""
        echo "Training is running! Monitor with:"
        echo "  ./gcp_train.sh status"
        echo "When done, download weights with:"
        echo "  ./gcp_train.sh download"
        echo "Then delete VM:"
        echo "  ./gcp_train.sh destroy"
        ;;
    help|*)
        echo "Usage: $0 {create|upload|train|status|download|destroy|ssh|all}"
        echo ""
        echo "Commands:"
        echo "  create   - Create GPU VM and install dependencies"
        echo "  upload   - Upload code and training data to VM"
        echo "  train    - Start training in background (tmux)"
        echo "  status   - Check training progress"
        echo "  download - Download best.pt weights"
        echo "  destroy  - Delete VM (saves money!)"
        echo "  ssh      - SSH into the VM"
        echo "  all      - Do everything (create → upload → train)"
        ;;
esac
