"""GCP VM management for training — no gcloud CLI needed.

Uses Google Cloud REST API directly via OAuth2 access token.
Designed for the competition GCP lab account.

Usage:
    python gcp_train.py create     # Create T4 GPU VM
    python gcp_train.py setup      # Install deps + upload code
    python gcp_train.py status     # Check VM / training status
    python gcp_train.py destroy    # Delete VM
    python gcp_train.py ssh-cmd    # Print SSH command to connect
"""
import argparse
import json
import time
from pathlib import Path
from urllib import request, error, parse

# ── GCP Configuration ──────────────────────────────────────────────────────
PROJECT_ID = "ai-nm26osl-1724"
ZONE = "europe-west4-a"
VM_NAME = "obj-detect-train"
MACHINE_TYPE = f"zones/{ZONE}/machineTypes/n1-standard-8"
GPU_TYPE = "nvidia-tesla-t4"
BOOT_DISK_SIZE_GB = 100

# Deep Learning VM image (PyTorch + CUDA preinstalled)
SOURCE_IMAGE = "projects/deeplearning-platform-release/global/images/family/pytorch-latest-gpu"

COMPUTE_API = "https://compute.googleapis.com/compute/v1"
PROJECT_URL = f"{COMPUTE_API}/projects/{PROJECT_ID}"


def get_access_token() -> str:
    """Get OAuth2 access token. Tries gcloud first, then prompts."""
    # Try gcloud token
    try:
        import subprocess
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Manual token — user gets this from Cloud Shell or OAuth playground
    print("=" * 60)
    print("ACCESS TOKEN NEEDED")
    print("=" * 60)
    print()
    print("Get a token from one of these methods:")
    print()
    print("1. Google Cloud Shell (easiest):")
    print("   Open https://shell.cloud.google.com")
    print("   Run: gcloud auth print-access-token")
    print()
    print("2. OAuth2 Playground:")
    print("   https://developers.google.com/oauthplayground/")
    print("   Scope: https://www.googleapis.com/auth/compute")
    print()
    token = input("Paste access token: ").strip()
    return token


def api_request(method: str, url: str, token: str, body: dict = None) -> dict:
    """Make an authenticated GCP API request."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body else None
    req = request.Request(url, data=data, headers=headers, method=method)

    try:
        with request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except error.HTTPError as e:
        err_body = e.read().decode()
        try:
            err = json.loads(err_body)
            msg = err.get("error", {}).get("message", err_body)
        except json.JSONDecodeError:
            msg = err_body
        print(f"API Error ({e.code}): {msg}")
        raise


def create_vm(token: str):
    """Create a GPU VM instance."""
    print(f"Creating VM: {VM_NAME} (n1-standard-8 + T4 GPU)")

    body = {
        "name": VM_NAME,
        "machineType": MACHINE_TYPE,
        "guestAccelerators": [{
            "acceleratorType": f"zones/{ZONE}/acceleratorTypes/{GPU_TYPE}",
            "acceleratorCount": 1,
        }],
        "scheduling": {
            "onHostMaintenance": "TERMINATE",
            "automaticRestart": True,
        },
        "disks": [{
            "boot": True,
            "autoDelete": True,
            "initializeParams": {
                "sourceImage": SOURCE_IMAGE,
                "diskSizeGb": str(BOOT_DISK_SIZE_GB),
                "diskType": f"zones/{ZONE}/diskTypes/pd-ssd",
            },
        }],
        "networkInterfaces": [{
            "network": f"projects/{PROJECT_ID}/global/networks/default",
            "accessConfigs": [{
                "type": "ONE_TO_ONE_NAT",
                "name": "External NAT",
            }],
        }],
        "metadata": {
            "items": [
                {"key": "install-nvidia-driver", "value": "True"},
            ],
        },
    }

    url = f"{PROJECT_URL}/zones/{ZONE}/instances"
    result = api_request("POST", url, token, body)
    op_name = result.get("name", "")
    print(f"Operation started: {op_name}")

    # Wait for completion
    print("Waiting for VM to be created...")
    for i in range(60):
        op_url = f"{PROJECT_URL}/zones/{ZONE}/operations/{op_name}"
        op = api_request("GET", op_url, token)
        status = op.get("status", "UNKNOWN")
        if status == "DONE":
            if "error" in op:
                print(f"ERROR: {op['error']}")
                return
            print("VM created successfully!")
            break
        time.sleep(5)
    else:
        print("Timeout waiting for VM creation")
        return

    # Get external IP
    vm = get_vm_info(token)
    if vm:
        ip = get_external_ip(vm)
        print(f"\nExternal IP: {ip}")
        print(f"\nSSH: gcloud compute ssh {VM_NAME} --zone={ZONE} --project={PROJECT_ID}")
        print(f"Or:  ssh {ip}")


def get_vm_info(token: str) -> dict:
    """Get VM instance details."""
    url = f"{PROJECT_URL}/zones/{ZONE}/instances/{VM_NAME}"
    try:
        return api_request("GET", url, token)
    except error.HTTPError:
        return None


def get_external_ip(vm: dict) -> str:
    """Extract external IP from VM info."""
    for iface in vm.get("networkInterfaces", []):
        for ac in iface.get("accessConfigs", []):
            if "natIP" in ac:
                return ac["natIP"]
    return "(no external IP)"


def check_status(token: str):
    """Check VM status."""
    vm = get_vm_info(token)
    if not vm:
        print(f"VM '{VM_NAME}' not found")
        return

    status = vm.get("status", "UNKNOWN")
    ip = get_external_ip(vm)
    machine = vm.get("machineType", "").split("/")[-1]
    gpus = vm.get("guestAccelerators", [])
    gpu_info = f"{gpus[0].get('acceleratorCount', 0)}x {gpus[0].get('acceleratorType', '').split('/')[-1]}" if gpus else "none"

    print(f"VM: {VM_NAME}")
    print(f"  Status:  {status}")
    print(f"  IP:      {ip}")
    print(f"  Machine: {machine}")
    print(f"  GPU:     {gpu_info}")
    print(f"  Zone:    {ZONE}")

    if status == "RUNNING":
        print(f"\nConnect via Cloud Shell:")
        print(f"  gcloud compute ssh {VM_NAME} --zone={ZONE} --project={PROJECT_ID}")


def destroy_vm(token: str):
    """Delete the VM."""
    print(f"Deleting VM: {VM_NAME}")
    confirm = input("Are you sure? (y/N) ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    url = f"{PROJECT_URL}/zones/{ZONE}/instances/{VM_NAME}"
    result = api_request("DELETE", url, token)
    print(f"Delete operation started: {result.get('name', '')}")
    print("VM will be deleted shortly.")


def print_ssh_setup():
    """Print the full training setup commands for SSH."""
    print("=" * 70)
    print("TRAINING SETUP — paste these commands after SSH into the VM")
    print("=" * 70)
    print("""
# 1. Install exact sandbox version of ultralytics
pip install ultralytics==8.1.0 pycocotools pyyaml

# 2. Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'CUDA: {torch.version.cuda}')"

# 3. Create project directory
mkdir -p ~/object-detection/data/coco
cd ~/object-detection

# 4. Upload training data (from your local machine, in a separate terminal):
#    gcloud compute scp NM_NGD_coco_dataset.zip VM_NAME:~/object-detection/data/ --zone=europe-west4-a
#    OR download directly if you have a URL

# 5. Extract COCO dataset
cd ~/object-detection/data/coco
unzip ../NM_NGD_coco_dataset.zip
cd ~/object-detection

# 6. Upload Python files (from your local machine):
#    gcloud compute scp run.py train.py prepare_data.py evaluate_local.py config.py VM_NAME:~/object-detection/ --zone=europe-west4-a

# 7. Convert COCO → YOLO format
python3 prepare_data.py --copy-images

# 8. Start training (in tmux so it survives disconnection)
tmux new-session -d -s train 'cd ~/object-detection && python3 train.py --data data/yolo/dataset.yaml --epochs 80 --batch 16 2>&1 | tee training.log'

# 9. Monitor training
tmux attach -t train       # Attach to training session (Ctrl+B D to detach)
tail -f training.log       # Watch log
nvidia-smi                 # Check GPU usage

# 10. When training is done, download weights (from your local machine):
#     gcloud compute scp VM_NAME:~/object-detection/runs/detect/train/weights/best.pt ./best.pt --zone=europe-west4-a
""")


def main():
    parser = argparse.ArgumentParser(description="GCP VM management for training")
    parser.add_argument("command", choices=["create", "status", "destroy", "ssh-cmd"],
                        help="VM management command")
    args = parser.parse_args()

    if args.command == "ssh-cmd":
        print_ssh_setup()
        return

    token = get_access_token()

    if args.command == "create":
        create_vm(token)
    elif args.command == "status":
        check_status(token)
    elif args.command == "destroy":
        destroy_vm(token)


if __name__ == "__main__":
    main()
