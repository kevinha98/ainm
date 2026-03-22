import subprocess, time, re, datetime

GCLOUD = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

while True:
    try:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        r = subprocess.run(
            [GCLOUD, "compute", "ssh", "obj-detect-train", "--zone=europe-west4-a",
             "--command=grep -oP '\\d+/400' ~/object-detection/train_v5x.log | tail -1; grep 'all' ~/object-detection/train_v5x.log | tail -1"],
            capture_output=True, text=True, timeout=60
        )
        out = r.stdout.strip()
        ep = re.search(r'(\d+)/400', out)
        epoch = int(ep.group(1)) if ep else "?"
        pct = f"{epoch/400*100:.1f}%" if isinstance(epoch, int) else "?"
        maps = [l for l in out.split('\n') if 'all' in l]
        last_map = maps[-1].strip() if maps else "no mAP yet"
        print(f"[{ts}]  Epoch {epoch}/400  ({pct})  |  {last_map}", flush=True)
    except Exception as e:
        print(f"[{datetime.datetime.now():%H:%M:%S}]  SSH error: {e}", flush=True)
    time.sleep(1800)  # 30 minutes
