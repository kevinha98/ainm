"""Local sandbox validator for AINM Object Detection submissions.

Simulates the competition sandbox environment WITHOUT Docker:
  1. Security scan (banned imports/calls)
  2. File structure validation
  3. Output format validation (JSON schema)
  4. Actually runs run.py and validates predictions

Usage:
    python validate_submission.py                        # test run.py in current dir
    python validate_submission.py --zip submissions/submission.zip  # test a zip
    python validate_submission.py --images data/coco/train/images --max-images 5
    python validate_submission.py --cpu                   # force CPU mode
"""
import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

# ── Sandbox rules from oppgaveteksten ───────────────────────────────
BANNED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "ctypes", "builtins", "importlib",
    "pickle", "marshal", "shelve", "shutil",
    "yaml",
    "requests", "urllib", "http",
    "multiprocessing", "threading", "signal", "gc",
    "code", "codeop", "pty",
}

BANNED_CALLS = {"eval", "exec", "compile", "__import__"}

MAX_ZIP_SIZE_UNCOMPRESSED = 420 * 1024 * 1024  # 420 MB
MAX_FILES = 1000
MAX_PYTHON_FILES = 10
MAX_WEIGHT_FILES = 3
MAX_WEIGHT_SIZE = 420 * 1024 * 1024  # 420 MB
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".cfg",
                      ".pt", ".pth", ".onnx", ".safetensors", ".npy"}
TIMEOUT_SECONDS = 300
CATEGORY_RANGE = (0, 355)


class ValidationResult:
    def __init__(self):
        self.checks = []

    def ok(self, msg):
        self.checks.append(("PASS", msg))
        print(f"  [PASS] {msg}")

    def fail(self, msg):
        self.checks.append(("FAIL", msg))
        print(f"  [FAIL] {msg}")

    def warn(self, msg):
        self.checks.append(("WARN", msg))
        print(f"  [WARN] {msg}")

    def info(self, msg):
        self.checks.append(("INFO", msg))
        print(f"  [INFO] {msg}")

    @property
    def passed(self):
        return all(status != "FAIL" for status, _ in self.checks)

    @property
    def n_fail(self):
        return sum(1 for s, _ in self.checks if s == "FAIL")


# ── 1. File structure validation ────────────────────────────────────
def validate_structure(submission_dir: Path, result: ValidationResult):
    print("\n=== File Structure ===")

    run_py = submission_dir / "run.py"
    if run_py.exists():
        result.ok("run.py found at root")
    else:
        result.fail("run.py NOT found at submission root")
        return

    all_files = list(submission_dir.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]
    result.info(f"{len(all_files)} files total")

    if len(all_files) > MAX_FILES:
        result.fail(f"Too many files: {len(all_files)} > {MAX_FILES}")
    else:
        result.ok(f"File count OK ({len(all_files)} <= {MAX_FILES})")

    py_files = [f for f in all_files if f.suffix == ".py"]
    if len(py_files) > MAX_PYTHON_FILES:
        result.fail(f"Too many Python files: {len(py_files)} > {MAX_PYTHON_FILES}")
    else:
        result.ok(f"Python files OK ({len(py_files)} <= {MAX_PYTHON_FILES})")

    weight_files = [f for f in all_files if f.suffix.lower() in WEIGHT_EXTENSIONS]
    if len(weight_files) > MAX_WEIGHT_FILES:
        result.fail(f"Too many weight files: {len(weight_files)} > {MAX_WEIGHT_FILES}")
    else:
        result.ok(f"Weight files OK ({len(weight_files)} <= {MAX_WEIGHT_FILES})")

    total_weight_size = sum(f.stat().st_size for f in weight_files)
    weight_mb = total_weight_size / (1024 * 1024)
    if total_weight_size > MAX_WEIGHT_SIZE:
        result.fail(f"Weight files too large: {weight_mb:.1f} MB > 420 MB")
    else:
        result.ok(f"Weight size OK ({weight_mb:.1f} MB <= 420 MB)")

    total_size = sum(f.stat().st_size for f in all_files)
    total_mb = total_size / (1024 * 1024)
    if total_size > MAX_ZIP_SIZE_UNCOMPRESSED:
        result.fail(f"Total uncompressed size too large: {total_mb:.1f} MB > 420 MB")
    else:
        result.ok(f"Total size OK ({total_mb:.1f} MB <= 420 MB)")

    # Check for disallowed file types
    bad_files = [f for f in all_files
                 if f.suffix.lower() not in ALLOWED_EXTENSIONS
                 and not f.name.startswith(".")]
    if bad_files:
        for bf in bad_files[:5]:
            result.fail(f"Disallowed file type: {bf.relative_to(submission_dir)}")
    else:
        result.ok("All file types allowed")


# ── 2. Security scan ────────────────────────────────────────────────
def validate_security(submission_dir: Path, result: ValidationResult):
    print("\n=== Security Scan ===")

    py_files = list(submission_dir.rglob("*.py"))

    for py_file in py_files:
        rel = py_file.relative_to(submission_dir)
        source = py_file.read_text(encoding="utf-8", errors="replace")

        try:
            tree = ast.parse(source, filename=str(rel))
        except SyntaxError as e:
            result.fail(f"{rel}: Syntax error at line {e.lineno}: {e.msg}")
            continue

        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".")[0] for alias in node.names]
                else:
                    modules = [node.module.split(".")[0]] if node.module else []

                for mod in modules:
                    if mod in BANNED_IMPORTS:
                        result.fail(f"{rel}:{node.lineno} — banned import: {mod}")

            # Check calls
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr

                if name in BANNED_CALLS:
                    result.fail(f"{rel}:{node.lineno} — banned call: {name}()")

        # Regex fallback for string-based evasion
        for i, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for mod in BANNED_IMPORTS:
                # Match   import os   or   from os import   but not   from pathlib.os
                if re.search(rf'\bimport\s+{re.escape(mod)}\b', stripped):
                    # Already caught by AST, but double-check
                    pass
                if re.search(rf'__import__\s*\(\s*["\']' + re.escape(mod), stripped):
                    result.fail(f"{rel}:{i} — dynamic import of banned module: {mod}")

    if result.n_fail == 0:
        result.ok("No banned imports or calls found")


# ── 3. Run the submission ───────────────────────────────────────────
def run_submission(submission_dir: Path, images_dir: Path, output_path: Path,
                   result: ValidationResult, timeout: int, cpu_only: bool):
    print("\n=== Running Submission ===")

    if not any(images_dir.iterdir()):
        result.fail(f"No images in {images_dir}")
        return None

    n_images = len([f for f in images_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    result.info(f"Input: {n_images} images from {images_dir}")

    cmd = [
        sys.executable, str(submission_dir / "run.py"),
        "--input", str(images_dir),
        "--output", str(output_path),
    ]

    env = None
    if cpu_only:
        import os
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result.info("Forcing CPU mode (CUDA_VISIBLE_DEVICES='')")

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(submission_dir),
            env=env,
        )
        elapsed = time.time() - start

        if proc.stdout:
            print("\n--- stdout ---")
            print(proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout)
        if proc.stderr:
            print("--- stderr ---")
            # Filter out ultralytics noise
            stderr_lines = proc.stderr.splitlines()
            important = [l for l in stderr_lines
                         if not any(x in l.lower() for x in ["warning", "fusing", "ultralytics"])]
            if important:
                print("\n".join(important[-50:]))

        if proc.returncode == 0:
            result.ok(f"Exit code 0 (elapsed: {elapsed:.1f}s)")
        else:
            result.fail(f"Exit code {proc.returncode} (elapsed: {elapsed:.1f}s)")
            if proc.stderr:
                # Show last error lines
                for line in proc.stderr.splitlines()[-10:]:
                    print(f"    {line}")

        if elapsed > TIMEOUT_SECONDS:
            result.fail(f"Would exceed sandbox timeout: {elapsed:.1f}s > {TIMEOUT_SECONDS}s")
        elif elapsed > TIMEOUT_SECONDS * 0.9:
            result.warn(f"Cutting it close: {elapsed:.1f}s / {TIMEOUT_SECONDS}s")
        else:
            result.ok(f"Within time budget ({elapsed:.1f}s / {TIMEOUT_SECONDS}s)")

        return elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        result.fail(f"TIMEOUT after {elapsed:.1f}s (limit: {timeout}s)")
        return elapsed


# ── 4. Validate output ──────────────────────────────────────────────
def validate_output(output_path: Path, images_dir: Path, result: ValidationResult):
    print("\n=== Output Validation ===")

    if not output_path.exists():
        result.fail("predictions.json not created")
        return

    result.ok("predictions.json exists")

    try:
        with open(output_path) as f:
            predictions = json.load(f)
    except json.JSONDecodeError as e:
        result.fail(f"Invalid JSON: {e}")
        return

    if not isinstance(predictions, list):
        result.fail(f"Expected JSON array, got {type(predictions).__name__}")
        return

    result.ok(f"Valid JSON array with {len(predictions)} predictions")

    if len(predictions) == 0:
        result.warn("Zero predictions — is this intentional?")
        return

    # Expected image IDs
    expected_ids = set()
    for img in images_dir.iterdir():
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                expected_ids.add(int(img.stem.split("_")[-1]))
            except ValueError:
                pass

    # Validate each prediction
    errors = []
    seen_ids = set()
    cat_counts = {}

    for i, pred in enumerate(predictions):
        # Required fields
        for field in ["image_id", "category_id", "bbox", "score"]:
            if field not in pred:
                errors.append(f"Prediction {i}: missing '{field}'")
                continue

        # image_id
        if not isinstance(pred.get("image_id"), (int, float)):
            errors.append(f"Prediction {i}: image_id must be int, got {type(pred.get('image_id')).__name__}")
        else:
            seen_ids.add(int(pred["image_id"]))

        # category_id
        cat = pred.get("category_id")
        if isinstance(cat, (int, float)):
            cat = int(cat)
            if cat < CATEGORY_RANGE[0] or cat > CATEGORY_RANGE[1]:
                errors.append(f"Prediction {i}: category_id {cat} outside [{CATEGORY_RANGE[0]}, {CATEGORY_RANGE[1]}]")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        # bbox
        bbox = pred.get("bbox")
        if isinstance(bbox, list):
            if len(bbox) != 4:
                errors.append(f"Prediction {i}: bbox must have 4 elements, got {len(bbox)}")
            elif any(not isinstance(v, (int, float)) for v in bbox):
                errors.append(f"Prediction {i}: bbox elements must be numbers")
            else:
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    errors.append(f"Prediction {i}: bbox w/h must be positive (got w={w}, h={h})")

        # score
        score = pred.get("score")
        if isinstance(score, (int, float)):
            if score < 0 or score > 1:
                errors.append(f"Prediction {i}: score {score} outside [0, 1]")

    # Report
    if not errors:
        result.ok("All predictions have valid format")
    else:
        unique_errors = list(dict.fromkeys(errors))  # dedupe preserving order
        for e in unique_errors[:10]:
            result.fail(e)
        if len(unique_errors) > 10:
            result.fail(f"... and {len(unique_errors) - 10} more format errors")

    # Coverage
    missing_ids = expected_ids - seen_ids
    if missing_ids:
        result.warn(f"Missing predictions for {len(missing_ids)} images: {sorted(missing_ids)[:5]}...")
    else:
        result.ok(f"All {len(expected_ids)} images have predictions")

    result.info(f"Unique image_ids: {len(seen_ids)}")
    result.info(f"Total predictions: {len(predictions)}")
    result.info(f"Avg per image: {len(predictions) / max(1, len(seen_ids)):.1f}")
    result.info(f"Unique categories used: {len(cat_counts)}")

    # Sample prediction
    if predictions:
        p = predictions[0]
        result.info(f"Sample: image_id={p.get('image_id')} cat={p.get('category_id')} "
                     f"bbox={p.get('bbox')} score={p.get('score')}")

    # Detection-only check
    if len(cat_counts) == 1 and 0 in cat_counts:
        result.warn("All predictions use category_id=0 — detection-only (max 70% score)")


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AINM sandbox validator")
    parser.add_argument("--zip", help="Path to submission.zip (alternative to --dir)")
    parser.add_argument("--dir", default=".", help="Submission directory (default: current)")
    parser.add_argument("--images", default="", help="Test images directory")
    parser.add_argument("--max-images", type=int, default=5, help="Max images to test with")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only mode")
    parser.add_argument("--skip-run", action="store_true", help="Only validate files, don't run")
    args = parser.parse_args()

    print("=" * 60)
    print("  AINM Object Detection — Local Sandbox Validator")
    print("=" * 60)

    # Determine submission directory
    temp_dir = None
    if args.zip:
        temp_dir = tempfile.mkdtemp(prefix="ainm_validate_")
        submission_dir = Path(temp_dir)
        print(f"\nExtracting {args.zip} → {temp_dir}")
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(temp_dir)
    else:
        submission_dir = Path(args.dir).resolve()

    print(f"Submission: {submission_dir}")

    result = ValidationResult()

    # 1. File structure
    validate_structure(submission_dir, result)

    # 2. Security scan
    validate_security(submission_dir, result)

    if args.skip_run:
        print("\n[SKIP] --skip-run: skipping execution")
    elif not result.passed:
        print("\n[SKIP] Structural/security failures — fix before running")
    else:
        # Prepare test images
        if args.images:
            images_dir = Path(args.images)
        else:
            # Try to find training images
            candidates = [
                submission_dir / "data" / "coco" / "train" / "images",
                submission_dir.parent / "data" / "coco" / "train" / "images",
                Path("data/coco/train/images"),
            ]
            images_dir = None
            for c in candidates:
                if c.exists():
                    images_dir = c
                    break

            if images_dir is None:
                print("\n[SKIP] No test images found. Use --images <path>")
                args.skip_run = True

        if not args.skip_run:
            # Copy subset of images to temp dir for testing
            test_images_dir = Path(tempfile.mkdtemp(prefix="ainm_images_"))
            import shutil
            imgs = sorted(images_dir.iterdir())
            imgs = [i for i in imgs if i.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            for img in imgs[:args.max_images]:
                shutil.copy2(img, test_images_dir / img.name)

            output_path = Path(tempfile.mkdtemp(prefix="ainm_output_")) / "predictions.json"

            # 3. Run
            run_submission(submission_dir, test_images_dir, output_path,
                           result, args.timeout, args.cpu)

            # 4. Validate output
            if output_path.exists():
                validate_output(output_path, test_images_dir, result)

            # Cleanup
            shutil.rmtree(test_images_dir, ignore_errors=True)
            if output_path.parent.exists():
                shutil.rmtree(output_path.parent, ignore_errors=True)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    n_pass = sum(1 for s, _ in result.checks if s == "PASS")
    n_fail = sum(1 for s, _ in result.checks if s == "FAIL")
    n_warn = sum(1 for s, _ in result.checks if s == "WARN")

    if result.passed:
        print(f"  RESULT: ALL CHECKS PASSED ({n_pass} passed, {n_warn} warnings)")
        print("  Ready to submit!")
    else:
        print(f"  RESULT: {n_fail} FAILURES ({n_pass} passed, {n_warn} warnings)")
        print("  Fix failures before submitting!")
    print("=" * 60)

    # Cleanup
    if temp_dir:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
