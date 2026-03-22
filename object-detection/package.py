"""Build a submission.zip with pre-flight validation.

Checks:
  - run.py exists at zip root
  - File count ≤ 1000, Python files ≤ 10, weight files ≤ 3
  - Uncompressed size ≤ 420 MB, weight size ≤ 420 MB
  - No disallowed file types
  - No banned imports/calls in .py files
  - No __MACOSX or hidden files

Usage:
    python package.py                       # Package run.py only (random baseline)
    python package.py --weights yolov8m.pt  # Include model weights
    python package.py --weights best.pt     # Include fine-tuned weights
"""
import argparse
import ast
import json
import re
import zipfile
from pathlib import Path

# Inline constants (no config import — this script must work standalone)
MAX_FILES = 1000
MAX_PYTHON_FILES = 10
MAX_WEIGHT_FILES = 3
MAX_UNCOMPRESSED_MB = 420
MAX_WEIGHT_MB = 420
ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".cfg",
                      ".pt", ".pth", ".onnx", ".safetensors", ".npy"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
BANNED_IMPORTS = {"os", "subprocess", "socket", "ctypes", "builtins"}
BANNED_CALLS = {"eval", "exec", "compile", "__import__"}


def scan_python_security(path: Path) -> list[str]:
    """Static analysis of a .py file for banned imports and calls."""
    issues = []
    source = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        issues.append(f"{path.name}: syntax error — {e}")
        return issues

    for node in ast.walk(tree):
        # Check import statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BANNED_IMPORTS:
                    issues.append(f"{path.name}:{node.lineno} — banned import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in BANNED_IMPORTS:
                    issues.append(f"{path.name}:{node.lineno} — banned import from '{node.module}'")
        # Check banned function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                issues.append(f"{path.name}:{node.lineno} — banned call '{node.func.id}()'")

    return issues


def validate_files(files: list[Path]) -> list[str]:
    """Validate a list of files against submission constraints."""
    issues = []

    # Check run.py exists
    names = [f.name for f in files]
    if "run.py" not in names:
        issues.append("run.py not found — it must be included at zip root")

    # Count by type
    py_files = [f for f in files if f.suffix == ".py"]
    weight_files = [f for f in files if f.suffix in WEIGHT_EXTENSIONS]

    if len(files) > MAX_FILES:
        issues.append(f"Too many files: {len(files)} > {MAX_FILES}")
    if len(py_files) > MAX_PYTHON_FILES:
        issues.append(f"Too many Python files: {len(py_files)} > {MAX_PYTHON_FILES}")
    if len(weight_files) > MAX_WEIGHT_FILES:
        issues.append(f"Too many weight files: {len(weight_files)} > {MAX_WEIGHT_FILES}")

    # Size checks
    total_bytes = sum(f.stat().st_size for f in files)
    total_mb = total_bytes / (1024 * 1024)
    if total_mb > MAX_UNCOMPRESSED_MB:
        issues.append(f"Total uncompressed size: {total_mb:.1f} MB > {MAX_UNCOMPRESSED_MB} MB")

    weight_bytes = sum(f.stat().st_size for f in weight_files)
    weight_mb = weight_bytes / (1024 * 1024)
    if weight_mb > MAX_WEIGHT_MB:
        issues.append(f"Weight files total: {weight_mb:.1f} MB > {MAX_WEIGHT_MB} MB")

    # Extension check
    for f in files:
        if f.suffix not in ALLOWED_EXTENSIONS:
            issues.append(f"Disallowed file type: {f.name} ({f.suffix})")

    # Security scan Python files
    for f in py_files:
        issues.extend(scan_python_security(f))

    return issues


def build_zip(files: list[Path], output: Path) -> None:
    """Create submission zip with files at the root level."""
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)  # Flat — all at root

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Created {output} ({size_mb:.1f} MB compressed)")

    # Verify structure
    with zipfile.ZipFile(output, "r") as zf:
        entries = zf.namelist()
        print(f"Contents ({len(entries)} files):")
        for e in entries:
            info = zf.getinfo(e)
            print(f"  {e} ({info.file_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Build submission.zip for NorgesGruppen competition")
    parser.add_argument("--weights", nargs="*", default=[], help="Model weight files to include")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra files to include (.py, .json, .yaml)")
    parser.add_argument("--output", default=None, help="Output zip path (default: submissions/submission.zip)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    submissions_dir = project_root / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    output = Path(args.output) if args.output else submissions_dir / "submission.zip"

    # Collect files
    files = [project_root / "run.py"]

    for w in args.weights:
        wp = Path(w) if Path(w).is_absolute() else project_root / w
        if not wp.exists():
            print(f"ERROR: Weight file not found: {wp}")
            return
        files.append(wp)

    for e in args.extra:
        ep = Path(e) if Path(e).is_absolute() else project_root / e
        if not ep.exists():
            print(f"ERROR: Extra file not found: {ep}")
            return
        files.append(ep)

    # Validate
    print("=" * 60)
    print("Pre-flight validation")
    print("=" * 60)
    issues = validate_files(files)

    if issues:
        print(f"\n✗ {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nFix these issues before uploading.")
        return

    print("✓ All checks passed\n")

    # Build
    build_zip(files, output)
    print(f"\nReady to upload: {output}")


if __name__ == "__main__":
    main()
