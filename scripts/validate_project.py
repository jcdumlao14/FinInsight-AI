from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "app",
    "data",
    "data/raw",
    "data/processed",
    "data/chunks",
    "data/metadata",
    "evaluation",
    "llm",
    "monitoring",
    "rag",
    "scripts",
    "tests",
    "docs",
    "config",
]

REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    "requirements.txt",
    "app/__init__.py",
    "evaluation/__init__.py",
    "llm/__init__.py",
    "monitoring/__init__.py",
    "rag/__init__.py",
    "scripts/__init__.py",
    "tests/__init__.py",
]

print("=" * 70)
print("FinInsight-AI — PHASE 1 PROJECT VALIDATION")
print("=" * 70)

errors = []

for directory in REQUIRED_DIRS:
    path = ROOT / directory

    if path.is_dir():
        print(f"[OK] Directory: {directory}")
    else:
        print(f"[FAIL] Missing directory: {directory}")
        errors.append(directory)

for filename in REQUIRED_FILES:
    path = ROOT / filename

    if path.is_file():
        print(f"[OK] File:      {filename}")
    else:
        print(f"[FAIL] Missing file: {filename}")
        errors.append(filename)

print()
print("-" * 70)

if errors:
    print(f"VALIDATION FAILED — {len(errors)} issue(s)")
    sys.exit(1)

print("VALIDATION PASSED")
print("Phase 1 project foundation is ready.")
