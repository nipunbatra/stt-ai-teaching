"""
TrackIO Demo — Clean Runner
============================
Clears old data and runs all 5 demo scripts in sequence.

Usage:
    python trackio_demo.py
    trackio show --project cs203-week08-demo
"""
from pathlib import Path
import subprocess, sys

PROJECT = "cs203-week08-demo"

# ── Clean slate ──
cache_dir = Path.home() / ".cache" / "huggingface" / "trackio"
for ext in [".db", ".db-shm", ".db-wal", ".lock"]:
    p = cache_dir / f"{PROJECT}{ext}"
    if p.exists():
        p.unlink()
print(f"Cleared old data for '{PROJECT}'\n")

# ── Run all demos ──
here = Path(__file__).parent
scripts = [
    "trackio_1_training_curves.py",
    "trackio_2_misclassified.py",
    "trackio_3_compare_hyperparams.py",
    "trackio_4_per_class_table.py",
    "trackio_5_overfitting_alert.py",
]

for script in scripts:
    print(f"\n{'=' * 60}")
    print(f"  {script}")
    print(f"{'=' * 60}")
    subprocess.run([sys.executable, str(here / script)], check=True)

print(f"\n{'=' * 60}")
print(f"  All done! View dashboard:")
print(f"  trackio show --project {PROJECT}")
print(f"{'=' * 60}")
