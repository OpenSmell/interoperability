"""Run alignment experiments 1-7 sequentially.

Each experiment resolves its own paths from its file location and writes
`result.txt` + `metrics.json` into `results/`. The runner subprocesses each
script with the current interpreter, so it works from the repo venv or any
Python with the required dependencies.

Usage:
    python alignment_experiments/run_alignment_experiments.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXPERIMENTS = [
    "experiment_1_calibration.py",
    "experiment_2_synthetic_probe.py",
    "experiment_3_calibration.py",
    "experiment_4_coral.py",
    "experiment_5_anchors.py",
    "experiment_6_taxonomy.py",
    "experiment_7_chemoprint.py",
]


def main() -> int:
    failures = []
    for name in EXPERIMENTS:
        script = HERE / name
        print(f"==> running {name}", flush=True)
        proc = subprocess.run([sys.executable, str(script)], cwd=HERE)
        status = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
        print(f"<== {name}: {status}", flush=True)
        if proc.returncode != 0:
            failures.append(name)
    if failures:
        print(f"FAILURES: {', '.join(failures)}")
        return 1
    print("All 7 alignment experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
