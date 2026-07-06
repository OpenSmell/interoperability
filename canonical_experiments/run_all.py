#!/usr/bin/env python3
"""Master runner: execute all canonical experiments sequentially."""

import sys, time, json, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, FIGURES_DIR, TABLES_DIR, METRICS_PATH


EXPERIMENTS = [
    ("Experiment 1: Session-Invariance", "01_session_invariance.py"),
    ("Experiment 2: Leave-Substance-Out", "02_leave_substance_out.py"),
    ("Experiment 3: UCI Drift Stability", "03_uci_drift.py"),
    ("Experiment 4: Normalization Comparison", "04_normalization_comparison.py"),
    ("Experiment 5: Ablation Study", "05_ablation.py"),
    ("Experiment 6: Baseline Comparison", "06_baseline_comparison.py"),
    ("Experiment 7: Cross-Device Sanity Check", "07_cross_device_sanity.py"),
]

POST_PROCESSING = [
    ("Generate Figures", "generate_figures.py"),
    ("Generate Tables", "generate_tables.py"),
]


def main():
    print("=" * 70)
    print("  COMPLETE PARADIGM FRAMEWORK: CANONICAL EXPERIMENTS")
    print("=" * 70)
    t_start = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results_summary = {}
    METRICS_PATH.write_text(json.dumps({"started": t_start}))

    for name, script in EXPERIMENTS:
        print(f"\n{'=' * 70}")
        print(f"  {name}")
        print(f"{'=' * 70}")
        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, script],
                cwd=Path(__file__).resolve().parent,
                capture_output=True, text=True, timeout=1800
            )
            print(result.stdout)
            if result.stderr:
                print(f"  STDERR: {result.stderr}")
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.1f}s with code {result.returncode}")
            results_summary[script] = {
                "returncode": result.returncode,
                "elapsed_s": round(elapsed, 1),
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 600s")
            results_summary[script] = {"returncode": -1, "elapsed_s": 600, "success": False, "error": "timeout"}
        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary[script] = {"returncode": -1, "elapsed_s": 0, "success": False, "error": str(e)}

    for name, script in POST_PROCESSING:
        print(f"\n{'─' * 70}")
        print(f"  {name}")
        try:
            result = subprocess.run(
                [sys.executable, script],
                cwd=Path(__file__).resolve().parent,
                capture_output=True, text=True, timeout=600
            )
            print(result.stdout)
            if result.stderr:
                print(f"  STDERR: {result.stderr}")
            results_summary[script] = {"returncode": result.returncode, "success": result.returncode == 0}
        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary[script] = {"returncode": -1, "success": False, "error": str(e)}

    t_total = time.time() - t_start
    n_success = sum(1 for v in results_summary.values() if v.get("success"))
    n_total = len(EXPERIMENTS)

    print(f"\n{'=' * 70}")
    print(f"  CANONICAL EXPERIMENTS COMPLETE")
    print(f"  Total time: {t_total:.1f}s")
    print(f"  Experiments: {n_success}/{n_total} successful")
    print(f"  Results: {METRICS_PATH}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print(f"{'=' * 70}")

    summary = {"run_completed": time.time(), "total_elapsed_s": round(t_total, 1),
               "n_success": n_success, "n_total": n_total, "experiments": results_summary}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))

    return 0 if n_success == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
