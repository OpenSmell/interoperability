"""Experiment 7: Cross-Device Sanity Check (Zero-Shot)

Documents the existing cross-device zero-shot results from the research/
directory.  These are informal tests between a 3-sensor OpenSmell user
device and the 6-sensor SmellNet device, with uncontrolled substance
matching and no standard enclosure protocol.

The result is consistent with Theorem 2: even with Rs/R₀ normalization,
different MQ sensor models (different a,b constants) produce incompatible
feature distributions.

This test serves as an empirical illustration of the theorem's prediction,
NOT as a rigorous experimental validation.
"""

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, TABLES_DIR, METRICS_PATH


EXISTING_RESULTS = [
    {
        "method": "30-dim paradigm features (6 ch, 4-class)",
        "accuracy": "11.9%",
        "chance": "25.0%",
        "p_value": "n.s.",
        "n_windows": 59,
        "source": "research/cross_device_paradigm_analysis/",
        "notes": "Full 6-channel device-agnostic features. Near chance.",
    },
    {
        "method": "145-dim framework features (6 ch, 4-class)",
        "accuracy": "18.6%",
        "chance": "25.0%",
        "p_value": "n.s.",
        "n_windows": 59,
        "source": "research/cross_device_145dim_analysis/",
        "notes": "Full framework with device-specific dimensions. Still near chance.",
    },
    {
        "method": "30-dim paradigm (garlic vs ginger, 2-class)",
        "accuracy": "~57%",
        "chance": "50.0%",
        "p_value": "~0.15",
        "n_windows": "~53",
        "source": "research/paradigm-experiments/",
        "notes": "2-class subset. Not statistically significant (1.0σ above chance).",
    },
    {
        "method": "15-dim shared sensors (MQ-3, MQ-7, MQ-135, 4-class)",
        "accuracy": "10.2%",
        "chance": "25.0%",
        "p_value": "0.999",
        "n_windows": 59,
        "source": "research/shared_sensors_experiment/",
        "notes": "Same physical sensor models on both devices. Still at chance.",
    },
    {
        "method": "MQ-3 only (5-dim, 4-class)",
        "accuracy": "6.8%",
        "chance": "25.0%",
        "p_value": "0.999",
        "n_windows": 59,
        "source": "research/ (test_mq3_only.py)",
        "notes": "MQ-3 is same model on both devices, yet performs below chance.",
    },
    {
        "method": "MQ-135 VOC only (5-dim, 4-class)",
        "accuracy": "40.7%",
        "chance": "25.0%",
        "p_value": "borderline",
        "n_windows": 59,
        "source": "research/cross_device_paradigm_analysis/",
        "notes": "Best single channel. Weak evidence of partial transferability.",
    },
]


def run():
    print("=" * 70)
    print("Experiment 7: Cross-Device Sanity Check (Zero-Shot)")
    print("=" * 70)
    print("""
  These results document our informal cross-device zero-shot tests
  between a 3-sensor OpenSmell user device and the 6-sensor SmellNet
  device.  They are NOT controlled experiments: substances were
  recorded at different times, with different concentrations, without
  a shared enclosure.  The results are consistent with Theorem 2:
  different MQ sensor models have different sensitivity curve constants
  (a, b), which Rs/R₀ normalization does NOT cancel.
  """)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Print table ─────────────────────────────────────────────────
    print(f"  {'Method':<55s} | {'Accuracy':>8s} | {'Chance':>8s} | {'p-value':>8s}")
    print(f"  {'─' * 55}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 8}")
    for r in EXISTING_RESULTS:
        print(f"  {r['method']:<55s} | {r['accuracy']:>8s} | {r['chance']:>8s} | {r['p_value']:>8s}")
    print(f"  {'─' * 55}─┴─{'─' * 8}─┴─{'─' * 8}─┴─{'─' * 8}")

    # ── Save CSV ────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "cross_device_sanity.csv"
    with open(csv_path, "w") as f:
        f.write("Method,Accuracy,Chance,p_value,N_Windows,Source,Notes\n")
        for r in EXISTING_RESULTS:
            f.write(f"{r['method']},{r['accuracy']},{r['chance']},{r['p_value']},{r['n_windows']},{r['source']},{r['notes']}\n")
    print(f"\n  Saved {csv_path}")

    # ── Save text table ─────────────────────────────────────────────
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 110)
    lines.append("Table 8: Cross-Device Sanity Check — Zero-Shot Transfer (Informal)")
    lines.append("=" * 110)
    lines.append(f"{'Method':<55s} | {'Accuracy':>8s} | {'Chance':>8s} | {'p-value':>8s} | {'N':>6s}")
    lines.append("-" * 110)
    for r in EXISTING_RESULTS:
        lines.append(f"{r['method']:<55s} | {r['accuracy']:>8s} | {r['chance']:>8s} | {r['p_value']:>8s} | {str(r['n_windows']):>6s}")
    lines.append("=" * 110)
    lines.append("")
    lines.append("Limitations:")
    lines.append("  - Different recording times and concentrations")
    lines.append("  - No standard enclosure or protocol")
    lines.append("  - Sensor arrays only partially overlap (3 vs 6 sensors)")
    lines.append("  - Substance identity matching is approximate")
    lines.append("")
    lines.append("Significance:")
    lines.append("  These results are consistent with Theorem 2: even when both devices")
    lines.append("  use the same MQ sensor model (e.g., MQ-3), unit-to-unit variation")
    lines.append("  in sensitivity curve constants (20-30% per datasheets) prevents")
    lines.append("  zero-shot transfer.  The framework's device-agnostic features are")
    lines.append("  necessary but not sufficient for cross-device interoperability —")
    lines.append("  calibration (adapter training or per-device calibration gas) is also")
    lines.append("  required.")

    table_path = TABLES_DIR / "table8_cross_device_sanity.txt"
    table_path.write_text("\n".join(lines))
    print(f"  Saved {table_path}")

    # ── Update metrics.json ──────────────────────────────────────────
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    metrics["experiment_7"] = {
        "experiment": "cross_device_sanity",
        "results": EXISTING_RESULTS,
        "runnable": True,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics["experiment_7"]


if __name__ == "__main__":
    run()
