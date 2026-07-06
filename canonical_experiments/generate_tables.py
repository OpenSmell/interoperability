"""Generate all tables for the paper from experiment results."""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, TABLES_DIR, METRICS_PATH


def table1_taxonomy():
    """Table 1: Framework taxonomy summary."""
    lines = []
    lines.append("=" * 90)
    lines.append("Table 1: Complete Framework Taxonomy Summary")
    lines.append("=" * 90)
    lines.append(f"{'Category':<20s} | {'Purpose':<30s} | {'Features/Ch':>12s} | {'Use Cases':<25s}")
    lines.append("-" * 90)
    categories = [
        ("Device-Agnostic", "Cross-device interoperability", "6 + N²/2 + 4", "Transfer learning"),
        ("Absolute", "Quantitative sensing", "4", "Concentration"),
        ("Temporal", "Fast dynamics / safety", "4", "Gas leak detection"),
        ("Health", "Predictive maintenance", "4", "Drift monitoring"),
        ("Hardware", "Device characterization", "4", "Fingerprinting"),
    ]
    for cat, purpose, n_feats, use_case in categories:
        lines.append(f"{cat:<20s} | {purpose:<30s} | {n_feats:>12s} | {use_case:<25s}")
    lines.append("-" * 90)
    lines.append("Total: 21 features per channel + cross-channel selectivity + global metrics")
    lines.append("=" * 90)
    return "\n".join(lines)


def table2_sensor_count():
    """Table 2: Sensor count vs distinguishable substances."""
    lines = []
    lines.append("=" * 60)
    lines.append("Table 2: Sensor Array Size vs Distinguishable Substances")
    lines.append("=" * 60)
    lines.append(f"{'Sensors':>8s} | {'Effective Dim.':>14s} | {'Distinguishable':>16s}")
    lines.append("-" * 60)
    rows = [
        (3, "~1.5", "4-6"),
        (6, "~4.0", "20-40"),
        (12, "~7.2", "200-400"),
        (24, "~14.4", "10,000+"),
    ]
    for n_sensors, eff_dim, distinguishable in rows:
        lines.append(f"{n_sensors:>8d} | {eff_dim:>14s} | {distinguishable:>16s}")
    lines.append("=" * 60)
    return "\n".join(lines)


def table3_chemical_boundaries():
    """Table 3: Chemical information boundaries."""
    lines = []
    lines.append("=" * 90)
    lines.append("Table 3: Chemical Information Boundaries for MOX Sensors")
    lines.append("=" * 90)
    lines.append(f"{'CAN Capture':<40s} | {'CANNOT Capture':<40s}")
    lines.append("-" * 90)
    boundaries = [
        ("Functional groups", "Molecular structure (isomers)"),
        ("Molecular size (small vs large)", "Absolute concentration (no calibration)"),
        ("Vapor pressure", "Non-redox-active molecules (N2, O2, CO2)"),
        ("Redox potential", "Trace concentrations (<1 ppm)"),
        ("Total reducing power", "Complex mixture decomposition"),
        ("Binding strength", "Chirality (L- vs D-forms)"),
    ]
    for can, cannot in boundaries:
        lines.append(f"{can:<40s} | {cannot:<40s}")
    lines.append("=" * 90)
    return "\n".join(lines)


def table4_leave_substance_out(metrics):
    """Table 4: Leave-substance-out generalization results."""
    exp2 = metrics.get("experiment_2", {})
    folds = exp2.get("fold_results", [])
    n_subs = exp2.get("n_substances", 0)
    lines = []
    lines.append("=" * 70)
    lines.append("Table 4: Leave-Substance-Out Generalization")
    lines.append("=" * 70)
    lines.append(f"{'Fold':>6s} | {'Substances Held Out':<35s} | {'Consistency':>11s}")
    lines.append("-" * 70)
    if folds:
        for f in folds:
            held = ", ".join(f.get('held_out_substances', [])[:3])
            n_held = len(f.get('held_out_substances', []))
            if n_held > 3:
                held += f" (+{n_held - 3})"
            cons = f.get('mean_consistency', 0) * 100
            lines.append(f"{f['fold']:>6d} | {held:<35s} | {cons:>10.1f}%")
        lines.append("-" * 70)
        mean_cons = exp2.get("mean_consistency", 0) * 100
        lines.append(f"{'Mean':>6s} | {'(all folds)':<35s} | {mean_cons:>10.1f}%")
        lines.append("-" * 70)
        baseline = 1.0 / n_subs * 100 if n_subs > 0 else 0
        lines.append(f"{'Random baseline':>32s} | {baseline:>10.1f}%")
    else:
        lines.append("  No data available")
    lines.append("=" * 70)
    return "\n".join(lines)


def table5_normalization_comparison(metrics):
    """Table 5: Normalization method comparison (LSO consistency)."""
    exp4 = metrics.get("experiment_4", {})
    comp = exp4.get("comparison", [])
    n_subs = exp4.get("n_substances", 0)
    lines = []
    lines.append("=" * 70)
    lines.append("Table 5: Leave-Substance-Out Normalization Comparison")
    lines.append("=" * 70)
    lines.append(f"{'Method':<40s} | {'Consistency':>11s} | {'Accuracy':>10s}")
    lines.append("-" * 70)
    if comp:
        for c in comp:
            cons = c.get('mean_consistency', 0) * 100
            acc = c.get('mean_accuracy', 0) * 100
            name = c['method'].replace('V/V0', 'R/R0')
            lines.append(f"{name:<40s} | {cons:>10.1f}% | {acc:>9.1f}%")
        lines.append("-" * 70)
        baseline = 1.0 / n_subs * 100 if n_subs > 0 else 0
        lines.append(f"{'Random baseline':<40s} | {baseline:>10.1f}% | {'--':>9s}")
    else:
        lines.append("  No data available")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Note: Held-out substances have unseen labels, so accuracy is 0%")
    lines.append("for all methods. Consistency measures within-substance agreement:")
    lines.append("fraction of held-out windows mapping to the same training class.")
    return "\n".join(lines)


def table6_sensor_roadmap():
    """Table 6: Next-gen sensor roadmap."""
    lines = []
    lines.append("=" * 90)
    lines.append("Table 6: Next-Generation Sensor Roadmap")
    lines.append("=" * 90)
    lines.append(f"{'Gen':>5s} | {'Technologies':<25s} | {'Adds':<30s} | {'Use Cases':<25s}")
    lines.append("-" * 90)
    roadmap = [
        ("1", "MOX (current)", "Redox chemistry, kinetics", "Pattern recognition"),
        ("2", "MOX + Electrochemical", "Absolute concentration, ppm", "Regulatory compliance"),
        ("3", "Gen 2 + Optical", "Vibrational spectra, isomers", "Molecular ID"),
        ("4", "Gen 3 + Chiral", "Enantiomer discrimination", "Pharma QC"),
        ("5", "Gen 4 + Mass Spec", "Molecular weight, fragments", "Forensics"),
    ]
    for gen, tech, adds, use in roadmap:
        lines.append(f"{gen:>5s} | {tech:<25s} | {adds:<30s} | {use:<25s}")
    lines.append("=" * 90)
    return "\n".join(lines)


def table7_ablation(metrics):
    """Table 7: Ablation study — Dimension 1 feature subgroups."""
    exp5 = metrics.get("experiment_5", {})
    baseline = exp5.get("baseline_accuracy")
    ablations = exp5.get("ablations", {})
    n_dim1 = exp5.get("n_dim1_features", 0)
    lines = []
    lines.append("=" * 80)
    lines.append("Table 7: Ablation Study — Dimension 1 (Device-Agnostic) Features")
    lines.append("=" * 80)
    lines.append(f"{'Ablation':<30s} | {'Mean Acc':>8s} | {'Std':>5s} | {'Δ from Dim1':>10s}")
    lines.append("-" * 80)
    if baseline is not None:
        lines.append(f"{'Dimension 1 (all ' + str(n_dim1) + ' feats)':<30s} | {baseline*100:>7.1f}% | {'—':>5s} | {'—':>10s}")
        for gname in sorted(ablations.keys()):
            d = ablations[gname]
            label = gname.replace("_", " ").replace("Remove ", "")
            delta = d["delta_from_dim1_baseline"]
            delta_str = f"{delta*100:+.1f}%"
            lines.append(f"{'  Without ' + label:<30s} | {d['mean_accuracy']*100:>7.1f}% | {d['std_accuracy']*100:>4.1f}% | {delta_str:>10s}")
    else:
        lines.append("  No data available")
    lines.append("=" * 80)
    return "\n".join(lines)


def table8_baseline_comparison():
    """Table 8: Baseline comparison across methods."""
    try:
        import csv
        csv_path = Path(__file__).resolve().parent / "results" / "baseline_comparison.csv"
        if not csv_path.exists():
            return "Table 8: No baseline comparison data. Run 06_baseline_comparison.py first."
        lines = []
        lines.append("=" * 100)
        lines.append("Table 8: Baseline Comparison — Session-Invariance Accuracy")
        lines.append("=" * 100)
        lines.append(f"{'Rank':>5s} | {'Method':<40s} | {'Accuracy':>9s}")
        lines.append("-" * 100)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                lines.append(f"{i:>5d} | {row['Method']:<40s} | {row['Accuracy']:>9s}")
        lines.append("=" * 100)
        return "\n".join(lines)
    except Exception:
        return "Table 8: Error loading baseline comparison."


def table9_cross_device_sanity():
    """Table 9: Cross-device sanity check results."""
    try:
        import csv
        csv_path = Path(__file__).resolve().parent / "results" / "cross_device_sanity.csv"
        if not csv_path.exists():
            return "Table 9: No cross-device data. Run 07_cross_device_sanity.py first."
        lines = []
        lines.append("=" * 110)
        lines.append("Table 9: Cross-Device Sanity Check — Zero-Shot Transfer (Informal)")
        lines.append("=" * 110)
        lines.append(f"{'Method':<55s} | {'Accuracy':>8s} | {'Chance':>8s} | {'p-value':>8s}")
        lines.append("-" * 110)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                lines.append(f"{row['Method']:<55s} | {row['Accuracy']:>8s} | {row['Chance']:>8s} | {row['p_value']:>8s}")
        lines.append("-" * 110)
        lines.append("Limitations: Different recording times/concentrations, no enclosure.")
        lines.append("" + "=" * 110)
        return "\n".join(lines)
    except Exception:
        return "Table 9: Error loading cross-device data."


def run():
    print("=" * 70)
    print("Generating Tables (1-9)")
    print("=" * 70)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())

    tables = {
        "table1_taxonomy": table1_taxonomy(),
        "table2_sensor_count": table2_sensor_count(),
        "table3_chemical_boundaries": table3_chemical_boundaries(),
        "table4_leave_substance_out": table4_leave_substance_out(metrics),
        "table5_normalization_comparison": table5_normalization_comparison(metrics),
        "table6_sensor_roadmap": table6_sensor_roadmap(),
        "table7_ablation": table7_ablation(metrics),
        "table8_baseline_comparison": table8_baseline_comparison(),
        "table9_cross_device_sanity": table9_cross_device_sanity(),
    }

    for name, content in tables.items():
        outpath = TABLES_DIR / f"{name}.txt"
        outpath.write_text(content)
        print(f"  Saved {outpath}")

    all_tables = []
    for name in ["table1_taxonomy", "table2_sensor_count", "table3_chemical_boundaries",
                  "table4_leave_substance_out", "table5_normalization_comparison",
                  "table6_sensor_roadmap", "table7_ablation", "table8_baseline_comparison",
                  "table9_cross_device_sanity"]:
        all_tables.append(tables[name])
        all_tables.append("")

    (TABLES_DIR / "all_tables.txt").write_text("\n".join(all_tables))
    print(f"  Saved {TABLES_DIR / 'all_tables.txt'}")
    print(f"\n  All tables saved to {TABLES_DIR}")


if __name__ == "__main__":
    run()
