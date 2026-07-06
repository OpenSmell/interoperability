"""Generate all figures for the paper from experiment results."""

import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, FIGURES_DIR, METRICS_PATH


HAS_MATPLOTLIB = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    pass


def fig1_session(metrics):
    """Figure 1: Per-substance session-invariance accuracy."""
    exp1 = metrics.get("experiment_1", {})
    accs = exp1.get("session_accuracies", [])
    if not accs:
        print("  SKIP fig1: no session accuracies")
        return
    if not HAS_MATPLOTLIB:
        print("  SKIP fig1: matplotlib not available")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(accs)), [a * 100 for a in accs], color="steelblue")
    chance = exp1.get("chance_level", 0.5) * 100
    ax.axhline(y=chance, color="red", linestyle="--", linewidth=2,
               label=f"Chance level ({chance:.0f}%)")
    ax.set_xlabel("Substance index")
    ax.set_ylabel("Session-invariance accuracy (%)")
    ax.set_title(f'Figure 1: Session-Invariance ({exp1.get("mean_session_accuracy", 0) * 100:.1f}%)')
    ax.set_ylim(0, 100)
    ax.legend()

    pv = exp1.get("p_value", 1.0)
    sig = " (p < 0.05)" if pv < 0.05 else " (ns)"
    ax.text(0.95, 0.1, f"t-test vs chance{sig}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, style="italic")
    plt.tight_layout()
    outpath = FIGURES_DIR / "fig1_session_invariance.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved {outpath}")


def fig2_leave_substance_out(metrics):
    """Figure 2: Per-fold leave-substance-out consistency."""
    exp2 = metrics.get("experiment_2", {})
    folds = exp2.get("fold_results", [])
    if not folds:
        print("  SKIP fig2: no LSO data")
        return
    if not HAS_MATPLOTLIB:
        print("  SKIP fig2: matplotlib not available")
        return
    cons = [f.get("mean_consistency", 0) * 100 for f in folds]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(cons)), cons, color="steelblue")
    mean_cons = exp2.get("mean_consistency", 0) * 100
    ax.axhline(y=mean_cons, color="red", linestyle="--", linewidth=2,
               label=f"Mean: {mean_cons:.1f}%")
    n_subs = exp2.get("n_substances", 50)
    baseline = 1.0 / n_subs * 100
    ax.axhline(y=baseline, color="gray", linestyle=":", linewidth=1.5,
               label=f"Random baseline: {baseline:.1f}%")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Within-substance prediction consistency (%)")
    ax.set_title("Figure 2: Leave-Substance-Out Generalization")
    ax.set_ylim(0, 100)
    ax.legend()
    for i, c in enumerate(cons):
        ax.text(i, c + 1, f"{c:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    outpath = FIGURES_DIR / "fig2_leave_substance_out.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved {outpath}")


def fig3_uci_drift(metrics):
    """Figure 3: Intra-gas vs inter-gas cosine similarity (real UCI data).

    Generates one figure per feature subset in feature_set_analysis.
    """
    exp3 = metrics.get("experiment_3", {})
    fsa = exp3.get("feature_set_analysis", {})
    if not fsa:
        print("  SKIP fig3: no UCI drift data")
        return
    if not HAS_MATPLOTLIB:
        print("  SKIP fig3: matplotlib not available")
        return
    for subset_name, subset_data in fsa.items():
        sep = subset_data.get("separation_ratios", {})
        if not sep:
            print(f"  SKIP fig3 subset '{subset_name}': incomplete data")
            continue
        gases = list(sep.keys())
        intra = [sep[g]["intra"] for g in gases]
        inter = [sep[g]["inter"] for g in gases]
        ratios = [sep[g]["ratio"] for g in gases]
        x = np.arange(len(gases))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        w = 0.35
        ax1.bar(x - w / 2, intra, w, label="Intra-gas (same gas, drift)", color="steelblue")
        ax1.bar(x + w / 2, inter, w, label="Inter-gas mean (different gases)", color="coral")
        ax1.set_xlabel("Gas")
        ax1.set_ylabel("Mean cosine similarity")
        ax1.set_title(f"Figure 3a: UCI Drift Similarities — {subset_name}")
        ax1.set_xticks(x)
        ax1.set_xticklabels(gases, rotation=45, ha="right")
        ax1.legend()
        colors = ["green" if r > 1.0 else "red" for r in ratios]
        ax2.bar(x, ratios, color=colors)
        ax2.set_xlabel("Gas")
        ax2.set_ylabel("Separation ratio (intra / inter)")
        msr = subset_data.get("mean_separation_ratio", 0)
        ax2.set_title(f"Figure 3b: Separation Ratios — {subset_name} (mean={msr:.2f})")
        ax2.set_xticks(x)
        ax2.set_xticklabels(gases, rotation=45, ha="right")
        ax2.axhline(y=1.0, color="black", linestyle="--", label="Stability threshold")
        ax2.legend()
        for i, r in enumerate(ratios):
            ax2.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
        plt.tight_layout()
        short_name = subset_name.replace(" ", "_")
        outpath = FIGURES_DIR / f"fig3_uci_drift_{short_name}.png"
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"  Saved {outpath}")


def fig4_normalization_comparison(metrics):
    """Figure 4: Bar chart comparing normalization methods (LSO consistency)."""
    exp4 = metrics.get("experiment_4", {})
    comp = exp4.get("comparison", [])
    if not comp:
        print("  SKIP fig4: no normalization comparison data")
        return
    if not HAS_MATPLOTLIB:
        print("  SKIP fig4: matplotlib not available")
        return
    cons = [c.get("mean_consistency", 0) * 100 for c in comp]
    short_names = ["Raw voltages", "R/R0 normalization", "Framework features"]
    colors = ["coral", "orange", "seagreen"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(short_names))
    bars = ax.bar(x, cons, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Leave-substance-out prediction consistency (%)")
    ax.set_title("Figure 4: Normalization Method Comparison (LSO)")
    for bar, c in zip(bars, cons):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{c:.1f}%", ha="center", fontsize=10, fontweight="bold")
    n_subs = exp4.get("n_substances", 50)
    baseline = 1.0 / n_subs * 100
    ax.axhline(y=baseline, color="gray", linestyle=":", linewidth=1.5,
               label=f"Random baseline ({baseline:.1f}%)")
    ax.set_ylim(0, max(cons) * 1.3)
    ax.legend()
    plt.tight_layout()
    outpath = FIGURES_DIR / "fig4_normalization_comparison.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved {outpath}")


def fig5_ablation(metrics):
    """Figure 5: Ablation study bar chart (Dimension 1 only)."""
    exp5 = metrics.get("experiment_5", {})
    ablations = exp5.get("ablations", {})
    if not ablations and exp5.get("baseline_accuracy") is None:
        print("  SKIP fig5: no ablation data")
        return
    if not HAS_MATPLOTLIB:
        print("  SKIP fig5: matplotlib not available")
        return

    labels = ["Dim 1 (all)"]
    means = [exp5["baseline_accuracy"] * 100]
    stds = [0.0]
    for group_name in sorted(ablations.keys()):
        label = group_name.replace("Remove_", "W/o ").replace("_", " ")
        labels.append(label)
        d = ablations[group_name]
        means.append(d["mean_accuracy"] * 100)
        stds.append(d["std_accuracy"] * 100)

    chance = exp5.get("mean_chance", 0.02) * 100
    colors = ["seagreen"] + ["coral"] * (len(labels) - 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Session-Invariance Accuracy (%)")
    ax.set_title("Figure 5: Ablation Study — Dimension 1 (Device-Agnostic) Subgroups")
    ax.axhline(y=chance, color="gray", linestyle=":", linewidth=1.5,
               label=f"Chance ({chance:.0f}%)")
    ax.legend()
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds + [1]) + 0.5,
                f"{m:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    outpath = FIGURES_DIR / "fig5_ablation.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved {outpath}")


def run():
    print("=" * 70)
    print("Generating Figures (1-5)")
    print("=" * 70)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())

    fig1_session(metrics)
    fig2_leave_substance_out(metrics)
    fig3_uci_drift(metrics)
    fig4_normalization_comparison(metrics)
    fig5_ablation(metrics)

    print(f"\n  All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run()
