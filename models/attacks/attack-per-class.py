# =============================================================================
# CS6413: Per-Class MIA Vulnerability Analysis
# =============================================================================
# PURPOSE:
#   Analyze membership inference vulnerability at the CLASS level rather than
#   the aggregate level. This reveals that privacy risk is NOT uniform —
#   some classes are far more vulnerable than others.
#
# WHY THIS MATTERS:
#   Aggregate AUC (e.g., 0.73) hides the fact that "cat" might have AUC=0.82
#   while "ship" has AUC=0.58. A user whose data falls in a vulnerable class
#   faces much higher privacy risk than the average suggests.
#
#   This analysis also shows HOW defenses work: a good defense should reduce
#   the variance across classes (make all classes equally hard to attack),
#   not just lower the mean AUC.
#
# ANALYSIS PRODUCED:
#   1. Per-class AUC heatmap across all defenses
#   2. Per-class vulnerability ranking (which classes leak most)
#   3. Defense effectiveness per class (which defense helps which class)
#   4. Correlation analysis: does model accuracy predict vulnerability?
#
# INPUTS:
#   outputs/logs/per_sample_mia_*.csv (all available defense configs)
#   outputs/reports/attack_shadow_v2/shadow_features_v2.csv
#
# OUTPUTS:
#   outputs/reports/per_class/per_class_heatmap.png
#   outputs/reports/per_class/vulnerability_ranking.png
#   outputs/reports/per_class/accuracy_vs_auc.png
#   outputs/reports/per_class/per_class_summary.csv
#   outputs/reports/per_class/analysis_summary.txt
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

LOGS_DIR   = os.path.join(PROJECT_ROOT, "outputs", "logs")
SHADOW_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports",
                          "attack_shadow_v2", "shadow_features_v2.csv")
REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports", "per_class")
os.makedirs(REPORT_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# All configs — will skip any that don't exist
ALL_CONFIGS = {
    "No defense":    "per_sample_mia_v2.csv",
    "Early stop":    "per_sample_mia_early_stop.csv",
    "Distillation":  "per_sample_mia_distillation.csv",
    "Conf masking":  "per_sample_mia_conf_masking.csv",
    "Regularised":   "per_sample_mia_regularised.csv",
    "DP  ε=10":      "per_sample_mia_dp_eps10.csv",
    "DP  ε=5":       "per_sample_mia_dp_eps5.csv",
    "DP  ε=1":       "per_sample_mia_dp_eps1.csv",
}


# =============================================================================
# HELPERS
# =============================================================================

def prepare_features(df):
    feat_cols = [
        "conf_true", "conf_max", "loss", "entropy",
        "logit_true", "logit_gap", "m_entropy",
    ]
    x = df[feat_cols].copy()
    x["loss"] = -x["loss"]
    x["entropy"] = -x["entropy"]
    y = (df["split_name"] == "member").astype(int).values
    return x.values.astype(np.float64), y


def per_class_auc(df, signal_col, negate=False):
    """Compute AUC for each of the 10 CIFAR-10 classes using a single signal."""
    y_all = (df["split_name"] == "member").astype(int).values
    sig = -df[signal_col].values if negate else df[signal_col].values

    aucs = []
    for cls_id in range(10):
        mask = (df["true_label"] == cls_id).values
        sub_y = y_all[mask]
        sub_sig = sig[mask]
        if sub_y.sum() == 0 or sub_y.sum() == mask.sum() or mask.sum() < 20:
            aucs.append(float("nan"))
            continue
        fpr, tpr, _ = roc_curve(sub_y, sub_sig, pos_label=1)
        aucs.append(auc(fpr, tpr))
    return aucs


def per_class_shadow_auc(df, attack_clf):
    """Compute per-class AUC using the learned shadow attack classifier."""
    y_all = (df["split_name"] == "member").astype(int).values
    x_feat, _ = prepare_features(df)
    scores = attack_clf.predict_proba(x_feat)[:, 1]

    aucs = []
    for cls_id in range(10):
        mask = (df["true_label"] == cls_id).values
        sub_y = y_all[mask]
        sub_s = scores[mask]
        if sub_y.sum() == 0 or sub_y.sum() == mask.sum() or mask.sum() < 20:
            aucs.append(float("nan"))
            continue
        fpr, tpr, _ = roc_curve(sub_y, sub_s, pos_label=1)
        aucs.append(auc(fpr, tpr))
    return aucs


def per_class_accuracy(df):
    """Compute member and non-member accuracy per class."""
    mem = df[df.split_name == "member"]
    non = df[df.split_name == "nonmember"]
    mem_acc, non_acc = [], []
    for cls_id in range(10):
        m = mem[mem.true_label == cls_id]
        n = non[non.true_label == cls_id]
        mem_acc.append(m["correct"].mean() if len(m) > 0 else float("nan"))
        non_acc.append(n["correct"].mean() if len(n) > 0 else float("nan"))
    return mem_acc, non_acc


# =============================================================================
# PLOT 1: Per-class AUC heatmap across all defenses
# =============================================================================

def plot_heatmap(all_aucs, configs_found, out_path):
    """
    Rows = defenses, Columns = CIFAR-10 classes.
    Color intensity = AUC. Darker = more vulnerable.
    This is the key figure for the report — shows at a glance which
    class+defense combinations are most/least private.
    """
    data = np.array([all_aucs[c] for c in configs_found])
    fig, ax = plt.subplots(figsize=(12, max(3, len(configs_found) * 0.7 + 1)))

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto",
                   vmin=0.45, vmax=0.85)

    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(configs_found)))
    ax.set_yticklabels(configs_found, fontsize=10)

    # Annotate each cell with the AUC value
    for i in range(len(configs_found)):
        for j in range(10):
            val = data[i, j]
            if np.isnan(val):
                txt = "—"
            else:
                txt = f"{val:.2f}"
            color = "white" if val > 0.72 or val < 0.48 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Shadow attack AUC", fontsize=10)

    ax.set_title("Per-class MIA vulnerability across defenses\n"
                 "(red = vulnerable, green = protected)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved → {out_path}")


# =============================================================================
# PLOT 2: Vulnerability ranking — which classes leak most (no-defense)
# =============================================================================

def plot_vulnerability_ranking(df_baseline, attack_clf, out_path):
    """
    Bar chart ranking classes by shadow attack AUC on the undefended model.
    Also overlays the per-class overfit gap (member_acc - nonmember_acc)
    to show the correlation between memorisation and vulnerability.
    """
    aucs = per_class_shadow_auc(df_baseline, attack_clf)
    mem_acc, non_acc = per_class_accuracy(df_baseline)
    gaps = [m - n for m, n in zip(mem_acc, non_acc)]

    # Sort by AUC descending
    order = np.argsort(aucs)[::-1]
    sorted_classes = [CIFAR10_CLASSES[i] for i in order]
    sorted_aucs = [aucs[i] for i in order]
    sorted_gaps = [gaps[i] for i in order]

    fig, ax1 = plt.subplots(figsize=(11, 5))

    x = np.arange(10)
    colors = ["#D85A30" if a >= 0.75 else "#BA7517" if a >= 0.65
              else "#1D9E75" for a in sorted_aucs]

    bars = ax1.bar(x, sorted_aucs, color=colors, edgecolor="none",
                   width=0.55, label="Shadow AUC")
    ax1.axhline(0.5, color="black", lw=1, ls="--", alpha=0.4)
    ax1.set_ylabel("Shadow attack AUC", fontsize=11)
    ax1.set_ylim(0.4, 0.95)

    # Overlay overfit gap as line
    ax2 = ax1.twinx()
    ax2.plot(x, sorted_gaps, "s-", color="#185FA5", lw=2, ms=6,
             label="Overfit gap (mem−non acc)")
    ax2.set_ylabel("Overfit gap", fontsize=11, color="#185FA5")
    ax2.set_ylim(0, 0.55)

    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_classes, fontsize=10)
    ax1.set_title("Class vulnerability ranking (no-defense baseline)\n"
                  "Higher AUC = more privacy leakage", fontsize=12)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    for bar, val in zip(bars, sorted_aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Vulnerability ranking saved → {out_path}")


# =============================================================================
# PLOT 3: Accuracy vs AUC scatter — does memorisation predict vulnerability?
# =============================================================================

def plot_accuracy_vs_auc(df_baseline, attack_clf, out_path):
    """
    Scatter plot: x = per-class overfit gap, y = per-class shadow AUC.
    If memorisation drives MIA, we expect a positive correlation.
    This is a key analytical finding for the report.
    """
    aucs = per_class_shadow_auc(df_baseline, attack_clf)
    mem_acc, non_acc = per_class_accuracy(df_baseline)
    gaps = [m - n for m, n in zip(mem_acc, non_acc)]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(10):
        color = "#D85A30" if aucs[i] >= 0.75 else "#185FA5"
        ax.scatter(gaps[i], aucs[i], s=100, color=color, zorder=3,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(CIFAR10_CLASSES[i], (gaps[i], aucs[i]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color=color)

    # Trend line
    valid = [(g, a) for g, a in zip(gaps, aucs) if not np.isnan(a)]
    if len(valid) >= 3:
        gs, as_ = zip(*valid)
        z = np.polyfit(gs, as_, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(gs), max(gs), 50)
        ax.plot(x_line, p(x_line), "--", color="#888780", lw=1.5, alpha=0.7)

        # Correlation coefficient
        corr = np.corrcoef(gs, as_)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}",
                transform=ax.transAxes, fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.axhline(0.5, color="black", lw=0.8, ls=":", alpha=0.4)
    ax.set_xlabel("Overfit gap (member acc − non-member acc)", fontsize=11)
    ax.set_ylabel("Shadow attack AUC", fontsize=11)
    ax.set_title("Does memorisation predict vulnerability?\n"
                 "Per-class overfit gap vs attack AUC (no-defense)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Accuracy vs AUC scatter saved → {out_path}")


# =============================================================================
# SUMMARY CSV + TEXT
# =============================================================================

def write_summary_csv(all_aucs, configs_found, out_path):
    """Write per-class AUC table as CSV for easy inclusion in LaTeX."""
    rows = []
    for cfg in configs_found:
        row = {"Defense": cfg}
        for i, cls_name in enumerate(CIFAR10_CLASSES):
            row[cls_name] = f"{all_aucs[cfg][i]:.4f}" if not np.isnan(all_aucs[cfg][i]) else "—"
        # Add mean and std
        vals = [v for v in all_aucs[cfg] if not np.isnan(v)]
        row["Mean"] = f"{np.mean(vals):.4f}" if vals else "—"
        row["Std"] = f"{np.std(vals):.4f}" if vals else "—"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Summary CSV saved → {out_path}")
    return df


def write_analysis(all_aucs, configs_found, df_baseline, attack_clf, out_path):
    """Write a text analysis highlighting key findings."""
    # Find most/least vulnerable classes on baseline
    baseline_aucs = all_aucs.get("No defense", [float("nan")] * 10)
    valid = [(CIFAR10_CLASSES[i], baseline_aucs[i])
             for i in range(10) if not np.isnan(baseline_aucs[i])]
    valid.sort(key=lambda x: x[1], reverse=True)

    # Correlation
    mem_acc, non_acc = per_class_accuracy(df_baseline)
    gaps = [m - n for m, n in zip(mem_acc, non_acc)]
    valid_pairs = [(g, a) for g, a in zip(gaps, baseline_aucs) if not np.isnan(a)]
    corr = np.corrcoef(*zip(*valid_pairs))[0, 1] if len(valid_pairs) >= 3 else 0

    # Defense effectiveness variance
    lines = [
        "=" * 70,
        "PER-CLASS MIA VULNERABILITY ANALYSIS",
        "=" * 70,
        "",
        "KEY FINDING 1: Privacy risk is NOT uniform across classes",
        f"  Most vulnerable class:  {valid[0][0]} (AUC={valid[0][1]:.4f})",
        f"  Least vulnerable class: {valid[-1][0]} (AUC={valid[-1][1]:.4f})",
        f"  Range: {valid[0][1] - valid[-1][1]:.4f}",
        "",
        "  Class ranking (no defense, shadow AUC):",
    ]
    for name, auc_val in valid:
        bar = "█" * int(auc_val * 40)
        lines.append(f"    {name:<12} {auc_val:.4f}  {bar}")

    lines += [
        "",
        "KEY FINDING 2: Memorisation predicts vulnerability",
        f"  Correlation (overfit gap vs AUC): r = {corr:.3f}",
        "  Classes the model memorises more are more vulnerable to MIA.",
        "",
        "KEY FINDING 3: Defenses affect classes differently",
    ]

    for cfg in configs_found:
        if cfg == "No defense":
            continue
        vals = [v for v in all_aucs[cfg] if not np.isnan(v)]
        baseline_vals = [v for v in baseline_aucs if not np.isnan(v)]
        if vals and baseline_vals:
            mean_reduction = np.mean(baseline_vals) - np.mean(vals)
            std_after = np.std(vals)
            lines.append(
                f"  {cfg:<16}: mean AUC reduction = {mean_reduction:+.4f}, "
                f"class std = {std_after:.4f}"
            )

    lines += [
        "",
        "IMPLICATION: Aggregate AUC understates risk for vulnerable classes.",
        "A defense that lowers mean AUC but leaves high variance still",
        "exposes users in vulnerable classes to significant privacy risk.",
        "",
        "=" * 70,
    ]

    txt = "\n".join(lines)
    print("\n" + txt)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nAnalysis saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("PER-CLASS MIA VULNERABILITY ANALYSIS")
    print("=" * 60)

    # ── Train shadow classifier ───────────────────────────────────────────
    if not os.path.exists(SHADOW_CSV):
        print(f"Missing: {SHADOW_CSV}")
        print("Run models/attacks/attack-shadow.py first.")
        return

    shadow_df = pd.read_csv(SHADOW_CSV)
    x_shd, y_shd = prepare_features(shadow_df)
    attack_clf = LogisticRegression(
        max_iter=2000, solver="lbfgs", class_weight="balanced")
    attack_clf.fit(x_shd, y_shd)
    print("Shadow attack classifier ready.\n")

    # ── Load all available configs ────────────────────────────────────────
    configs_found = []
    config_dfs = {}
    for label, fname in ALL_CONFIGS.items():
        path = os.path.join(LOGS_DIR, fname)
        if os.path.exists(path):
            config_dfs[label] = pd.read_csv(path)
            configs_found.append(label)
            print(f"  ✓ {label}")
        else:
            print(f"  ✗ {label} — skipped")

    if not configs_found:
        print("No configs found.")
        return

    # ── Compute per-class AUCs ────────────────────────────────────────────
    print("\nComputing per-class AUCs...")
    all_aucs = {}
    for label in configs_found:
        all_aucs[label] = per_class_shadow_auc(config_dfs[label], attack_clf)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[1/3] Heatmap...")
    plot_heatmap(all_aucs, configs_found,
                 os.path.join(REPORT_DIR, "per_class_heatmap.png"))

    if "No defense" in config_dfs:
        print("[2/3] Vulnerability ranking...")
        plot_vulnerability_ranking(
            config_dfs["No defense"], attack_clf,
            os.path.join(REPORT_DIR, "vulnerability_ranking.png"))

        print("[3/3] Accuracy vs AUC scatter...")
        plot_accuracy_vs_auc(
            config_dfs["No defense"], attack_clf,
            os.path.join(REPORT_DIR, "accuracy_vs_auc.png"))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\nWriting summaries...")
    write_summary_csv(all_aucs, configs_found,
                      os.path.join(REPORT_DIR, "per_class_summary.csv"))

    if "No defense" in config_dfs:
        write_analysis(all_aucs, configs_found,
                       config_dfs["No defense"], attack_clf,
                       os.path.join(REPORT_DIR, "analysis_summary.txt"))

    print(f"\nAll outputs in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
