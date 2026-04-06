# =============================================================================
# CS6413: Generate All Report Figures
# =============================================================================
# Produces publication-ready figures for the final ACM report.
# Run after all defenses and attacks have completed.
#
# OUTPUTS (all in outputs/reports/figures/):
#   fig1_roc_all_defenses.png      — ROC curves overlaid
#   fig2_privacy_utility_scatter.png — scatter: AUC vs holdout acc
#   fig3_heatmap_per_class.png     — per-class AUC heatmap
#   fig4_radar_chart.png           — radar/spider chart of defense profiles
#   fig5_signal_violin.png         — violin plots: member vs non-member signals
#   fig6_overfit_gap_vs_auc.png    — bar chart: gap predicts vulnerability
#   fig7_training_curves.png       — training loss/acc curves across defenses
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score


# =============================================================================
# PATHS & CONFIG
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

LOGS_DIR   = os.path.join(PROJECT_ROOT, "outputs", "logs")
SHADOW_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports",
                          "attack_shadow_v2", "shadow_features_v2.csv")
FIG_DIR    = os.path.join(PROJECT_ROOT, "outputs", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ALL_CONFIGS = {
    "No defense":    "per_sample_mia_v2.csv",
    "Regularised":   "per_sample_mia_regularised.csv",
    "Early stop":    "per_sample_mia_early_stop.csv",
    "Distillation":  "per_sample_mia_distillation.csv",
    "Conf masking":  "per_sample_mia_conf_masking.csv",
    "DP  ε=10":      "per_sample_mia_dp_eps10.csv",
    "DP  ε=5":       "per_sample_mia_dp_eps5.csv",
    "DP  ε=1":       "per_sample_mia_dp_eps1.csv",
}

PALETTE = {
    "No defense":   "#2C2C2A",
    "Regularised":  "#1D9E75",
    "Early stop":   "#7F77DD",
    "Distillation": "#E07BAA",
    "Conf masking": "#4ECDC4",
    "DP  ε=10":     "#185FA5",
    "DP  ε=5":      "#BA7517",
    "DP  ε=1":      "#D85A30",
}

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

TRAIN_LOGS = {
    "No defense":   "train_log_v2.csv",
    "Regularised":  "train_log_regularised.csv",
    "Early stop":   "train_log_early_stop.csv",
    "Distillation": "train_log_distillation.csv",
}


# =============================================================================
# HELPERS
# =============================================================================

def prepare_features(df):
    fc = ["conf_true","conf_max","loss","entropy","logit_true","logit_gap","m_entropy"]
    x = df[fc].copy()
    x["loss"] = -x["loss"]; x["entropy"] = -x["entropy"]
    y = (df["split_name"] == "member").astype(int).values
    return x.values.astype(np.float64), y

def tpr_at_fpr(fpr, tpr, max_fpr):
    m = fpr <= max_fpr
    return float(tpr[m].max()) if m.any() else 0.0

def load_configs():
    dfs = {}
    for label, fname in ALL_CONFIGS.items():
        path = os.path.join(LOGS_DIR, fname)
        if os.path.exists(path):
            dfs[label] = pd.read_csv(path)
    return dfs

def get_attack_clf():
    shd = pd.read_csv(SHADOW_CSV)
    xs, ys = prepare_features(shd)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    clf.fit(xs, ys)
    return clf

def evaluate_all(dfs, clf):
    results = {}
    for label, df in dfs.items():
        yt = (df["split_name"] == "member").astype(int).values
        # Threshold
        ft, tt, _ = roc_curve(yt, df["logit_gap"].values, pos_label=1)
        # Shadow
        xf, _ = prepare_features(df)
        sc = clf.predict_proba(xf)[:, 1]
        fs, ts, ths = roc_curve(yt, sc, pos_label=1)
        ba = max(accuracy_score(yt, (sc >= t).astype(int)) for t in ths)

        results[label] = {
            "fpr_t": ft, "tpr_t": tt, "auc_t": auc(ft, tt),
            "tpr1_t": tpr_at_fpr(ft, tt, 0.01),
            "fpr_s": fs, "tpr_s": ts, "auc_s": auc(fs, ts),
            "tpr1_s": tpr_at_fpr(fs, ts, 0.01),
            "advantage": 2 * (ba - 0.5),
            "mem_acc": df[df.split_name == "member"]["correct"].mean(),
            "non_acc": df[df.split_name == "nonmember"]["correct"].mean(),
            "gap": (df[df.split_name == "member"]["correct"].mean() -
                    df[df.split_name == "nonmember"]["correct"].mean()),
        }
    return results


# =============================================================================
# FIG 1: ROC Curves — all defenses on one plot (full + zoom)
# =============================================================================

def fig1_roc(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Shadow Attack ROC Curves — All Defense Configurations", fontsize=13)

    for ax_i, ax in enumerate(axes):
        xlim = (0, 1.0) if ax_i == 0 else (0, 0.05)
        ylim = (0, 1.05) if ax_i == 0 else (0, 0.50)

        for label, r in results.items():
            c = PALETTE.get(label, "#888")
            ls = "--" if label == "No defense" else "-"
            lw = 2.5 if label == "No defense" else 1.8
            ax.plot(r["fpr_s"], r["tpr_s"], color=c, lw=lw, ls=ls,
                    label=f"{label}  (AUC={r['auc_s']:.3f})")

        ax.plot([0,1],[0,1], "k:", lw=1, alpha=0.3)
        ax.axvline(0.01, color="gray", lw=0.7, ls=":", alpha=0.5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title("Full ROC" if ax_i == 0 else "Zoom: FPR 0–5%", fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
        if ax_i == 0:
            ax.legend(fontsize=7.5, loc="lower right")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(out)}")


# =============================================================================
# FIG 2: Privacy-Utility Scatter — the key tradeoff plot
# =============================================================================

def fig2_scatter(results, out):
    """
    x = holdout accuracy (utility), y = shadow AUC (privacy risk).
    Ideal defense is bottom-right: high utility, low AUC.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, r in results.items():
        c = PALETTE.get(label, "#888")
        marker = "D" if "DP" in label else "o"
        size = 160 if label == "No defense" else 120
        ax.scatter(r["non_acc"], r["auc_s"], s=size, color=c,
                   marker=marker, edgecolors="white", linewidth=1.5, zorder=3)
        # Offset labels to avoid overlap
        offset_x, offset_y = 0.008, 0.008
        if "DP" in label:
            offset_y = -0.015
        ax.annotate(label, (r["non_acc"], r["auc_s"]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=8.5, color=c, fontweight="bold")

    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.3,
               label="Random baseline (AUC=0.5)")
    ax.set_xlabel("Holdout Accuracy (Utility) →", fontsize=11)
    ax.set_ylabel("← Shadow Attack AUC (Privacy Risk)", fontsize=11)
    ax.set_title("Privacy–Utility Trade-off\n"
                 "Ideal: bottom-right (high utility, low attack AUC)", fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")

    # Shade the "ideal zone"
    ax.axhspan(0.48, 0.55, alpha=0.06, color="green")
    ax.axvspan(0.65, 0.80, alpha=0.06, color="green")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(out)}")


# =============================================================================
# FIG 3: Per-Class AUC Heatmap
# =============================================================================

def fig3_heatmap(dfs, clf, out):
    labels = list(dfs.keys())
    data = np.zeros((len(labels), 10))

    for i, label in enumerate(labels):
        df = dfs[label]
        yt = (df["split_name"] == "member").astype(int).values
        xf, _ = prepare_features(df)
        scores = clf.predict_proba(xf)[:, 1]

        for cls_id in range(10):
            mask = (df["true_label"] == cls_id).values
            sub_y, sub_s = yt[mask], scores[mask]
            if sub_y.sum() == 0 or sub_y.sum() == mask.sum():
                data[i, cls_id] = np.nan
            else:
                fp, tp, _ = roc_curve(sub_y, sub_s, pos_label=1)
                data[i, cls_id] = auc(fp, tp)

    fig, ax = plt.subplots(figsize=(12, max(3.5, len(labels) * 0.65 + 1)))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0.42, vmax=0.88)

    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(10):
            val = data[i, j]
            txt = f"{val:.2f}" if not np.isnan(val) else "—"
            color = "white" if (not np.isnan(val) and (val > 0.75 or val < 0.47)) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Shadow Attack AUC", fontsize=10)
    ax.set_title("Per-Class MIA Vulnerability Across Defenses\n"
                 "(Red = vulnerable, Green = protected)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(out)}")


# =============================================================================
# FIG 4: Radar / Spider Chart — defense profiles
# =============================================================================

def fig4_radar(results, out):
    """
    Each defense gets a polygon on the radar chart.
    Axes: Privacy (1-AUC), Utility (holdout acc), Low Overfit,
          Low TPR@1%, Low Advantage.
    All normalised to [0,1] where 1 = best.
    """
    categories = ["Privacy\n(1−AUC)", "Utility\n(Holdout acc)",
                  "Low Overfit\n(1−gap)", "Low TPR@1%\n(1−TPR)",
                  "Low Advantage\n(1−adv)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for label, r in results.items():
        if "DP" in label and "ε=5" in label:
            continue  # skip ε=5 to reduce clutter (ε=1 and ε=10 enough)

        values = [
            1 - r["auc_s"],                    # privacy: lower AUC = better
            r["non_acc"],                       # utility: higher = better
            1 - max(0, r["gap"]),               # low overfit: lower gap = better
            1 - r["tpr1_s"],                    # low TPR@1%: lower = better
            1 - r["advantage"],                 # low advantage: lower = better
        ]
        values += values[:1]

        c = PALETTE.get(label, "#888")
        ls = "--" if label == "No defense" else "-"
        ax.plot(angles, values, color=c, lw=2, ls=ls, label=label)
        ax.fill(angles, values, color=c, alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="gray")
    ax.set_title("Defense Profile Comparison\n(larger area = better overall)", fontsize=12, y=1.08)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(out)}")


# =============================================================================
# FIG 5: Violin Plots — signal distributions member vs non-member
# =============================================================================

def fig5_violin(dfs, out):
    """
    Violin plots showing logit_gap distribution for members vs non-members
    across selected defenses. The overlap between the two violins shows
    how well the defense masks membership.
    """
    # Pick representative configs
    show = ["No defense", "Regularised", "Early stop", "DP  ε=1"]
    show = [s for s in show if s in dfs]

    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 5), sharey=True)
    if len(show) == 1:
        axes = [axes]

    fig.suptitle("logit_gap Distribution: Member vs Non-member", fontsize=13, y=1.02)

    for ax, label in zip(axes, show):
        df = dfs[label]
        mem = df[df.split_name == "member"]["logit_gap"].values
        non = df[df.split_name == "nonmember"]["logit_gap"].values

        # Clip outliers for cleaner display
        lo, hi = np.percentile(np.concatenate([mem, non]), [1, 99])
        mem_c = mem[(mem >= lo) & (mem <= hi)]
        non_c = non[(non >= lo) & (non <= hi)]

        parts = ax.violinplot([mem_c, non_c], positions=[0, 1],
                              showmeans=True, showmedians=True)

        colors = ["#3B8BD4", "#D85A30"]
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(1)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Member", "Non-member"], fontsize=10)
        ax.set_title(label, fontsize=11, color=PALETTE.get(label, "#333"))
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("logit_gap", fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(out)}")


# =============================================================================
# FIG 6: Grouped Bar Chart — overfit gap vs attack AUC per defense
# =============================================================================

def fig6_gap_vs_auc(results, out):
    """
    Side-by-side bars: overfit gap and shadow AUC for each defense.
    Shows the direct relationship between memorisation and vulnerability.
    """
    labels = list(results.keys())
    gaps = [max(0, results[l]["gap"]) for l in labels]
    aucs = [results[l]["auc_s"] for l in labels]
    colors = [PALETTE.get(l, "#888") for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(13, 5.5))

    bars1 = ax1.bar(x - width/2, gaps, width, label="Overfit gap (mem−non acc)",
                    color=colors, alpha=0.55, edgecolor="none")
    bars2 = ax1.bar(x + width/2, aucs, width, label="Shadow attack AUC",
                    color=colors, alpha=0.90, edgecolor="none")

    ax1.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.3,
                label="Random AUC baseline")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Value", fontsize=11)
    ax1.set_ylim(0, 0.85)
    ax1.set_title("Overfit Gap vs Attack AUC Per Defense\n"
                  "(gap drives vulnerability — they move together)", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars2, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(out)}")


# =============================================================================
# FIG 7: Training Curves — loss and overfit gap over epochs
# =============================================================================

def fig7_training_curves(out):
    """
    Plot training loss and overfit gap across epochs for each defense
    that has a training log. Shows HOW each defense evolves during training.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Dynamics Across Defenses", fontsize=13)

    for label, fname in TRAIN_LOGS.items():
        path = os.path.join(LOGS_DIR, fname)
        if not os.path.exists(path):
            continue
        log = pd.read_csv(path)
        c = PALETTE.get(label, "#888")
        ls = "--" if label == "No defense" else "-"

        ax1.plot(log["epoch"], log["train_loss"], color=c, lw=1.5, ls=ls,
                 label=label)

        if "overfit_gap" in log.columns:
            ax2.plot(log["epoch"], log["overfit_gap"], color=c, lw=1.5, ls=ls,
                     label=label)
        elif "member_eval_acc" in log.columns and "holdout_acc" in log.columns:
            gap = log["member_eval_acc"].astype(float) - log["holdout_acc"].astype(float)
            ax2.plot(log["epoch"], gap, color=c, lw=1.5, ls=ls, label=label)

    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Training Loss", fontsize=10)
    ax1.set_title("Training Loss Over Epochs", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Overfit Gap (member − holdout acc)", fontsize=10)
    ax2.set_title("Memorisation (Overfit Gap) Over Epochs", fontsize=11)
    ax2.axhline(0.08, color="gray", lw=1, ls=":", alpha=0.5,
                label="Early stop threshold (0.08)")
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(out)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("GENERATING REPORT FIGURES")
    print("=" * 60)

    dfs = load_configs()
    clf = get_attack_clf()
    results = evaluate_all(dfs, clf)

    print(f"\nLoaded {len(dfs)} configs: {', '.join(dfs.keys())}\n")

    print("Generating figures:")
    fig1_roc(results, os.path.join(FIG_DIR, "fig1_roc_all_defenses.png"))
    fig2_scatter(results, os.path.join(FIG_DIR, "fig2_privacy_utility_scatter.png"))
    fig3_heatmap(dfs, clf, os.path.join(FIG_DIR, "fig3_heatmap_per_class.png"))
    fig4_radar(results, os.path.join(FIG_DIR, "fig4_radar_chart.png"))
    fig5_violin(dfs, os.path.join(FIG_DIR, "fig5_signal_violin.png"))
    fig6_gap_vs_auc(results, os.path.join(FIG_DIR, "fig6_overfit_gap_vs_auc.png"))
    fig7_training_curves(os.path.join(FIG_DIR, "fig7_training_curves.png"))

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
