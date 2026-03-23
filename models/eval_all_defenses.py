# =============================================================================
# CS6413: Final Defense Comparison — No-DP vs Regularised vs DP-SGD
# =============================================================================
# PURPOSE:
#   Produce the final summary table and comparison plots for the report,
#   covering all defense configurations:
#     - No defense (baseline)
#     - Heuristic defense (L2 + label smoothing + dropout)
#     - DP-SGD at ε = 1, 5, 10
#
#   Uses the same shadow attack classifier (trained once on shadow_features_v2.csv)
#   applied consistently across all configs — realistic threat model.
#
# INPUTS:
#   outputs/logs/per_sample_mia_v2.csv
#   outputs/logs/per_sample_mia_regularised.csv
#   outputs/logs/per_sample_mia_dp_eps{1,5,10}.csv
#   outputs/reports/attack_shadow_v2/shadow_features_v2.csv
#
# OUTPUTS:
#   outputs/reports/final/final_comparison.png
#   outputs/reports/final/final_summary.txt
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LOGS_DIR   = os.path.join(PROJECT_ROOT, "outputs", "logs")
SHADOW_CSV = os.path.join(PROJECT_ROOT, "outputs", "reports",
                          "attack_shadow_v2", "shadow_features_v2.csv")
REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports", "final")
os.makedirs(REPORT_DIR, exist_ok=True)

# All configs: label → csv filename
CONFIGS = {
    "No defense"   : "per_sample_mia_v2.csv",
    "Regularised"  : "per_sample_mia_regularised.csv",
    "DP  ε=10"     : "per_sample_mia_dp_eps10.csv",
    "DP  ε=5"      : "per_sample_mia_dp_eps5.csv",
    "DP  ε=1"      : "per_sample_mia_dp_eps1.csv",
}

# Colours per config for consistent plotting
PALETTE = {
    "No defense" : "#2C2C2A",
    "Regularised": "#1D9E75",
    "DP  ε=10"   : "#185FA5",
    "DP  ε=5"    : "#BA7517",
    "DP  ε=1"    : "#D85A30",
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
    x["loss"]    = -x["loss"]
    x["entropy"] = -x["entropy"]
    y = (df["split_name"] == "member").astype(int).values
    return x.values.astype(np.float64), y


def tpr_at_fpr(fpr_arr, tpr_arr, max_fpr):
    mask = fpr_arr <= max_fpr
    return float(tpr_arr[mask].max()) if mask.any() else 0.0


def evaluate(csv_path, attack_clf):
    df     = pd.read_csv(csv_path)
    y_true = (df["split_name"] == "member").astype(int).values

    # Threshold attack — logit_gap
    fpr_t, tpr_t, _ = roc_curve(y_true, df["logit_gap"].values, pos_label=1)
    auc_thresh       = auc(fpr_t, tpr_t)
    tpr1_thresh      = tpr_at_fpr(fpr_t, tpr_t, 0.01)

    # Shadow attack — learned classifier
    x_feat, _ = prepare_features(df)
    score      = attack_clf.predict_proba(x_feat)[:, 1]
    fpr_s, tpr_s, thresholds = roc_curve(y_true, score, pos_label=1)
    auc_shadow  = auc(fpr_s, tpr_s)
    tpr1_shadow = tpr_at_fpr(fpr_s, tpr_s, 0.01)

    best_acc  = max(accuracy_score(y_true, (score >= t).astype(int))
                    for t in thresholds)
    advantage = 2 * (best_acc - 0.5)

    mem_acc = df[df.split_name == "member"]["correct"].mean()
    non_acc = df[df.split_name == "nonmember"]["correct"].mean()

    return dict(
        auc_thresh=auc_thresh,   tpr1_thresh=tpr1_thresh,
        auc_shadow=auc_shadow,   tpr1_shadow=tpr1_shadow,
        advantage=advantage,
        mem_acc=mem_acc,         non_acc=non_acc,
        overfit_gap=mem_acc - non_acc,
        fpr_s=fpr_s,             tpr_s=tpr_s,
    )


# =============================================================================
# PLOTS
# =============================================================================

def plot_final(results, out_path):
    """
    Three-panel figure:
      Left   — ROC curves (shadow attack) for all configs
      Centre — AUC bar chart
      Right  — Utility (holdout acc) bar chart
    """
    configs  = list(results.keys())
    colors   = [PALETTE[c] for c in configs]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Defense comparison — privacy vs utility", fontsize=13)

    # ── Panel 1: ROC curves ───────────────────────────────────────────────
    ax = axes[0]
    for cfg, res in results.items():
        ls = "--" if cfg == "No defense" else "-"
        ax.plot(res["fpr_s"], res["tpr_s"],
                color=PALETTE[cfg], lw=2, ls=ls,
                label=f"{cfg}  ({res['auc_shadow']:.3f})")
    ax.plot([0,1],[0,1],"k:",lw=1,alpha=0.3)
    ax.axvline(0.01, color="gray", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Shadow attack ROC", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)

    # ── Panel 2: Attack AUC bar ───────────────────────────────────────────
    ax2 = axes[1]
    aucs = [results[c]["auc_shadow"] for c in configs]
    bars = ax2.bar(range(len(configs)), aucs, color=colors,
                   edgecolor="none", width=0.6)
    ax2.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.4,
                label="Random baseline")
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    ax2.set_ylim(0.4, 0.85)
    ax2.set_ylabel("Shadow attack AUC")
    ax2.set_title("Attack AUC per config\n(lower = better privacy)", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # ── Panel 3: Utility (non-member acc) bar ────────────────────────────
    ax3 = axes[2]
    non_accs = [results[c]["non_acc"] for c in configs]
    bars3 = ax3.bar(range(len(configs)), non_accs, color=colors,
                    edgecolor="none", width=0.6)
    ax3.axhline(results["No defense"]["non_acc"],
                color="#2C2C2A", lw=1, ls=":", alpha=0.5,
                label="No-defense baseline")
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    ax3.set_ylim(0.0, 0.85)
    ax3.set_ylabel("Holdout accuracy (utility)")
    ax3.set_title("Model utility per config\n(higher = better utility)", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars3, non_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Final comparison plot saved → {out_path}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def write_summary(results, out_path):
    lines = [
        "=" * 82,
        "FINAL DEFENSE COMPARISON SUMMARY",
        "=" * 82,
        "",
        f"{'Config':<16} {'Thresh AUC':>11} {'Shadow AUC':>11} "
        f"{'TPR@1%(shd)':>12} {'Advantage':>10} "
        f"{'Holdout acc':>12} {'Overfit gap':>12}",
        "-" * 82,
    ]
    for cfg, r in results.items():
        lines.append(
            f"{cfg:<16} "
            f"{r['auc_thresh']:>11.4f} "
            f"{r['auc_shadow']:>11.4f} "
            f"{r['tpr1_shadow']:>12.4f} "
            f"{r['advantage']:>10.4f} "
            f"{r['non_acc']:>12.4f} "
            f"{r['overfit_gap']:>12.4f}"
        )
    lines += [
        "",
        "Key findings:",
        "  Privacy  : Shadow AUC closer to 0.5 = better privacy",
        "  Utility  : Holdout acc closer to no-defense = better utility",
        "  Best tradeoff = high holdout acc AND low shadow AUC",
        "",
        "=" * 82,
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nSummary saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("FINAL DEFENSE EVALUATION — ALL CONFIGS")
    print("=" * 60)

    # Train attack classifier once on shadow data
    print("\nTraining attack classifier on shadow features...")
    shadow_df = pd.read_csv(SHADOW_CSV)
    x_shd, y_shd = prepare_features(shadow_df)
    attack_clf = LogisticRegression(
        max_iter=2000, solver="lbfgs", class_weight="balanced")
    attack_clf.fit(x_shd, y_shd)
    print("Classifier ready.\n")

    # Evaluate all configs
    results = {}
    for label, fname in CONFIGS.items():
        csv_path = os.path.join(LOGS_DIR, fname)
        if not os.path.exists(csv_path):
            print(f"  SKIP (not found): {fname}")
            continue
        print(f"Evaluating: {label}")
        results[label] = evaluate(csv_path, attack_clf)

    if len(results) < 2:
        print("\nNeed at least 2 configs. Run train_regularised.py first.")
        return

    plot_final(results, os.path.join(REPORT_DIR, "final_comparison.png"))
    write_summary(results, os.path.join(REPORT_DIR, "final_summary.txt"))
    print(f"\nAll outputs in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
