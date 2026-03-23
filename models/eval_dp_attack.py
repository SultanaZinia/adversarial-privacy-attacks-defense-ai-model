# =============================================================================
# CS6413: DP Defense Evaluation — Attack AUC Before vs After DP
# =============================================================================
# PURPOSE:
#   Load the per_sample_mia_dp_eps{ε}.csv files produced by train_dp.py,
#   re-run both the threshold attack AND the shadow attack classifier
#   (trained on shadow data, not re-trained per ε) on each DP model,
#   and produce a single comparison table + plot.
#
# WHY THE SAME SHADOW CLASSIFIER?
#   In the real threat model the attacker trains their attack classifier
#   ONCE on a shadow model and then uses it against any target. Re-training
#   the attack classifier per ε would be unrealistic and would understate
#   how well DP defends against a fixed attacker.
#
# INPUTS:
#   outputs/logs/per_sample_mia_v2.csv          (no-DP baseline)
#   outputs/logs/per_sample_mia_dp_eps{ε}.csv   (one per ε)
#   outputs/reports/attack_shadow_v2/shadow_features_v2.csv
#
# OUTPUTS:
#   outputs/reports/dp_defense/dp_attack_comparison.png
#   outputs/reports/dp_defense/dp_attack_summary.txt
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

LOGS_DIR      = os.path.join(PROJECT_ROOT, "outputs", "logs")
SHADOW_CSV    = os.path.join(PROJECT_ROOT, "outputs", "reports",
                             "attack_shadow_v2", "shadow_features_v2.csv")
REPORT_DIR    = os.path.join(PROJECT_ROOT, "outputs", "reports", "dp_defense")
os.makedirs(REPORT_DIR, exist_ok=True)

BASELINE_CSV  = os.path.join(LOGS_DIR, "per_sample_mia_v2.csv")
EPSILONS      = [1, 5, 10]


# =============================================================================
# HELPERS
# =============================================================================

def prepare_features(df):
    """Same feature preparation as attack_shadow_v2.py — must match exactly."""
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


def evaluate_csv(csv_path, attack_clf):
    """
    Run both threshold (logit_gap) and shadow (LR classifier) attacks
    on the features in csv_path. Returns a dict of metrics.
    """
    df     = pd.read_csv(csv_path)
    y_true = (df["split_name"] == "member").astype(int).values

    # ── Threshold attack (logit_gap) ──────────────────────────────────────
    thresh_score    = df["logit_gap"].values
    fpr_t, tpr_t, _ = roc_curve(y_true, thresh_score, pos_label=1)
    auc_thresh       = auc(fpr_t, tpr_t)
    tpr1_thresh      = tpr_at_fpr(fpr_t, tpr_t, 0.01)

    # ── Shadow attack (trained classifier) ───────────────────────────────
    x_feat, _ = prepare_features(df)
    score      = attack_clf.predict_proba(x_feat)[:, 1]
    fpr_s, tpr_s, thresholds = roc_curve(y_true, score, pos_label=1)
    auc_shadow  = auc(fpr_s, tpr_s)
    tpr1_shadow = tpr_at_fpr(fpr_s, tpr_s, 0.01)

    best_acc = max(
        accuracy_score(y_true, (score >= t).astype(int))
        for t in thresholds
    )
    advantage = 2 * (best_acc - 0.5)

    # Also compute overfit gap from the CSV
    mem_acc = df[df.split_name == "member"]["correct"].mean()
    non_acc = df[df.split_name == "nonmember"]["correct"].mean()
    gap     = mem_acc - non_acc

    return dict(
        auc_thresh=auc_thresh,   tpr1_thresh=tpr1_thresh,
        auc_shadow=auc_shadow,   tpr1_shadow=tpr1_shadow,
        advantage=advantage,     overfit_gap=gap,
        mem_acc=mem_acc,         non_acc=non_acc,
        fpr_s=fpr_s,             tpr_s=tpr_s,
        fpr_t=fpr_t,             tpr_t=tpr_t,
    )


# =============================================================================
# PLOT: ROC curves for all configs on one figure
# =============================================================================

def plot_comparison(all_results, out_path):
    """
    Two panels:
      Left  — full ROC curves for shadow attack across all ε + baseline
      Right — bar chart: TPR@1%FPR and AUC side by side per config
    """
    configs = list(all_results.keys())
    palette = {
        "No DP":   "#2C2C2A",
        "ε = 10":  "#185FA5",
        "ε = 5":   "#1D9E75",
        "ε = 1":   "#D85A30",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DP-SGD defense — attack effectiveness before and after",
                 fontsize=13)

    # ── Left: ROC curves ──────────────────────────────────────────────────
    ax = axes[0]
    for cfg, res in all_results.items():
        color = palette.get(cfg, "#888780")
        ax.plot(res["fpr_s"], res["tpr_s"], color=color, lw=2,
                label=f"{cfg}  AUC={res['auc_shadow']:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.3,label="Random")
    ax.axvline(0.01, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Shadow attack ROC (all configs)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)

    # ── Right: metric bars ────────────────────────────────────────────────
    ax2 = axes[1]
    x       = np.arange(len(configs))
    width   = 0.35
    aucs    = [all_results[c]["auc_shadow"]  for c in configs]
    tprs    = [all_results[c]["tpr1_shadow"] for c in configs]
    colors  = [palette.get(c, "#888780") for c in configs]

    bars1 = ax2.bar(x - width/2, aucs, width, label="AUC (shadow)",
                    color=colors, alpha=0.85, edgecolor="none")
    bars2 = ax2.bar(x + width/2, tprs, width, label="TPR@1%FPR (shadow)",
                    color=colors, alpha=0.45, edgecolor="none")

    ax2.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.3,
                label="Random AUC baseline")
    ax2.set_xticks(x); ax2.set_xticklabels(configs)
    ax2.set_ylabel("Value"); ax2.set_ylim(0, 1.0)
    ax2.set_title("AUC and TPR@1%FPR per config", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.spines[["top","right"]].set_visible(False)

    for bar, val in zip(bars1, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, tprs):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved → {out_path}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def write_summary(all_results, out_path):
    lines = [
        "=" * 78,
        "DP-SGD DEFENSE — ATTACK EVALUATION SUMMARY",
        "=" * 78,
        "",
        f"{'Config':<12} {'Thresh AUC':>11} {'Shadow AUC':>11} "
        f"{'TPR@1%(thr)':>12} {'TPR@1%(shd)':>12} "
        f"{'Advantage':>10} {'Gap':>7}",
        "-" * 78,
    ]
    for cfg, r in all_results.items():
        lines.append(
            f"{cfg:<12} "
            f"{r['auc_thresh']:>11.4f} "
            f"{r['auc_shadow']:>11.4f} "
            f"{r['tpr1_thresh']:>12.4f} "
            f"{r['tpr1_shadow']:>12.4f} "
            f"{r['advantage']:>10.4f} "
            f"{r['overfit_gap']:>7.4f}"
        )
    lines += [
        "",
        "Gap = member_acc − nonmember_acc on the DP model.",
        "As ε decreases, expect: AUC → 0.5, TPR → 0, Gap → 0.",
        "",
        "=" * 78,
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
    print("=" * 55)
    print("DP DEFENSE EVALUATION")
    print("=" * 55)

    # ── Train attack classifier once on shadow features ───────────────────
    print("\nLoading shadow features and training attack classifier...")
    shadow_df  = pd.read_csv(SHADOW_CSV)
    x_shd, y_shd = prepare_features(shadow_df)
    attack_clf = LogisticRegression(
        max_iter=2000, solver="lbfgs", class_weight="balanced")
    attack_clf.fit(x_shd, y_shd)
    print("Attack classifier ready.")

    # ── Evaluate baseline (no DP) ─────────────────────────────────────────
    print(f"\nEvaluating: No DP  ({BASELINE_CSV})")
    all_results = {}
    all_results["No DP"] = evaluate_csv(BASELINE_CSV, attack_clf)

    # ── Evaluate each ε ───────────────────────────────────────────────────
    for eps in sorted(EPSILONS, reverse=True):
        csv_path = os.path.join(LOGS_DIR, f"per_sample_mia_dp_eps{eps}.csv")
        if not os.path.exists(csv_path):
            print(f"  MISSING: {csv_path} — skipping ε={eps}")
            print(f"  Run train_dp.py first.")
            continue
        print(f"\nEvaluating: ε={eps}  ({csv_path})")
        all_results[f"ε = {eps}"] = evaluate_csv(csv_path, attack_clf)

    if len(all_results) < 2:
        print("\nNot enough results to plot. Run train_dp.py first.")
        return

    # ── Plots and summary ─────────────────────────────────────────────────
    plot_comparison(all_results,
                    os.path.join(REPORT_DIR, "dp_attack_comparison.png"))
    write_summary(all_results,
                  os.path.join(REPORT_DIR, "dp_attack_summary.txt"))

    print(f"\nAll outputs in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
