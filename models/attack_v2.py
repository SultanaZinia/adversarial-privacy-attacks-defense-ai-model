# =============================================================================
# CS6413: MIA Threshold Attack v2 — Logit-Space Signals
# =============================================================================
# PURPOSE:
#   Run the threshold attack using the logit-space signals extracted by
#   extract_logits.py.  These bypass the softmax saturation problem and
#   produce clean separation at the distribution tails where tight-threshold
#   (low-FPR) attacks must operate.
#
# KEY DIFFERENCE FROM v1:
#   v1 used softmax outputs  → saturates at 1.0, TPR@1%FPR = 0
#   v2 adds logit_gap and m_entropy → no saturation, tail is separable
#
# INPUT:  outputs/logs/per_sample_mia_v2.csv
# OUTPUT: outputs/reports/attack_v2/
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, accuracy_score

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MIA_CSV_V2  = os.path.join(PROJECT_ROOT, "outputs", "logs",
                            "per_sample_mia_v2.csv")
REPORT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "reports", "attack_v2")
os.makedirs(REPORT_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# =============================================================================
# LOAD
# =============================================================================

def load_data(csv_path):
    df     = pd.read_csv(csv_path)
    y_true = (df["split_name"] == "member").astype(int).values
    n_mem  = y_true.sum()
    n_non  = len(y_true) - n_mem
    print(f"Loaded {len(df):,} samples  ({n_mem:,} members, {n_non:,} non-members)")

    signals = {
        # ── v1 signals (softmax-based) ────────────────────────────────────
        "conf_true"        :  df["conf_true"].values,
        "loss (negated)"   : -df["loss"].values,
        "entropy (negated)": -df["entropy"].values,
        # ── v2 signals (logit-space) ──────────────────────────────────────
        # logit_gap: margin between true-class logit and runner-up
        # no saturation → full range of values at the tail
        "logit_gap"        :  df["logit_gap"].values,
        # logit_true: raw true-class logit (monotone with confidence but unsaturated)
        "logit_true"       :  df["logit_true"].values,
        # modified entropy (Song & Mittal 2021):
        # -entropy for correct preds, +entropy for incorrect preds
        # pushes members (mostly correct + low entropy) to high values
        "m_entropy"        :  df["m_entropy"].values,
    }
    return df, y_true, signals


# =============================================================================
# THRESHOLD ATTACK — single signal
# =============================================================================

def run_attack(signal, y_true, name):
    fpr, tpr, thresholds = roc_curve(y_true, signal, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # TPR at 1% FPR
    mask        = fpr <= 0.01
    tpr_at_1fpr = float(tpr[mask].max()) if mask.any() else 0.0

    # TPR at 0.1% FPR (stricter)
    mask_01     = fpr <= 0.001
    tpr_at_01fpr = float(tpr[mask_01].max()) if mask_01.any() else 0.0

    # Best accuracy + advantage
    best_acc, best_τ = 0.0, thresholds[0]
    for τ in thresholds:
        acc = accuracy_score(y_true, (signal >= τ).astype(int))
        if acc > best_acc:
            best_acc, best_τ = acc, τ
    advantage = 2 * (best_acc - 0.5)

    print(f"\n  [{name}]")
    print(f"    AUC          : {roc_auc:.4f}")
    print(f"    TPR@0.1%FPR  : {tpr_at_01fpr:.4f}  ({tpr_at_01fpr*100:.1f}%)")
    print(f"    TPR@1%FPR    : {tpr_at_1fpr:.4f}  ({tpr_at_1fpr*100:.1f}%)")
    print(f"    Best acc     : {best_acc:.4f}  (threshold={best_τ:.4f})")
    print(f"    Advantage    : {advantage:.4f}")

    return dict(name=name, fpr=fpr, tpr=tpr, thresholds=thresholds,
                roc_auc=roc_auc, tpr_at_1fpr=tpr_at_1fpr,
                tpr_at_01fpr=tpr_at_01fpr,
                best_acc=best_acc, best_τ=best_τ, advantage=advantage)


# =============================================================================
# PLOT: distribution comparison v1 vs v2 signals (side-by-side)
# =============================================================================

def plot_distributions_comparison(df, out_path):
    """
    Compare softmax signal (conf_true) vs logit_gap side by side.
    The point: both humps pile up at conf_true≈1.0; logit_gap separates them.
    """
    mem = df[df.split_name == "member"]
    non = df[df.split_name == "nonmember"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Why logit signals fix the saturation problem", fontsize=13, y=1.02)

    specs = [
        ("conf_true",  "conf_true  (v1 — saturates at 1.0)",  False),
        ("logit_gap",  "logit_gap  (v2 — no saturation)",      True),
        ("m_entropy",  "m_entropy  (v2 — modified entropy)",   True),
    ]

    for ax, (col, title, clip) in zip(axes, specs):
        vals_m = mem[col].values
        vals_n = non[col].values
        if clip:
            lo = np.percentile(np.concatenate([vals_m, vals_n]), 1)
            hi = np.percentile(np.concatenate([vals_m, vals_n]), 99)
            rng = (lo, hi)
        else:
            rng = None

        ax.hist(vals_m, bins=80, alpha=0.55, density=True,
                color="#3B8BD4", label="member",     edgecolor="none",
                range=rng)
        ax.hist(vals_n, bins=80, alpha=0.55, density=True,
                color="#D85A30", label="non-member", edgecolor="none",
                range=rng)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distribution comparison saved → {out_path}")


# =============================================================================
# PLOT: ROC curves — v1 vs v2 on same axes
# =============================================================================

def plot_roc_comparison(results_v1, results_v2, out_path):
    """
    Overlay v1 (softmax) and v2 (logit) ROC curves.
    Two insets: overall zoom and tight low-FPR (0–1%) view.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ROC curves — v1 (softmax) vs v2 (logit-space signals)", fontsize=13)

    # Colour mapping
    v1_palette = {"conf_true": "#B5D4F4", "loss (negated)": "#B5D4F4",
                  "entropy (negated)": "#B5D4F4"}
    v2_palette = {"logit_gap": "#185FA5", "logit_true": "#0F6E56",
                  "m_entropy": "#BA7517", "conf_true": "#B5D4F4",
                  "loss (negated)": "#C0D0E0", "entropy (negated)": "#C0D0E0"}

    for ax_idx, ax in enumerate(axes):
        fpr_lim = (0, 1.0) if ax_idx == 0 else (0, 0.05)
        tpr_lim = (0, 1.05) if ax_idx == 0 else (0, 0.70)

        # v1 in muted blue
        for r in results_v1:
            ax.plot(r["fpr"], r["tpr"],
                    color="#B5D4F4", lw=1.5, ls="--", alpha=0.7,
                    label=f"v1: {r['name']}  (AUC={r['roc_auc']:.3f})")

        # v2 in strong colours
        v2_colours = ["#185FA5", "#0F6E56", "#BA7517"]
        for r, c in zip(results_v2, v2_colours):
            ax.plot(r["fpr"], r["tpr"], color=c, lw=2,
                    label=f"v2: {r['name']}  (AUC={r['roc_auc']:.3f})")

        ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.35,label="Random (0.500)")
        ax.axvline(0.01, color="gray", lw=0.8, ls=":", alpha=0.6)

        ax.set_xlim(*fpr_lim)
        ax.set_ylim(*tpr_lim)
        ax.set_xlabel("FPR (non-members wrongly flagged)")
        ax.set_ylabel("TPR (members correctly identified)")
        title = "Full ROC" if ax_idx == 0 else "Zoom: FPR 0–5%  (practical regime)"
        ax.set_title(title, fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC comparison saved → {out_path}")


# =============================================================================
# PLOT: per-class AUC for best signal
# =============================================================================

def plot_per_class(df, signal_col, out_path):
    y_all = (df["split_name"] == "member").astype(int).values
    sig   = df[signal_col].values

    class_aucs = []
    for cls_id in range(10):
        mask = (df["true_label"] == cls_id).values
        sub_y   = y_all[mask]
        sub_sig = sig[mask]
        if sub_y.sum() == 0 or sub_y.sum() == mask.sum():
            class_aucs.append(float("nan"))
            continue
        fp, tp, _ = roc_curve(sub_y, sub_sig, pos_label=1)
        class_aucs.append(auc(fp, tp))

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["#185FA5" if a >= 0.70 else "#D85A30" if a < 0.60 else "#888780"
              for a in class_aucs]
    bars = ax.bar(CIFAR10_CLASSES, class_aucs, color=colors,
                  edgecolor="none", width=0.6)
    ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.4)
    ax.axhline(0.7, color="#1D9E75", lw=0.8, ls=":", alpha=0.5)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Per-class attack AUC  ({signal_col})", fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, class_aucs):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class AUC saved → {out_path}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def write_summary(all_results, out_path):
    lines = [
        "=" * 75,
        "THRESHOLD MIA v2 — EVALUATION SUMMARY",
        "=" * 75,
        "",
        f"{'Signal':<28} {'Ver':>4} {'AUC':>7} {'TPR@0.1%':>10} "
        f"{'TPR@1%FPR':>11} {'Best acc':>10} {'Advantage':>11}",
        "-" * 75,
    ]
    for r in all_results:
        ver = "v2" if r["name"] in ("logit_gap","logit_true","m_entropy") else "v1"
        lines.append(
            f"{r['name']:<28} {ver:>4} "
            f"{r['roc_auc']:>7.4f} "
            f"{r['tpr_at_01fpr']:>10.4f} "
            f"{r['tpr_at_1fpr']:>11.4f} "
            f"{r['best_acc']:>10.4f} "
            f"{r['advantage']:>11.4f}"
        )
    lines += ["", "=" * 75]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(out_path, "w") as f:
        f.write(txt + "\n")
    print(f"\nSummary saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("THRESHOLD MIA ATTACK v2 — LOGIT-SPACE SIGNALS")
    print("=" * 60)

    df, y_true, signals = load_data(MIA_CSV_V2)

    # ── Run all attacks ───────────────────────────────────────────────────
    print("\n[1/4] Running threshold attacks...")
    v1_names = ["conf_true", "loss (negated)", "entropy (negated)"]
    v2_names = ["logit_gap", "logit_true", "m_entropy"]

    all_results = []
    for name, sig in signals.items():
        all_results.append(run_attack(sig, y_true, name))

    results_v1 = [r for r in all_results if r["name"] in v1_names]
    results_v2 = [r for r in all_results if r["name"] in v2_names]

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[2/4] Plotting distribution comparison...")
    plot_distributions_comparison(
        df, os.path.join(REPORT_DIR, "distributions_v2.png"))

    print("\n[3/4] Plotting ROC comparison...")
    plot_roc_comparison(
        results_v1, results_v2,
        os.path.join(REPORT_DIR, "roc_comparison.png"))

    # Per-class for best v2 signal
    best_v2 = max(results_v2, key=lambda r: r["roc_auc"])
    plot_per_class(df, best_v2["name"],
                   os.path.join(REPORT_DIR, "per_class_v2.png"))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[4/4] Writing summary...")
    write_summary(all_results, os.path.join(REPORT_DIR, "summary_v2.txt"))

    print(f"\nAll outputs in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
