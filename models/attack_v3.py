# =============================================================================
# CS6413: MIA Threshold Attack v3 — Logit-Space Signals (Balanced)
# =============================================================================
# PURPOSE:
#   Run the threshold attack using the v3-trained model's logit-space signals.
#   v3 (Balanced & Fast) uses:
#     - Moderate-capacity CNN (approx. 0.8M params)
#     - 60 epochs (fast training on CPU/GPU)
#     - DATA AUGMENTATION: RandomCrop + RandomHorizontalFlip
#     - WEIGHT_DECAY=0.0005 (light regularization)
#     - BATCH_SIZE=128
#     - 6000 member set (standard size, same as v2)
#   This balanced approach aims for:
#     - Reasonable generalisation with visible train/holdout gap
#     - Logit gaps large enough for informative MIA signals
#
# INPUT:  outputs/logs/per_sample_mia_v3.csv  (from train_v3.py with balanced settings)
# OUTPUT: outputs/reports/attack_v3/
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

MIA_CSV_V3  = os.path.join(PROJECT_ROOT, "outputs", "logs",
                            "per_sample_mia_v3.csv")
REPORT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "reports", "attack_v3")
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
        # ── v2/v3 signals (logit-space) ───────────────────────────────────
        # logit_gap: margin between true-class logit and runner-up
        # Larger gaps in v3 due to stronger memorisation
        "logit_gap"        :  df["logit_gap"].values,
        # logit_true: raw true-class logit (monotone with confidence)
        "logit_true"       :  df["logit_true"].values,
        # modified entropy (Song & Mittal 2021)
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
# PLOT: distribution comparison v1 vs v3 signals
# =============================================================================

def plot_distributions_comparison(df, out_path):
    """
    Compare softmax (v1) vs logit-space (v3) signals.
    v3 model should show MORE SEPARATION than v2 due to stronger memorisation.
    """
    mem = df[df.split_name == "member"]
    non = df[df.split_name == "nonmember"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("v3 Balanced: Restored augmentation + weight decay + batch size optimization",
                 fontsize=13, y=1.02)

    specs = [
        ("conf_true",  "conf_true  (v1 — saturates at 1.0)",  False),
        ("logit_gap",  "logit_gap  (v3 — larger gaps)",       True),
        ("m_entropy",  "m_entropy  (v3 — modified entropy)",  True),
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
                color="#185FA5", label="member",     edgecolor="none",
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
# PLOT: ROC curves — all v3 signals
# =============================================================================

def plot_roc_curves(results, out_path):
    """
    Overlay all ROC curves for v3 signals.
    Inset: zoom on low-FPR region (0–5%) which is the practical attack regime.
    """
    fig = plt.figure(figsize=(10, 7))
    gs  = gridspec.GridSpec(1, 1)
    ax  = fig.add_subplot(gs[0])

    # Inset: zoom on low FPR
    ax_inset = ax.inset_axes([0.38, 0.08, 0.55, 0.42])

    palette = ["#185FA5", "#0F6E56", "#BA7517", "#B5D4F4", "#C0D0E0", "#888780"]
    signal_order = ["logit_gap", "logit_true", "m_entropy",
                    "conf_true", "loss (negated)", "entropy (negated)"]

    # Plot in specific order
    results_dict = {r["name"]: r for r in results}
    for i, signal_name in enumerate(signal_order):
        if signal_name in results_dict:
            res = results_dict[signal_name]
            color = palette[i % len(palette)]
            label = f"{res['name']}  (AUC={res['roc_auc']:.3f})"
            ax.plot(res["fpr"], res["tpr"], color=color, lw=2, label=label)
            ax_inset.plot(res["fpr"], res["tpr"], color=color, lw=1.5)

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC=0.500)")
    ax_inset.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)

    # 1% FPR reference line
    ax.axvline(0.01, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.text(0.013, 0.05, "1% FPR", fontsize=8, color="gray", rotation=90)

    ax.set_xlabel("False Positive Rate (FPR)\n(non-members wrongly labelled as member)",
                  fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR)\n(members correctly identified)", fontsize=11)
    ax.set_title("ROC curves — v3 model threshold MIA", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.spines[["top","right"]].set_visible(False)

    # Inset formatting
    ax_inset.set_xlim(0, 0.05)
    ax_inset.set_ylim(0, 0.8)
    ax_inset.set_title("Zoom: FPR 0–5%  (practical regime)", fontsize=8)
    ax_inset.axvline(0.01, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax_inset.spines[["top","right"]].set_visible(False)
    ax_inset.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve plot saved → {out_path}")


# =============================================================================
# PLOT: per-class AUC for best signal
# =============================================================================

def plot_per_class(df, signal_col, out_path):
    y_all = (df["split_name"] == "member").astype(int).values

    # Map display names to underlying CSV column names
    if signal_col == "loss (negated)":
        col_name = "loss"
    elif signal_col == "entropy (negated)":
        col_name = "entropy"
    else:
        col_name = signal_col

    sig = df[col_name].values

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
    ax.set_title(f"Per-class attack AUC (v3 model)  ({signal_col})", fontsize=12)
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
        "=" * 80,
        "THRESHOLD MIA v3 (BALANCED) — EVALUATION SUMMARY",
        "v3 Balanced: moderate CNN + 120 epochs + augmentation + weight_decay=0.0 + batch_size=128",
        "=" * 80,
        "",
        f"{'Signal':<28} {'Type':>6} {'AUC':>7} {'TPR@0.1%':>10} "
        f"{'TPR@1%FPR':>11} {'Best acc':>10} {'Advantage':>11}",
        "-" * 80,
    ]
    for r in all_results:
        signal_type = "logit" if r["name"] in ("logit_gap","logit_true","m_entropy") else "softmax"
        lines.append(
            f"{r['name']:<28} {signal_type:>6} "
            f"{r['roc_auc']:>7.4f} "
            f"{r['tpr_at_01fpr']:>10.4f} "
            f"{r['tpr_at_1fpr']:>11.4f} "
            f"{r['best_acc']:>10.4f} "
            f"{r['advantage']:>11.4f}"
        )
    
    best_auc_result = max(all_results, key=lambda r: r["roc_auc"])
    lines += [
        "",
        "=" * 80,
        f"BEST SIGNAL: {best_auc_result['name']} (AUC={best_auc_result['roc_auc']:.4f})",
        "",
        "TARGET METRICS FOR STRONG MIA:",
        "  ✓ AUC > 0.85       (strong separation)",
        "  ✓ TPR@1%FPR > 0.5  (identifies >50% of members at 1% false alarm)",
        f"  Current best: AUC={best_auc_result['roc_auc']:.4f}, "
        f"TPR@1%FPR={best_auc_result['tpr_at_1fpr']:.4f}",
        "=" * 80,
    ]
    
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(out_path, "w") as f:
        f.write(txt + "\n")
    print(f"\nSummary saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("THRESHOLD MIA ATTACK v3 (BALANCED) — LOGIT-SPACE SIGNALS")
    print("=" * 80)

    df, y_true, signals = load_data(MIA_CSV_V3)

    # ── Run all attacks ───────────────────────────────────────────────────
    print("\n[1/4] Running threshold attacks...")
    all_results = []
    for name, sig in signals.items():
        all_results.append(run_attack(sig, y_true, name))

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[2/4] Plotting distribution comparison...")
    plot_distributions_comparison(
        df, os.path.join(REPORT_DIR, "distributions_v3.png"))

    print("\n[3/4] Plotting ROC curves...")
    plot_roc_curves(all_results, os.path.join(REPORT_DIR, "roc_curves_v3.png"))

    # Per-class for best signal
    best_result = max(all_results, key=lambda r: r["roc_auc"])
    plot_per_class(df, best_result["name"],
                   os.path.join(REPORT_DIR, "per_class_v3.png"))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[4/4] Writing summary...")
    write_summary(all_results, os.path.join(REPORT_DIR, "summary_v3.txt"))

    print(f"\nAll outputs in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
