# PURPOSE:
#   Implement the classic threshold-based MIA (Yeom et al., 2018).
#   For each per-sample signal (loss, confidence, entropy) we sweep every
#   possible threshold and ask: "does this signal alone let an adversary
#   distinguish training members from non-members?"
#
# THEORY IN ONE PARAGRAPH:
#   A model trained with gradient descent memorises its training data.
#   Members (samples the model WAS trained on) typically have:
#       • lower cross-entropy loss
#       • higher softmax confidence on the true class
#       • lower prediction entropy
#   If we pick a threshold τ and predict "member" when loss < τ (or
#   confidence > τ), we get a binary classifier. Sweeping τ traces the
#   ROC curve. The area under that curve (AUC) measures how well the
#   signal separates members from non-members.
#   AUC = 0.5  →  no better than random guessing
#   AUC = 1.0  →  perfect attack (complete privacy failure)
#
# METRICS REPORTED:
#   • ROC-AUC    — overall attack power
#   • TPR@1%FPR  — practical privacy risk: at a 1% false alarm rate, how
#                  many real members does the attacker correctly identify?
#   • Attacker advantage = 2 × (best_attack_acc − 0.5)
#                  Range [0, 1]; 0 = no advantage; 1 = perfect
#
# INPUT:  outputs/logs/per_sample_mia.csv   (produced by train_baseline.py)
# OUTPUT: outputs/reports/threshold_attack/ (PNG plots + text summary)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, confusion_matrix
)


# =============================================================================
# SECTION 1: PATHS
# =============================================================================

# Resolve paths relative to THIS file so the script works from any cwd.
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MIA_CSV      = os.path.join(PROJECT_ROOT, "outputs", "logs", "per_sample_mia.csv")
REPORT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "reports", "threshold_attack")
os.makedirs(REPORT_DIR, exist_ok=True)


# =============================================================================
# SECTION 2: LOAD DATA
# =============================================================================

def load_mia_data(csv_path: str):
    """
    Load the per-sample MIA CSV and return features + binary membership labels.

    The adversary only has access to the model's *outputs* for a given sample
    — they do NOT know the true membership label (that's what they're trying
    to infer). Here we have ground truth labels so we can evaluate the attack.

    Returns
    -------
    df      : full DataFrame
    y_true  : np.ndarray of shape (N,), dtype int
              1 = member, 0 = nonmember
    signals : dict[str → np.ndarray]
              Each array is a scalar score per sample. Higher score = model
              predicts "member". We flip loss and entropy so ALL signals
              are oriented in the same direction (higher → more likely member).
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} samples  "
          f"({(df.split_name=='member').sum():,} members, "
          f"{(df.split_name=='nonmember').sum():,} non-members)")

    y_true = (df["split_name"] == "member").astype(int).values

    signals = {
        # Higher confidence on the true class → more likely a member
        "confidence (conf_true)" : df["conf_true"].values,
        # Higher max confidence → more likely a member
        "max confidence (conf_max)": df["conf_max"].values,
        # Lower loss → more likely a member  →  negate so higher = member
        "loss (negated)"         : -df["loss"].values,
        # Lower entropy → more likely a member  →  negate
        "entropy (negated)"      : -df["entropy"].values,
    }
    return df, y_true, signals


# =============================================================================
# SECTION 3: SINGLE-SIGNAL THRESHOLD ATTACK
# =============================================================================

def run_threshold_attack(signal: np.ndarray, y_true: np.ndarray,
                         signal_name: str) -> dict:
    """
    Sweep all thresholds for one signal and compute attack metrics.

    Parameters
    ----------
    signal      : model output score; higher → predict "member"
    y_true      : ground truth membership (1 = member, 0 = non-member)
    signal_name : human-readable name for printing

    Returns
    -------
    dict with keys: fpr, tpr, thresholds, roc_auc, tpr_at_1fpr,
                    best_acc, best_threshold, advantage, signal_name
    """
    # sklearn's roc_curve treats the POSITIVE class as "member" (label=1).
    # It returns fpr, tpr arrays and the threshold that produces each point.
    fpr, tpr, thresholds = roc_curve(y_true, signal, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # --- TPR at 1% FPR ---
    # Find the highest TPR achievable when FPR ≤ 0.01.
    # This is the most practically meaningful metric: at a 1-in-100 false
    # alarm rate, how many real members does the attacker catch?
    mask_1fpr  = fpr <= 0.01
    tpr_at_1fpr = tpr[mask_1fpr].max() if mask_1fpr.any() else 0.0

    # --- Best attack accuracy + attacker advantage ---
    # Sweep thresholds, predict member if signal ≥ τ, measure accuracy.
    best_acc, best_τ = 0.0, thresholds[0]
    for τ in thresholds:
        y_pred = (signal >= τ).astype(int)
        acc    = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_τ   = τ

    # Attacker advantage ∈ [0,1]:  0 = random, 1 = perfect
    advantage = 2 * (best_acc - 0.5)

    print(f"\n  {signal_name}")
    print(f"    ROC-AUC        : {roc_auc:.4f}")
    print(f"    TPR @ 1% FPR   : {tpr_at_1fpr:.4f}  "
          f"({tpr_at_1fpr*100:.1f}% of members identified at 1% false alarm)")
    print(f"    Best threshold : {best_τ:.6f}")
    print(f"    Best attack acc: {best_acc:.4f}")
    print(f"    Adv. (2×(acc-½)): {advantage:.4f}")

    return dict(
        signal_name   = signal_name,
        fpr           = fpr,
        tpr           = tpr,
        thresholds    = thresholds,
        roc_auc       = roc_auc,
        tpr_at_1fpr   = tpr_at_1fpr,
        best_acc      = best_acc,
        best_threshold= best_τ,
        advantage     = advantage,
    )


# =============================================================================
# SECTION 4: COMBINED SCORE ATTACK (OPTIONAL ENSEMBLE)
# =============================================================================

def combined_score(signals: dict) -> np.ndarray:
    """
    Combine all signals into one score via z-score normalisation + sum.

    Why: individual signals may capture different aspects of memorisation.
    Averaging normalised scores often beats any single signal.

    Note: this is still a threshold attack on the combined score — it
    doesn't require a shadow model or labelled attack data.
    """
    normed = []
    for arr in signals.values():
        mu, sigma = arr.mean(), arr.std() + 1e-9
        normed.append((arr - mu) / sigma)
    return np.stack(normed, axis=0).mean(axis=0)


# =============================================================================
# SECTION 5: CONFUSION MATRIX AT BEST THRESHOLD
# =============================================================================

def compute_confusion(signal: np.ndarray, y_true: np.ndarray,
                      threshold: float) -> np.ndarray:
    """Binary confusion matrix at a fixed threshold."""
    y_pred = (signal >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


# =============================================================================
# SECTION 6: DISTRIBUTION PLOT — member vs non-member overlap
# =============================================================================

def plot_distributions(df: pd.DataFrame, out_path: str):
    """
    For each raw signal, overlay the member and non-member distributions.
    The more separated the two humps, the more effective the attack.
    """
    raw_signals = {
        "loss"     : "loss",
        "conf_true": "conf_true",
        "conf_max" : "conf_max",
        "entropy"  : "entropy",
    }
    members    = df[df.split_name == "member"]
    nonmembers = df[df.split_name == "nonmember"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Signal distributions — member vs non-member", fontsize=14, y=1.01)
    colors = {"member": "#3B8BD4", "nonmember": "#D85A30"}

    for ax, (label, col) in zip(axes.flatten(), raw_signals.items()):
        ax.hist(members[col],    bins=60, alpha=0.6, density=True,
                color=colors["member"],    label="member",    edgecolor="none")
        ax.hist(nonmembers[col], bins=60, alpha=0.6, density=True,
                color=colors["nonmember"], label="non-member", edgecolor="none")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nDistribution plot saved → {out_path}")


# =============================================================================
# SECTION 7: ROC CURVE PLOT
# =============================================================================

def plot_roc_curves(results: list[dict], out_path: str):
    """
    Overlay ROC curves for all signals on one plot.

    The diagonal (AUC=0.5) is the random-guess baseline.
    Every curve above it represents a real privacy leak.
    The inset zooms into the low-FPR region (0–5%) which is the
    practically relevant regime for real-world attacks.
    """
    fig = plt.figure(figsize=(10, 7))
    gs  = gridspec.GridSpec(1, 1)
    ax  = fig.add_subplot(gs[0])

    # Inset: zoom on low FPR (0–0.05)
    ax_inset = ax.inset_axes([0.38, 0.08, 0.55, 0.42])

    palette = ["#3B8BD4", "#D85A30", "#1D9E75", "#BA7517", "#7F77DD"]

    for i, res in enumerate(results):
        color = palette[i % len(palette)]
        label = f"{res['signal_name']}  (AUC={res['roc_auc']:.3f})"
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
    ax.set_title("ROC curves — threshold membership inference attack", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.spines[["top","right"]].set_visible(False)

    # Inset formatting
    ax_inset.set_xlim(0, 0.05)
    ax_inset.set_ylim(0, 0.6)
    ax_inset.set_title("Zoom: FPR 0–5%", fontsize=8)
    ax_inset.axvline(0.01, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax_inset.spines[["top","right"]].set_visible(False)
    ax_inset.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve plot saved   → {out_path}")


# =============================================================================
# SECTION 8: PER-CLASS BREAKDOWN
# =============================================================================

def per_class_attack(df: pd.DataFrame, signal_col: str,
                     negate: bool, out_path: str):
    """
    Run the threshold attack separately for each of the 10 CIFAR-10 classes.
    Some classes are harder to memorise (e.g. deer, cat) so the attack
    may be stronger on certain classes.

    Reveals whether privacy risk is uniform or concentrated in some classes.
    """
    classes = [
        "airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"
    ]
    class_aucs = []
    for cls_id in range(10):
        sub = df[df.true_label == cls_id].copy()
        if len(sub) < 20:
            class_aucs.append(float("nan"))
            continue
        y   = (sub.split_name == "member").astype(int).values
        sig = (-sub[signal_col] if negate else sub[signal_col]).values
        if y.sum() == 0 or y.sum() == len(y):
            class_aucs.append(float("nan"))
            continue
        fpr, tpr, _ = roc_curve(y, sig, pos_label=1)
        class_aucs.append(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = ["#3B8BD4" if a >= 0.6 else "#D85A30" if a < 0.55 else "#888780"
               for a in class_aucs]
    bars = ax.bar(classes, class_aucs, color=colors, edgecolor="none", width=0.6)
    ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.4, label="Random baseline")
    ax.axhline(0.7, color="#1D9E75", lw=0.8, ls=":", alpha=0.5, label="AUC=0.70 reference")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Per-class attack AUC  ({signal_col})", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, class_aucs):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class AUC plot saved → {out_path}")


# =============================================================================
# SECTION 9: TEXT SUMMARY REPORT
# =============================================================================

def write_summary(results: list[dict], out_path: str):
    """Write a plain-text summary suitable for pasting into a report."""
    lines = [
        "=" * 70,
        "THRESHOLD MIA — EVALUATION SUMMARY",
        "=" * 70,
        "",
        "Metric definitions:",
        "  ROC-AUC    : Area under ROC curve. 0.5 = random, 1.0 = perfect.",
        "  TPR@1%FPR  : True positive rate when false positive rate ≤ 1%.",
        "               Practical question: of all members, what fraction does",
        "               the attacker correctly identify with only 1 in 100",
        "               false alarms?",
        "  Advantage  : 2×(best_acc − 0.5). Range [0,1].",
        "               Advantage=0 means attack is no better than coin flip.",
        "               Advantage=1 means attacker is always correct.",
        "",
        f"{'Signal':<30} {'AUC':>7} {'TPR@1%FPR':>11} {'Best acc':>10} {'Advantage':>11}",
        "-" * 70,
    ]
    for r in results:
        lines.append(
            f"{r['signal_name']:<30} "
            f"{r['roc_auc']:>7.4f} "
            f"{r['tpr_at_1fpr']:>11.4f} "
            f"{r['best_acc']:>10.4f} "
            f"{r['advantage']:>11.4f}"
        )
    lines += ["", "=" * 70]

    txt = "\n".join(lines)
    print("\n" + txt)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nSummary saved → {out_path}")


# =============================================================================
# SECTION 10: MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("THRESHOLD MEMBERSHIP INFERENCE ATTACK")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    df, y_true, signals = load_mia_data(MIA_CSV)

    # Add combined score
    signals["combined (z-score avg)"] = combined_score(signals)

    # ── Distribution plots ────────────────────────────────────────────────
    print("\n[1/4] Plotting signal distributions...")
    plot_distributions(df, os.path.join(REPORT_DIR, "distributions.png"))

    # ── Threshold attacks ─────────────────────────────────────────────────
    print("\n[2/4] Running threshold attacks...")
    results = []
    for name, sig in signals.items():
        res = run_threshold_attack(sig, y_true, name)
        results.append(res)

    # ── ROC curves ────────────────────────────────────────────────────────
    print("\n[3/4] Plotting ROC curves...")
    plot_roc_curves(results, os.path.join(REPORT_DIR, "roc_curves.png"))

    # Per-class breakdown using loss signal (strongest single signal typically)
    per_class_attack(df, "loss", negate=True,
                     out_path=os.path.join(REPORT_DIR, "per_class_auc.png"))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[4/4] Writing summary report...")
    write_summary(results, os.path.join(REPORT_DIR, "summary.txt"))

    print("\n" + "=" * 60)
    print("Done. Outputs in:", REPORT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
