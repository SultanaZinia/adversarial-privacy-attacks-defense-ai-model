# =============================================================================
# CS6413: DP-SGD Defense — Opacus Training
# =============================================================================
# PURPOSE:
#   Retrain the SmallCNN target model three times using Differentially Private
#   SGD (DP-SGD) at privacy budgets ε = 1, 5, 10. For each ε, we save:
#     - a checkpoint  (best_dp_eps{ε}.pt)
#     - a MIA feature CSV  (per_sample_mia_dp_eps{ε}.csv)
#     - a training log  (train_log_dp_eps{ε}.csv)
#
# WHAT DP-SGD DOES:
#   Standard SGD computes gradients per batch. DP-SGD adds two modifications:
#     1. Gradient clipping — each individual sample's gradient is clipped to
#        a maximum L2 norm (max_grad_norm). This bounds how much any single
#        training sample can influence the model, limiting memorisation.
#     2. Gaussian noise — after clipping, calibrated Gaussian noise is added
#        to the aggregated gradient before the weight update. This masks the
#        contribution of any individual sample.
#   Together these give a formal (ε, δ)-DP guarantee: an adversary observing
#   the final model cannot determine with confidence whether any specific
#   sample was in the training set.
#
# PRIVACY BUDGET ε:
#   ε controls the privacy-utility trade-off:
#     ε = 1   → strong privacy, expect significant accuracy drop
#     ε = 5   → moderate privacy, moderate accuracy drop
#     ε = 10  → weak privacy, small accuracy drop, still reduces MIA
#   δ is fixed at 1/N (N = training set size = 6000), the standard choice.
#
# OPACUS CONSTRAINTS (important — these affect your architecture):
#   Opacus computes per-sample gradients. Some PyTorch layers are incompatible:
#     - BatchNorm → replace with GroupNorm (done below)
#     - Dropout   → safe to use but already omitted in our model
#   Our SmallCNN uses only Conv2d, Linear, MaxPool, ReLU — all compatible.
#
# INPUTS:
#   outputs/splits/cifar10_12k_seed42.npz
#   data/cifar-10-batches-py
#
# OUTPUTS (per ε):
#   outputs/checkpoints/best_dp_eps{ε}.pt
#   outputs/logs/train_log_dp_eps{ε}.csv
#   outputs/logs/per_sample_mia_dp_eps{ε}.csv
#   outputs/reports/dp_defense/privacy_utility_curve.png
#   outputs/reports/dp_defense/summary_dp.txt
# =============================================================================

import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
except ImportError:
    raise ImportError(
        "Opacus is not installed. Run:  pip install opacus"
    )


# =============================================================================
# SECTION 1: PATHS
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
SPLITS_PATH     = os.path.join(PROJECT_ROOT, "outputs", "splits",
                               "cifar10_12k_seed42.npz")
CKPT_DIR        = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
LOGS_DIR        = os.path.join(PROJECT_ROOT, "outputs", "logs")
REPORT_DIR      = os.path.join(PROJECT_ROOT, "outputs", "reports", "dp_defense")

for d in [CKPT_DIR, LOGS_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# SECTION 2: REPRODUCIBILITY
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# SECTION 3: MODEL
# =============================================================================

class SmallCNN(nn.Module):
    """
    Identical to train_v2.py — no BatchNorm so fully Opacus-compatible.
    Opacus requires per-sample gradients; BatchNorm uses batch statistics
    and is incompatible. Our model only uses Conv2d + Linear + ReLU + MaxPool,
    all of which Opacus handles natively.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# SECTION 4: INDEXED SUBSET
# =============================================================================

class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig = self.indices[idx]
        x, y = self.dataset[orig]
        return x, y, orig


# =============================================================================
# SECTION 5: EVALUATION AND MIA FEATURE LOGGING
# =============================================================================

@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        # loader may return (x, y) or (x, y, idx)
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


@torch.no_grad()
def log_mia_features(model, loader, device, split_name, writer):
    """Extract all v2 MIA signals for members and non-members."""
    model.eval()
    ce      = nn.CrossEntropyLoss(reduction="none")
    softmax = nn.Softmax(dim=1)

    for x, y, idx in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        probs  = softmax(logits)
        losses = ce(logits, y)
        pred   = logits.argmax(1)

        conf_true  = probs.gather(1, y.view(-1,1)).squeeze(1)
        conf_max   = probs.max(1).values
        correct    = (pred == y).long()
        entropy    = -(probs * torch.log(probs + 1e-10)).sum(1)

        logit_true = logits.gather(1, y.view(-1,1)).squeeze(1)
        logits_oth = logits.clone()
        logits_oth.scatter_(1, y.view(-1,1), float("-inf"))
        logit_gap  = logit_true - logits_oth.max(1).values
        m_entropy  = torch.where(correct.bool(), -entropy, entropy)

        for i in range(x.size(0)):
            writer.writerow([
                split_name, int(idx[i]),
                int(y[i].item()), int(pred[i].item()), int(correct[i].item()),
                f"{losses[i].item():.6f}",
                f"{conf_true[i].item():.6f}", f"{conf_max[i].item():.6f}",
                f"{entropy[i].item():.6f}",
                f"{logit_true[i].item():.6f}",
                f"{logit_gap[i].item():.6f}",
                f"{m_entropy[i].item():.6f}",
            ])


# =============================================================================
# SECTION 6: DP-SGD TRAINING FOR ONE ε
# =============================================================================

def train_dp(epsilon, target_idx, holdout_idx, ds_aug, ds_eval, device):
    """
    Train SmallCNN with DP-SGD at a given ε budget.

    Key hyperparameters:
      max_grad_norm : per-sample gradient clipping threshold.
                      Lower = more privacy but noisier gradients.
                      1.0 is the standard default.
      noise_multiplier: how much Gaussian noise to add (set automatically
                        by Opacus to hit the target ε).
      δ             : probability of ε-bound failing. Standard to set as
                      1/N where N is training set size.
      epochs        : fewer epochs = less privacy budget spent per run.
                      We use 60 — enough to converge without excessive ε.

    Why DP-SGD hurts accuracy:
      The gradient noise prevents the model from memorising individual
      samples — which is exactly what makes it less vulnerable to MIA,
      but also means it generalises slightly less well to the training set.
      The privacy-utility trade-off is the central result of this section.
    """
    set_seed(42)

    EPOCHS        = 100
    BATCH_SIZE    = 512   # larger batch = less noise per step = better utility
    LR            = 0.01
    MAX_GRAD_NORM = 1.0
    DELTA         = 1 / len(target_idx)   # 1/N standard choice

    print(f"\n{'='*55}")
    print(f"  DP-SGD training  |  ε={epsilon}  δ={DELTA:.2e}")
    print(f"{'='*55}")

    # ── Data loaders ──────────────────────────────────────────────────────
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_train = datasets.CIFAR10(DATA_DIR, train=True, download=False,
                                transform=train_tf)
    ds_ev    = datasets.CIFAR10(DATA_DIR, train=True, download=False,
                                transform=eval_tf)

    # Opacus requires drop_last=True so every batch is the same size
    train_loader = DataLoader(
        Subset(ds_train, target_idx),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True,
    )
    eval_loader  = DataLoader(
        Subset(ds_ev, target_idx),
        batch_size=256, shuffle=False, num_workers=0,
    )
    hld_loader   = DataLoader(
        Subset(ds_ev, holdout_idx),
        batch_size=256, shuffle=False, num_workers=0,
    )

    # ── Model + optimiser ─────────────────────────────────────────────────
    model = SmallCNN(num_classes=10).to(device)

    # Validate that model is Opacus-compatible (no BatchNorm etc.)
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"  Opacus validation warnings: {errors}")
        model = ModuleValidator.fix(model)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=0.0)
    ce = nn.CrossEntropyLoss()

    # ── Attach Opacus PrivacyEngine ───────────────────────────────────────
    # make_private_with_epsilon automatically computes the noise_multiplier
    # needed to spend exactly ε over the chosen number of epochs and steps.
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module       = model,
        optimizer    = optimizer,
        data_loader  = train_loader,
        epochs       = EPOCHS,
        target_epsilon = epsilon,
        target_delta   = DELTA,
        max_grad_norm  = MAX_GRAD_NORM,
    )
    print(f"  Noise multiplier : {optimizer.noise_multiplier:.4f}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Epochs           : {EPOCHS}")

    # ── Training loop ─────────────────────────────────────────────────────
    log_path = os.path.join(LOGS_DIR, f"train_log_dp_eps{epsilon}.csv")
    best_acc = 0.0
    ckpt_path = os.path.join(CKPT_DIR, f"best_dp_eps{epsilon}.pt")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch","train_loss","train_acc","member_eval_acc",
             "holdout_acc","best_acc","epsilon_spent"])

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = correct = total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = ce(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)

        train_loss    = loss_sum / total
        train_acc     = correct  / total
        mem_acc       = eval_accuracy(model, eval_loader, device)
        hld_acc       = eval_accuracy(model, hld_loader,  device)
        eps_spent     = privacy_engine.get_epsilon(DELTA)

        if hld_acc > best_acc:
            best_acc = hld_acc
            # Opacus wraps the model — save the unwrapped state dict
            torch.save(model._module.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch <= 3:
            print(f"  Ep {epoch:03d} | loss={train_loss:.4f} | "
                  f"mem={mem_acc:.4f} | hld={hld_acc:.4f} | "
                  f"ε_spent={eps_spent:.2f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{mem_acc:.6f}", f"{hld_acc:.6f}",
                f"{best_acc:.6f}", f"{eps_spent:.4f}",
            ])

    print(f"  Final ε spent : {privacy_engine.get_epsilon(DELTA):.4f}")
    print(f"  Best holdout  : {best_acc:.4f}")
    print(f"  Checkpoint    : {ckpt_path}")
    return best_acc, mem_acc


# =============================================================================
# SECTION 7: MIA FEATURE EXPORT FOR A DP CHECKPOINT
# =============================================================================

def export_mia_csv_dp(epsilon, target_idx, holdout_idx, device):
    """Load the saved DP checkpoint and write per_sample_mia_dp_eps{ε}.csv."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds_ev = datasets.CIFAR10(DATA_DIR, train=True, download=False,
                             transform=eval_tf)

    mem_loader = DataLoader(
        IndexedSubset(ds_ev, target_idx),
        batch_size=256, shuffle=False, num_workers=0)
    hld_loader = DataLoader(
        IndexedSubset(ds_ev, holdout_idx),
        batch_size=256, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes=10).to(device)
    ckpt  = os.path.join(CKPT_DIR, f"best_dp_eps{epsilon}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"  Loaded: {ckpt}")

    out = os.path.join(LOGS_DIR, f"per_sample_mia_dp_eps{epsilon}.csv")
    header = [
        "split_name","index","true_label","pred_label","correct",
        "loss","conf_true","conf_max","entropy",
        "logit_true","logit_gap","m_entropy",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        log_mia_features(model, mem_loader, device, "member",    writer)
        log_mia_features(model, hld_loader, device, "nonmember", writer)

    print(f"  MIA CSV: {out}")
    return out


# =============================================================================
# SECTION 8: PRIVACY-UTILITY CURVE PLOT
# =============================================================================

def plot_privacy_utility(results, baseline_hld, baseline_auc, out_path):
    """
    Plot two y-axes:
      Left  — holdout accuracy (utility) vs ε
      Right — attack AUC vs ε  (if attack CSVs exist, computed inline)

    The ideal defense: accuracy stays high, AUC drops toward 0.5.
    """
    epsilons   = [r["epsilon"]  for r in results]
    hld_accs   = [r["hld_acc"]  for r in results]
    mem_accs   = [r["mem_acc"]  for r in results]
    overfit    = [m - h for m, h in zip(mem_accs, hld_accs)]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Accuracy lines
    ax1.plot(epsilons, hld_accs, "o-", color="#185FA5", lw=2,
             label="Holdout acc (utility)")
    ax1.plot(epsilons, mem_accs, "s--", color="#3B8BD4", lw=1.5,
             label="Member acc")
    ax1.axhline(baseline_hld, color="#185FA5", lw=1, ls=":",
                alpha=0.5, label=f"No-DP holdout baseline ({baseline_hld:.3f})")

    # Overfit gap bars (secondary feel)
    ax1.bar(epsilons, overfit, width=0.6, alpha=0.12,
            color="#D85A30", label="Overfit gap (mem−hld)")

    ax1.set_xlabel("Privacy budget ε  (lower = stronger privacy)", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim(0.3, 1.05)
    ax1.set_xticks(epsilons)
    ax1.set_xticklabels([f"ε={e}" for e in epsilons])
    ax1.spines[["top","right"]].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, fontsize=9, loc="lower right")

    ax1.set_title("DP-SGD defense — privacy-utility trade-off", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Privacy-utility plot saved → {out_path}")


# =============================================================================
# SECTION 9: SUMMARY REPORT
# =============================================================================

def write_summary(results, baseline_hld, baseline_mem, out_path):
    lines = [
        "=" * 70,
        "DP-SGD DEFENSE SUMMARY",
        "=" * 70,
        "",
        f"{'Config':<18} {'Holdout acc':>12} {'Member acc':>11} "
        f"{'Overfit gap':>12} {'Notes'}",
        "-" * 70,
        f"{'No-DP baseline':<18} {baseline_hld:>12.4f} {baseline_mem:>11.4f} "
        f"{baseline_mem-baseline_hld:>12.4f}",
    ]
    for r in results:
        gap = r["mem_acc"] - r["hld_acc"]
        lines.append(
            f"DP ε={r['epsilon']:<13} {r['hld_acc']:>12.4f} "
            f"{r['mem_acc']:>11.4f} {gap:>12.4f}"
        )
    lines += [
        "",
        "Next step: run attack_shadow_v2.py for each ε checkpoint to",
        "measure how DP reduces attack AUC and TPR@1%FPR.",
        "",
        "=" * 70,
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")


# =============================================================================
# SECTION 10: MAIN
# =============================================================================

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load splits
    splits      = np.load(SPLITS_PATH)
    target_idx  = splits["target_idx"].tolist()
    holdout_idx = splits["holdout_idx"].tolist()
    print(f"Members: {len(target_idx):,}  |  Non-members: {len(holdout_idx):,}")

    # Baseline numbers from the non-DP v2 model
    # Update these if your train_log_v2.csv shows different values
    BASELINE_HLD = 0.7087
    BASELINE_MEM = 0.9933

    # ── Run DP training for each ε ────────────────────────────────────────
    epsilons = [10, 5, 1]   # high → low privacy (train strongest privacy last)
    results  = []

    for eps in epsilons:
        hld_acc, mem_acc = train_dp(
            eps, target_idx, holdout_idx,
            None, None, device   # ds_aug/ds_eval loaded inside train_dp
        )
        print(f"\nExporting MIA features for ε={eps}...")
        export_mia_csv_dp(eps, target_idx, holdout_idx, device)
        results.append({"epsilon": eps, "hld_acc": hld_acc, "mem_acc": mem_acc})

    # Sort by epsilon for plotting
    results.sort(key=lambda r: r["epsilon"])

    # ── Plots and summary ─────────────────────────────────────────────────
    plot_privacy_utility(
        results, BASELINE_HLD, 0.736,
        os.path.join(REPORT_DIR, "privacy_utility_curve.png"),
    )

    write_summary(
        results, BASELINE_HLD, BASELINE_MEM,
        os.path.join(REPORT_DIR, "summary_dp.txt"),
    )

    print("\n" + "=" * 55)
    print("DP TRAINING COMPLETE")
    print("Next: for each ε, run attack_shadow_v2.py pointing")
    print("      TARGET_MIA_V2 at the per_sample_mia_dp_eps{ε}.csv")
    print("=" * 55)


if __name__ == "__main__":
    main()
