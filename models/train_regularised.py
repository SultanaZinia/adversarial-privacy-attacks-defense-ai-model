# =============================================================================
# CS6413: Heuristic Defense — L2 Regularisation + Label Smoothing + Dropout
# =============================================================================
# PURPOSE:
#   Retrain the SmallCNN with three heuristic defenses that reduce
#   memorisation without the utility collapse seen in DP-SGD:
#
#     1. L2 regularisation (weight_decay=1e-3)
#        Penalises large weights in the loss function. Large weights are
#        what allow the model to memorise individual samples — shrinking
#        them forces the model to learn more general features instead.
#        Direct effect: reduces logit_gap magnitude for members, making
#        them harder to distinguish from non-members.
#
#     2. Label smoothing (smoothing=0.1)
#        Replaces hard targets (0 or 1) with soft targets (0.05 or 0.95).
#        Prevents the model from becoming overconfident on training samples.
#        Direct effect: caps maximum softmax confidence, reducing the
#        conf_true signal that threshold attacks exploit.
#
#     3. Dropout (p=0.5) after FC(256)
#        Randomly disables 50% of neurons during training. Forces the
#        network to learn redundant representations rather than memorising
#        specific training samples through individual neurons.
#        Direct effect: reduces overfitting gap (member_acc - holdout_acc).
#
#   These three are combined because they attack different aspects of
#   memorisation simultaneously. Each alone gives partial protection;
#   together they provide meaningful privacy without destroying utility.
#
# COMPARISON WITH DP-SGD:
#   DP-SGD: formal (ε,δ)-DP guarantee, but utility collapses on small
#           datasets (33% accuracy on our 6K training set).
#   This script: no formal guarantee, but maintains ~65-70% holdout
#           accuracy while measurably reducing attack AUC.
#   The contrast between the two is a core finding of this project.
#
# INPUTS:
#   outputs/splits/cifar10_12k_seed42.npz
#   data/cifar-10-batches-py
#
# OUTPUTS:
#   outputs/checkpoints/best_regularised.pt
#   outputs/logs/train_log_regularised.csv
#   outputs/logs/per_sample_mia_regularised.csv
# =============================================================================

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


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

for d in [CKPT_DIR, LOGS_DIR]:
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
# SECTION 3: MODEL — with Dropout restored as defense
# =============================================================================

class SmallCNN(nn.Module):
    """
    Same architecture as train_v2.py with one change:
    Dropout(0.5) is added before the final FC layer.

    In train_v2.py dropout was intentionally REMOVED to maximise
    memorisation for attack experiments. Here we ADD it back as
    a defense — it was always in the original proposal architecture.
    """
    def __init__(self, num_classes=10, dropout_p=0.5):
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
            nn.Dropout(p=dropout_p),          # DEFENSE: prevents neuron co-adaptation
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
# SECTION 5: EVALUATION HELPERS
# =============================================================================

@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


@torch.no_grad()
def log_mia_features(model, loader, device, split_name, writer):
    """Extract all v2 MIA signals — identical to train_v2.py and train_dp.py."""
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
# SECTION 6: TRAINING LOOP
# =============================================================================

def train(model, train_loader, eval_loader, holdout_loader,
          device, ckpt_path, log_path, epochs=150):
    """
    Three defense changes vs train_v2.py:
      1. weight_decay=1e-3  → L2 regularisation
      2. label_smoothing=0.1 in CrossEntropyLoss
      3. Dropout(0.5) is in the model architecture above

    Everything else (LR schedule, epochs, seed) is identical so
    results are directly comparable to the no-defense baseline.
    """
    # DEFENSE 1: L2 regularisation via weight_decay
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                    weight_decay=1e-3)

    sched = optim.lr_scheduler.MultiStepLR(
        opt, milestones=[40, 80, 110], gamma=0.5)

    # DEFENSE 2: label smoothing — soft targets instead of hard 0/1
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc",
            "member_eval_acc", "holdout_acc", "best_acc", "overfit_gap"
        ])

    print(f"Defenses active: L2 (wd=1e-3) | "
          f"Label smoothing (0.1) | Dropout (p=0.5)")
    print(f"Epochs: {epochs} | LR schedule: decay at 40, 80, 110\n")

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss   = ce(logits, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)

        sched.step()

        train_loss = loss_sum / total
        train_acc  = correct  / total
        mem_acc    = eval_accuracy(model, eval_loader,    device)
        hld_acc    = eval_accuracy(model, holdout_loader, device)
        gap        = mem_acc - hld_acc

        if hld_acc > best_acc:
            best_acc = hld_acc
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"Ep {epoch:03d} | loss={train_loss:.4f} | "
                  f"mem={mem_acc:.4f} | hld={hld_acc:.4f} | "
                  f"gap={gap:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{mem_acc:.6f}", f"{hld_acc:.6f}",
                f"{best_acc:.6f}", f"{gap:.6f}",
            ])

    print(f"\nBest holdout acc : {best_acc:.4f}")
    print(f"Checkpoint saved : {ckpt_path}")
    return best_acc


# =============================================================================
# SECTION 7: MIA FEATURE EXPORT
# =============================================================================

def export_mia_csv(model, ckpt_path, target_idx, holdout_idx, device):
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

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    print(f"Loaded checkpoint: {ckpt_path}")

    out = os.path.join(LOGS_DIR, "per_sample_mia_regularised.csv")
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

    print(f"MIA features saved: {out}")
    return out


# =============================================================================
# SECTION 8: MAIN
# =============================================================================

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load splits
    splits      = np.load(SPLITS_PATH)
    target_idx  = splits["target_idx"].tolist()
    holdout_idx = splits["holdout_idx"].tolist()
    print(f"Members: {len(target_idx):,}  |  Non-members: {len(holdout_idx):,}\n")

    # Data loaders
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

    ds_aug  = datasets.CIFAR10(DATA_DIR, train=True, download=False,
                               transform=train_tf)
    ds_eval = datasets.CIFAR10(DATA_DIR, train=True, download=False,
                               transform=eval_tf)

    train_loader = DataLoader(
        Subset(ds_aug,  target_idx),
        batch_size=128, shuffle=True,  num_workers=2)
    eval_loader  = DataLoader(
        Subset(ds_eval, target_idx),
        batch_size=256, shuffle=False, num_workers=2)
    hld_loader   = DataLoader(
        Subset(ds_eval, holdout_idx),
        batch_size=256, shuffle=False, num_workers=2)

    ckpt_path = os.path.join(CKPT_DIR, "best_regularised.pt")
    log_path  = os.path.join(LOGS_DIR, "train_log_regularised.csv")

    print("=" * 60)
    print("STEP 1: Training with heuristic defenses")
    print("=" * 60)
    model = SmallCNN(num_classes=10, dropout_p=0.5).to(device)
    train(model, train_loader, eval_loader, hld_loader,
          device, ckpt_path, log_path, epochs=150)

    print("\n" + "=" * 60)
    print("STEP 2: Exporting MIA features")
    print("=" * 60)
    export_mia_csv(model, ckpt_path, target_idx, holdout_idx, device)

    print("\n" + "=" * 60)
    print("DONE — next: run eval_all_defenses.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
