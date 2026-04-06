# =============================================================================
# CS6413: Early Stopping Defense
# =============================================================================
# PURPOSE:
#   The simplest possible defense against MIA: stop training before the
#   model has time to memorise individual training samples.
#
# WHY IT WORKS AGAINST MIA:
#   Memorisation is a progressive process. In the early epochs, the model
#   learns general features (edges, textures, shapes) that apply to both
#   members and non-members equally. As training continues, the model
#   starts fitting to sample-specific noise in the training set — this is
#   when the member/non-member gap opens up.
#
#   By stopping early, we freeze the model in the "general features" phase
#   before significant memorisation occurs. The overfit gap (member_acc -
#   holdout_acc) is the direct indicator: when it starts growing rapidly,
#   memorisation is happening.
#
# STOPPING CRITERION:
#   We monitor the overfit gap = member_eval_acc - holdout_acc.
#   Training stops when EITHER:
#     1. The gap exceeds a threshold (default: 0.08), OR
#     2. Holdout accuracy hasn't improved for `patience` epochs
#   Whichever comes first.
#
# COMPARISON WITH OTHER DEFENSES:
#   - DP-SGD: formal guarantee but destroys utility on small datasets
#   - Regularisation: reduces memorisation but still trains full epochs
#   - Distillation: requires training two models
#   - Early stopping: zero overhead, zero hyperparameters beyond the
#     gap threshold, but no formal privacy guarantee
#
# INPUTS:
#   outputs/splits/cifar10_12k_seed42.npz
#   data/cifar-10-batches-py
#
# OUTPUTS:
#   outputs/checkpoints/best_early_stop.pt
#   outputs/logs/train_log_early_stop.csv
#   outputs/logs/per_sample_mia_early_stop.csv
# =============================================================================

import os, csv, random
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
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

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
# SECTION 3: MODEL (identical to train_v2.py)
# =============================================================================

class SmallCNN(nn.Module):
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
    """Extract all v2 MIA signals."""
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
# SECTION 6: TRAINING WITH EARLY STOPPING
# =============================================================================

def train_with_early_stopping(model, train_loader, eval_loader, holdout_loader,
                              device, ckpt_path, log_path,
                              max_epochs=150, gap_threshold=0.08, patience=15):
    """
    Train with two early stopping criteria:

    1. OVERFIT GAP THRESHOLD (gap_threshold=0.08):
       Stop when member_acc - holdout_acc exceeds this value.
       The gap directly measures how differently the model treats members
       vs non-members — which is exactly what MIA exploits.
       0.08 is chosen as a balance: enough training to learn useful features,
       but stopped before deep memorisation sets in.

    2. PATIENCE (patience=15):
       Stop if holdout accuracy hasn't improved for 15 consecutive epochs.
       This catches the case where the model is overfitting (holdout acc
       plateaus or drops) even if the gap hasn't hit the threshold yet.

    The model checkpoint saved is the one with the BEST holdout accuracy
    seen before either stopping criterion fires.
    """
    opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 80, 110], gamma=0.5)
    ce    = nn.CrossEntropyLoss()

    best_acc = 0.0
    epochs_no_improve = 0
    stopped_epoch = max_epochs

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc",
            "member_eval_acc", "holdout_acc", "best_acc", "overfit_gap"
        ])

    print(f"Early stopping: gap_threshold={gap_threshold} | "
          f"patience={patience} | max_epochs={max_epochs}")

    for epoch in range(1, max_epochs + 1):
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

        # Track best holdout accuracy
        if hld_acc > best_acc:
            best_acc = hld_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch <= 5:
            print(f"Ep {epoch:03d} | loss={train_loss:.4f} | "
                  f"mem={mem_acc:.4f} | hld={hld_acc:.4f} | "
                  f"gap={gap:.4f} | no_improve={epochs_no_improve}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{mem_acc:.6f}", f"{hld_acc:.6f}",
                f"{best_acc:.6f}", f"{gap:.6f}",
            ])

        # ── Check stopping criteria ──────────────────────────────────────
        if gap > gap_threshold:
            print(f"\n  EARLY STOP at epoch {epoch}: "
                  f"overfit gap {gap:.4f} > threshold {gap_threshold}")
            stopped_epoch = epoch
            break

        if epochs_no_improve >= patience:
            print(f"\n  EARLY STOP at epoch {epoch}: "
                  f"no holdout improvement for {patience} epochs")
            stopped_epoch = epoch
            break

    print(f"  Stopped at epoch {stopped_epoch}/{max_epochs}")
    print(f"  Best holdout acc: {best_acc:.4f}")
    return model, stopped_epoch


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
    print(f"  Loaded checkpoint: {ckpt_path}")

    out = os.path.join(LOGS_DIR, "per_sample_mia_early_stop.csv")
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

    print(f"  MIA features saved: {out}")
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

    ckpt_path = os.path.join(CKPT_DIR, "best_early_stop.pt")
    log_path  = os.path.join(LOGS_DIR, "train_log_early_stop.csv")

    # ── Train with early stopping ─────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Training with early stopping defense")
    print("=" * 60)
    model = SmallCNN(num_classes=10).to(device)
    model, stopped_epoch = train_with_early_stopping(
        model, train_loader, eval_loader, hld_loader,
        device, ckpt_path, log_path,
        max_epochs=150, gap_threshold=0.08, patience=15,
    )

    # ── Export MIA features ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Exporting MIA features")
    print("=" * 60)
    export_mia_csv(model, ckpt_path, target_idx, holdout_idx, device)

    print("\n" + "=" * 60)
    print(f"DONE — stopped at epoch {stopped_epoch}")
    print("Next: run eval_all_defenses.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
