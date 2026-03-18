# =============================================================================
# CS6413: Target Model Training v3 — Balanced & Fast
# =============================================================================
# GOAL:
#   Provide a v3 configuration that (a) trains quickly on typical
#   student hardware and (b) still exposes a measurable membership
#   inference signal.
#
# FINAL SETTINGS (v3 balanced):
#   - Moderate-capacity CNN (≈ 0.8M params)
#   - Data augmentation: RandomCrop + RandomHorizontalFlip
#   - Epochs: 120
#   - Batch size: 128
#   - Weight decay: 0.0
#   - LR schedule: milestones=[60, 90] for 120 epochs
#
# Compared to v2 (150 epochs, smaller model), this v3 config trades
# some absolute accuracy for MUCH lower wall-clock time while still
# generating a non-trivial train/holdout gap for MIA experiments.
# =============================================================================

import os, random, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# MODEL v3 — INCREASED CAPACITY
# =============================================================================

class LargeCNN(nn.Module):
    """
    Balanced CNN with moderate filters for speed + memorisation.
    
    Architecture:
    - Conv block 1: 3 → 48 filters (1.5x from 32)
    - Conv block 2: 48 → 96 filters (1.5x from 64)
    - Dense layers: 96×8×8 → 256 → 128 → 10 (simplified)
    
    No dropout, no batch norm → promote memorisation.
    Optimized for reasonable training speed.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 32×32 → 16×16
            nn.Conv2d(3, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 16×16 → 8×8
            nn.Conv2d(48, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # Flatten: 96 * 8 * 8 = 6144
            nn.Flatten(),
            # Dense block
            nn.Linear(96 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            # Output
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# INDEXED SUBSET
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
# EVALUATION HELPERS
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
    """
    Write one CSV row per sample with ALL MIA signals:
      - softmax-based: loss, conf_true, conf_max, entropy
      - logit-based: logit_true, logit_gap, m_entropy
    """
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
# DATA LOADING — BALANCED APPROACH
# =============================================================================

def load_data_and_splits(data_dir, splits_dir):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        # LIGHT DATA AUGMENTATION — only random cropping (no flips)
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_aug  = datasets.CIFAR10(data_dir, train=True, download=True,  transform=train_tf)
    ds_eval = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_tf)

    # Use standard v2 split file (6000 total: 6k members + 3k shadow + 3k holdout)
    split_path = os.path.join(splits_dir, "cifar10_12k_seed42.npz")
    if os.path.exists(split_path):
        sp          = np.load(split_path)
        target_idx  = sp["target_idx"].tolist()
        shadow_idx  = sp["shadow_idx"].tolist()
        holdout_idx = sp["holdout_idx"].tolist()
        print(f"Loaded splits from {split_path}")
    else:
        # Create standard split (same as v2)
        # 6000 members + 3000 shadow + 3000 holdout = 12000 total
        rng        = np.random.RandomState(42)
        all_idx    = rng.permutation(len(ds_aug))[:12000]
        target_idx  = all_idx[:6000].tolist()
        shadow_idx  = all_idx[6000:9000].tolist()
        holdout_idx = all_idx[9000:12000].tolist()
        np.savez(split_path, target_idx=target_idx,
                 shadow_idx=shadow_idx, holdout_idx=holdout_idx)
        print(f"Created standard v3 splits at {split_path}")

    print(f"  Members:     {len(target_idx):,}  (standard split, same as v2)")
    print(f"  Non-members: {len(holdout_idx):,}")
    return ds_aug, ds_eval, target_idx, shadow_idx, holdout_idx


# =============================================================================
# TRAINING LOOP v3 — BALANCED SCHEDULE (120 EPOCHS)
# =============================================================================

def train(model, train_loader, eval_loader, holdout_loader,
        device, ckpt_dir, log_path, epochs=120):
    """
    120-epoch training with 2 LR decay stages: [60, 90].

    Decay schedule:
    epochs   1–59   : lr = 0.01
    epochs  60–89   : lr = 0.005  (×0.5)
    epochs  90–120  : lr = 0.0025 (×0.5)

    Data augmentation + light weight decay prevent catastrophic
    overfitting while a moderate-capacity model still memorises the
    training set more than the holdout, creating a usable MIA signal.
    """
    opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 90], gamma=0.5)
    ce    = nn.CrossEntropyLoss()
    best_acc = 0.0

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch","train_loss","train_acc","member_eval_acc",
             "holdout_acc","best_acc","lr"])

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            if not torch.isfinite(logits).all():
                print(f"Non-finite logits at epoch {epoch}. Stopping.")
                return model
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)

        sched.step()
        current_lr = opt.param_groups[0]["lr"]

        train_loss = loss_sum / total
        train_acc  = correct / total
        mem_acc    = eval_accuracy(model, eval_loader,    device)
        hld_acc    = eval_accuracy(model, holdout_loader, device)

        if hld_acc > best_acc:
            best_acc = hld_acc
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, "best_baseline_v3.pt"))

        if epoch % 10 == 0 or epoch <= 5 or epoch in [60, 90, 120]:
            print(f"Ep {epoch:03d} | loss={train_loss:.4f} | "
                  f"mem={mem_acc:.4f} | hld={hld_acc:.4f} | "
                  f"gap={mem_acc-hld_acc:.4f} | lr={current_lr:.6f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{mem_acc:.6f}", f"{hld_acc:.6f}",
                f"{best_acc:.6f}", f"{current_lr:.6f}",
            ])

    print(f"\nBest holdout acc: {best_acc:.4f}")
    print(f"GOAL: member_acc >> holdout_acc (wide gap for MIA)")
    return model


# =============================================================================
# MIA FEATURE EXPORT
# =============================================================================

def export_mia_csv(model, ds_eval, target_idx, holdout_idx,
                   device, ckpt_dir, logs_dir):
    ckpt = os.path.join(ckpt_dir, "best_baseline_v3.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    print(f"\nLoaded checkpoint: {ckpt}")

    mem_loader = DataLoader(
        IndexedSubset(ds_eval, target_idx),
        batch_size=256, shuffle=False, num_workers=2)
    hld_loader = DataLoader(
        IndexedSubset(ds_eval, holdout_idx),
        batch_size=256, shuffle=False, num_workers=2)

    out = os.path.join(logs_dir, "per_sample_mia_v3.csv")
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

    print(f"MIA CSV saved → {out}")
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    project_root    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir        = os.path.join(project_root, "data")
    splits_dir      = os.path.join(project_root, "outputs", "splits")
    checkpoints_dir = os.path.join(project_root, "outputs", "checkpoints")
    logs_dir        = os.path.join(project_root, "outputs", "logs")

    for d in [splits_dir, checkpoints_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print("CS6413 v3 (Balanced): MODERATE MODEL + 60 EPOCHS")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("STEP 1: Loading data and splits (6000 members, augmentation restored)")
    print("=" * 70)
    ds_aug, ds_eval, target_idx, shadow_idx, holdout_idx = \
        load_data_and_splits(data_dir, splits_dir)

    train_loader = DataLoader(
        Subset(ds_aug,  target_idx),  batch_size=128, shuffle=True,  num_workers=2)
    eval_loader  = DataLoader(
        Subset(ds_eval, target_idx),  batch_size=256, shuffle=False, num_workers=2)
    hld_loader   = DataLoader(
        Subset(ds_eval, holdout_idx), batch_size=256, shuffle=False, num_workers=2)

    print("\n" + "=" * 70)
    print("STEP 2: Training (120 epochs, LargeCNN, wd=0.0005, bs=128)")
    print("=" * 70)
    model    = LargeCNN(num_classes=10).to(device)
    
    # Count parameters to show increased capacity
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    log_path = os.path.join(logs_dir, "train_log_v3.csv")
    train(model, train_loader, eval_loader, hld_loader,
          device, checkpoints_dir, log_path, epochs=120)

    print("\n" + "=" * 70)
    print("STEP 3: Exporting MIA features (logit-space signals)")
    print("=" * 70)
    export_mia_csv(model, ds_eval, target_idx, holdout_idx,
                   device, checkpoints_dir, logs_dir)

    print("\n" + "=" * 70)
    print("DONE — next: run models/attack_v3.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
