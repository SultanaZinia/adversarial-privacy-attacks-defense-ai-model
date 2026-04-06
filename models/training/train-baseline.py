# =============================================================================
# CS6413: Membership Inference Attack — Target Model Training
# =============================================================================
# PURPOSE:
#   Train a SmallCNN on a fixed 6,000-sample CIFAR-10 subset (the "members").
#   After training, log per-sample model outputs (loss, confidence) for both
#   members and non-members. This CSV is the foundation for all MIA attacks.
#
# OUTPUT FILES:
#   outputs/splits/cifar10_12k_seed42.npz   — saved train/shadow/holdout indices
#   outputs/checkpoints/best_baseline.pt    — best model checkpoint
#   outputs/logs/train_log.csv              — epoch-level training metrics
#   outputs/logs/per_sample_mia.csv         — per-sample MIA features
# =============================================================================

import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# =============================================================================
# SECTION 1: REPRODUCIBILITY
# =============================================================================
# Seed every RNG so results are identical across runs.
# deterministic=True forces cuDNN to use the same algorithm every time.

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# SECTION 2: MODEL ARCHITECTURE
# =============================================================================
# SmallCNN — a lightweight 4-layer CNN designed for CIFAR-10 (32x32 RGB).
#
# Architecture:
#   Block 1: Conv(3→32) → ReLU → Conv(32→32) → ReLU → MaxPool  [32x32 → 16x16]
#   Block 2: Conv(32→64) → ReLU → Conv(64→64) → ReLU → MaxPool [16x16 → 8x8]
#   Head:    Flatten → FC(4096→256) → ReLU → Dropout(0.5) → FC(256→10)
#
# The small size is intentional — we WANT moderate overfitting to make
# members and non-members distinguishable for MIA.

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # --- Convolutional Block 1 ---
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x32 → 16x16

            # --- Convolutional Block 2 ---
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x16 → 8x8

            # --- Classification Head ---
            nn.Flatten(),                                 # 64 * 8 * 8 = 4096
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)                   # raw logits (no softmax)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# SECTION 3: CUSTOM DATASET — IndexedSubset
# =============================================================================
# PyTorch's built-in Subset returns (image, label) per sample.
# We need (image, label, original_index) so we can track WHICH sample
# produced which model output in the MIA CSV.

class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]       # map local idx → original dataset idx
        x, y = self.dataset[orig_idx]
        return x, y, orig_idx              # return index alongside data


# =============================================================================
# SECTION 4: EVALUATION HELPERS
# =============================================================================

@torch.no_grad()
def eval_accuracy(model, loader, device):
    """Compute overall accuracy on a dataloader (no per-sample logging)."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def log_per_sample_mia_features(model, loader, device, split_name, csv_writer):
    """
    For every sample in the loader, record MIA-relevant features to CSV.

    Features logged per sample:
      - split_name : "member" or "nonmember"
      - index      : original CIFAR-10 dataset index
      - true_label : ground truth class (0–9)
      - pred_label : model's predicted class
      - correct    : 1 if prediction matches true label, else 0
      - loss       : cross-entropy loss (lower = model is more confident/correct)
      - conf_true  : model's softmax probability assigned to the TRUE class
      - conf_max   : model's softmax probability for its TOP prediction

    Members typically have lower loss and higher conf_true — this gap
    is exactly what MIA threshold attacks exploit.
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="none")   # per-sample loss (not averaged)
    softmax = nn.Softmax(dim=1)

    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs  = softmax(logits)
        losses = ce(logits, y)
        pred   = logits.argmax(dim=1)

        # Confidence on the TRUE label (even if model predicted wrong)
        conf_true = probs.gather(1, y.view(-1, 1)).squeeze(1)
        # Confidence on the TOP predicted label
        conf_max  = probs.max(dim=1).values
        correct   = (pred == y).long()
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)


        for i in range(x.size(0)):
            csv_writer.writerow([
                split_name,
                int(idx[i]),
                int(y[i].item()),
                int(pred[i].item()),
                int(correct[i].item()),
                f"{losses[i].item():.6f}",
                f"{conf_true[i].item():.6f}",
                f"{conf_max[i].item():.6f}",
                f"{entropy[i].item():.6f}",   # NEW column

            ])


# =============================================================================
# SECTION 5: DATA LOADING AND SPLITTING
# =============================================================================

def load_data_and_splits(data_dir, splits_dir):
    """
    Load CIFAR-10 and return the three fixed index splits.

    Two dataset objects are created from the same underlying data:
      - train_ds_aug  : with random crop + flip (used during training)
      - train_ds_eval : without augmentation   (used during evaluation/logging)

    Why two versions?
      Augmentation introduces randomness — during eval we want deterministic,
      consistent outputs so loss/confidence values are comparable across runs.

    Split structure (12K samples total, fixed seed):
      target_idx  (6K) — model IS trained on these → "members"
      shadow_idx  (3K) — reserved for shadow model training (future work)
      holdout_idx (3K) — model is NOT trained on these → "non-members"
    """
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

    train_ds_aug  = datasets.CIFAR10(data_dir, train=True, download=True,  transform=train_tf)
    train_ds_eval = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_tf)

    # Load splits from disk if they exist, otherwise create and save them
    split_path = os.path.join(splits_dir, "cifar10_12k_seed42.npz")
    if os.path.exists(split_path):
        splits      = np.load(split_path)
        target_idx  = splits["target_idx"].tolist()
        shadow_idx  = splits["shadow_idx"].tolist()
        holdout_idx = splits["holdout_idx"].tolist()
        print(f"Loaded existing splits from {split_path}")
    else:
        rng        = np.random.RandomState(42)
        all_idx    = rng.permutation(len(train_ds_aug))
        subset_idx = all_idx[:12000]
        target_idx  = subset_idx[:6000].tolist()
        shadow_idx  = subset_idx[6000:9000].tolist()
        holdout_idx = subset_idx[9000:12000].tolist()
        np.savez(split_path, target_idx=target_idx,
                 shadow_idx=shadow_idx, holdout_idx=holdout_idx)
        print(f"Created and saved new splits to {split_path}")

    print(f"  Members (target):  {len(target_idx):,}")
    print(f"  Shadow (reserved): {len(shadow_idx):,}")
    print(f"  Non-members:       {len(holdout_idx):,}")

    return train_ds_aug, train_ds_eval, target_idx, shadow_idx, holdout_idx


# =============================================================================
# SECTION 6: TRAINING LOOP
# =============================================================================

def train_target_model(model, train_loader, target_eval_loader,
                       holdout_loader, device, checkpoints_dir, log_path,
                       epochs=120):
    """
    Train the target model and save the best checkpoint.

    Optimizer: SGD with momentum=0.9, weight_decay=0.0
    Scheduler: LR decays by 0.2x at epochs 60, 90, 110
    Checkpoint: saved when holdout_acc improves (best generalization)

    Gradient clipping (max_norm=1.0) prevents exploding gradients,
    which can occur when training on only 6K samples.

    We track BOTH member accuracy (train eval) and holdout accuracy each
    epoch — the gap between them reveals how much the model is overfitting,
    which is the key signal for MIA attacks.
    """
    opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[120,160], gamma=0.5)
    ce    = nn.CrossEntropyLoss()

    best_acc = 0.0

    # Initialise log file with header
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "member_eval_acc", "holdout_acc", "best_acc"]
        )

    for epoch in range(1, epochs + 1):

        # --- Training pass ---
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)

            # Safety checks — stop early if numerics break
            if not torch.isfinite(logits).all():
                print(f"Non-finite logits at epoch {epoch}, batch {batch_idx}. Stopping.")
                return model
            loss = ce(logits, y)
            if not torch.isfinite(loss):
                print(f"Non-finite loss at epoch {epoch}, batch {batch_idx}. Stopping.")
                return model

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # for overfitting
            opt.step()

            train_loss_sum += loss.item() * x.size(0)
            train_correct  += (logits.argmax(dim=1) == y).sum().item()
            train_total    += y.size(0)

        sched.step()

        # --- Epoch-level metrics ---
        train_loss      = train_loss_sum / train_total
        train_acc       = train_correct / train_total
        member_eval_acc = eval_accuracy(model, target_eval_loader, device)
        holdout_acc     = eval_accuracy(model, holdout_loader, device)
        overfit_gap     = member_eval_acc - holdout_acc

        # Save best checkpoint based on holdout accuracy
        if holdout_acc > best_acc:
            best_acc = holdout_acc
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir, "best_baseline.pt"))

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"member_eval_acc={member_eval_acc:.4f} | holdout_acc={holdout_acc:.4f} | "
            f"overfit_gap={overfit_gap:.4f}"
        )

        # Append to log CSV
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{member_eval_acc:.6f}",
                f"{holdout_acc:.6f}",
                f"{best_acc:.6f}",
            ])

    print(f"\nTraining complete. Best holdout accuracy: {best_acc:.4f}")
    return model


# =============================================================================
# SECTION 7: PER-SAMPLE MIA FEATURE LOGGING
# =============================================================================

def generate_mia_csv(model, train_ds_eval, target_idx, holdout_idx,
                     device, logs_dir, checkpoints_dir):
    """
    Load the best checkpoint and log per-sample MIA features for:
      - All 6,000 members  (split_name = "member")
      - All 3,000 holdouts (split_name = "nonmember")

    Uses eval transform (no augmentation) for consistent, reproducible outputs.
    The resulting CSV is used by attack scripts to compute ROC-AUC, accuracy,
    and attacker advantage metrics.

    CSV columns:
      split_name | index | true_label | pred_label | correct | loss | conf_true | conf_max
    """
    # Load best checkpoint
    ckpt_path = os.path.join(checkpoints_dir, "best_baseline.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    print(f"\nLoaded best checkpoint from {ckpt_path}")

    # Build indexed loaders (returns orig index alongside x, y)
    member_loader  = DataLoader(
        IndexedSubset(train_ds_eval, target_idx),
        batch_size=256, shuffle=False, num_workers=2
    )
    holdout_loader = DataLoader(
        IndexedSubset(train_ds_eval, holdout_idx),
        batch_size=256, shuffle=False, num_workers=2
    )

    per_sample_path = os.path.join(logs_dir, "per_sample_mia.csv")
    with open(per_sample_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split_name", "index", "true_label", "pred_label",
            "correct", "loss", "conf_true", "conf_max", "entropy"
        ])
        log_per_sample_mia_features(model, member_loader,  device, "member",    writer)
        log_per_sample_mia_features(model, holdout_loader, device, "nonmember", writer)

    print(f"Per-sample MIA CSV saved to: {per_sample_path}")
    return per_sample_path


# =============================================================================
# SECTION 8: MAIN ENTRY POINT
# =============================================================================

def main():
    # --- Setup ---
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # This script lives in models/, so step up one level to the project root.
    project_root    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir        = os.path.join(project_root, "data")
    splits_dir      = os.path.join(project_root, "outputs", "splits")
    checkpoints_dir = os.path.join(project_root, "outputs", "checkpoints")
    logs_dir        = os.path.join(project_root, "outputs", "logs")

    for d in [splits_dir, checkpoints_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    # --- Data ---
    print("=" * 60)
    print("STEP 1: Loading data and splits")
    print("=" * 60)
    train_ds_aug, train_ds_eval, target_idx, shadow_idx, holdout_idx = \
        load_data_and_splits(data_dir, splits_dir)

    # Build DataLoaders
    target_train_loader = DataLoader(
        Subset(train_ds_aug,  target_idx),
        batch_size=128, shuffle=True,  num_workers=2
    )
    target_eval_loader  = DataLoader(
        Subset(train_ds_eval, target_idx),
        batch_size=256, shuffle=False, num_workers=2
    )
    holdout_loader      = DataLoader(
        Subset(train_ds_eval, holdout_idx),
        batch_size=256, shuffle=False, num_workers=2
    )

    # --- Train ---
    print("\n" + "=" * 60)
    print("STEP 2: Training target model (120 epochs)")
    print("=" * 60)
    model = SmallCNN(num_classes=10).to(device)
    log_path = os.path.join(logs_dir, "train_log.csv")
    train_target_model(
        model, target_train_loader, target_eval_loader,
        holdout_loader, device, checkpoints_dir, log_path, epochs=120
    )

    # --- MIA Logging ---
    print("\n" + "=" * 60)
    print("STEP 3: Generating per-sample MIA features")
    print("=" * 60)
    per_sample_path = generate_mia_csv(
        model, train_ds_eval, target_idx, holdout_idx, device, logs_dir, checkpoints_dir
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DONE — Output files:")
    print(f"  Training log : outputs/logs/train_log.csv")
    print(f"  MIA features : outputs/logs/per_sample_mia.csv")
    print(f"  Checkpoint   : outputs/checkpoints/best_baseline.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
