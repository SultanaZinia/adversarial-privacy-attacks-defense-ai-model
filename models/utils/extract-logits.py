# =============================================================================
# CS6413: MIA Feature Re-extraction — Raw Logit Signals
# =============================================================================
# PURPOSE:
#   The original per_sample_mia.csv only stored softmax probabilities.
#   Softmax SATURATES — logit=3 and logit=30 both map to confidence≈1.0,
#   making members and non-members indistinguishable at the extreme tail
#   where a tight threshold (TPR@1%FPR) must operate.
#
#   This script reloads the existing best_baseline.pt checkpoint and
#   re-runs inference to extract PRE-SOFTMAX logit signals:
#
#     logit_true   — raw logit of the ground-truth class
#     logit_gap    — logit_true minus the second-highest logit
#                    (a.k.a. margin / confidence gap in logit space)
#
#   Why logit_gap works for TPR@1%FPR:
#     A perfectly memorised member might have logit_gap = 20–40.
#     A non-member the model happens to classify correctly might have
#     logit_gap = 1–5.  Both give confidence ≈ 1.0 after softmax, but
#     their logit gaps are clearly separated at the tail — exactly the
#     region where a low-FPR threshold lives.
#
# INPUT:  outputs/checkpoints/best_baseline.pt
#         outputs/splits/cifar10_12k_seed42.npz
# OUTPUT: outputs/logs/per_sample_mia_v2.csv  (all original cols + new ones)
# =============================================================================

import os, csv, random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =============================================================================
# SECTION 1: PATHS
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
SPLITS_PATH = os.path.join(PROJECT_ROOT, "outputs", "splits",
                           "cifar10_12k_seed42.npz")
CKPT_PATH   = os.path.join(PROJECT_ROOT, "outputs", "checkpoints",
                           "best_baseline.pt")
OUT_CSV     = os.path.join(PROJECT_ROOT, "outputs", "logs",
                           "per_sample_mia_v2.csv")


# =============================================================================
# SECTION 2: MODEL (must match train_baseline.py exactly)
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
# SECTION 3: INDEXED DATASET (returns orig index alongside x, y)
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
# SECTION 4: FEATURE EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_features(model, loader, device, split_name, writer):
    """
    For every sample, extract BOTH the original softmax-based signals
    AND the new logit-space signals.

    New columns
    -----------
    logit_true  : raw logit for the ground-truth class (no saturation)
    logit_gap   : logit_true − max(logits for all other classes)
                  Positive = model prefers the true class.
                  Large positive = strong memorisation signal.
    logit_margin: same as logit_gap when prediction is correct;
                  negative when prediction is wrong.
    m_entropy   : modified entropy (Song & Mittal, 2021)
                  For CORRECT predictions: −entropy  (lower = more confident)
                  For INCORRECT predictions: +entropy (higher = more confused)
                  Sign flip means both directions push toward "member" for
                  high-confidence correct predictions. This resolves the
                  saturation problem for the correct-prediction subset.
    """
    model.eval()
    ce      = nn.CrossEntropyLoss(reduction="none")
    softmax = nn.Softmax(dim=1)

    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                        # shape (B, 10)
        probs  = softmax(logits)
        losses = ce(logits, y)
        pred   = logits.argmax(dim=1)

        # ── original signals ──────────────────────────────────────────────
        conf_true = probs.gather(1, y.view(-1, 1)).squeeze(1)
        conf_max  = probs.max(dim=1).values
        correct   = (pred == y).long()
        entropy   = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

        # ── new logit-space signals ───────────────────────────────────────
        # True class raw logit
        logit_true = logits.gather(1, y.view(-1, 1)).squeeze(1)

        # For logit_gap: max logit among ALL OTHER classes
        # We create a copy and set the true class column to -inf, then take max
        logits_other = logits.clone()
        logits_other.scatter_(1, y.view(-1, 1), float("-inf"))
        logit_second  = logits_other.max(dim=1).values
        logit_gap     = logit_true - logit_second   # margin in logit space

        # Modified entropy (Song & Mittal 2021):
        # flips sign for misclassified samples so the signal is monotone
        # with membership likelihood
        m_ent = torch.where(correct.bool(), -entropy, entropy)

        for i in range(x.size(0)):
            writer.writerow([
                split_name,
                int(idx[i]),
                int(y[i].item()),
                int(pred[i].item()),
                int(correct[i].item()),
                f"{losses[i].item():.6f}",
                f"{conf_true[i].item():.6f}",
                f"{conf_max[i].item():.6f}",
                f"{entropy[i].item():.6f}",
                f"{logit_true[i].item():.6f}",
                f"{logit_gap[i].item():.6f}",
                f"{m_ent[i].item():.6f}",
            ])


# =============================================================================
# SECTION 5: MAIN
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load splits ───────────────────────────────────────────────────────
    splits      = np.load(SPLITS_PATH)
    target_idx  = splits["target_idx"].tolist()
    holdout_idx = splits["holdout_idx"].tolist()
    print(f"Members:     {len(target_idx):,}")
    print(f"Non-members: {len(holdout_idx):,}")

    # ── Dataset (eval transform — no augmentation) ────────────────────────
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = datasets.CIFAR10(DATA_DIR, train=True, download=False,
                               transform=eval_tf)

    member_loader = DataLoader(
        IndexedSubset(dataset, target_idx),
        batch_size=256, shuffle=False, num_workers=2,
    )
    holdout_loader = DataLoader(
        IndexedSubset(dataset, holdout_idx),
        batch_size=256, shuffle=False, num_workers=2,
    )

    # ── Load checkpoint ───────────────────────────────────────────────────
    model = SmallCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    print(f"Loaded checkpoint: {CKPT_PATH}")

    # ── Extract and write ─────────────────────────────────────────────────
    header = [
        "split_name", "index", "true_label", "pred_label", "correct",
        "loss", "conf_true", "conf_max", "entropy",
        "logit_true", "logit_gap", "m_entropy",
    ]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        print("Extracting member features...")
        extract_features(model, member_loader,  device, "member",    writer)
        print("Extracting non-member features...")
        extract_features(model, holdout_loader, device, "nonmember", writer)

    print(f"\nSaved → {OUT_CSV}")
    print("Next step: run attack_v2.py")


if __name__ == "__main__":
    main()
