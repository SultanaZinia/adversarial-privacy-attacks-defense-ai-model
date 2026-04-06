# =============================================================================
# CS6413: Confidence Masking Defense (MemGuard-inspired)
# =============================================================================
# PURPOSE:
#   Post-hoc defense that does NOT retrain the model. Instead, it adds
#   calibrated noise to the model's output probabilities before they are
#   released to the querier. This masks the confidence gap between members
#   and non-members that MIA threshold attacks exploit.
#
# WHY IT WORKS AGAINST MIA:
#   MIA attacks rely on the observation that members produce higher
#   confidence and lower loss than non-members. By adding noise to the
#   output probabilities, we blur this distinction:
#     - conf_true for members gets pulled DOWN
#     - conf_true for non-members gets pushed UP (or stays noisy)
#     - The distributions overlap more → AUC drops toward 0.5
#
# APPROACH:
#   1. Load the existing best_baseline_v2.pt checkpoint (no retraining)
#   2. For each sample, compute logits → softmax → add Laplace noise
#      to the probability vector → re-normalise to valid distribution
#   3. Export the noised probabilities as MIA features
#
#   This is inspired by MemGuard (Jia et al., 2019) which adds carefully
#   crafted noise to confidence vectors. Our version uses simpler Laplace
#   noise with a tunable scale parameter.
#
# NOISE SCALE:
#   scale = 0.1  → mild noise, small privacy gain
#   scale = 0.2  → moderate noise (our default)
#   scale = 0.5  → aggressive noise, significant utility drop
#
# KEY ADVANTAGE:
#   No retraining required — can be applied to ANY existing model.
#   The model's internal weights are unchanged; only the released
#   outputs are perturbed. This makes it practical for deployment.
#
# INPUTS:
#   outputs/checkpoints/best_baseline_v2.pt
#   outputs/splits/cifar10_12k_seed42.npz
#   data/cifar-10-batches-py
#
# OUTPUTS:
#   outputs/logs/per_sample_mia_conf_masking.csv
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
                           "best_baseline_v2.pt")
LOGS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


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
# SECTION 3: MODEL (must match train_v2.py)
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
# SECTION 5: CONFIDENCE MASKING — CORE DEFENSE
# =============================================================================

def add_noise_to_probs(probs, noise_scale=0.2):
    """
    Add Laplace noise to a probability vector and re-normalise.

    Steps:
      1. Draw Laplace noise with scale=noise_scale for each logit dimension
      2. Add noise to the probability vector
      3. Clamp to [ε, 1] to avoid negative or zero probabilities
      4. Re-normalise so the vector sums to 1.0

    The Laplace distribution has heavier tails than Gaussian, which means
    occasional large perturbations — this is desirable because it prevents
    an attacker from simply averaging out the noise over multiple queries.

    Parameters
    ----------
    probs       : tensor of shape (B, C) — softmax probabilities
    noise_scale : float — Laplace scale parameter (higher = more noise)

    Returns
    -------
    noised_probs : tensor of shape (B, C) — valid probability distribution
    """
    noise = torch.zeros_like(probs).uniform_(-1, 1)
    # Convert uniform to Laplace: F^{-1}(u) = -b·sign(u)·ln(1-2|u|)
    # Simpler: use torch.distributions
    laplace = torch.distributions.Laplace(0, noise_scale)
    noise = laplace.sample(probs.shape).to(probs.device)

    noised = probs + noise
    noised = noised.clamp(min=1e-7)                    # no negatives
    noised = noised / noised.sum(dim=1, keepdim=True)  # re-normalise
    return noised


# =============================================================================
# SECTION 6: FEATURE EXTRACTION WITH MASKING
# =============================================================================

@torch.no_grad()
def log_masked_mia_features(model, loader, device, split_name, writer,
                            noise_scale=0.2):
    """
    Extract MIA features using NOISED probabilities.

    The logits are computed normally, but the softmax probabilities are
    perturbed before computing conf_true, conf_max, entropy, etc.
    This simulates what an attacker would see if the model API returned
    noised confidence scores.

    For logit-space signals (logit_true, logit_gap), we add noise to the
    raw logits as well — otherwise an attacker could bypass the defense
    by requesting logits instead of probabilities.
    """
    model.eval()
    softmax = nn.Softmax(dim=1)

    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        # ── Add noise to logits (defense against logit-space attacks) ─────
        logit_noise = torch.distributions.Laplace(0, noise_scale * 2).sample(
            logits.shape).to(device)
        noised_logits = logits + logit_noise

        # ── Noised probabilities ──────────────────────────────────────────
        probs = softmax(logits)
        noised_probs = add_noise_to_probs(probs, noise_scale=noise_scale)

        # ── Compute MIA signals from noised outputs ──────────────────────
        # Loss computed against noised logits (what attacker would measure)
        ce_per_sample = nn.CrossEntropyLoss(reduction="none")
        losses = ce_per_sample(noised_logits, y)

        pred      = noised_logits.argmax(1)
        conf_true = noised_probs.gather(1, y.view(-1,1)).squeeze(1)
        conf_max  = noised_probs.max(1).values
        correct   = (pred == y).long()
        entropy   = -(noised_probs * torch.log(noised_probs + 1e-10)).sum(1)

        logit_true = noised_logits.gather(1, y.view(-1,1)).squeeze(1)
        logits_oth = noised_logits.clone()
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
# SECTION 7: MAIN
# =============================================================================

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    NOISE_SCALE = 0.2  # Tunable: 0.1 (mild) → 0.5 (aggressive)

    # Load splits
    splits      = np.load(SPLITS_PATH)
    target_idx  = splits["target_idx"].tolist()
    holdout_idx = splits["holdout_idx"].tolist()
    print(f"Members: {len(target_idx):,}  |  Non-members: {len(holdout_idx):,}")

    # Dataset (eval transform — no augmentation)
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

    # Load existing model — NO retraining
    model = SmallCNN(num_classes=10).to(device)
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Missing checkpoint: {CKPT_PATH}\n"
            f"Run train_v2.py first to produce the baseline model."
        )
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    print(f"Loaded checkpoint: {CKPT_PATH}")
    print(f"Noise scale: {NOISE_SCALE}")

    # Export noised MIA features
    out = os.path.join(LOGS_DIR, "per_sample_mia_conf_masking.csv")
    header = [
        "split_name","index","true_label","pred_label","correct",
        "loss","conf_true","conf_max","entropy",
        "logit_true","logit_gap","m_entropy",
    ]

    print("\n" + "=" * 60)
    print("Extracting MIA features with confidence masking")
    print("=" * 60)

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        print("  Processing members...")
        log_masked_mia_features(model, mem_loader, device, "member",
                                writer, noise_scale=NOISE_SCALE)
        print("  Processing non-members...")
        log_masked_mia_features(model, hld_loader, device, "nonmember",
                                writer, noise_scale=NOISE_SCALE)

    print(f"\nMIA features saved: {out}")
    print("\n" + "=" * 60)
    print("DONE — next: run eval_all_defenses.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
