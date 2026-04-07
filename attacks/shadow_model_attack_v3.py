#!/usr/bin/env python3
# =============================================================================
# Shadow Model Membership Inference Attack  (Shokri et al., 2017)
# =============================================================================
# OVERVIEW
# --------
# Classic MIA in which an attacker trains "shadow" copies of the target model
# to generate labelled (feature_vector → member / non-member) training data
# for a set of binary attack classifiers — one per CIFAR-10 class.
#
# IMPROVEMENTS OVER BASELINE
# --------------------------
#  1. Richer 23-dim feature vector per sample:
#       - 10-dim softmax confidence vector
#       - 10-dim sorted (descending) confidence vector
#       - 1-dim Shannon entropy
#       - 1-dim max confidence
#       - 1-dim per-sample cross-entropy loss  (single strongest MIA signal)
#  2. More shadow models (8 vs 4) trained for more epochs (120 vs 80) to
#     match the target's memorisation level.  Weight decay removed so shadows
#     overfit at least as much as the target.
#  3. Multi-augmentation querying: each sample is queried N_AUG_QUERIES times
#     with random augmentation; the averaged confidence vector is more robust.
#  4. Deeper AttackMLP (→128→64→1) with Dropout(0.3) per hidden layer.
#  5. Threshold corrected for the 2:1 member/non-member evaluation imbalance:
#     decision boundary shifted to log(2) ≈ 0.693 (Bayes-optimal under 2:1
#     prior given a classifier trained on balanced data).
#
# PIPELINE
# --------
#  1. Load the pre-trained target model and the fixed index splits.
#  2. Build a shadow pool from all CIFAR-10 training indices NOT used by the
#     target (≈44 K samples).
#  3. Train NUM_SHADOWS shadow models (same LargeCNN architecture).
#     Each shadow model trains on SHADOW_IN_SIZE random samples from the pool;
#     the complementary SHADOW_OUT_SIZE samples act as its "non-members".
#  4. Collect (feature_vector, class_label, is_member) from every shadow model
#     using multi-augmentation querying and the enriched feature extractor.
#  5. Train one binary AttackMLP per CIFAR-10 class on those shadow outputs.
#  6. Evaluate the attack on the target model:
#       - members   : target_idx  (6 K samples the target was trained on)
#       - non-members: holdout_idx (3 K samples the target never saw)
#  7. Report accuracy, precision, recall, F1, AUC-ROC; save CSV results.
#
# OUTPUT FILES
# ------------
#   outputs/checkpoints/shadow_models/shadow_<n>.pt
#   outputs/checkpoints/attack_models/attack_class_<c>.pt
#   outputs/logs/shadow_mia_v3_results.csv
#   outputs/logs/shadow_mia_v3_per_sample.csv
# =============================================================================

import os
import csv
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


# =============================================================================
# PATHS  (resolve relative to this file's location)
# =============================================================================
ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data"
OUT_DIR       = ROOT / "outputs"
SPLITS_FILE   = OUT_DIR / "splits"  / "cifar10_12k_seed42.npz"
TARGET_CKPT   = OUT_DIR / "checkpoints" / "best_baseline_v3.pt"
SHADOW_DIR    = OUT_DIR / "checkpoints" / "shadow_models_v3"
ATTACK_DIR    = OUT_DIR / "checkpoints" / "attack_models_v3"
RESULTS_DIR   = OUT_DIR / "logs"

for _d in [SHADOW_DIR, ATTACK_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HYPER-PARAMETERS
# =============================================================================
SEED            = 42
NUM_SHADOWS     = 8           # was 4 — more shadows → better attack coverage
SHADOW_IN_SIZE  = 3_000       # training ("in") samples per shadow model
SHADOW_OUT_SIZE = 3_000       # held-out ("out") samples per shadow model
SHADOW_EPOCHS   = 120         # was 80 — match target training to replicate its memorisation
SHADOW_LR       = 0.01
ATTACK_EPOCHS   = 100         # was 60
ATTACK_LR       = 1e-3
NUM_CLASSES     = 10
BATCH_SIZE      = 128
N_AUG_QUERIES   = 10          # number of augmented queries averaged per sample
FEATURE_DIM     = 23          # 10 conf + 10 sorted + entropy + max_conf + loss

# Threshold corrected for 2:1 member/non-member evaluation imbalance.
# A classifier trained on balanced (1:1) data outputs logits calibrated to a
# 50/50 prior.  Under the actual 2:1 prior the Bayes-optimal decision boundary
# shifts by log(P(member)/P(non-member)) = log(2).
MEMBER_THRESHOLD = np.log(2.0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed()


# =============================================================================
# MODEL ARCHITECTURE  (identical to target model in train_baseline.py)
# =============================================================================
class LargeCNN(nn.Module):
    """Balanced CNN with moderate filters for speed + memorisation."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(96 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# ATTACK CLASSIFIER  (one per CIFAR-10 class)
# =============================================================================
class AttackMLP(nn.Module):
    """
    Binary MLP: feature_vector (dim=FEATURE_DIM) → logit for P(is_member).

    Input  : 23-dim enriched feature vector (softmax, sorted softmax, entropy,
             max confidence, per-sample cross-entropy loss).
    Output : scalar logit  (positive → predict "member").

    Following Shokri et al. we train one AttackMLP per class and at
    inference time route each sample through the classifier that matches
    its (predicted) true class.

    Compared to the baseline:
      - Input dimension 23 (was 10 — raw softmax only)
      - Deeper: 128 → 64 → 1  (was 64 → 32 → 1)
      - Dropout(0.3) after each hidden layer to prevent overfitting on
        the (much richer) feature space
    """

    def __init__(self, in_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,     64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,       1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (batch,) logits


# =============================================================================
# DATA / TRANSFORMS
# =============================================================================
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

_train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
_eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def load_cifar10_train(transform):
    return datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=False, transform=transform
    )


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def make_attack_features(conf: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Build the enriched 23-dim attack feature vector from a softmax matrix.

    Components
    ----------
    conf        (N, 10) : raw softmax confidence vector
    sorted_conf (N, 10) : same values sorted in descending order — the attack
                          model learns relative confidence gaps without needing
                          to know which class is which
    entropy     (N,  1) : Shannon entropy H = -Σ p·log(p); low entropy → the
                          model is very confident → likely a member
    max_conf    (N,  1) : maximum softmax probability (redundant with sorted[0]
                          but keeps the signal explicitly available)
    loss        (N,  1) : per-sample cross-entropy = -log(conf[true_label]);
                          members tend to have low loss, non-members high loss —
                          this is the single strongest MIA signal
    """
    entropy     = -(conf * (conf + 1e-9).log()).sum(dim=1, keepdim=True)
    max_conf    = conf.max(dim=1, keepdim=True).values
    sorted_conf = conf.sort(dim=1, descending=True).values
    loss        = -conf[torch.arange(len(labels)), labels].log().unsqueeze(1)
    return torch.cat([conf, sorted_conf, entropy, max_conf, loss], dim=1)  # (N, 23)


# =============================================================================
# HELPER: extract enriched attack features from a model
# =============================================================================
@torch.no_grad()
def get_attack_features(
    model:       nn.Module,
    ds_aug,                   # augmented dataset (for multi-aug querying)
    ds_eval,                  # clean eval dataset (for label extraction only)
    indices,
    n_aug:       int = N_AUG_QUERIES,
    batch_size:  int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Query `model` on `dataset[indices]` and return enriched attack features.

    Multi-augmentation querying
    ---------------------------
    Each sample is passed through the model `n_aug` times with independent
    random augmentations (RandomCrop + RandomHorizontalFlip).  The resulting
    confidence vectors are averaged before feature engineering.  This reduces
    noise in the confidence estimate and gives the attack model a cleaner
    signal — especially useful at the decision boundary.

    Returns
    -------
    features : (N, FEATURE_DIM)  float32 — enriched feature vectors
    labels   : (N,)              int64   — true CIFAR-10 class labels
    """
    idx_list = list(indices)

    # True labels from clean (non-augmented) dataset
    label_loader = DataLoader(
        Subset(ds_eval, idx_list),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    all_labels = []
    for _, lbl in label_loader:
        all_labels.append(lbl)
    all_labels = torch.cat(all_labels)   # (N,)

    # Average confidence over n_aug independent augmented forward passes
    model.eval()
    avg_conf = None
    for _ in range(n_aug):
        loader = DataLoader(
            Subset(ds_aug, idx_list),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
        batch_conf = []
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            probs = torch.softmax(model(imgs), dim=1).cpu()
            batch_conf.append(probs)
        conf = torch.cat(batch_conf)                       # (N, C)
        avg_conf = conf if avg_conf is None else avg_conf + conf

    avg_conf = avg_conf / n_aug                            # (N, C)

    features = make_attack_features(avg_conf, all_labels)  # (N, FEATURE_DIM)
    return features, all_labels


# =============================================================================
# SHADOW POOL CONSTRUCTION
# =============================================================================
def build_shadow_pool(target_idx: np.ndarray, full_train_size: int = 50_000):
    """
    Return all CIFAR-10 training indices that the target model was never
    trained on.  This is the attacker's "shadow dataset".

    excluded = target_idx ∪ shadow_idx ∪ holdout_idx (all 12 K splits)
    We exclude all 12 K to keep a clean separation: shadow models are built
    from data the attacker is assumed to have independent access to.
    """
    splits      = np.load(SPLITS_FILE)
    used        = set(splits["target_idx"].tolist()  +
                      splits["shadow_idx"].tolist()  +
                      splits["holdout_idx"].tolist())
    shadow_pool = np.array([i for i in range(full_train_size) if i not in used])
    print(f"[INFO] Shadow pool: {len(shadow_pool):,} samples "
          f"(CIFAR-10 train minus the 12 K fixed split)")
    return shadow_pool


# =============================================================================
# SHADOW MODEL TRAINING
# =============================================================================
def train_one_shadow(
    model: LargeCNN,
    in_idx,        # indices to train on  ("members")
    ds_aug,        # augmented dataset for training
    epochs: int,
    lr: float,
    desc: str,
):
    loader = DataLoader(
        Subset(ds_aug, list(in_idx)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    opt   = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9,
        weight_decay=0.0,
    )
    sched = optim.lr_scheduler.MultiStepLR(
        opt, milestones=[60, 90], gamma=0.5
    )
    ce = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = ce(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(imgs)
        sched.step()
        if ep % 20 == 0:
            n = len(loader.dataset)
            print(f"    [{desc}] epoch {ep:3d}/{epochs}  "
                  f"avg_loss={total_loss / n:.4f}")
    return model


def collect_shadow_data(shadow_pool: np.ndarray):
    """
    Train NUM_SHADOWS shadow models, each on a fresh random SHADOW_IN_SIZE
    subset of the shadow pool.  Collect and return the combined attack
    training dataset as tensors using the enriched feature extractor.

    Returns
    -------
    features : (N_total, FEATURE_DIM)  float32 — enriched attack feature vectors
    labels   : (N_total,)              int64   — true class labels
    members  : (N_total,)              float32 — 1 = in, 0 = out
    """
    ds_aug  = load_cifar10_train(_train_tf)
    ds_eval = load_cifar10_train(_eval_tf)

    rng             = np.random.default_rng(SEED)
    all_feats, all_labels, all_members = [], [], []

    for s in range(1, NUM_SHADOWS + 1):
        print(f"\n── Shadow model {s}/{NUM_SHADOWS} "
              f"{'─' * (50 - len(str(s)))}")

        perm    = rng.permutation(len(shadow_pool))
        in_idx  = shadow_pool[perm[:SHADOW_IN_SIZE]]
        out_idx = shadow_pool[perm[SHADOW_IN_SIZE : SHADOW_IN_SIZE + SHADOW_OUT_SIZE]]

        model = LargeCNN().to(DEVICE)
        train_one_shadow(model, in_idx, ds_aug,
                         SHADOW_EPOCHS, SHADOW_LR, f"shadow-{s}")

        # Persist checkpoint
        torch.save(model.state_dict(), SHADOW_DIR / f"shadow_{s}.pt")

        # Extract enriched feature vectors (multi-aug querying)
        in_feats,  in_lbl  = get_attack_features(model, ds_aug, ds_eval, in_idx)
        out_feats, out_lbl = get_attack_features(model, ds_aug, ds_eval, out_idx)

        all_feats.append(torch.cat([in_feats,  out_feats]))
        all_labels.append(torch.cat([in_lbl,   out_lbl]))
        all_members.append(torch.cat([
            torch.ones(len(in_idx),  dtype=torch.float32),
            torch.zeros(len(out_idx), dtype=torch.float32),
        ]))

        print(f"  Collected {len(in_idx):,} in + {len(out_idx):,} out samples")

    return (
        torch.cat(all_feats),    # (N, FEATURE_DIM)
        torch.cat(all_labels),   # (N,)
        torch.cat(all_members),  # (N,)
    )


# =============================================================================
# ATTACK CLASSIFIER TRAINING (per class)
# =============================================================================
def train_attack_classifiers(
    feats:   torch.Tensor,   # (N, FEATURE_DIM)
    labels:  torch.Tensor,   # (N,)
    members: torch.Tensor,   # (N,)  float 0/1
):
    """
    Train one binary AttackMLP per CIFAR-10 class.

    Shokri et al. route samples through the class-specific attack model
    because each class may have a distinctive confidence profile, and
    training class-specific classifiers exploits that structure.

    Returns a list of NUM_CLASSES AttackMLP objects (None if a class had
    fewer than 10 samples in the shadow data — practically never happens).
    """
    attack_models = []

    for c in range(NUM_CLASSES):
        mask = (labels == c)
        n_c  = mask.sum().item()

        if n_c < 10:
            print(f"  [class {c:2d}] only {n_c} samples — skipping")
            attack_models.append(None)
            continue

        X = feats[mask]    # (n_c, FEATURE_DIM)
        y = members[mask]  # (n_c,)

        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

        m    = AttackMLP().to(DEVICE)
        opt  = optim.Adam(m.parameters(), lr=ATTACK_LR)
        loss_fn = nn.BCEWithLogitsLoss()

        m.train()
        for _ in range(ATTACK_EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss_fn(m(xb), yb).backward()
                opt.step()

        torch.save(m.state_dict(), ATTACK_DIR / f"attack_class_{c}.pt")
        attack_models.append(m)
        print(f"  [class {c:2d}] attack model trained  "
              f"({n_c:,} samples, "
              f"{int(y.sum()):,} in / {int((1 - y).sum()):,} out)")

    return attack_models


# =============================================================================
# EVALUATION ON TARGET MODEL
# =============================================================================
def evaluate_attack(attack_models, target_idx, holdout_idx):
    """
    Run each per-class attack classifier against the real target model.

    Members   = target_idx  (label 1)
    Non-members = holdout_idx (label 0)

    Decision threshold is set to MEMBER_THRESHOLD = log(2) ≈ 0.693 rather
    than 0 to correct for the 2:1 member/non-member evaluation imbalance.
    The attack classifiers are trained on balanced (1:1) shadow data, so
    their logits assume a 50/50 prior.  Shifting the threshold by log(2)
    gives the Bayes-optimal boundary under the actual 2:1 evaluation prior.
    """
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix,
        )
    except ImportError:
        print("[ERROR] scikit-learn is not installed.  Run:  pip install scikit-learn")
        sys.exit(1)

    print("\n── Loading target model ─────────────────────────────────────────")
    target = LargeCNN().to(DEVICE)
    target.load_state_dict(torch.load(TARGET_CKPT, map_location=DEVICE))
    target.eval()

    ds_aug  = load_cifar10_train(_train_tf)
    ds_eval = load_cifar10_train(_eval_tf)

    print("[INFO] Extracting enriched features from target model (multi-aug) …")
    mem_feats,  mem_lbl  = get_attack_features(target, ds_aug, ds_eval, target_idx)
    non_feats,  non_lbl  = get_attack_features(target, ds_aug, ds_eval, holdout_idx)

    all_feats   = torch.cat([mem_feats,  non_feats])   # (N, FEATURE_DIM)
    all_labels  = torch.cat([mem_lbl,   non_lbl])      # (N,)
    all_members = torch.cat([
        torch.ones(len(target_idx),  dtype=torch.float32),
        torch.zeros(len(holdout_idx), dtype=torch.float32),
    ])

    # Route each sample through its class-specific attack model
    preds  = torch.zeros(len(all_members))
    logits = torch.zeros(len(all_members))

    for c in range(NUM_CLASSES):
        m = attack_models[c]
        if m is None:
            continue
        mask = (all_labels == c)
        if mask.sum() == 0:
            continue
        X = all_feats[mask].to(DEVICE)
        with torch.no_grad():
            lg = m(X).cpu()
        logits[mask] = lg
        # Threshold corrected for 2:1 member/non-member imbalance
        preds[mask]  = (lg > MEMBER_THRESHOLD).float()

    y_true  = all_members.numpy()
    y_pred  = preds.numpy()
    y_score = torch.sigmoid(logits).numpy()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred,    zero_division=0)
    f1   = f1_score(y_true, y_pred,        zero_division=0)
    auc  = roc_auc_score(y_true, y_score)
    cm   = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    adv       = 2.0 * (rec - fpr)        # attacker advantage ∈ [−1, 1]

    # ── Print results ──────────────────────────────────────────────────────
    width = 58
    print("\n" + "═" * width)
    print("  Shadow Model Membership Inference Attack — Results")
    print("═" * width)
    print(f"  Attack dataset  : {len(y_true):,} samples "
          f"({int(y_true.sum()):,} members / {int((1-y_true).sum()):,} non-members)")
    print(f"  Shadow models   : {NUM_SHADOWS}  ×  "
          f"{SHADOW_IN_SIZE:,} in / {SHADOW_OUT_SIZE:,} out")
    print(f"  Aug queries     : {N_AUG_QUERIES}  per sample")
    print(f"  Feature dim     : {FEATURE_DIM}  (conf + sorted + entropy + max + loss)")
    print(f"  Decision thresh : {MEMBER_THRESHOLD:.4f}  (corrected for 2:1 prior)")
    print("─" * width)
    print(f"  Accuracy        : {acc:.4f}   "
          f"(baseline ≈ {int(y_true.sum())/len(y_true):.4f})")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall (TPR)    : {rec:.4f}")
    print(f"  False Pos. Rate : {fpr:.4f}")
    print(f"  F1              : {f1:.4f}")
    print(f"  AUC-ROC         : {auc:.4f}   (random = 0.5000)")
    print(f"  Attacker Adv.   : {adv:.4f}   (random = 0.0000)")
    print("─" * width)
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"    TN={tn:5d}  FP={fp:5d}")
    print(f"    FN={fn:5d}  TP={tp:5d}")
    print("═" * width + "\n")

    # ── Save aggregate results ─────────────────────────────────────────────
    results = {
        "attack":             "shadow_model_mia",
        "num_shadows":        NUM_SHADOWS,
        "shadow_in_size":     SHADOW_IN_SIZE,
        "shadow_out_size":    SHADOW_OUT_SIZE,
        "shadow_epochs":      SHADOW_EPOCHS,
        "n_aug_queries":      N_AUG_QUERIES,
        "feature_dim":        FEATURE_DIM,
        "member_threshold":   round(MEMBER_THRESHOLD, 6),
        "n_members":          int(y_true.sum()),
        "n_nonmembers":       int((1 - y_true).sum()),
        "accuracy":           round(acc,  6),
        "precision":          round(prec, 6),
        "recall_tpr":         round(rec,  6),
        "fpr":                round(fpr,  6),
        "f1":                 round(f1,   6),
        "auc_roc":            round(auc,  6),
        "attacker_advantage": round(adv,  6),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    csv_path = RESULTS_DIR / "shadow_mia_v3_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results.keys()))
        w.writeheader()
        w.writerow(results)
    print(f"[INFO] Aggregate results   → {csv_path}")

    # ── Save per-sample scores ─────────────────────────────────────────────
    per_sample_path = RESULTS_DIR / "shadow_mia_v3_per_sample.csv"
    with open(per_sample_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_member", "predicted_member", "attack_score", "true_class"])
        for i in range(len(y_true)):
            w.writerow([
                int(y_true[i]),
                int(y_pred[i]),
                f"{y_score[i]:.6f}",
                int(all_labels[i].item()),
            ])
    print(f"[INFO] Per-sample scores   → {per_sample_path}")

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  Shadow Model Membership Inference Attack")
    print("  (Shokri et al., 2017 — improved)")
    print("=" * 60)

    # 1. Load fixed splits ────────────────────────────────────────────────────
    if not SPLITS_FILE.exists():
        print(f"[ERROR] Splits file not found: {SPLITS_FILE}")
        print("        Run  models/train_baseline.py  first.")
        sys.exit(1)
    if not TARGET_CKPT.exists():
        print(f"[ERROR] Target checkpoint not found: {TARGET_CKPT}")
        print("        Run  models/train_baseline.py  first.")
        sys.exit(1)

    splits      = np.load(SPLITS_FILE)
    target_idx  = splits["target_idx"]
    holdout_idx = splits["holdout_idx"]
    print(f"[INFO] Target members   : {len(target_idx):,}")
    print(f"[INFO] Non-members      : {len(holdout_idx):,}")

    # 2. Shadow pool ──────────────────────────────────────────────────────────
    shadow_pool = build_shadow_pool(target_idx)

    # 3. Train shadow models and collect attack training data ─────────────────
    t0 = time.time()
    print(f"\n[STEP 1/3] Training {NUM_SHADOWS} shadow models "
          f"({SHADOW_EPOCHS} epochs each) …")
    feats, labels, members = collect_shadow_data(shadow_pool)
    elapsed = (time.time() - t0) / 60
    print(f"\n[INFO] Shadow training complete in {elapsed:.1f} min")
    print(f"[INFO] Attack training set: {len(members):,} samples  "
          f"({int(members.sum()):,} in / {int((1 - members).sum()):,} out)")

    # 4. Train per-class attack classifiers ───────────────────────────────────
    print(f"\n[STEP 2/3] Training {NUM_CLASSES} per-class attack classifiers …")
    attack_models = train_attack_classifiers(feats, labels, members)

    # 5. Evaluate on the real target model ────────────────────────────────────
    print(f"\n[STEP 3/3] Evaluating attack on target model …")
    results = evaluate_attack(attack_models, target_idx, holdout_idx)

    print("[DONE] Shadow model attack complete.")
    return results


if __name__ == "__main__":
    main()
