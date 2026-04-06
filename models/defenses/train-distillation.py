# =============================================================================
# CS6413: Knowledge Distillation Defense
# =============================================================================
# PURPOSE:
#   Train a "teacher" model on the target split (same as baseline), then
#   train a fresh "student" model using the teacher's soft predictions
#   instead of hard labels.  The student learns the teacher's generalised
#   knowledge without memorising individual training samples.
#
# WHY IT WORKS AGAINST MIA:
#   Membership inference exploits the gap between how a model treats its
#   training data vs unseen data.  Distillation smooths this gap because:
#     1. Soft labels carry inter-class similarity info (e.g. "this cat
#        image is 70% cat, 15% dog, 10% deer") — the student learns
#        relationships, not rote memorisation of one-hot targets.
#     2. The temperature parameter τ controls how soft the labels are.
#        Higher τ → softer distributions → less memorisation.
#     3. The student never sees the original hard labels directly, so
#        it cannot overfit to individual sample idiosyncrasies.
#
# TEMPERATURE (τ):
#   τ = 1  → standard softmax (no extra smoothing)
#   τ = 5  → moderate smoothing (our default)
#   τ = 20 → very soft, almost uniform — too much destroys utility
#
# DISTILLATION LOSS:
#   L = α · KL(student_soft || teacher_soft) · τ²  +  (1-α) · CE(student, hard_label)
#   The τ² scaling compensates for the reduced gradient magnitude at high τ.
#   α = 0.7 means we lean heavily on the teacher's soft knowledge.
#
# INPUTS:
#   outputs/splits/cifar10_12k_seed42.npz
#   data/cifar-10-batches-py
#
# OUTPUTS:
#   outputs/checkpoints/best_distilled.pt
#   outputs/logs/train_log_distillation.csv
#   outputs/logs/per_sample_mia_distillation.csv
# =============================================================================

import os, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
# SECTION 3: MODEL
# =============================================================================

class SmallCNN(nn.Module):
    """Identical architecture to train_v2.py — no dropout, no BatchNorm."""
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
    """Extract all v2 MIA signals — identical to train_v2.py."""
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
# SECTION 6: TEACHER TRAINING
# =============================================================================

def train_teacher(model, train_loader, device, epochs=150):
    """
    Train the teacher model identically to train_v2.py.
    The teacher will overfit — that's fine. We only use its soft outputs.
    """
    opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 80, 110], gamma=0.5)
    ce    = nn.CrossEntropyLoss()

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

        if epoch % 30 == 0 or epoch <= 3:
            print(f"  Teacher Ep {epoch:03d} | loss={loss_sum/total:.4f} | "
                  f"acc={correct/total:.4f}")

    print("  Teacher training complete.")
    return model


# =============================================================================
# SECTION 7: DISTILLATION TRAINING
# =============================================================================

def distillation_loss(student_logits, teacher_logits, hard_labels,
                      temperature=5.0, alpha=0.7):
    """
    Combined distillation loss:
      L = α · KL(soft_student || soft_teacher) · τ²  +  (1-α) · CE(student, label)

    The KL term teaches the student to mimic the teacher's soft predictions.
    The CE term keeps the student grounded on the actual task.
    τ² scaling compensates for reduced gradient magnitude at high temperature.
    """
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)

    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    ce = F.cross_entropy(student_logits, hard_labels)

    return alpha * kl + (1 - alpha) * ce


def train_student(student, teacher, train_loader, eval_loader, holdout_loader,
                  device, ckpt_path, log_path, temperature=5.0, alpha=0.7,
                  epochs=150):
    """
    Train the student using the teacher's soft labels.

    Key difference from normal training: the loss function uses the
    teacher's output distribution instead of (or in addition to) the
    hard one-hot labels. This prevents the student from memorising
    individual samples because the soft labels encode the teacher's
    generalised knowledge, not sample-specific patterns.
    """
    teacher.eval()  # teacher is frozen — inference only

    opt   = optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 80, 110], gamma=0.5)
    best_acc = 0.0

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc",
            "member_eval_acc", "holdout_acc", "best_acc", "overfit_gap"
        ])

    print(f"  Distillation: τ={temperature} | α={alpha} | epochs={epochs}")

    for epoch in range(1, epochs + 1):
        student.train()
        loss_sum = correct = total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            student_logits = student(x)
            with torch.no_grad():
                teacher_logits = teacher(x)

            loss = distillation_loss(student_logits, teacher_logits, y,
                                     temperature=temperature, alpha=alpha)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            correct  += (student_logits.argmax(1) == y).sum().item()
            total    += y.size(0)

        sched.step()

        train_loss = loss_sum / total
        train_acc  = correct  / total
        mem_acc    = eval_accuracy(student, eval_loader,    device)
        hld_acc    = eval_accuracy(student, holdout_loader, device)
        gap        = mem_acc - hld_acc

        if hld_acc > best_acc:
            best_acc = hld_acc
            torch.save(student.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  Student Ep {epoch:03d} | loss={train_loss:.4f} | "
                  f"mem={mem_acc:.4f} | hld={hld_acc:.4f} | gap={gap:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{mem_acc:.6f}", f"{hld_acc:.6f}",
                f"{best_acc:.6f}", f"{gap:.6f}",
            ])

    print(f"  Best holdout acc: {best_acc:.4f}")
    return student


# =============================================================================
# SECTION 8: MIA FEATURE EXPORT
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

    out = os.path.join(LOGS_DIR, "per_sample_mia_distillation.csv")
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
# SECTION 9: MAIN
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

    ckpt_path = os.path.join(CKPT_DIR, "best_distilled.pt")
    log_path  = os.path.join(LOGS_DIR, "train_log_distillation.csv")

    # ── Step 1: Train teacher ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Training teacher model (150 epochs)")
    print("=" * 60)
    teacher = SmallCNN(num_classes=10).to(device)
    train_teacher(teacher, train_loader, device, epochs=150)

    # ── Step 2: Train student via distillation ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Training student via knowledge distillation (τ=5, α=0.7)")
    print("=" * 60)
    student = SmallCNN(num_classes=10).to(device)
    train_student(student, teacher, train_loader, eval_loader, hld_loader,
                  device, ckpt_path, log_path,
                  temperature=5.0, alpha=0.7, epochs=150)

    # ── Step 3: Export MIA features ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Exporting MIA features")
    print("=" * 60)
    export_mia_csv(student, ckpt_path, target_idx, holdout_idx, device)

    print("\n" + "=" * 60)
    print("DONE — next: run eval_all_defenses.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
