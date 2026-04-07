import os
import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Add project root to path so we can cleanly import from models.train_v3
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.train_v3 import (
    LargeCNN, IndexedSubset, set_seed, load_data_and_splits, eval_accuracy, log_mia_features
)


def distillation_loss_fn(student_logits, teacher_logits, labels, T, alpha):
    """
    Computes KD loss given student logits, teacher logits, and true labels.
    - Soft loss is the KL divergence between softened probabilities.
    - Hard loss is standard Cross Entropy.
    """
    hard_loss = F.cross_entropy(student_logits, labels)
    
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    soft_probs   = F.log_softmax(student_logits / T, dim=1)
    
    # kl_div expects log probabilities for input and probabilities for target.
    # Multiply by T^2 to scale the gradients to be roughly the same magnitude as the hard loss.
    soft_loss = F.kl_div(soft_probs, soft_targets, reduction='batchmean') * (T * T)
    
    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def train_kd(student, teacher, train_loader, eval_loader, holdout_loader,
             device, ckpt_dir, log_path, epochs=120, T=3.0, alpha=0.7):
    """
    Training loop that replaces standard CE Loss with Knowledge Distillation.
    teacher model is in strictly eval() mode.
    """
    opt   = optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 90], gamma=0.5)
    
    best_acc = 0.0

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch","train_loss","train_acc","member_eval_acc",
            "holdout_acc","best_acc","lr"
        ])

    for epoch in range(1, epochs + 1):
        student.train()
        loss_sum = correct = total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # 1. Forward passes
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)
            
            # 2. KD Loss extraction
            loss = distillation_loss_fn(student_logits, teacher_logits, y, T=T, alpha=alpha)
            
            if not torch.isfinite(loss):
                print(f"Non-finite loss at epoch {epoch}. Stopping.")
                return student

            # 3. Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_sum += loss.item() * x.size(0)
            correct  += (student_logits.argmax(1) == y).sum().item()
            total    += y.size(0)

        sched.step()
        current_lr = opt.param_groups[0]["lr"]

        train_loss = loss_sum / total
        train_acc  = correct / total
        mem_acc    = eval_accuracy(student, eval_loader,    device)
        hld_acc    = eval_accuracy(student, holdout_loader, device)

        # Save checkpoint strictly on holdout/generalization accuracy
        if hld_acc > best_acc:
            best_acc = hld_acc
            torch.save(student.state_dict(), os.path.join(ckpt_dir, "best_student_kd_v3.pt"))

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

    print(f"\nBest student holdout acc: {best_acc:.4f}")
    print(f"GOAL (KD Defended): member_acc should be close to holdout_acc (narrow gap)")
    return student


def export_mia_csv_kd(student, ds_eval, target_idx, holdout_idx, device, ckpt_dir, logs_dir):
    ckpt = os.path.join(ckpt_dir, "best_student_kd_v3.pt")
    student.load_state_dict(torch.load(ckpt, map_location=device))
    student.to(device)
    print(f"\nLoaded KD Student checkpoint: {ckpt}")

    mem_loader = DataLoader(
        IndexedSubset(ds_eval, target_idx),
        batch_size=256, shuffle=False, num_workers=2)
    hld_loader = DataLoader(
        IndexedSubset(ds_eval, holdout_idx),
        batch_size=256, shuffle=False, num_workers=2)

    out = os.path.join(logs_dir, "per_sample_mia_v3_kd.csv")
    header = [
        "split_name","index","true_label","pred_label","correct",
        "loss","conf_true","conf_max","entropy",
        "logit_true","logit_gap","m_entropy",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        log_mia_features(student, mem_loader, device, "member",    writer)
        log_mia_features(student, hld_loader, device, "nonmember", writer)

    print(f"KD Defended MIA CSV saved → {out}")
    return out


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    data_dir        = os.path.join(PROJECT_ROOT, "data")
    splits_dir      = os.path.join(PROJECT_ROOT, "outputs", "splits")
    checkpoints_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    logs_dir        = os.path.join(PROJECT_ROOT, "outputs", "logs")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 80)
    print("KNOWLEDGE DISTILLATION DEFENSE (v3 Architecture)")
    print("=" * 80)

    # 1. Provide identical data 
    print("\n[1/3] Loading data and splits")
    ds_aug, ds_eval, target_idx, shadow_idx, holdout_idx = load_data_and_splits(data_dir, splits_dir)

    train_loader = DataLoader(Subset(ds_aug,  target_idx),  batch_size=128, shuffle=True,  num_workers=2)
    eval_loader  = DataLoader(Subset(ds_eval, target_idx),  batch_size=256, shuffle=False, num_workers=2)
    hld_loader   = DataLoader(Subset(ds_eval, holdout_idx), batch_size=256, shuffle=False, num_workers=2)

    # 2. Setup Models
    print(f"\n[2/3] Setting up Teacher and Student networks")
    teacher = LargeCNN(num_classes=10).to(device)
    teacher_ckpt = os.path.join(checkpoints_dir, "best_baseline_v3.pt")
    if not os.path.exists(teacher_ckpt):
        print(f"ERROR: Teacher model {teacher_ckpt} not found! Run train_v3.py first.")
        sys.exit(1)
    
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval() # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    student = LargeCNN(num_classes=10).to(device)

    # KD params
    TEMPERATURE = 3.0
    ALPHA = 0.7  # 70% soft label focus, 30% hard label
    
    print(f"Distillation active: T={TEMPERATURE}, Alpha={ALPHA}")

    log_path = os.path.join(logs_dir, "train_log_v3_kd.csv")
    train_kd(student, teacher, train_loader, eval_loader, hld_loader,
             device, checkpoints_dir, log_path, epochs=120, T=TEMPERATURE, alpha=ALPHA)

    # 3. Export
    print("\n[3/3] Exporting KD-defended MIA features for evaluation")
    export_mia_csv_kd(student, ds_eval, target_idx, holdout_idx, device, checkpoints_dir, logs_dir)

    print("\nDONE — Model logic implemented correctly. Next: run defense/evaluate_kd_v3.py")

if __name__ == "__main__":
    main()
