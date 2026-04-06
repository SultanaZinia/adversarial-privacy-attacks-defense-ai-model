# =============================================================================
# CS6413: Shadow-Model MIA v2 — Learned Attack Classifier
# =============================================================================
# PURPOSE:
#   Train a shadow model on the train_v2 fixed split, extract v2 MIA features
#   (including logit signals), train a learned attack classifier, and evaluate
#   it on target-model features from per_sample_mia_v2.csv.
#
# INPUTS:
#   outputs/splits/cifar10_12k_seed42.npz
#   outputs/logs/per_sample_mia_v2.csv
#   data/cifar-10-batches-py
#
# OUTPUTS:
#   outputs/reports/attack_shadow_v2/shadow_features_v2.csv
#   outputs/reports/attack_shadow_v2/attack_eval_target.csv
#   outputs/reports/attack_shadow_v2/shadow_attack_roc.png
#   outputs/reports/attack_shadow_v2/summary_shadow_v2.txt
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLITS_PATH = os.path.join(PROJECT_ROOT, "outputs", "splits", "cifar10_12k_seed42.npz")
TARGET_MIA_V2 = os.path.join(PROJECT_ROOT, "outputs", "logs", "per_sample_mia_v2.csv")

REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports", "attack_shadow_v2")
SHADOW_FEATURES_CSV = os.path.join(REPORT_DIR, "shadow_features_v2.csv")
TARGET_EVAL_CSV = os.path.join(REPORT_DIR, "attack_eval_target.csv")
ROC_PNG = os.path.join(REPORT_DIR, "shadow_attack_roc.png")
SUMMARY_TXT = os.path.join(REPORT_DIR, "summary_shadow_v2.txt")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


@torch.no_grad()
def collect_v2_features(model, loader, device, split_name):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="none")
    softmax = nn.Softmax(dim=1)
    rows = []

    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = softmax(logits)
        losses = ce(logits, y)
        pred = logits.argmax(dim=1)

        conf_true = probs.gather(1, y.view(-1, 1)).squeeze(1)
        conf_max = probs.max(dim=1).values
        correct = (pred == y).long()
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

        logit_true = logits.gather(1, y.view(-1, 1)).squeeze(1)
        logits_other = logits.clone()
        logits_other.scatter_(1, y.view(-1, 1), float("-inf"))
        logit_gap = logit_true - logits_other.max(dim=1).values
        m_entropy = torch.where(correct.bool(), -entropy, entropy)

        for i in range(x.size(0)):
            rows.append(
                {
                    "split_name": split_name,
                    "index": int(idx[i]),
                    "true_label": int(y[i].item()),
                    "pred_label": int(pred[i].item()),
                    "correct": int(correct[i].item()),
                    "loss": float(losses[i].item()),
                    "conf_true": float(conf_true[i].item()),
                    "conf_max": float(conf_max[i].item()),
                    "entropy": float(entropy[i].item()),
                    "logit_true": float(logit_true[i].item()),
                    "logit_gap": float(logit_gap[i].item()),
                    "m_entropy": float(m_entropy[i].item()),
                }
            )
    return rows


def train_shadow_model(model, train_loader, device, epochs=120):
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 80, 110], gamma=0.5)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        sched.step()

        if epoch % 20 == 0 or epoch <= 5:
            print(
                f"Shadow Ep {epoch:03d} | "
                f"loss={loss_sum / total:.4f} | acc={correct / total:.4f} | "
                f"lr={opt.param_groups[0]['lr']:.5f}"
            )


def tpr_at_fpr(y_true, score, max_fpr):
    fpr, tpr, _ = roc_curve(y_true, score, pos_label=1)
    mask = fpr <= max_fpr
    return float(tpr[mask].max()) if mask.any() else 0.0


def prepare_attack_matrix(df):
    feat_cols = [
        "conf_true", "conf_max", "loss", "entropy",
        "logit_true", "logit_gap", "m_entropy",
    ]
    x = df[feat_cols].copy()
    # Make monotone direction approximately "higher = more likely member".
    x["loss"] = -x["loss"]
    x["entropy"] = -x["entropy"]
    y = (df["split_name"] == "member").astype(int).values
    return x.values.astype(np.float64), y, feat_cols


def save_summary(result_dict, out_path):
    lines = [
        "=" * 74,
        "SHADOW-MODEL ATTACK v2 — SUMMARY",
        "=" * 74,
        "",
        f"Attack model         : {result_dict['attack_model']}",
        f"Target ROC-AUC       : {result_dict['auc']:.4f}",
        f"Target TPR@0.1%FPR   : {result_dict['tpr_01']:.4f}",
        f"Target TPR@1%FPR     : {result_dict['tpr_1']:.4f}",
        f"Target best acc      : {result_dict['best_acc']:.4f}",
        f"Target advantage     : {result_dict['advantage']:.4f}",
        "",
        "Interpretation:",
        "  - Higher TPR@1%FPR means stronger practical privacy leakage.",
        "  - Compare against threshold attack v2 to see whether learned",
        "    shadow attack improves low-FPR member detection.",
        "",
        "=" * 74,
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    set_seed(42)
    os.makedirs(REPORT_DIR, exist_ok=True)

    if not os.path.exists(SPLITS_PATH):
        raise FileNotFoundError(f"Missing splits file: {SPLITS_PATH}")
    if not os.path.exists(TARGET_MIA_V2):
        raise FileNotFoundError(
            f"Missing target v2 features: {TARGET_MIA_V2}. Run train_v2.py first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    splits = np.load(SPLITS_PATH)
    target_idx = splits["target_idx"].tolist()
    shadow_idx = splits["shadow_idx"].tolist()
    holdout_idx = splits["holdout_idx"].tolist()
    print(f"Splits loaded | target={len(target_idx):,} shadow={len(shadow_idx):,} holdout={len(holdout_idx):,}")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
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

    ds_shadow_train = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=train_tf)
    ds_eval = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=eval_tf)

    shadow_train_loader = DataLoader(
        Subset(ds_shadow_train, shadow_idx),
        batch_size=128, shuffle=True, num_workers=2,
    )
    shadow_member_loader = DataLoader(
        IndexedSubset(ds_eval, shadow_idx),
        batch_size=256, shuffle=False, num_workers=2,
    )
    # Use target split as shadow non-members; disjoint from shadow members.
    shadow_nonmember_loader = DataLoader(
        IndexedSubset(ds_eval, target_idx),
        batch_size=256, shuffle=False, num_workers=2,
    )

    print("\n[1/5] Training shadow model...")
    shadow_model = SmallCNN(num_classes=10).to(device)
    train_shadow_model(shadow_model, shadow_train_loader, device, epochs=120)

    print("\n[2/5] Extracting shadow member/non-member features...")
    shadow_rows = []
    shadow_rows.extend(collect_v2_features(shadow_model, shadow_member_loader, device, "member"))
    shadow_rows.extend(collect_v2_features(shadow_model, shadow_nonmember_loader, device, "nonmember"))
    shadow_df = pd.DataFrame(shadow_rows)
    shadow_df.to_csv(SHADOW_FEATURES_CSV, index=False)
    print(f"Shadow features saved -> {SHADOW_FEATURES_CSV}")

    print("\n[3/5] Training learned attack classifier (logistic regression)...")
    x_shadow, y_shadow, feat_cols = prepare_attack_matrix(shadow_df)
    attack = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    attack.fit(x_shadow, y_shadow)
    print(f"Attack features: {feat_cols}")

    print("\n[4/5] Evaluating attack on target model features...")
    target_df = pd.read_csv(TARGET_MIA_V2)
    x_target, y_target, _ = prepare_attack_matrix(target_df)

    target_score = attack.predict_proba(x_target)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_target, target_score, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    tpr_01 = tpr_at_fpr(y_target, target_score, 0.001)
    tpr_1 = tpr_at_fpr(y_target, target_score, 0.01)

    best_acc = 0.0
    for thr in thresholds:
        pred = (target_score >= thr).astype(int)
        acc = accuracy_score(y_target, pred)
        if acc > best_acc:
            best_acc = acc
    advantage = 2 * (best_acc - 0.5)

    eval_df = pd.DataFrame(
        {
            "score": target_score,
            "y_true": y_target,
        }
    )
    eval_df.to_csv(TARGET_EVAL_CSV, index=False)
    print(f"Target attack scores saved -> {TARGET_EVAL_CSV}")

    print("\n[5/5] Writing plots and summary...")

    # Load threshold attack best signal for comparison
    thresh_score = target_df["logit_gap"].values
    y_thresh = y_target
    fpr_t, tpr_t, _ = roc_curve(y_thresh, thresh_score, pos_label=1)
    auc_t = auc(fpr_t, tpr_t)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Shadow attack vs threshold attack", fontsize=13)

    for ax_i, ax in enumerate(axes):
        xlim = (0, 1.0) if ax_i == 0 else (0, 0.05)
        ylim = (0, 1.05) if ax_i == 0 else (0, 0.50)

        ax.plot(fpr, tpr, color="#185FA5", lw=2.2,
                label=f"Shadow (LR)  AUC={roc_auc:.3f}")
        ax.plot(fpr_t, tpr_t, color="#D85A30", lw=2, ls="--",
                label=f"Threshold (logit_gap)  AUC={auc_t:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.35, label="Random")
        ax.axvline(0.01, color="gray", lw=0.8, ls=":", alpha=0.6)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("Full ROC" if ax_i == 0 else "Zoom: FPR 0–5%", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        if ax_i == 0:
            ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(ROC_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC saved -> {ROC_PNG}")

    result = {
        "attack_model": "LogisticRegression",
        "auc": roc_auc,
        "tpr_01": tpr_01,
        "tpr_1": tpr_1,
        "best_acc": best_acc,
        "advantage": advantage,
    }
    save_summary(result, SUMMARY_TXT)

    print("\n" + "=" * 60)
    print("SHADOW ATTACK v2 COMPLETE")
    print(f"AUC={roc_auc:.4f} | TPR@0.1%={tpr_01:.4f} | TPR@1%={tpr_1:.4f}")
    print(f"Outputs: {REPORT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
