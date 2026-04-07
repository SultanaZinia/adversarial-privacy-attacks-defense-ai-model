import os
import sys
import csv
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from attacks.shadow_model_attack_v3 import (
    AttackMLP, get_attack_features, load_cifar10_train, _train_tf, _eval_tf, 
    NUM_CLASSES, FEATURE_DIM, DEVICE, MEMBER_THRESHOLD
)
from models.train_v3 import LargeCNN

# Check securely since this machine might not have scikit learn directly loaded locally
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
except ImportError:
    print("[ERROR] scikit-learn is not installed. You must install it to run metrics.")
    sys.exit(1)


def main():
    print("=" * 80)
    print("EVALUATING KNOWLEDGE DISTILLATION DEFENSE")
    print("=" * 80)

    # 1. Load splits
    SPLITS_FILE = os.path.join(PROJECT_ROOT, "outputs", "splits", "cifar10_12k_seed42.npz")
    if not os.path.exists(SPLITS_FILE):
        print(f"[ERROR] Splits not found at {SPLITS_FILE}")
        sys.exit(1)
        
    splits = np.load(SPLITS_FILE)
    target_idx  = splits["target_idx"]
    holdout_idx = splits["holdout_idx"]

    # 2. Load Defended Target Model (Student)
    TARGET_CKPT = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_student_kd_v3.pt")
    if not os.path.exists(TARGET_CKPT):
        print(f"[ERROR] KD Model not found at {TARGET_CKPT}")
        print("Please run python defense/train_distillation_v3.py first!")
        sys.exit(1)
        
    print("\n[1/3] Loading KD Defended Student Model...")
    target_model = LargeCNN().to(DEVICE)
    target_model.load_state_dict(torch.load(TARGET_CKPT, map_location=DEVICE))
    target_model.eval()

    # 3. Load pre-trained attacker MLPs (Shadow Models bypass)
    print("\n[2/3] Loading pre-trained Attack Models (bypassing hours of shadow training!)...")
    ATTACK_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "attack_models_v3")
    attack_models = []
    missing_any = False
    
    for c in range(NUM_CLASSES):
        m_path = os.path.join(ATTACK_DIR, f"attack_class_{c}.pt")
        if not os.path.exists(m_path):
            missing_any = True
            break
        m = AttackMLP(in_dim=FEATURE_DIM).to(DEVICE)
        m.load_state_dict(torch.load(m_path, map_location=DEVICE))
        m.eval()
        attack_models.append(m)

    if missing_any:
        print("[ERROR] Could not find pre-trained attack models.")
        print("You must generate the shadow attack first: python attacks/shadow_model_attack_v3.py")
        sys.exit(1)

    # 4. Extract capabilities using the borrowed logic from the attack script
    print("\n[3/3] Quering student model to extract logit-features and testing attack survival...")
    ds_aug  = load_cifar10_train(_train_tf)
    ds_eval = load_cifar10_train(_eval_tf)

    mem_feats, mem_lbl = get_attack_features(target_model, ds_aug, ds_eval, target_idx)
    non_feats, non_lbl = get_attack_features(target_model, ds_aug, ds_eval, holdout_idx)

    all_feats   = torch.cat([mem_feats,  non_feats])
    all_labels  = torch.cat([mem_lbl,    non_lbl])
    all_members = torch.cat([
        torch.ones(len(target_idx), dtype=torch.float32),
        torch.zeros(len(holdout_idx), dtype=torch.float32)
    ])

    preds  = torch.zeros(len(all_members))
    logits = torch.zeros(len(all_members))

    for c in range(NUM_CLASSES):
        m = attack_models[c]
        mask = (all_labels == c)
        if mask.sum() == 0: continue
        X = all_feats[mask].to(DEVICE)
        with torch.no_grad():
            lg = m(X).cpu()
        logits[mask] = lg
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
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    adv = 2.0 * (rec - fpr)

    RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
    
    print("\n" + "=" * 80)
    print("  KD DEFENSE EVALUATION vs (v3 Shadow Attack)")
    print("=" * 80)
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Recall (TPR)    : {rec:.4f}")
    print(f"  False Pos. Rate : {fpr:.4f}")
    print(f"  AUC-ROC         : {auc:.4f}  (Was heavily leaking at 0.8147!)")
    print(f"  Attacker Adv.   : {adv:.4f}")
    print("=" * 80 + "\n")

    results = {
        "attack": "shadow_model_mia_vs_kd",
        "n_members": int(y_true.sum()),
        "n_nonmembers": int((1 - y_true).sum()),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall_tpr": round(rec, 6),
        "fpr": round(fpr, 6),
        "f1": round(f1, 6),
        "auc_roc": round(auc, 6),
        "attacker_advantage": round(adv, 6),
        "tn": str(tn), "fp": str(fp), "fn": str(fn), "tp": str(tp),
    }
    
    csv_path = os.path.join(RESULTS_DIR, "shadow_mia_v3_kd_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results.keys()))
        w.writeheader()
        w.writerow(results)

    per_sample_path = os.path.join(RESULTS_DIR, "shadow_mia_v3_kd_per_sample.csv")
    with open(per_sample_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_member", "predicted_member", "attack_score", "true_class"])
        for i in range(len(y_true)):
            w.writerow([int(y_true[i]), int(y_pred[i]), f"{y_score[i]:.6f}", int(all_labels[i].item())])

    print(f"Metrics saved to {csv_path}")
    print("DONE. Defended model evaluation complete.")

if __name__ == "__main__":
    main()
