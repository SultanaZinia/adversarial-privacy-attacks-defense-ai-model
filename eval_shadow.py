#!/usr/bin/env python3
"""Re-run only the evaluation step of the shadow model attack using saved checkpoints."""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "attacks"))
from shadow_model_attack import (
    AttackMLP, evaluate_attack,
    SPLITS_FILE, ATTACK_DIR, NUM_CLASSES, DEVICE
)

splits      = np.load(SPLITS_FILE)
target_idx  = splits["target_idx"]
holdout_idx = splits["holdout_idx"]

print(f"[INFO] Loading {NUM_CLASSES} attack classifiers from {ATTACK_DIR}")
attack_models = []
for c in range(NUM_CLASSES):
    m = AttackMLP().to(DEVICE)
    m.load_state_dict(torch.load(ATTACK_DIR / f"attack_class_{c}.pt", map_location=DEVICE))
    m.eval()
    attack_models.append(m)

evaluate_attack(attack_models, target_idx, holdout_idx)
