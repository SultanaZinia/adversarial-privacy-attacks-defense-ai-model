# Project Instructions — CS6413 Project 25
# Adversarial Privacy Attacks and Defenses in AI Models

## What This Project Does

We train a small CNN on a 6,000-sample subset of CIFAR-10 and investigate how much
an adversary can learn about which samples were used for training. This is called a
Membership Inference Attack (MIA). We then implement and compare multiple defense
mechanisms that reduce this privacy leakage, measuring the tradeoff between model
utility (accuracy) and privacy protection (attack resistance).

---

## Project Structure

```
models/
  training/
    train-baseline.py            # v1: baseline CNN, softmax-only MIA features
    train-baseline-v2.py         # v2: fixed LR schedule, adds logit-space signals
  defenses/
    train-regularised.py         # L2 weight decay + label smoothing + dropout
    train-early-stopping.py      # stops when overfit gap > 0.08 or patience exhausted
    train-distillation.py        # teacher-student knowledge distillation (τ=5, α=0.7)
    apply-confidence-masking.py  # post-hoc Laplace noise on model outputs (no retraining)
    train-dp-sgd.py              # DP-SGD via Opacus at ε = 1, 5, 10
  attacks/
    attack-threshold.py          # v1 threshold attack (softmax signals)
    attack-threshold-v2.py       # v2 threshold attack (logit-space signals)
    attack-shadow.py             # shadow-model learned attack (logistic regression)
    attack-per-class.py          # per-class vulnerability analysis + correlation
  evaluation/
    eval-all-defenses.py         # final comparison table + plots across all defenses
    eval-dp-attack.py            # DP-specific before/after evaluation
    generate-report-figures.py   # generates all 7 publication-ready figures
  utils/
    extract-logits.py            # re-extract logit signals from existing checkpoint
notebooks/
  defense-comparison.ipynb       # interactive notebook for exploring results
data/                            # CIFAR-10 (auto-downloaded)
outputs/
  splits/                        # fixed train/shadow/holdout index splits
  checkpoints/                   # saved model weights (.pt files)
  logs/                          # training logs + per-sample MIA feature CSVs
  reports/                       # plots, summaries, figures for the report
```

---

## Setup

```bash
conda create -n adv-privacy python=3.10 -y
conda activate adv-privacy
pip install --upgrade pip
pip install -r requirements.txt
pip install opacus    # needed only for DP-SGD defense
```

---

## Execution Order

### Phase 1: Train baseline models

```bash
# v1 baseline (original, softmax signals only)
python models/training/train-baseline.py

# v2 baseline (fixed LR schedule, logit signals — used as "no defense" reference)
python models/training/train-baseline-v2.py
```

### Phase 2: Run attacks on baseline

```bash
# Threshold attack on v1 features
python models/attacks/attack-threshold.py

# Threshold attack on v2 features (logit-space signals)
python models/attacks/attack-threshold-v2.py

# Shadow-model learned attack (trains shadow model + logistic regression classifier)
python models/attacks/attack-shadow.py
```

### Phase 3: Train defenses

Each defense produces a per-sample MIA CSV in `outputs/logs/` with the same format,
so they all plug into the same evaluation pipeline.

```bash
# Heuristic defenses (no formal privacy guarantee)
python models/defenses/train-regularised.py       # ~15 min CPU
python models/defenses/train-early-stopping.py     # ~5-10 min (stops early)
python models/defenses/train-distillation.py       # ~20 min (trains 2 models)

# Output perturbation (no retraining, needs v2 checkpoint)
python models/defenses/apply-confidence-masking.py # seconds

# Formal privacy guarantee (needs opacus installed)
python models/defenses/train-dp-sgd.py             # ~30 min (trains 3 models: ε=1,5,10)
```

### Phase 4: Evaluate and compare

```bash
# Final comparison across all defenses
python models/evaluation/eval-all-defenses.py

# DP-specific evaluation
python models/evaluation/eval-dp-attack.py

# Per-class vulnerability analysis
python models/attacks/attack-per-class.py

# Generate all report figures (7 figures)
python models/evaluation/generate-report-figures.py
```

---

## What Each Defense Does

### 1. Regularised (L2 + Label Smoothing + Dropout)
- L2 weight decay (1e-3) penalises large weights that enable memorisation
- Label smoothing (0.1) prevents overconfident predictions on training samples
- Dropout (0.5) forces redundant representations instead of sample-specific neurons
- Result: best privacy-utility tradeoff (AUC 0.597, holdout acc 72.8%)

### 2. Early Stopping
- Monitors overfit gap (member_acc − holdout_acc) during training
- Stops when gap exceeds 0.08 or holdout accuracy stalls for 15 epochs
- Freezes model before deep memorisation occurs
- Result: strong privacy (AUC 0.546) but lower utility (63.7%)

### 3. Knowledge Distillation
- Trains a teacher model normally (it overfits)
- Trains a fresh student on teacher's soft probability outputs (temperature τ=5)
- Student learns inter-class relationships, not sample-specific patterns
- Result: preserves utility (73.7%) but still leaks (AUC 0.698)

### 4. Confidence Masking (MemGuard-inspired)
- No retraining — loads existing v2 checkpoint
- Adds Laplace noise to both logits and softmax probabilities
- Simulates a model API that returns noised outputs
- Result: minimal privacy gain (AUC 0.681), utility preserved (71.2%)

### 5. DP-SGD (Differential Privacy)
- Clips per-sample gradients + adds Gaussian noise during training
- Provides formal (ε,δ)-differential privacy guarantee
- Trained at three privacy budgets: ε = 1 (strong), 5 (moderate), 10 (weak)
- Result: near-random AUC (~0.50) but accuracy collapses to ~32%

---

## Attack Types

### Threshold Attack
- Sweeps a threshold on a single signal (loss, confidence, logit_gap)
- Predicts "member" if signal exceeds threshold
- v2 uses logit-space signals that avoid softmax saturation

### Shadow-Model Attack
- Trains a separate model on the shadow split (3,000 samples)
- Extracts same MIA features from shadow model
- Trains logistic regression to classify member vs non-member
- More realistic threat model — attacker doesn't need access to target training data

### Per-Class Analysis
- Runs shadow attack separately for each CIFAR-10 class
- Reveals non-uniform vulnerability (cat AUC=0.846 vs ship AUC=0.618)
- Shows r=0.98 correlation between memorisation and vulnerability

---

## Key Results

| Defense        | Holdout acc | Shadow AUC | Overfit gap | TPR@1% |
|----------------|-------------|------------|-------------|--------|
| No defense     | 0.717       | 0.727      | 0.280       | 0.033  |
| Regularised    | 0.728       | 0.597      | 0.173       | 0.009  |
| Early stop     | 0.637       | 0.546      | 0.080       | 0.012  |
| Distillation   | 0.737       | 0.698      | 0.250       | 0.034  |
| Conf masking   | 0.712       | 0.681      | 0.282       | 0.018  |
| DP ε=10        | 0.328       | 0.495      | −0.002      | 0.012  |
| DP ε=5         | 0.329       | 0.495      | −0.005      | 0.012  |
| DP ε=1         | 0.309       | 0.503      | 0.006       | 0.011  |

---

## Report Figures

Generated by `python models/evaluation/generate-report-figures.py`:

| File | Chart Type | Use In Report |
|------|-----------|---------------|
| fig1_roc_all_defenses.png | ROC curves (line) | Section 3/4: attack evaluation |
| fig2_privacy_utility_scatter.png | Scatter plot | Section 4: key tradeoff figure |
| fig3_heatmap_per_class.png | Heatmap | Section 4: per-class vulnerability |
| fig4_radar_chart.png | Radar/spider chart | Section 4 or presentation slides |
| fig5_signal_violin.png | Violin plots | Section 3: signal distributions |
| fig6_overfit_gap_vs_auc.png | Grouped bar chart | Section 4: gap predicts vulnerability |
| fig7_training_curves.png | Line chart | Section 3: training dynamics |

---

## Data Splits

- Total: 12,000 samples from CIFAR-10 training set (seed=42)
- Target (members): 6,000 — model is trained on these
- Shadow: 3,000 — used to train the shadow attack model
- Holdout (non-members): 3,000 — model never sees these

---

## Reproducibility

- All scripts use `seed=42` for random, numpy, and torch
- `torch.backends.cudnn.deterministic = True`
- Splits are saved to `outputs/splits/cifar10_12k_seed42.npz` and reused
- Evaluation uses deterministic preprocessing (no augmentation)
