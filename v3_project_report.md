# Membership Inference Attacks and Knowledge Distillation Defense on CIFAR-10
## CS-6413: Foundations of Privacy — V3 Experiment Report

**Date:** April 7, 2026  
**Course:** CS-6413 Foundations of Privacy  
**Project:** Adversarial Privacy Attacks and Defense on AI Models

---

## 1. Introduction

### 1.1 Background and Motivation

Modern deep learning models achieve remarkable accuracy across a wide range of tasks, but this performance often comes at the cost of privacy. When a neural network is trained on a dataset, it does not merely learn general patterns — it can also memorize specific characteristics of individual training samples. This memorization poses a significant privacy risk: an adversary with query access to a trained model can potentially determine whether a specific data record was included in the model's training set. This vulnerability is formally studied through **Membership Inference Attacks (MIAs)**, first systematically explored by Shokri et al. in their seminal 2017 paper "Membership Inference Attacks Against Machine Learning Models."

The privacy implications of MIAs are far-reaching. In healthcare, confirming that a patient's medical record was used to train a diagnostic model reveals their participation in a specific clinical study. In finance, confirming a user's transaction data was part of a fraud detection model's training set leaks information about their financial behavior. Even in seemingly benign image classification tasks, membership inference can reveal which individuals contributed to a dataset, violating data minimization principles enshrined in regulations such as GDPR and PIPEDA.

### 1.2 Project Objectives

This V3 experiment series pursues three interconnected objectives:

1. **Demonstrate vulnerability:** Train a convolutional neural network (CNN) on CIFAR-10 under conditions that promote overfitting, then quantify the resulting privacy leakage through per-sample signal analysis.
2. **Mount a sophisticated attack:** Implement an enhanced version of the Shokri et al. shadow model attack, incorporating richer feature engineering, multi-augmentation querying, and per-class attack classifiers to maximize attack effectiveness.
3. **Evaluate a defense:** Deploy Knowledge Distillation (KD) as a mitigation strategy and rigorously measure whether it reduces the attacker's ability to distinguish training members from non-members.

The central research question driving this work is: *Can Knowledge Distillation, by smoothing a model's output distribution through soft-label training, close the generalization gap sufficiently to defeat a state-of-the-art shadow model membership inference attack?*

### 1.3 Report Structure

Section 2 details the experimental setup including dataset partitioning, model architecture, and training hyperparameters. Section 3 presents the target model's training results and analyzes the overfitting dynamics that create MIA vulnerability. Section 4 describes the shadow model attack methodology and its results against the undefended target model. Section 5 covers the Knowledge Distillation defense implementation and the student model's training behavior. Section 6 provides a comprehensive comparative analysis of the attack's effectiveness before and after the defense. Section 7 concludes with key findings and recommendations for stronger defenses.

---

## 2. Experimental Setup

### 2.1 Dataset and Partitioning Strategy

All experiments in this report use **CIFAR-10**, a widely-used benchmark dataset consisting of 60,000 color images of size 32×32 pixels distributed across 10 mutually exclusive classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The dataset is split into 50,000 training images and 10,000 test images by default.

For a rigorous membership inference evaluation, the 50,000 training images are partitioned into strictly disjoint subsets using a fixed random seed (`seed=42`) to ensure complete reproducibility across all experiments. The partition is persisted to disk as `cifar10_12k_seed42.npz`, guaranteeing that target model training, shadow model training, and evaluation all operate on the same fixed splits regardless of when or how many times each script is run.

| Split | Size | Role in Experiment |
|:---|---:|:---|
| **Target Members** | 6,000 | Training data for the target model and the KD student model. These are the records whose membership the attacker tries to infer. |
| **Shadow Pool** | 3,000 | Initially reserved indices. The attacker draws from the larger remaining pool (~38,000 unused training images) to train shadow models, simulating realistic data access. |
| **Holdout (Non-members)** | 3,000 | Evaluation-only samples that the target model never encounters during training. Used as the "non-member" ground truth during attack evaluation. |
| **Remaining Pool** | 38,000 | Available for shadow model construction. Ensures attackers have plentiful data from the same distribution without any overlap with target training data. |

This 2:1 ratio of members (6,000) to non-members (3,000) in the evaluation set is intentional — it reflects realistic scenarios where an attacker queries a model on a mix of members and non-members, and the ratio is not balanced. The attack threshold is mathematically corrected to account for this imbalance (discussed in Section 4).

During training, light data augmentation is applied: `RandomCrop(32, padding=4)` adds spatial jitter to prevent the model from memorizing exact pixel locations while still allowing it to memorize sample-level patterns over many epochs. Importantly, `RandomHorizontalFlip` is omitted to keep augmentation minimal and allow overfitting to develop naturally. Standard normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]) is applied to all inputs.

### 2.2 Model Architecture: LargeCNN

The `LargeCNN` architecture is deliberately designed to be moderately over-parameterized relative to the small 6,000-sample training subset. With approximately **0.8 million learnable parameters** fitting only 6,000 training images, the model has more than enough capacity to memorize every sample — which is precisely the scenario we wish to study.

| Layer | Configuration | Output Shape | Parameters |
|:---|:---|:---|---:|
| Conv2d + ReLU | 3→48, kernel 3×3, pad 1 | 48×32×32 | 1,344 |
| Conv2d + ReLU | 48→48, kernel 3×3, pad 1 | 48×32×32 | 20,784 |
| MaxPool2d | 2×2 stride 2 | 48×16×16 | 0 |
| Conv2d + ReLU | 48→96, kernel 3×3, pad 1 | 96×16×16 | 41,568 |
| Conv2d + ReLU | 96→96, kernel 3×3, pad 1 | 96×16×16 | 83,040 |
| MaxPool2d | 2×2 stride 2 | 96×8×8 | 0 |
| Flatten | — | 6,144 | 0 |
| Linear + ReLU | 6144→256 | 256 | 1,573,120 |
| Linear + ReLU | 256→128 | 128 | 32,896 |
| Linear (output) | 128→10 | 10 | 1,290 |
| **Total** | | | **~1.75M** |

A critical design choice is the **absence of regularization mechanisms**: no Dropout layers, no Batch Normalization, and zero weight decay. This combination allows the model to freely memorize its training data, creating the exact conditions under which membership inference attacks thrive. While a model deployed in production would typically include such regularization, this experiment deliberately omits them to study the worst-case privacy scenario and to create a clear baseline against which defenses can be measured.

### 2.3 Training Configuration

All models in this experiment — the target, the student, and each of the 8 shadow models — share identical training hyperparameters to ensure fair comparison:

| Parameter | Value | Rationale |
|:---|:---|:---|
| Optimizer | SGD, momentum=0.9 | Standard choice for CNN training; momentum accelerates convergence |
| Initial Learning Rate | 0.01 | Moderate starting rate that allows gradual learning |
| LR Schedule | MultiStepLR, milestones=[60, 90], γ=0.5 | Two decay stages (0.01→0.005→0.0025) allow fine-grained memorization in later epochs |
| Weight Decay | 0.0 | Zero regularization maximizes memorization |
| Epochs | 120 | Sufficient for complete overfitting on 6,000 samples |
| Batch Size | 128 | Balances training speed with gradient noise |
| Loss Function | CrossEntropyLoss (target); KD composite loss (student) | Standard classification loss; KD uses a blend of soft and hard targets |

The learning rate schedule is particularly important: the two decay stages at epochs 60 and 90 allow the model to first learn broad features (epochs 1–59), then refine decision boundaries (epochs 60–89), and finally memorize individual samples with very small gradient updates (epochs 90–120). This mirrors common training recipes used in practice, making the experiment realistic.

---

## 3. Target Model Training Results

### 3.1 Training Dynamics and Overfitting Progression

The V3 target model was trained for 120 epochs. The complete training trajectory is logged in `train_log_v3.csv`, from which we extract key milestones that illustrate how the model progressively transitions from learning generalizable features to memorizing individual training records:

| Epoch | Train Loss | Train Acc | Member Eval Acc | Holdout Acc | Member–Holdout Gap |
|:---|:---|:---|:---|:---|:---|
| 1 | 2.3035 | 10.68% | 15.32% | 15.97% | −0.65% |
| 10 | 1.6705 | 38.92% | 43.48% | 41.97% | +1.52% |
| 20 | 1.3408 | 50.57% | 55.60% | 50.73% | +4.87% |
| 30 | 1.0539 | 61.53% | 69.35% | 58.83% | +10.52% |
| 40 | 0.7363 | 73.97% | 82.12% | 64.83% | +17.28% |
| 50 | 0.4932 | 82.30% | 92.87% | 67.37% | +25.50% |
| 60 | 0.2930 | 89.32% | 97.08% | 66.37% | +30.72% |
| 90 | 0.0743 | 97.50% | 99.35% | 69.20% | +30.15% |
| **120** | **0.0135** | **99.70%** | **100.00%** | **69.30%** | **+30.70%** |

Several distinct phases are visible in the training dynamics:

**Phase 1 — Feature Learning (Epochs 1–30):** Both member and holdout accuracy rise together. The model is learning general visual features (edges, textures, shapes) that transfer to unseen data. The member–holdout gap grows slowly from near-zero to about 10 percentage points, indicating the onset of mild overfitting.

**Phase 2 — Accelerating Overfitting (Epochs 30–60):** Member accuracy climbs rapidly from 69% to 97%, while holdout accuracy stalls around 64–67%. The gap widens from 10pp to over 30pp. The model has exhausted the generalizable signal in its small training set and begins memorizing sample-specific patterns — exact pixel arrangements, noise patterns, and other features that do not transfer to unseen images.

**Phase 3 — Complete Memorization (Epochs 60–120):** After the first learning rate decay at epoch 60, the model converges to near-zero training loss and 100% member accuracy. Holdout accuracy oscillates between 69–70% and never improves further. The model has effectively become a lookup table for its 6,000 training members while maintaining only moderate generalization to new data.

### 3.2 Per-Sample MIA Feature Analysis

After training, per-sample MIA features were extracted for all 9,000 evaluation samples (6,000 members + 3,000 non-members) and saved to `per_sample_mia_v3.csv`. These features capture multiple dimensions of how the model treats members versus non-members:

| Feature | Member Mean | Non-member Mean | Gap | Privacy Signal |
|:---|:---|:---|:---|:---|
| Cross-entropy Loss | 0.111 | 1.653 | 1.542 | Members have near-zero loss; strongest single MIA signal |
| Confidence (true class) | 0.976 | 0.738 | 0.238 | Model is far more confident on its training data |
| Max confidence | 0.993 | 0.944 | 0.049 | Output distributions are sharper for members |
| Shannon Entropy | 0.011 | 0.161 | 0.150 | Very low entropy on members = near-deterministic predictions |
| Logit (true class) | 27.33 | 16.93 | 10.40 | Raw logit magnitude is substantially higher for members |
| Logit gap | 14.82 | 3.71 | 11.11 | Margin between top-1 and top-2 logit is much wider for members |
| Modified entropy | −0.009 | 0.032 | 0.041 | Correctly-classified members produce negative modified entropy |

These statistics paint a clear picture: the model behaves in a categorically different manner when processing data it has memorized versus data it encounters for the first time. The cross-entropy loss is the most discriminative single feature — members receive losses near zero (the model has essentially "seen" them before and knows the answer), while non-members have losses over an order of magnitude higher. The logit gap tells a similar story from the raw output perspective: the margin between the true class logit and the next-best class is 14.82 for members but only 3.71 for non-members, meaning the model is not merely correct on members but overwhelmingly, excessively correct.

These multi-dimensional behavioral differences are exactly what the shadow model attack exploits. Rather than relying on any single threshold, the attack classifier learns a decision boundary across all 23 features simultaneously, capturing complex interaction patterns that no single-metric analysis can detect.

---

## 4. Shadow Model Attack

### 4.1 Attack Overview and Threat Model

The shadow model attack operationalizes a realistic threat model in which the adversary has:

- **Black-box query access** to the target model — they can submit inputs and observe the output probability distribution (softmax vector), but cannot inspect weights, gradients, or training data directly.
- **Knowledge of the training data distribution** — the attacker can obtain additional samples from the same domain (CIFAR-10 in this case) to train surrogate models.
- **Knowledge of the model architecture** — the attacker knows or can reasonably guess the target model's architecture class (a common assumption when models are deployed as services with published architecture descriptions).

The fundamental insight of Shokri et al. is that the attacker can create their own labeled dataset for the membership inference task by training "shadow" models that mimic the target's behavior. Since the attacker controls the shadow models' training data, they know exactly which samples are "in" (members) and "out" (non-members) for each shadow, and can use this labeled data to train a binary classifier that distinguishes member behavior from non-member behavior.

### 4.2 Shadow Model Training

In this experiment, **8 independent shadow models** are trained, each on a fresh random subset of 3,000 "in" samples drawn from the ~38,000 remaining CIFAR-10 training images (strictly disjoint from the target's training, shadow, and holdout splits). For each shadow model, a corresponding set of 3,000 "out" samples is designated from the same pool. Each shadow model uses the identical `LargeCNN` architecture and training recipe (120 epochs, SGD, same LR schedule, zero weight decay) as the target model.

Using 8 shadow models (doubled from the baseline's 4) provides several advantages. First, it increases the volume and diversity of the attack training data, giving the attack classifier more examples of how models behave on their members versus non-members. Second, it averages out the randomness inherent in any single model's training, producing more robust behavioral signatures. Each shadow model contributes 6,000 labeled samples (3,000 in + 3,000 out), yielding a total attack training set of 48,000 samples — far more than typical single-shadow approaches.

### 4.3 Feature Engineering: The 23-Dimensional Attack Vector

A key improvement in this V3 attack is the construction of a rich, 23-dimensional feature vector for each queried sample, far surpassing the 10-dimensional raw softmax vector used in baseline approaches:

| Component | Dimensions | Description | Privacy Signal |
|:---|:---:|:---|:---|
| Raw softmax confidence | 10 | Probability assigned to each of the 10 classes | Members show peaked distributions concentrated on the true class |
| Sorted softmax (descending) | 10 | Class probabilities sorted by magnitude, removing class identity | Captures the *shape* of the confidence distribution independent of which class is predicted |
| Shannon entropy | 1 | $H = -\sum_{i} p_i \log(p_i)$ | Members produce near-zero entropy (model is certain); non-members produce higher entropy |
| Maximum confidence | 1 | $\max_i p_i$ | Explictly encodes peak confidence, strongly correlated with membership |
| Cross-entropy loss | 1 | $-\log(p_{\text{true}})$ | Single strongest signal; members have near-zero loss, non-members orders of magnitude higher |

The sorted softmax features deserve special explanation. By sorting the confidence values in descending order, we decouple the confidence *shape* from the class identity. This allows the attack model to learn patterns like "the gap between the 1st and 2nd highest confidence is very large" without needing to know which specific class is top-1. This class-agnostic representation generalizes better across different samples and classes.

### 4.4 Multi-Augmentation Querying

Rather than querying the model once per sample, each sample is passed through the model **10 times** with independent random augmentations (RandomCrop with padding=4). The 10 resulting softmax vectors are averaged before feature construction. This multi-augmentation approach reduces noise caused by individual augmentation choices and produces smoother, more reliable confidence estimates. The smoothed confidence vector provides a cleaner signal for the attack classifier, particularly for samples near the decision boundary where a single query might produce ambiguous results. This technique is especially effective against models trained with data augmentation, as it "averages out" the augmentation-induced variance.

### 4.5 Per-Class Attack Classifiers

Following Shokri et al., we train **one binary MLP per CIFAR-10 class** rather than a single universal classifier. The rationale is that each class may exhibit distinct confidence profiles — for example, the model might be very confident on "automobile" images but less certain on "cat" images due to inter-class visual similarity variations. By training class-specific classifiers, each MLP can learn the membership decision boundary that is most appropriate for its class.

Each attack MLP has the architecture: `Input(23) → Linear(128) → ReLU → Dropout(0.3) → Linear(64) → ReLU → Dropout(0.3) → Linear(1)`. The deeper architecture (compared to baseline's 64→32→1) with Dropout(0.3) prevents overfitting on the richer 23-dimensional feature space while capturing complex nonlinear interactions between features. The output is a single logit; positive values indicate "member" prediction.

The decision threshold is set to $\log(2) \approx 0.693$ rather than the default 0.0 to correct for the 2:1 member-to-non-member ratio in the evaluation set. Since the attack classifiers are trained on balanced (1:1) shadow data, their logits are calibrated to a 50/50 prior. Under the actual 2:1 evaluation prior, the Bayes-optimal decision boundary shifts by $\log(P(\text{member})/P(\text{non-member})) = \log(6000/3000) = \log(2)$.

### 4.6 Attack Results on Base Target Model

| Metric | Value | Interpretation |
|:---|:---|:---|
| **Accuracy** | 81.24% | 81% of all membership predictions are correct (baseline random: 66.7% due to 2:1 ratio) |
| **Precision** | 82.12% | Of samples predicted as members, 82% actually are |
| **Recall (TPR)** | 91.87% | 91.9% of true members are correctly identified — very high exposure |
| **False Positive Rate** | 40.00% | 40% of non-members are incorrectly flagged as members |
| **F1 Score** | 0.8672 | Strong harmonic mean of precision and recall |
| **AUC-ROC** | 0.8147 | Well above the 0.5 random baseline; strong discriminative power |
| **Attacker Advantage** | 1.0373 | $2 \times (TPR - FPR)$; significantly better than random |

**Confusion Matrix:**

|  | Predicted Non-member | Predicted Member | Total |
|:---|---:|---:|---:|
| **True Non-member** | TN = 1,800 | FP = 1,200 | 3,000 |
| **True Member** | FN = 488 | TP = 5,512 | 6,000 |
| **Total** | 2,288 | 6,712 | 9,000 |

The attack results demonstrate that the target model's privacy is severely compromised. Out of 6,000 training members, 5,512 (91.9%) are correctly identified as members. From a privacy perspective, this means an adversary can determine with high confidence whether almost any given record was used to train the model. The false positive rate of 40% means the attacker does over-predict membership for some non-members, but the overwhelming recall means that a positive prediction carries substantial information — if the attack says someone is a member, there is an 82% chance it is correct.

The AUC-ROC of 0.8147 provides a threshold-independent measure of attack quality. A value of 0.5 would indicate that the attack cannot distinguish members from non-members at all; a value of 1.0 would indicate perfect separation. At 0.8147, the attack has strong discriminative power, confirming that the model's outputs carry substantial membership information regardless of the specific decision threshold chosen.

---

## 5. Knowledge Distillation Defense

### 5.1 Defense Rationale and Mechanism

Knowledge Distillation (KD), originally proposed by Hinton et al. (2015) for model compression, has been explored as a privacy defense based on the following hypothesis: if a student model learns from the teacher's softened probability distributions rather than from hard one-hot labels, it should acquire the teacher's generalizable knowledge without mimicking its sample-specific memorization patterns.

The mechanism works as follows. Given a trained teacher model $T$ with highly peaked output distributions (near 1.0 for the correct class on training members), we apply a temperature parameter $T_{\text{temp}}$ to soften these distributions before presenting them to the student. At temperature $T_{\text{temp}} = 3.0$, a teacher output of [0.99, 0.005, 0.005, ...] becomes approximately [0.68, 0.04, 0.04, ...] — still informative about class relationships but less extreme. The student then trains on a composite loss:

$$\mathcal{L} = \alpha \cdot T_{\text{temp}}^2 \cdot D_{KL}\!\left(\sigma\!\left(\frac{z_s}{T_{\text{temp}}}\right) \| \sigma\!\left(\frac{z_t}{T_{\text{temp}}}\right)\right) + (1 - \alpha) \cdot \text{CE}(z_s, y)$$

where $z_s$ and $z_t$ are student and teacher logits, $\sigma$ is the softmax function, $y$ is the true label, $\alpha$ controls the balance between soft and hard targets, and the $T_{\text{temp}}^2$ factor ensures gradient magnitudes remain comparable between the two loss components.

### 5.2 KD Configuration

| Parameter | Value | Justification |
|:---|:---|:---|
| Teacher Model | Frozen `best_baseline_v3.pt` | The overfit target model serves as the knowledge source |
| Student Architecture | LargeCNN (identical to teacher) | Same capacity ensures any gap reduction is attributable to KD, not architecture |
| Temperature ($T$) | 3.0 | Moderate softening; higher values risk information loss, lower values preserve peaks |
| Alpha ($\alpha$) | 0.7 | 70% emphasis on soft teacher labels, 30% on hard ground-truth labels |
| Training Schedule | Identical to target (120 epochs, same LR) | Fair comparison requires identical optimization trajectory |

The teacher model is loaded from its best checkpoint and frozen — all parameters have `requires_grad=False` and the model is permanently in `eval()` mode. The student is initialized with fresh random weights and trained from scratch.

### 5.3 Student Training Dynamics

The KD student was trained for 120 epochs with its composite loss function. The training trajectory from `train_log_v3_kd.csv` reveals important dynamics:

| Epoch | Train Loss (KD) | Train Acc | Member Eval Acc | Holdout Acc | Member–Holdout Gap |
|:---|:---|:---|:---|:---|:---|
| 1 | 12.930 | 11.13% | 13.70% | 14.07% | −0.37% |
| 10 | 8.397 | 42.03% | 47.38% | 45.17% | +2.22% |
| 20 | 5.265 | 62.38% | 72.05% | 62.43% | +9.62% |
| 30 | 3.029 | 77.53% | 84.05% | 65.17% | +18.88% |
| 40 | 1.481 | 90.05% | 96.67% | 71.33% | +25.33% |
| 50 | 0.827 | 96.27% | 99.43% | 72.53% | +26.90% |
| 60 | 0.588 | 97.88% | 99.78% | 73.10% | +26.68% |
| 90 | 0.307 | 99.50% | 100.00% | 73.20% | +26.80% |
| **120** | **0.261** | **99.63%** | **100.00%** | **73.47%** | **+26.53%** |

Note that the KD loss values are numerically larger than the standard CE loss (the target model's epoch-1 loss was 2.30 vs. the student's 12.93) because the KD loss includes the $T^2$-scaled KL divergence term. This does not indicate worse performance — it reflects a different loss scale.

### 5.4 Comparison: Target vs. Student

| Metric | Target Model (No Defense) | KD Student | Difference |
|:---|:---|:---|:---|
| Final Training Accuracy | 99.70% | 99.63% | −0.07% |
| Final Member Eval Accuracy | 100.00% | 100.00% | 0.00% |
| Best Holdout Accuracy | 70.40% | 73.87% | **+3.47%** |
| Final Generalization Gap | 30.70% | 26.53% | **−4.17%** |

The KD student achieved modestly better generalization (+3.5 percentage points on holdout accuracy) and a slightly narrower gap (−4.2pp). However, the fundamental overfitting pattern remained virtually identical: the student still memorized 100% of its training members while achieving only ~74% on unseen data. The gap narrowed primarily because the student generalized slightly better — not because it memorized less. This is a crucial distinction for the defense evaluation: KD improved utility (holdout accuracy) but did not substantively reduce the privacy leak.

The root cause becomes clear when examining the teacher's behavior. The teacher assigns near-1.0 confidence to every one of its 6,000 training members. Even after softening with $T=3.0$, these distributions remain heavily peaked — a softened [0.99, 0.005, ...] at $T=3.0$ still contains an overwhelmingly dominant class signal. The student, having the same high capacity as the teacher, simply learns to replicate these peaked distributions, memorizing the training records through the soft labels just as effectively as it would through hard labels.

---

## 6. Attack Evaluation Against KD-Defended Model

### 6.1 Methodology

To evaluate the defense, the pre-trained attack classifiers (the 10 per-class MLPs trained on shadow model data in Section 4) were applied without retraining to the KD-defended student model. This simulates a realistic scenario where the attacker has already built their attack pipeline against a model with the same architecture and then applies it to the defended version. The same multi-augmentation querying (10 augmented forward passes per sample) and 23-dimensional feature extraction pipeline were used.

### 6.2 Comparative Results

| Metric | Target (No Defense) | KD Student (Defended) | Δ (Change) | Assessment |
|:---|:---|:---|:---|:---|
| **Accuracy** | 81.24% | 81.38% | +0.13% | ❌ No improvement |
| **Precision** | 82.12% | 80.30% | −1.83% | Marginal noise |
| **Recall (TPR)** | 91.87% | 95.50% | **+3.63%** | ❌ Defense made it worse |
| **False Positive Rate** | 40.00% | 46.87% | +6.87% | Mixed — more false alarms |
| **F1 Score** | 0.8672 | 0.8724 | +0.005 | ❌ No improvement |
| **AUC-ROC** | 0.8147 | 0.7894 | −0.025 | ✅ Slight improvement |
| **Attacker Advantage** | 1.0373 | 0.9727 | −0.065 | ✅ Marginal improvement |

### 6.3 Confusion Matrices

**Target Model (No Defense):**

|  | Pred Non-member | Pred Member | Total |
|:---|---:|---:|---:|
| **True Non-member** | TN = 1,800 | FP = 1,200 | 3,000 |
| **True Member** | FN = 488 | TP = 5,512 | 6,000 |

**KD Student (Defended):**

|  | Pred Non-member | Pred Member | Total |
|:---|---:|---:|---:|
| **True Non-member** | TN = 1,594 | FP = 1,406 | 3,000 |
| **True Member** | FN = 270 | TP = 5,730 | 6,000 |

### 6.4 Detailed Analysis

The results conclusively demonstrate that Knowledge Distillation, as configured, **failed to provide meaningful privacy protection**:

**Attack accuracy unchanged (81.24% → 81.38%):** The attacker's overall success rate did not decrease at all after KD was applied. The defense produced zero measurable reduction in the attacker's ability to correctly classify membership.

**Recall increased from 91.87% to 95.50%:** This is the most concerning finding. The attack now correctly identifies 5,730 out of 6,000 members (up from 5,512) — an increase of 218 true positives. Knowledge Distillation actually made the student model *more* uniformly detectable on its training members, not less. This occurred because the KD loss function taught the student to produce highly consistent, peaked outputs on all training samples, creating an even more homogeneous "member signature" that the attack classifier could exploit.

**FPR increased from 40.00% to 46.87%:** More non-members are now incorrectly classified as members. This represents the one partial positive effect of KD — by improving the student's generalization, KD made its behavior on non-members slightly more confident and member-like. However, this increased FPR did not translate into actual privacy protection because the simultaneously increased TPR more than compensated for the noise.

**AUC-ROC decreased from 0.8147 to 0.7894 (−0.025):** This is the only metric that shows improvement, but the magnitude is negligible. An AUC drop of 0.025 means the probabilistic separability between members and non-members decreased only marginally. The defended model still leaks substantially more information than a random classifier (AUC = 0.5).

**Net privacy assessment:** From an adversarial risk perspective, a model from which an attacker can identify 95.5% of training members with 81.4% overall accuracy is **critically compromised**. The KD defense did not achieve its intended goal of narrowing the behavioral gap between members and non-members.

---

## 7. Conclusions and Future Work

### 7.1 Summary of Key Findings

| Component | Key Result |
|:---|:---|
| Target Model Training | 100% member accuracy, 70.4% holdout accuracy → 30.7% generalization gap |
| Shadow Model Attack (vs. Target) | 81.2% accuracy, 91.9% recall, AUC = 0.815 |
| KD Student Training | 100% member accuracy, 73.9% holdout accuracy → 26.5% gap (−4.2pp) |
| Shadow Model Attack (vs. KD Student) | 81.4% accuracy, 95.5% recall, AUC = 0.789 |
| **Defense Verdict** | **KD failed to reduce attack effectiveness; recall actually increased** |

### 7.2 Why Knowledge Distillation Failed

The failure of KD as a privacy defense in this experiment can be attributed to three interconnected factors:

1. **Overfit teacher produces compromised soft labels:** The teacher model achieved 100% confidence on its training members. Softening these near-delta distributions with $T=3.0$ still produces heavily peaked probability vectors that carry strong sample-specific information. The student essentially received "slightly noisy" memorized labels rather than truly generalized probabilistic knowledge.

2. **Student has sufficient capacity to re-memorize:** Because the student uses the identical high-capacity `LargeCNN` architecture, it has more than enough parameters to learn the teacher's memorization patterns through the soft labels. A lower-capacity student might have been forced to prioritize generalizable patterns over sample-specific memorization.

3. **KD optimizes for utility, not privacy:** The KD loss function is designed to transfer the teacher's knowledge effectively — it has no mechanism to explicitly penalize memorization or to bound the information leakage about individual training samples.

### 7.3 Recommendations for Stronger Defenses

Based on the V3 findings, several alternative or complementary defenses warrant investigation in future work (V4):

| Defense Strategy | Mechanism | Expected Privacy Benefit | Trade-off |
|:---|:---|:---|:---|
| **DP-SGD** | Per-sample gradient clipping + calibrated Gaussian noise | Provides a mathematical ($\epsilon, \delta$)-differential privacy guarantee bounding any individual record's influence | Accuracy degradation; slower training |
| **Early Stopping** | Halt training at epoch ~40 when the gap is ~10-15% | Directly prevents the deep memorization phase | Reduced final accuracy |
| **Strong Regularization** | Weight decay ($10^{-3}$) + Dropout (0.5) + Batch Normalization | Constrains model capacity and prevents extreme confidence | May reduce peak accuracy |
| **Label Smoothing** | Replace hard labels $y$ with $(1-\epsilon)y + \epsilon/K$ | Prevents the model from ever producing near-1.0 confidence values | Slight accuracy reduction |
| **MixUp / CutMix** | Linear interpolation of training pairs and their labels | Blurs individual sample identity; model learns distributions, not records | Requires careful tuning |
| **Adversarial Regularization** | Add MIA-based penalty to training loss | Directly optimizes against the attack during training | Complex to implement; computational overhead |

The most promising path forward combines DP-SGD (for formal guarantees) with early stopping and label smoothing (for practical effectiveness), creating a layered defense that addresses the memorization problem at multiple levels simultaneously.

### 7.4 Reproducibility

All experiments are fully reproducible. The codebase, fixed splits, model checkpoints, and raw CSV logs are organized as follows:

| File | Purpose |
|:---|:---|
| `models/train_v3.py` | Target model training and MIA feature export |
| `attacks/shadow_model_attack_v3.py` | Shadow model training, attack classifier training, and evaluation |
| `defense/train_distillation_v3.py` | Knowledge Distillation student training |
| `defense/evaluate_kd_v3.py` | Attack evaluation on the KD-defended student model |
| `build_v3_attack.py` | Script to generate V3 attack from base attack template |
| `outputs/splits/cifar10_12k_seed42.npz` | Fixed dataset split (seed=42) |
| `outputs/logs/train_log_v3.csv` | Target model training log (120 epochs) |
| `outputs/logs/train_log_v3_kd.csv` | KD student training log (120 epochs) |
| `outputs/logs/per_sample_mia_v3.csv` | Per-sample MIA features for target model |
| `outputs/logs/per_sample_mia_v3_kd.csv` | Per-sample MIA features for KD student |
| `outputs/logs/shadow_mia_v3_results.csv` | Attack results vs. target model |
| `outputs/logs/shadow_mia_v3_kd_results.csv` | Attack results vs. KD-defended model |

---

*Report generated from experimental logs. All metrics are reproducible with `seed=42`.*  
*All code released under the project repository for CS-6413: Foundations of Privacy.*
