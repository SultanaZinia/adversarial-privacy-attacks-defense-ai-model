# Adversarial Privacy Attacks and Defense

This project trains a CNN on a fixed subset of CIFAR-10 and exports per-sample statistics for Membership Inference Attack (MIA) experiments.

The repository currently includes:

- baseline target-model training
- per-sample feature export for members and non-members

## Repository Layout

- `models/train_baseline.py`: trains the baseline CNN and writes MIA features
- `data/`: CIFAR-10 download location
- `outputs/`: training outputs such as splits, checkpoints, and logs
- `reports/`: evaluation reports

## Requirements

- Python 3.10 or newer
- `pip` or Conda
- Optional: CUDA-capable GPU for faster training

## Installation

Clone the repository, move into the project root, create an environment, and install dependencies.

### Option 1: Virtual Environment

```bash
git clone <your-repository-url>
cd adversarial-privacy-attacks-defense
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Conda

```bash
git clone <your-repository-url>
cd adversarial-privacy-attacks-defense
conda create -n adv-privacy python=3.10 -y
conda activate adv-privacy
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Baseline Training

From the project root:

```bash
python models/train_baseline.py
```

What the training script does:

- downloads CIFAR-10 automatically if it is not already present in `data/`
- creates fixed member, shadow, and holdout splits
- trains the baseline target model
- saves the best checkpoint
- exports per-sample MIA features for members and non-members

The script uses GPU automatically when CUDA is available. Otherwise, it falls back to CPU.


## Generated Files

After a successful training run, the main outputs are:

- `outputs/splits/cifar10_12k_seed42.npz`
- `outputs/checkpoints/best_baseline.pt`
- `outputs/logs/train_log.csv`
- `outputs/logs/per_sample_mia.csv`


## Reproducibility Notes

- fixed random seeds are used in training
- the target, shadow, and holdout splits are saved and reused
- evaluation uses deterministic preprocessing without data augmentation

## Troubleshooting

If `pip install -r requirements.txt` fails when installing PyTorch on your machine, install `torch` and `torchvision` using the official PyTorch instructions for your OS, Python version, and CUDA setup, then rerun:

```bash
pip install -r requirements.txt
```

This project has been kept intentionally lightweight and does not require any local absolute paths to run.
