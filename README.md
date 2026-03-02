# Adversarial Privacy Attacks and Defense

This project trains a CIFAR-10 target model and exports per-sample features used for Membership Inference Attack (MIA) experiments.

## Project Structure

- `models/train_baseline.py`: trains the baseline CNN and saves MIA features.
- `data/`: CIFAR-10 data directory (downloaded automatically if missing).
- `outputs/`: generated splits, checkpoints, and logs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Training + Feature Export

From the project root:

```bash
python models/train_baseline.py
```

## Generated Files

After training, you should see:

- `outputs/splits/cifar10_12k_seed42.npz`
- `outputs/checkpoints/best_baseline.pt`
- `outputs/logs/train_log.csv`
- `outputs/logs/per_sample_mia.csv`

## Notes

- The script uses GPU automatically when CUDA is available; otherwise it runs on CPU.
- Fixed random seeds are used for reproducibility.
