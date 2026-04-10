# Generative Augmentation for Pneumonia Chest X-ray Classification

This repository studies whether synthetic pneumonia chest X-rays can replace part of the real pneumonia training data without hurting downstream binary classification performance. The workflow trains two lightweight generative models, builds balanced real/synthetic training sets, and then evaluates a ResNet-18 classifier on held-out chest X-ray images.

## Data Source

The experiments use the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:

<https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

The raw dataset is not included in this public repository.

The local experiment copy used the following class counts:

- `train`: 1,341 `NORMAL`, 3,875 `PNEUMONIA`
- `val`: 8 `NORMAL`, 8 `PNEUMONIA`
- `test`: 234 `NORMAL`, 390 `PNEUMONIA`

## How The Data Is Used

- `WGAN_GP.py` trains a WGAN-GP on `chest_xray/train/PNEUMONIA` only.
- `mini_ddpm.py` trains a lightweight DDPM on the same `train/PNEUMONIA` folder.
- `build_datasets.py` uses `random.seed(42)` and creates three balanced downstream training sets from the training split.
- In the local experiment copy, the 1,341 real normal images define the target class balance for every downstream experiment.
- `Exp_A_Baseline` uses 1,341 real normal images and 1,341 real pneumonia images.
- `Exp_B_WGAN` uses 1,341 real normal images, 670 shared real pneumonia images, and 671 WGAN-generated pneumonia images.
- `Exp_C_DDPM` uses 1,341 real normal images, 670 shared real pneumonia images, and 671 DDPM-generated pneumonia images.
- `resnet.py` trains on one of the constructed experiment folders and evaluates on `chest_xray/test`.
- `fid.py` recalculates FID scores for WGAN or DDPM checkpoints against the real pneumonia image folder.

## Key Findings

- **Synthetic replacement** at a 1:1 pneumonia-training scale stays close to the **real-only baseline** in the archived downstream experiments.
- In the archived **scratch ResNet-18** runs, **DDPM replacement** reaches the highest best accuracy at **0.8766**, while **WGAN-GP replacement** reaches the highest best AUC at **0.9449**.
- In the archived **pretrained ResNet-18** runs, **DDPM replacement** reaches the highest best accuracy at **0.8910**, and both **WGAN-GP** and **DDPM** slightly outperform the **baseline** in best AUC.
- Across both evaluation settings, the results suggest that **generated pneumonia images** can serve as a practical supplement when building balanced downstream training sets.

## Repository Structure

- `WGAN_GP.py`: WGAN-GP training and synthetic pneumonia generation.
- `mini_ddpm.py`: lightweight DDPM training and synthetic pneumonia generation.
- `build_datasets.py`: experiment dataset construction for real-only and real/synthetic mixtures.
- `resnet.py`: downstream binary classification with ResNet-18.
- `fid.py`: checkpoint-by-checkpoint FID evaluation.
- `Project Description - Proposal.pdf`, `Project Description - Midterm.pdf`, `Project Description - Final.pdf`: project writeups and milestones.
- `results/scratch_classifier/`: metrics and plots from the non-pretrained ResNet-18 runs.
- `results/pretrained_classifier/`: metrics and plots from the pretrained ResNet-18 runs.
- `results/wgan_distance_curve.png`: representative WGAN training visualization.

## Typical Workflow

Train WGAN-GP on the pneumonia subset:

```bash
python WGAN_GP.py \
  --data_dir ./chest_xray/train/PNEUMONIA \
  --out_dir ./output_wgan \
  --epochs 100 \
  --batch_size 32 \
  --device cuda
```

Train the lightweight DDPM on the same subset:

```bash
python mini_ddpm.py \
  --data_dir ./chest_xray/train/PNEUMONIA \
  --out_dir ./output_ddpm \
  --epochs 100 \
  --batch_size 32 \
  --device cuda
```

Build downstream training sets:

```bash
python build_datasets.py
```

Train the classifier on one experiment split:

```bash
python resnet.py \
  --train_dir ./Experiment_Datasets/Exp_B_WGAN \
  --test_dir ./chest_xray/test \
  --epochs 15 \
  --batch_size 32 \
  --device cuda
```

## Notes

- The public repository intentionally excludes raw chest X-rays, generated image folders, constructed experiment datasets, and large checkpoints.
- `resnet.py` currently defaults to `weights=None`; the archived pretrained comparison in `results/pretrained_classifier/` was produced by manually switching the model weights inside the script.
