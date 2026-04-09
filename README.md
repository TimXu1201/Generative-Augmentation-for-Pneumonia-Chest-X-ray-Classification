# ECE 285: Generative Modeling for Chest X-ray Augmentation

This repository contains my ECE 285 project work on **synthetic pneumonia chest X-ray generation** and **downstream classification evaluation**.

The core idea is to compare two lightweight generative pipelines:

- **WGAN-GP**
- **Mini-DDPM**

and then evaluate whether synthetic pneumonia images can replace part of the real pneumonia training set without hurting downstream binary classification performance.

## Project Pipeline

1. Train **WGAN-GP** on pneumonia-only chest X-ray images.
2. Train **Mini-DDPM** on the same target class.
3. Generate synthetic pneumonia samples from the best checkpoints.
4. Build downstream experiment datasets with different real/synthetic mixtures.
5. Train **ResNet-18** classifiers for normal-vs-pneumonia classification.
6. Compare training behavior and downstream performance.

## Main Files

- `WGAN_GP.py`
  WGAN-GP training and sample generation.
- `mini_ddpm.py`
  Lightweight DDPM training and sample generation.
- `build_datasets.py`
  Builds experiment datasets for downstream classification.
- `resnet.py`
  ResNet-18 training and evaluation script.
- `fid.py`
  FID-related helper script.
- `wgan_w_dist_academic.png`
  Example training visualization.

## Supporting Course Work

This folder also keeps several related course deliverables:

- `285_final_report.pdf`
- `285_midterm_report.pdf`
- `project proposal.pdf`
- `hw2/`
- `hw3/`
- `none_weight_result/`
- `pretrain_results/`

## Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:

<https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

The dataset is **not included** in this public repository.

Expected source structure before running the pipeline:

```text
chest_xray/
|-- train/
|   |-- NORMAL/
|   `-- PNEUMONIA/
`-- test/
    |-- NORMAL/
    `-- PNEUMONIA/
```

## Environment

Typical dependencies include:

- Python 3.10+
- torch
- torchvision
- matplotlib
- tqdm
- pillow
- pytorch-ignite
- scikit-learn

## Example Commands

### Train WGAN-GP

```bash
python WGAN_GP.py \
  --data_dir ./chest_xray/train/PNEUMONIA \
  --out_dir ./output_wgan \
  --epochs 100 \
  --batch_size 32 \
  --device cuda
```

### Train Mini-DDPM

```bash
python mini_ddpm.py \
  --data_dir ./chest_xray/train/PNEUMONIA \
  --out_dir ./output_ddpm \
  --epochs 100 \
  --batch_size 32 \
  --device cuda
```

### Build Downstream Experiment Datasets

```bash
python build_datasets.py
```

### Train the ResNet Classifier

```bash
python resnet.py \
  --train_dir ./Experiment_Datasets/Exp_A_Baseline \
  --test_dir ./chest_xray/test \
  --epochs 15 \
  --batch_size 32 \
  --device cuda
```

## What Is Excluded

This public repository intentionally excludes:

- raw chest X-ray images
- generated synthetic image folders
- built experiment datasets
- large training checkpoints
- local archives and cache outputs

## Notes

- In `resnet.py`, the pretrained-vs-scratch setting is selected manually inside the script.
- The current GitHub version of this project already exists as `TimXu1201/ECE-285`; this local README keeps the repository aligned with the cleaned public version.
