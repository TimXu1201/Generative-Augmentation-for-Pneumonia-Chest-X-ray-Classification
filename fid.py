#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import math
import os
import random
import re
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from ignite.engine import Engine
from ignite.metrics import FID


# =========================================================
# Utilities
# =========================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_epoch(path: str) -> int:
    """
    Extract the last integer from checkpoint filename.
    Example:
        netG_wgan_epoch_96.pth -> 96
        unet_ddpm_epoch_100.pth -> 100
    """
    name = Path(path).stem
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1


def list_checkpoints(
    ckpt_glob: str,
    min_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
) -> List[str]:
    paths = sorted(glob.glob(ckpt_glob), key=lambda p: extract_epoch(p))
    if not paths:
        raise FileNotFoundError(f"No checkpoint matched: {ckpt_glob}")

    if min_epoch is not None:
        paths = [p for p in paths if extract_epoch(p) >= min_epoch]
    if max_epoch is not None:
        paths = [p for p in paths if extract_epoch(p) <= max_epoch]

    if not paths:
        raise FileNotFoundError(
            f"No checkpoint remained after epoch filtering: "
            f"glob={ckpt_glob}, min_epoch={min_epoch}, max_epoch={max_epoch}"
        )

    return paths


# =========================================================
# Dataset
# =========================================================
class SimpleImageFolder(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths = (
            glob.glob(os.path.join(data_dir, "*.jpg"))
            + glob.glob(os.path.join(data_dir, "*.jpeg"))
            + glob.glob(os.path.join(data_dir, "*.png"))
            + glob.glob(os.path.join(data_dir, "*.JPG"))
            + glob.glob(os.path.join(data_dir, "*.JPEG"))
            + glob.glob(os.path.join(data_dir, "*.PNG"))
        )
        self.image_paths = sorted(list(set(self.image_paths)))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in: {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img


# =========================================================
# WGAN-GP Generator
# =========================================================
class Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


# =========================================================
# Mini-DDPM
# =========================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SimpleUNet(nn.Module):
    def __init__(
        self,
        image_channels=1,
        down_channels=(64, 128, 256),
        up_channels=(256, 128, 64),
        time_emb_dim=32,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1], time_emb_dim, up=False)
                for i in range(len(down_channels) - 1)
            ]
        )

        self.ups = nn.ModuleList(
            [
                Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
                for i in range(len(up_channels) - 1)
            ]
        )

        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)

        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)

        for up in self.ups:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = up(x, t)

        return self.output(x)


class DiffusionModel(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (
            1.0 - self.alphas_cumprod
        )

    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def sample(self, num_samples, image_size, device):
        was_training = self.model.training
        self.model.eval()

        x = torch.randn((num_samples, 1, image_size, image_size), device=device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            betas_t = self.get_index_from_list(self.betas, t, x.shape)
            sqrt_one_minus_t = self.get_index_from_list(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            sqrt_recip_alphas_t = self.get_index_from_list(
                self.sqrt_recip_alphas, t, x.shape
            )

            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, t) / sqrt_one_minus_t
            )

            if i > 0:
                noise = torch.randn_like(x)
                post_var_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
                x = model_mean + torch.sqrt(post_var_t) * noise
            else:
                x = model_mean

        if was_training:
            self.model.train()

        return x


# =========================================================
# FID helpers
# =========================================================
def to_fid_input(batch: torch.Tensor, from_neg1: bool) -> torch.Tensor:
    """
    Convert grayscale [B,1,H,W] to RGB [B,3,299,299] in [0,1] for FID.
    """
    x = batch.detach().cpu()

    if from_neg1:
        x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    else:
        x = torch.clamp(x, 0.0, 1.0)

    x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x


def build_model(model_type: str, device: torch.device, nz: int):
    if model_type == "wgan":
        model = Generator(nz=nz, nc=1).to(device)
    elif model_type == "ddpm":
        unet = SimpleUNet(image_channels=1).to(device)
        model = DiffusionModel(unet, timesteps=1000).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model


def load_ckpt_into_model(model_type: str, model, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)

    if model_type == "wgan":
        model.load_state_dict(state)
        model.eval()
    else:
        model.model.load_state_dict(state)
        model.eval()


def make_evaluator(model_type: str, model, device: torch.device, nz: int, img_size: int):
    def evaluation_step(engine, batch):
        real = batch.to(device)
        bsz = real.size(0)

        with torch.no_grad():
            if model_type == "wgan":
                noise = torch.randn(bsz, nz, 1, 1, device=device)
                fake = model(noise)
            else:
                fake = model.sample(bsz, img_size, device)

        fake_fid = to_fid_input(fake, from_neg1=True).to(device)
        real_fid = to_fid_input(real, from_neg1=False).to(device)
        return fake_fid, real_fid

    evaluator = Engine(evaluation_step)
    fid_metric = FID(device=device)
    fid_metric.attach(evaluator, "fid")
    return evaluator


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Recalculate FID for WGAN or DDPM checkpoints.")

    parser.add_argument("--model", choices=["wgan", "ddpm"], required=True, help="Model type.")
    parser.add_argument("--real_dir", required=True, help="Folder of real pneumonia images.")
    parser.add_argument("--ckpt_glob", required=True, help="Checkpoint glob.")
    parser.add_argument("--min_epoch", type=int, default=None, help="Minimum epoch to evaluate.")
    parser.add_argument("--max_epoch", type=int, default=None, help="Maximum epoch to evaluate.")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--csv_out", default=None, help="Optional CSV output path.")

    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpts = list_checkpoints(args.ckpt_glob, args.min_epoch, args.max_epoch)

    real_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),   # keep real images in [0,1]
    ])

    real_dataset = SimpleImageFolder(args.real_dir, transform=real_transform)
    real_loader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    print(f"Using device: {device}")
    print(f"Real image folder: {args.real_dir}")
    print(f"Number of real images: {len(real_dataset)}")
    print(f"Evaluating checkpoints:")
    for p in ckpts:
        print(f"  - epoch {extract_epoch(p)}: {p}")

    model = build_model(args.model, device, args.nz)

    results = []
    best_fid = float("inf")
    best_ckpt = None
    best_epoch = None

    for ckpt in ckpts:
        seed_everything(args.seed)

        print("\n" + "=" * 80)
        print(f"Evaluating checkpoint: {ckpt}")

        load_ckpt_into_model(args.model, model, ckpt, device)

        evaluator = make_evaluator(args.model, model, device, args.nz, args.img_size)
        evaluator.run(real_loader, max_epochs=1)
        fid_score = float(evaluator.state.metrics["fid"])

        epoch = extract_epoch(ckpt)
        print(f"Epoch {epoch}: FID = {fid_score:.6f}")

        row = {
            "epoch": epoch,
            "checkpoint": ckpt,
            "fid": fid_score,
        }
        results.append(row)

        if fid_score < best_fid:
            best_fid = fid_score
            best_ckpt = ckpt
            best_epoch = epoch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "#" * 80)
    print("FID ranking (best to worst):")
    for row in sorted(results, key=lambda x: x["fid"]):
        print(f"Epoch {row['epoch']:>3} | FID = {row['fid']:.6f} | {row['checkpoint']}")

    print("\nBest checkpoint:")
    print(f"  Epoch    : {best_epoch}")
    print(f"  Checkpoint: {best_ckpt}")
    print(f"  Best FID : {best_fid:.6f}")

    if args.csv_out is None:
        parent = Path(ckpts[0]).parent
        csv_out = str(parent / f"fid_results_{args.model}_{args.min_epoch}_{args.max_epoch}.csv")
    else:
        csv_out = args.csv_out

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "checkpoint", "fid"])
        writer.writeheader()
        for row in sorted(results, key=lambda x: x["epoch"]):
            writer.writerow(row)

    print(f"\nSaved CSV to: {csv_out}")


if __name__ == "__main__":
    main()