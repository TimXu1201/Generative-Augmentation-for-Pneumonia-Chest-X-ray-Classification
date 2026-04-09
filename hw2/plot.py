import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 先设后端，避免 Windows Tk 问题
import matplotlib.pyplot as plt
import torch


def plot_split(history, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_total"]) + 1))

    # 1) total + recon（放一张图）
    plt.figure()
    plt.plot(epochs, history["train_total"], label="train_total")
    plt.plot(epochs, history["val_total"], label="val_total")
    plt.plot(epochs, history["train_recon"], label="train_recon")
    plt.plot(epochs, history["val_recon"], label="val_recon")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_total_recon.png", dpi=200)
    plt.close()

    # 2) KL（单独一张图，y轴就能看清变化）
    plt.figure()
    plt.plot(epochs, history["train_kl"], label="train_kl")
    plt.plot(epochs, history["val_kl"], label="val_kl")
    plt.xlabel("epoch")
    plt.ylabel("KL loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_kl.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/vae_ckpt.pt")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    history = ckpt["history"]
    plot_split(history, Path(args.out_dir))
    print("Saved:", Path(args.out_dir) / "loss_total_recon.png")
    print("Saved:", Path(args.out_dir) / "loss_kl.png")


if __name__ == "__main__":
    main()
