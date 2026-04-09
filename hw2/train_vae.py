import os
import math
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from tqdm import tqdm

from vae_model import ConvVAE, vae_loss

import matplotlib
matplotlib.use("Agg")  # 关键：禁用 Tk 后端


def save_image_grid(tensor, path, nrow=8):
    """
    tensor: (B,1,H,W) in [0,1]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = utils.make_grid(tensor, nrow=nrow)
    utils.save_image(grid, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="path to chest_xray folder")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transforms: grayscale -> resize -> tensor in [0,1]
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # Use train split for training; optionally use val for monitoring
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=tfm)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_total": [], "train_recon": [], "train_kl": [],
               "val_total": [], "val_recon": [], "val_kl": []}

    # Save a few real images for reference
    x0, _ = next(iter(train_loader))
    save_image_grid(x0[:64], out_dir / "real_grid.png", nrow=8)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_total = tr_recon = tr_kl = 0.0
        n_tr = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for x, _ in pbar:
            x = x.to(device)
            x_logits, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_logits, x, mu, logvar, beta=args.beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            tr_total += loss.item() * bs
            tr_recon += recon.item() * bs
            tr_kl += kl.item() * bs
            n_tr += bs
            pbar.set_postfix(loss=loss.item(), recon=recon.item(), kl=kl.item())

        history["train_total"].append(tr_total / n_tr)
        history["train_recon"].append(tr_recon / n_tr)
        history["train_kl"].append(tr_kl / n_tr)

        # Validation
        model.eval()
        va_total = va_recon = va_kl = 0.0
        n_va = 0
        with torch.no_grad():
            for x, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                x = x.to(device)
                x_logits, mu, logvar = model(x)
                loss, recon, kl = vae_loss(x_logits, x, mu, logvar, beta=args.beta)

                bs = x.size(0)
                va_total += loss.item() * bs
                va_recon += recon.item() * bs
                va_kl += kl.item() * bs
                n_va += bs

        history["val_total"].append(va_total / n_va)
        history["val_recon"].append(va_recon / n_va)
        history["val_kl"].append(va_kl / n_va)

        # Save recon + samples
        with torch.no_grad():
            # reconstructions
            x, _ = next(iter(val_loader))
            x = x.to(device)[:32]
            x_logits, _, _ = model(x)
            x_recon = torch.sigmoid(x_logits).clamp(0, 1)
            save_image_grid(x, out_dir / f"epoch{epoch:03d}_val_real.png", nrow=8)
            save_image_grid(x_recon, out_dir / f"epoch{epoch:03d}_val_recon.png", nrow=8)

            # random samples
            z = torch.randn(64, args.latent_dim, device=device)
            x_samp = torch.sigmoid(model.decode(z)).clamp(0, 1)
            save_image_grid(x_samp, out_dir / f"epoch{epoch:03d}_samples.png", nrow=8)

        # checkpoint
        ckpt = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "history": history,
            "epoch": epoch
        }
        torch.save(ckpt, out_dir / "vae_ckpt.pt")

        # plot curves each epoch
        plot_losses(history, out_dir / "loss_curve.png")

        print(f"Epoch {epoch}: "
              f"train={history['train_total'][-1]:.4f} "
              f"val={history['val_total'][-1]:.4f}")

    print("Done. Outputs saved to:", out_dir.resolve())


def plot_losses(history, save_path):
    import matplotlib.pyplot as plt
    epochs = list(range(1, len(history["train_total"]) + 1))
    plt.figure()
    plt.plot(epochs, history["train_total"], label="train_total")
    plt.plot(epochs, history["val_total"], label="val_total")
    plt.plot(epochs, history["train_recon"], label="train_recon")
    plt.plot(epochs, history["val_recon"], label="val_recon")
    plt.plot(epochs, history["train_kl"], label="train_kl")
    plt.plot(epochs, history["val_kl"], label="val_kl")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
