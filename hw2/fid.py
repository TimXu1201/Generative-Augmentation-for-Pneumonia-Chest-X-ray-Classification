import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vae_model import ConvVAE

# Metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="path to chest_xray folder")
    parser.add_argument("--ckpt", type=str, default="outputs/vae_ckpt.pt", help="path to vae_ckpt.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows稳一点
    parser.add_argument("--n_gen", type=int, default=2048, help="number of generated images")
    parser.add_argument("--save_gen_dir", type=str, default="", help="optional folder to save generated samples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ----- Load checkpoint -----
    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})
    latent_dim = int(saved_args.get("latent_dim", 128))
    img_size = int(saved_args.get("img_size", 64))

    model = ConvVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ----- Real images loader (from split) -----
    split_dir = os.path.join(args.data_root, args.split)

    # IMPORTANT:
    # - training used grayscale + resize(img_size) + ToTensor([0,1])
    # - BUT Inception-based metrics expect 3-channel images.
    #   We'll convert grayscale->3ch by repeating channels later.
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0,1]
    ])

    real_ds = datasets.ImageFolder(split_dir, transform=tfm)
    real_loader = DataLoader(
        real_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False
    )
    print(f"Real split: {args.split}, num_real={len(real_ds)}")

    # ----- Metrics (Inception-based) -----
    # torchmetrics FID expects uint8 images in [0,255], shape (N,3,H,W)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)

    # ----- Update FID with real images -----
    with torch.no_grad():
        for x, _ in real_loader:
            x = x.to(device)  # (B,1,H,W) float [0,1]
            x3 = x.repeat(1, 3, 1, 1)  # -> (B,3,H,W)
            x_u8 = (x3.clamp(0, 1) * 255).to(torch.uint8)
            fid.update(x_u8, real=True)

    # ----- Generate images and update metrics -----
    save_dir = Path(args.save_gen_dir) if args.save_gen_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    gen_done = 0
    bs = args.batch_size
    with torch.no_grad():
        while gen_done < args.n_gen:
            cur_bs = min(bs, args.n_gen - gen_done)
            z = torch.randn(cur_bs, latent_dim, device=device)
            x_logits = model.decode(z)                     # (B,1,H,W) logits
            x = torch.sigmoid(x_logits).clamp(0, 1)        # (B,1,H,W) float [0,1]

            x3 = x.repeat(1, 3, 1, 1)                      # (B,3,H,W)
            x_u8 = (x3 * 255).to(torch.uint8)              # uint8 for FID
            fid.update(x_u8, real=False)

            # IS expects float [0,1] in many implementations; torchmetrics handles normalize=True
            is_metric.update(x3)

            # optional save some images
            if save_dir and gen_done < 256:  # just save first 256
                from torchvision.utils import save_image
                for i in range(cur_bs):
                    idx = gen_done + i
                    if idx >= 256:
                        break
                    save_image(x[i], str(save_dir / f"gen_{idx:04d}.png"))

            gen_done += cur_bs

    fid_score = fid.compute().item()
    is_mean, is_std = is_metric.compute()
    is_mean = is_mean.item()
    is_std = is_std.item()

    print("\n===== Metrics =====")
    print(f"FID: {fid_score:.4f}  (lower is better)")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}  (higher is better)")
    print("===================\n")


if __name__ == "__main__":
    main()
