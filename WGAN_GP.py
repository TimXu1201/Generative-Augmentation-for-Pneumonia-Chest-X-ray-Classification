import argparse
import os
from pathlib import Path
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ignite.engine import Engine, Events
from ignite.metrics import FID
import PIL.Image as Image


class SimpleImageFolder(Dataset):
    # Load grayscale images from a folder
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = (
            glob.glob(os.path.join(data_dir, '*.jpg')) +
            glob.glob(os.path.join(data_dir, '*.jpeg')) +
            glob.glob(os.path.join(data_dir, '*.png')) +
            glob.glob(os.path.join(data_dir, '*.JPG')) +
            glob.glob(os.path.join(data_dir, '*.PNG'))
        )
        self.image_paths = sorted(list(set(self.image_paths)))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {data_dir}")
        print(f"Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img


class Generator(nn.Module):
    # DCGAN-style generator
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
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Critic(nn.Module):
    # WGAN-GP critic
    def __init__(self, nc=1, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).view(-1)


def weights_init(m):
    # Initialize conv and norm layers
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def compute_gradient_penalty(netD, real_samples, fake_samples, device):
    # Gradient penalty for WGAN-GP
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = netD(interpolates)
    grad_outputs = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def fid_preprocess(batch):
    # Convert [-1,1] grayscale to [0,1] RGB 299x299
    x = batch.detach().cpu()
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    return x


def smooth_curve(points, factor=0.95):
    # Smooth a training curve
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed


def generate_single_fakes(generator, num_images, save_dir, device, nz=128):
    # Save final images from the best generator
    generator.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {num_images} final images")
    with torch.no_grad():
        for i in range(num_images):
            noise = torch.randn(1, nz, 1, 1, device=device)
            fake_img = generator(noise).detach().cpu()
            fake_img = (fake_img + 1.0) / 2.0
            vutils.save_image(fake_img, save_dir / f'synthetic_pneumonia_{i:04d}.png')

            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1} / {num_images}")

    print(f"Saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nz', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--lambda_gp', type=float, default=10.0)
    parser.add_argument('--num_gen', type=int, default=3875)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    netG = Generator(nz=args.nz, nc=1).to(device)
    netD = Critic(nc=1).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = SimpleImageFolder(args.data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.0, 0.9))

    w_distances = []
    best_fid = float('inf')
    best_epoch = -1

    def training_step(engine, batch):
        netD.train()
        netG.train()
        real = batch.to(device)
        b_size = real.size(0)

        optimizerD.zero_grad()
        noise = torch.randn(b_size, args.nz, 1, 1, device=device)
        fake = netG(noise).detach()
        d_real = netD(real)
        d_fake = netD(fake)
        gp = compute_gradient_penalty(netD, real, fake, device)
        d_loss = d_fake.mean() - d_real.mean() + args.lambda_gp * gp
        d_loss.backward()
        optimizerD.step()

        w_dist = (d_real.mean() - d_fake.mean()).item()

        if engine.state.iteration % args.n_critic == 0:
            optimizerG.zero_grad()
            noise_g = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake_g = netG(noise_g)
            g_loss = -netD(fake_g).mean()
            g_loss.backward()
            optimizerG.step()

        return {'W_Dist': w_dist}

    trainer = Engine(training_step)

    @trainer.on(Events.ITERATION_COMPLETED)
    def store_losses(engine):
        w_distances.append(engine.state.output["W_Dist"])

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        epoch = engine.state.epoch
        torch.save(netG.state_dict(), out_dir / f'netG_epoch_{epoch}.pth')

    def evaluation_step(engine, batch):
        netG.eval()
        with torch.no_grad():
            real = batch.to(device)
            noise = torch.randn(real.size(0), args.nz, 1, 1, device=device)
            fake = netG(noise)
            fake = fid_preprocess(fake).to(device)
            real = fid_preprocess(real).to(device)
            return fake, real

    evaluator = Engine(evaluation_step)
    fid_metric = FID(device=device)
    fid_metric.attach(evaluator, "fid")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        nonlocal best_fid, best_epoch
        epoch = engine.state.epoch

        if epoch > args.epochs - 5:
            print(f"Epoch [{epoch}/{args.epochs}] computing FID...")
            evaluator.run(train_loader, max_epochs=1)
            fid_score = evaluator.state.metrics['fid']
            print(f"Epoch [{epoch}/{args.epochs}] FID: {fid_score:.4f}")

            if fid_score < best_fid:
                best_fid = fid_score
                best_epoch = epoch
                torch.save(netG.state_dict(), out_dir / 'netG_BEST_FID.pth')
                print("New best FID checkpoint saved")

    print("Starting WGAN-GP training")
    trainer.run(train_loader, max_epochs=args.epochs)

    plt.figure(figsize=(10, 5))
    plt.plot(w_distances, alpha=0.3, color='tab:blue', label='Raw W-Distance')
    smoothed_w = smooth_curve(w_distances, factor=0.98)
    plt.plot(smoothed_w, color='tab:blue', linewidth=2, label='Smoothed W-Distance')
    plt.title("WGAN-GP Training Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Wasserstein Distance")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / 'wgan_distance_curve_smoothed.png', dpi=300)
    plt.close()

    print(f"\nTraining finished. Best epoch: {best_epoch}, best FID: {best_fid:.4f}")
    best_model_path = out_dir / 'netG_BEST_FID.pth'

    if best_model_path.exists():
        netG.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    generate_single_fakes(netG, args.num_gen, out_dir / 'synthetic_pneumonia_fakes_best', device, args.nz)


if __name__ == '__main__':
    main()