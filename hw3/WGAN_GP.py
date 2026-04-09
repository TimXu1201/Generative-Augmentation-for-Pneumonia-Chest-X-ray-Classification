#!/usr/bin/env python3
"""
GAN_WGAN.py - WGAN-GP 架构实现 (针对胸部 X 光片)
已适配 64x64 灰度肺炎 X 光片生成。
"""

import argparse
import os
from pathlib import Path
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ignite.engine import Engine, Events
from ignite.metrics import FID, InceptionScore
import PIL.Image as Image

# ================= 数据集类 =================
class SimpleImageFolder(Dataset):
    """加载平铺文件夹中的所有图片"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(data_dir, '*.jpg')) + \
                          glob.glob(os.path.join(data_dir, '*.jpeg')) + \
                          glob.glob(os.path.join(data_dir, '*.png')) + \
                          glob.glob(os.path.join(data_dir, '*.JPG')) + \
                          glob.glob(os.path.join(data_dir, '*.PNG'))
        self.image_paths = sorted(list(set(self.image_paths)))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"在 {data_dir} 中未找到图片")
        print(f"找到 {len(self.image_paths)} 张图片")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # X光片直接转换为灰度图加载
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img

# ================= WGAN-GP 模型 =================
class Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=1): # nc=1 对应灰度图
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

    def forward(self, input):
        return self.main(input)

class Critic(nn.Module): 
    def __init__(self, nc=1, ndf=64): # nc=1 对应灰度图
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
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input).view(-1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ================= 辅助函数 =================
def compute_gradient_penalty(netD, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = netD(interpolates)
    
    fake = torch.ones(d_interpolates.size(), requires_grad=False, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def interpolate_img(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        # FID 要求 3 通道。将生成的单通道图像转回 RGB 以供评估。
        pil_img_rgb = pil_img.convert('RGB')
        resized_img = pil_img_rgb.resize((299, 299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)

# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='图片数据目录 (请指向 PNEUMONIA 文件夹)')
    parser.add_argument('--out_dir', required=True, help='输出文件夹')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nz', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001) 
    parser.add_argument('--n_critic', type=int, default=5, help='Critic训练频率')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='梯度惩罚系数')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    netG = Generator(nz=args.nz, nc=1).to(device)
    netD = Critic(nc=1).to(device)
    
    if args.resume is not None:
        print("不支持在更换架构后继续旧模型，将重新开始训练！")
        
    netG.apply(weights_init)
    netD.apply(weights_init)

    # 增加了 Resize 到 64x64 并转换为单通道灰度
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # 单通道标准化
    ])
    
    dataset = SimpleImageFolder(args.data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.0, 0.9))

    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    G_losses = []
    D_losses = []
    W_distances = []

    def training_step(engine, batch):
        netD.train()
        netG.train()
        real = batch.to(device)
        b_size = real.size(0)

        # 1. 训练 Critic
        optimizerD.zero_grad()
        noise = torch.randn(b_size, args.nz, 1, 1, device=device)
        fake = netG(noise).detach()
        
        d_real = netD(real)
        d_fake = netD(fake)
        
        gradient_penalty = compute_gradient_penalty(netD, real, fake, device)
        d_loss = d_fake.mean() - d_real.mean() + args.lambda_gp * gradient_penalty
        d_loss.backward()
        optimizerD.step()

        w_dist = (d_real.mean() - d_fake.mean()).item()

        # 2. 训练 Generator
        g_loss_val = getattr(engine.state, 'last_g_loss', 0.0)
        
        if engine.state.iteration % args.n_critic == 0:
            optimizerG.zero_grad()
            noise_g = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake_g = netG(noise_g)
            g_loss = -netD(fake_g).mean()
            g_loss.backward()
            optimizerG.step()
            g_loss_val = g_loss.item()
            engine.state.last_g_loss = g_loss_val

        return {
            'Loss_D': d_loss.item(),
            'Loss_G': g_loss_val,
            'W_Dist': w_dist
        }

    trainer = Engine(training_step)

    @trainer.on(Events.ITERATION_COMPLETED)
    def store_losses(engine):
        o = engine.state.output
        D_losses.append(o["Loss_D"])
        G_losses.append(o["Loss_G"])
        W_distances.append(o["W_Dist"])

    @trainer.on(Events.ITERATION_COMPLETED(every=500))
    def store_images(engine):
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            vutils.save_image((fake + 1) / 2.0, out_dir / f'wgan_fake_iter_{engine.state.iteration:06d}.png', nrow=8, normalize=True)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        epoch = engine.state.epoch
        torch.save(netG.state_dict(), out_dir / f'netG_wgan_epoch_{epoch}.pth')
        torch.save(netD.state_dict(), out_dir / f'netD_wgan_epoch_{epoch}.pth')
        print(f'Epoch [{epoch}/{args.epochs}]  Critic_loss: {D_losses[-1]:.4f}  G_loss: {G_losses[-1]:.4f}  W-Dist: {W_distances[-1]:.4f}')

    # ================= 评估步骤 =================
    def evaluation_step(engine, batch):
        netG.eval()
        with torch.no_grad():
            batch_size = batch.size(0)
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake_batch = netG(noise)
            fake = interpolate_img(fake_batch)
            real = interpolate_img(batch)
            return fake, real

    evaluator = Engine(evaluation_step)
    fid_metric = FID(device=device)
    is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])
    fid_metric.attach(evaluator, "fid")
    is_metric.attach(evaluator, "is")

    fid_values = []
    is_values = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        epoch = engine.state.epoch
        if epoch % 10 == 0 or epoch == args.epochs:
            evaluator.run(train_loader, max_epochs=1)
            metrics = evaluator.state.metrics
            fid_score = metrics['fid']
            is_score = metrics['is']
            fid_values.append(fid_score)
            is_values.append(is_score)
            print(f"Epoch [{epoch}/{args.epochs}] Metrics - FID: {fid_score:.4f}, IS: {is_score:.4f}")
        else:
            print(f"Epoch [{epoch}/{args.epochs}] (metrics skipped)")

    print('Starting WGAN-GP Training Loop for Chest X-Rays...')
    trainer.run(train_loader, max_epochs=args.epochs)

    plt.figure()
    plt.plot(W_distances, label='Wasserstein Distance')
    plt.xlabel('iterations')
    plt.ylabel('W-Dist')
    plt.legend()
    plt.savefig(out_dir / 'wgan_distance_curve.png')
    plt.close()

if __name__ == '__main__':
    main()