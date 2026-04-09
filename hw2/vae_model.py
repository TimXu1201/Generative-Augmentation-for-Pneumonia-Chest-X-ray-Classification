import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    A simple convolutional VAE for 1-channel 64x64 images.
    Encoder: (1,64,64) -> latent z
    Decoder: z -> (1,64,64)
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x64x64 -> 256x4x4 (if using 4 downsamples)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 32x32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 128x8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1) # 256x4x4
        )

        self.enc_out_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # Decoder: latent -> 256x4x4 -> 1x64x64
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 128x8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 32x32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)     # 1x64x64
            # Output logits (no sigmoid here if using BCEWithLogitsLoss)
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        # z = mu + std * eps
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), 256, 4, 4)
        x_logits = self.dec(h)
        return x_logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decode(z)
        return x_logits, mu, logvar


def vae_loss(x_logits, x, mu, logvar, beta: float = 1.0):
    """
    x is in [0,1]. We use BCEWithLogits for stability.
    ELBO loss = recon + beta * KL
    """
    recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum") / x.size(0)
    # KL divergence between q(z|x)=N(mu, sigma^2) and p(z)=N(0,I)
    kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / x.size(0)
    return recon + beta * kl, recon.detach(), kl.detach()
