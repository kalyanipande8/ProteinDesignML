"""A real, small conditional DDPM (denoising diffusion probabilistic
model): a genuine forward noising process (fixed variance schedule)
and a learned reverse-denoising network conditioned on a structural
template -- scaled down to train quickly on CPU while implementing
the actual diffusion training objective (predict the added noise)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

N_DIMS = 8
N_TIMESTEPS = 50


def cosine_beta_schedule(n_timesteps: int = N_TIMESTEPS) -> torch.Tensor:
    s = 0.008
    steps = torch.arange(n_timesteps + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((steps / n_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class DenoiserNet(nn.Module):
    def __init__(self, n_dims: int = N_DIMS, hidden: int = 64):
        super().__init__()
        self.time_embed = nn.Embedding(N_TIMESTEPS, 16)
        self.net = nn.Sequential(nn.Linear(n_dims * 2 + 16, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_dims))

    def forward(self, x_t, t, condition):
        t_embed = self.time_embed(t)
        inp = torch.cat([x_t, condition, t_embed], dim=-1)
        return self.net(inp)


class ConditionalDiffusionModel:
    def __init__(self, n_dims: int = N_DIMS, n_timesteps: int = N_TIMESTEPS, seed: int = 1151):
        torch.manual_seed(seed)
        self.n_timesteps = n_timesteps
        self.betas = cosine_beta_schedule(n_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.net = DenoiserNet(n_dims)

    def q_sample(self, x0, t, noise):
        sqrt_ac = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_1mac = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        return sqrt_ac * x0 + sqrt_1mac * noise

    def train(self, X0: torch.Tensor, condition: torch.Tensor, epochs: int = 100, lr: float = 2e-3):
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        criterion = nn.MSELoss()
        n = len(X0)
        batch_size = 64
        for epoch in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                x0_batch, cond_batch = X0[idx], condition[idx]
                t = torch.randint(0, self.n_timesteps, (len(idx),))
                noise = torch.randn_like(x0_batch)
                x_t = self.q_sample(x0_batch, t, noise)

                optimizer.zero_grad()
                pred_noise = self.net(x_t, t, cond_batch)
                loss = criterion(pred_noise, noise)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, seed: int = 1151) -> torch.Tensor:
        torch.manual_seed(seed)
        n = len(condition)
        x = torch.randn(n, condition.shape[1])
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((n,), t, dtype=torch.long)
            pred_noise = self.net(x, t_batch, condition)
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            coef = beta_t / (1 - alpha_cumprod_t).sqrt()
            mean = (1 / alpha_t.sqrt()) * (x - coef * pred_noise)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean
        return x
