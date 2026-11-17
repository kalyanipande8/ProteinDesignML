"""Real multi-objective scoring: a learned reward-model score (a
small MLP trained to predict design quality from template-conditioned
features) combined with a deterministic energy-based score, ranking
all generated candidates -- validated against a hidden true quality
value the simulation tracks independently."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LearnedRewardModel(nn.Module):
    def __init__(self, n_dims: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_dims, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_reward_model(X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 150) -> LearnedRewardModel:
    model = LearnedRewardModel(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
    return model


def energy_based_score(designs: np.ndarray) -> np.ndarray:
    """A deterministic physics-style energy score: penalizes extreme
    feature magnitudes (standing in for steric clash / packing
    energy), independent of the learned reward model."""
    return -np.sum(designs ** 2, axis=1)


def combined_score(learned_scores: np.ndarray, energy_scores: np.ndarray, weight: float = 0.6) -> np.ndarray:
    def normalize(s):
        return (s - s.mean()) / (s.std() + 1e-8)
    return weight * normalize(learned_scores) + (1 - weight) * normalize(energy_scores)


def validate_ranking(combined: np.ndarray, true_quality: np.ndarray, top_k: int = 50) -> dict:
    predicted_top_k = set(np.argsort(combined)[::-1][:top_k])
    true_top_k = set(np.argsort(true_quality)[::-1][:top_k])
    hits = len(predicted_top_k & true_top_k)
    from scipy.stats import spearmanr
    rho, p_value = spearmanr(combined, true_quality)
    return {"top_k": top_k, "n_hits": hits, "top_k_hit_rate": round(hits / top_k, 3),
            "spearman_rho": round(float(rho), 4), "p_value": float(p_value)}
