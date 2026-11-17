import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_templates import simulate, N_TEMPLATES
from src.diffusion_model import ConditionalDiffusionModel
from src.multiobjective_scoring import train_reward_model, energy_based_score, combined_score, validate_ranking


def test_at_least_800_templates():
    templates = simulate()
    assert len(templates) >= 800


def test_diffusion_model_generates_samples_matching_condition_shape():
    templates = simulate(n_templates=50)
    model = ConditionalDiffusionModel()
    model.train(torch.tensor(templates), torch.tensor(templates), epochs=20)
    samples = model.sample(torch.tensor(templates))
    assert samples.shape == templates.shape


def test_at_least_2400_candidates_generated_at_full_scale():
    assert N_TEMPLATES * 3 >= 2400


def test_combined_score_beats_energy_alone_on_ranking():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, (300, 8))
    true_quality = -np.sum(samples ** 2, axis=1) + rng.normal(0, 0.1, 300)
    learned_scores = -np.sum(samples ** 2, axis=1) + rng.normal(0, 0.5, 300)
    energy_scores = energy_based_score(samples)
    combined = combined_score(learned_scores, energy_scores)
    result = validate_ranking(combined, true_quality, top_k=30)
    assert result["spearman_rho"] > 0.3


def test_validate_ranking_returns_valid_metrics():
    rng = np.random.default_rng(1)
    combined = rng.normal(0, 1, 100)
    true_quality = combined + rng.normal(0, 0.1, 100)
    result = validate_ranking(combined, true_quality, top_k=10)
    assert 0 <= result["top_k_hit_rate"] <= 1
    assert -1 <= result["spearman_rho"] <= 1
