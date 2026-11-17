import json
import os

import numpy as np
import torch

from src.simulate_templates import simulate, N_TEMPLATES, N_DIMS
from src.diffusion_model import ConditionalDiffusionModel
from src.multiobjective_scoring import train_reward_model, energy_based_score, combined_score, validate_ranking

N_SAMPLES_PER_TEMPLATE = 3  # 820 templates x 3 = 2,460 candidates


def main():
    rng = np.random.default_rng(1151)
    templates = simulate()

    # true "native-like" design features for training the diffusion
    # model: a real function of the template plus noise, independent
    # of whatever the reward model later learns.
    true_weights = rng.normal(0, 1, N_DIMS)
    X0 = (templates * 0.7 + rng.normal(0, 0.3, templates.shape)).astype(np.float32)

    model = ConditionalDiffusionModel()
    model.train(torch.tensor(X0), torch.tensor(templates), epochs=150)

    repeated_condition = np.repeat(templates, N_SAMPLES_PER_TEMPLATE, axis=0).astype(np.float32)
    samples = model.sample(torch.tensor(repeated_condition)).numpy()

    # hidden true design quality: a real function of the sampled
    # design's closeness to its conditioning template plus a
    # biophysical-style term, independent of the scoring pipeline
    true_quality = -np.sum((samples - repeated_condition) ** 2, axis=1) - 0.3 * np.sum(samples ** 2, axis=1)

    # train the learned reward model on a held-out labeled subset
    n_label = 400
    samples32 = samples.astype(np.float32)
    quality32 = true_quality.astype(np.float32)
    reward_model = train_reward_model(torch.tensor(samples32[:n_label]), torch.tensor(quality32[:n_label]))
    with torch.no_grad():
        learned_scores = reward_model(torch.tensor(samples32)).numpy()

    energy_scores = energy_based_score(samples)
    combined = combined_score(learned_scores, energy_scores)
    ranking_validation = validate_ranking(combined, true_quality)

    summary = {"n_templates": N_TEMPLATES, "n_candidates_generated": len(samples),
               "ranking_validation": ranking_validation}

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
