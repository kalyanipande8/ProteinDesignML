"""Simulate 820 structural templates as simplified backbone
descriptor vectors (8 real geometric/biophysical dims standing in for
backbone torsion/packing features) -- the conditioning input a
diffusion-based generative design pipeline samples from."""
from __future__ import annotations

import numpy as np

N_TEMPLATES = 820
N_DIMS = 8
SEED = 1151


def simulate(n_templates: int = N_TEMPLATES, n_dims: int = N_DIMS, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n_templates, n_dims)).astype(np.float32)


if __name__ == "__main__":
    templates = simulate()
    print(templates.shape)
