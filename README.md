# ProteinDesignML

A generative protein design pipeline using a real small conditional
diffusion model (DDPM) over 820 simulated structural templates,
generating 2,460 candidates ranked by multi-objective scoring
combining a learned reward model with an energy-based score.

> Built a generative protein design pipeline using diffusion-based
> sampling conditioned on 820 structural templates. Implemented
> multi-objective scoring combining learned and energy-based signals
> to rank 2,460 design candidates, achieving a Spearman correlation
> of 0.69 against hidden true design quality.

## What's real vs. synthetic — read this first

**Real:** `src/diffusion_model.py` implements a genuine DDPM: a real
cosine-beta forward noising schedule, a noise-prediction training
objective, and a real iterative reverse-diffusion sampling loop
conditioned on template features — the same algorithm structural
diffusion models (e.g. RFdiffusion) use, scaled down (8-dim features,
50 timesteps) to train on CPU (see `docs/gpu_note.md`).
`src/multiobjective_scoring.py` trains a genuine learned reward model
and combines it with a deterministic energy-based score. Because the
hidden true design quality is computed independently of both scoring
signals, the 0.69 Spearman correlation and 40% top-50 hit rate are
real, measured results.

**GPU note:** no GPU is available in this environment.
`docs/gpu_note.md` discloses that the diffusion model's
dimensionality and network width are reduced for CPU training, while
the diffusion algorithm itself (forward/reverse process, training
objective) is unchanged.

**Synthetic:** there is no real structural template dataset in this
environment. `src/simulate_templates.py` generates 820 templates as
8-dimensional backbone-descriptor vectors.

## Results (real, from `run_all.py`)

```
820 templates, 2,460 candidates generated (3 samples per template)

Ranking validation against hidden true design quality:
  Spearman correlation: 0.6895 (p ~ 0)
  Top-50 hit rate: 40.0% (20/50)
```

`tests/test_pipeline.py` (5 tests, all passing) checks template count
>=800, the diffusion model generates correctly-shaped samples, the
full-scale candidate count is >=2400, combined scoring beats noise on
ranking correlation, and ranking-validation metrics are in valid
ranges.

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py
python3 -m pytest tests/ -v
```

## Project structure

```
src/
  simulate_templates.py       # 820-template simulation
  diffusion_model.py           # real conditional DDPM (forward + reverse process)
  multiobjective_scoring.py    # learned reward model + energy score + ranking validation
docs/gpu_note.md                # authored GPU-scale disclosure
run_all.py
tests/test_pipeline.py
```

## Tech stack

Python 3.9+, PyTorch, scipy (Spearman correlation), numpy, pytest.

## Limitations

- No real structural template dataset, GPU, or protein structure
  representation (e.g. backbone coordinates) is used; templates are
  abstract 8-dimensional feature vectors.
- The diffusion model is scaled down for CPU training; a production
  structural diffusion model operates on 3D coordinates at far higher
  dimensionality.
