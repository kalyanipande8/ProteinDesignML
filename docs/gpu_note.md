# GPU Note (authored disclosure, not executed)

No GPU is available in this environment. `src/diffusion_model.py`'s
model is deliberately scaled down (8-dim features, 50 diffusion
timesteps, a 2-layer MLP denoiser) so that both training and the full
reverse-diffusion sampling loop run on CPU in a few seconds. The
diffusion process itself — the forward noising schedule (cosine beta
schedule), the noise-prediction training objective, and the real
iterative reverse-sampling loop — is the same real DDPM algorithm a
GPU-scale structural diffusion model (e.g. RFdiffusion) uses; only the
feature dimensionality and network width are reduced, not the
algorithm.
