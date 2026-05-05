# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DiffPIR (CVPR NTIRE 2023): plug-and-play image restoration using denoising diffusion models as generative denoiser prior. Tasks: super-resolution (SISR), deblurring (Gaussian/motion), inpainting (box/random). Built on OpenAI Guided Diffusion + DPIR.

## Setup

```bash
pip install -r requirements.txt          # PyTorch 1.13 + cu117
bash download.sh                          # fetches 256x256_diffusion_uncond.pt + diffusion_ffhq_10m.pt into model_zoo/
# rename ffhq_10m -> diffusion_ffhq_m for code consistency (per README)
```

Motion blur extra: clone `https://github.com/LeviBorodenko/motionblur` into repo root.

## Run

Two entry-point styles, both produce same algorithm:

```bash
# Per-task scripts (hardcoded args at top of file)
python main_ddpir_sisr.py
python main_ddpir_deblur.py
python main_ddpir_inpainting.py

# Unified driver via YAML
python main_ddpir.py --opt configs/sisr.yaml
python main_ddpir.py --opt configs/deblur.yaml
python main_ddpir.py --opt configs/inpaint.yaml
```

Outputs land under `results/` (timestamped subdir). Test images: `testsets/`. Kernels: `kernels/` (`.mat` files via hdf5storage). Inpaint masks: `masks/` (or generated via `create_mask_demo.py`).

## Architecture

**Sampling loop** = diffusion reverse process where each step splits into:
1. Prior sub-problem: pretrained unconditional diffusion model denoises x_t → x0_pred (generative denoiser).
2. Data sub-problem: closed-form / iterative solve coupling x0_pred to measurement y under operator H (SR downsample, blur kernel, mask). Result fed back as x_{t-1}.

Key knobs in configs: `iter_num` (NFEs, typ 20–100), `lambda_`, `zeta` (data-fidelity / noise-injection weights), `sub_1_analytic` (closed-form data step), `model_output_type`, `skip_type` (quad), `eta` (DDIM stochasticity), `noise_level_img`.

**Code layout**:
- `main_ddpir.py` — unified pipeline, `CustomDataset` builds (img_L, kernel, mask) tuples per `config.task`, then runs DiffPIR loop. The other `main_ddpir_*.py` are task-specialized variants (predate the unified driver, kept for reproducibility).
- `guided_diffusion/` — vendored from OpenAI guided-diffusion: UNet (`unet.py`), Gaussian diffusion math (`gaussian_diffusion.py`), spaced-timestep wrapper (`respace.py`), model factory (`script_util.py`). Treat as upstream — avoid edits here unless porting.
- `utils/utils_model.py` — diffusion model wrappers + `model_fn` adapters used by the sampler.
- `utils/utils_sisr.py` — SISR data sub-problem (analytic FFT-domain solve under bicubic / Gaussian kernels).
- `utils/utils_deblur.py` — `GaussialBlurOperator`, `MotionBlurOperator` (the latter requires the `motionblur` package).
- `utils/utils_inpaint.py` — `mask_generator` (box / random / extreme).
- `utils/utils_resizer.py` — differentiable resizer used in SR forward op.
- `utils/utils_image.py` — I/O, tensor↔np conversions, PSNR/SSIM, save grids.
- `service/` — thin programmatic wrappers (`sr.py`, `deblur.py`, `inpaint.py`) + `*_demo.py` runnable demos. Use these when embedding DiffPIR in another app rather than as CLI.
- `configs/*.yaml` — per-task hyperparameters; keys map 1:1 to attributes on the `config` object inside `main_ddpir.py`.

## Notes

- Kernels are loaded from `.mat` via `hdf5storage`; `kernels_bicubicx234.mat` indexed by `sf-2` for SR scales 2/3/4.
- `testset_name` resolves to `testsets/<name>/` for input images. Model checkpoints expected under `model_zoo/<model_name>.pt`.
- Reproducibility: `np.random.seed(idx*10)` per-image when generating DIY blur kernels.
- GPU assumed (cu117 wheels). Code uses `device` from config; CPU not exercised.
