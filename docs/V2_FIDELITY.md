# SRALNIK v2/v3 fidelity stack

User-facing documentation for the architectural interventions added on top of
the matched M0/M1/M2/M3 ablation. Everything here is **opt-in**; without the
relevant CLI flags, training runs identically to the matched-ablation
behaviour described in `ARCHITECTURE.md`.

## Why

The matched M0/M1 results showed `M1 ≈ M0` to within 1–2% on every aggregate
(overall L1, Phase-C L1, both scenes). Open-loop reconstruction GIFs revealed
the cause: the model produces grey-brown averages of the training distribution
with no scene-specific content. Three diagnoses converged:

1. **Posterior collapse.** KL is pinned at the `free_bits = 1.0` per-element
   floor for all 75k steps. The encoder posterior `q(z | x_t)` matches the
   prior `p(z | h_{t-1})` to within the floor, so `z` carries no
   frame-specific information.
2. **Decoder capacity ceiling.** The from-scratch CNN decoder has
   `render_channels=64`, `feat_hw=8×8` → 4096 floats compressing to a
   256×256×3 = 196,608-pixel frame (48:1). It physically cannot render fine
   detail at this scale.
3. **L1 loss bias.** Pure L1 reconstruction biases toward the conditional
   median; when the decoder is uncertain it blurs.

The v2/v3 stack addresses all three with opt-in flags that compose.

## What changes (per flag)

| Flag | Default | What it does | Mechanism |
|---|---|---|---|
| `--lpips` | off | Adds VGG-feature-space perceptual loss to L1. | Punishes blur in feature space; complementary to L1. |
| `--lpips-weight 0.5` | 0.5 | Weight on the LPIPS term. | `loss_rec = L1 + λ·LPIPS`. |
| `--free-bits 0.0` | (1.0 in cfg) | Override KL clamp floor. 0.0 lets KL exceed the floor. | KL term gets nonzero gradient; encoder is incentivised to encode information into z. |
| `--kl-balance 0.2` | (0.8 in cfg) | Override KL balance direction. 0.2 in this code's convention = Dreamer α=0.8 favouring the prior. | Prior network receives stronger gradient; encoder is allowed to encode informative latents. |
| `--pixel-shuffle` | off | Decoder upsampling via PixelShuffle (sub-pixel conv). | Avoids ConvTranspose checkerboard. **Has no effect when `--sd-vae` is on** (SD-VAE replaces the upsampling stack entirely). |
| `--sd-vae` | off | Replace from-scratch CNN decoder with frozen `stabilityai/sd-vae-ft-mse`. | World model now predicts a `(4, 32, 32)` SD-VAE latent; the VAE's pretrained natural-image decoder produces RGB. |

The first four flags are **v2** (training-side fixes). The fifth replaces the
upsampling stack but is now subsumed by SD-VAE in v3. The sixth is **v3**
(decoder swap).

## Architecture comparison

```
Matched M0–M3 (default config):

  x_t -> Encoder(CNN) -> u_t -> MLP -> z_t (32-d Gaussian)
  prior(z|h_{t-1}) ----------------- KL (free_bits=1.0, balance=0.8)
  Memory(M0/M1/M2/M3) -> h_mem
  GRUCell([z, action, succ], h_mem) -> h_t
  Decoder(h_t, z_t)  [from-scratch CNN, 4096-float bottleneck] -> x_hat
  Loss: L1(x_hat, x_t) + KL


v3 (with --sd-vae --lpips --free-bits 0 --kl-balance 0.2):

  x_t -> Encoder(CNN) -> u_t -> MLP -> z_t (32-d Gaussian, NOW INFORMATIVE)
  prior(z|h_{t-1}) ----------------- KL (free_bits=0, balance=0.2)
  Memory(M0/M1/M2/M3) -> h_mem
  GRUCell([z, action, succ], h_mem) -> h_t
  SDVAEDecoder.proj(h_t, z_t) -> (4, 32, 32) SD-VAE latent
  frozen SD-VAE.decode(latent) -> x_hat (256x256 RGB, sharp)
  Loss: L1(x_hat, x_t) + 0.5 * LPIPS(x_hat, x_t) + KL
```

The world-model trunk (encoder, memory, dynamics) is **identical**. Only the
KL clamp values, the loss term, and the rendering pathway change.

## How to run

After the v3 code lands on the instance via `git pull` + `uv sync`:

```bash
# Verification (~3 min, single-GPU benchmark, real data slice)
uv run python verify_v2_arch.py

# v3 headline run: M3 + LPIPS + KL fixes + SD-VAE decoder, 50k steps (~3.5 h on 8x H100)
bash scripts/run_v3.sh gated 50000

# Optional fair-comparison baseline: same v3 stack on no-memory mode (~3.5 h)
bash scripts/run_v3.sh none 50000
```

GIFs land in `runs/m_<mode>_v3_<ts>/rollout_eval/gifs/` and sync to S3
within 120 s of training completion.

## Tuning knobs (env-var overrides on the launcher)

```bash
LPIPS_WEIGHT=0.3 bash scripts/run_v3.sh gated 50000   # lighter perceptual term
FREE_BITS=0.5    bash scripts/run_v3.sh gated 50000   # partial KL relaxation
KL_BALANCE=0.5   bash scripts/run_v3.sh gated 50000   # neutral balance
```

## Caveats

- **v3 runs are NOT directly comparable to matched M0/M1/M2/M3.** The
  decoder, loss function, and KL configuration all differ. Use them as a
  separate "architectural recovery" results section in the paper, not as
  additional points in the matched ablation table.

- **SD-VAE has a domain shift.** The VAE was trained on natural photographs;
  AI2-THOR produces Unity-rendered scenes that are photo-realistic-ish but
  have telltale game-engine signatures. Expect minor texture artefacts. A
  short LoRA fine-tune of the VAE decoder on AI2-THOR frames would mitigate
  this; saved for v2 of the paper.

- **`--pixel-shuffle` is a no-op when `--sd-vae` is on.** SD-VAE replaces the
  entire upsampling stack. Don't pass both flags together; the launcher
  `scripts/run_v3.sh` only sets `--sd-vae` for this reason.

- **`free_bits=0.0` may produce volatile early training.** With KL no longer
  clamped, the loss can briefly spike in the first 100–500 steps as the
  encoder finds an information-encoding equilibrium. Expected. Watch
  `metrics.jsonl` via `notebooks/live_metrics.py` and confirm `loss_kl`
  exceeds 1.0 within the first ~500 steps; if it stays at exactly 1.0,
  the KL balance change didn't take effect and the run should be aborted.

## Future work

The v3 stack stops short of two more substantial interventions worth
exploring once a paper is in:

1. **LoRA-fine-tune the SD-VAE decoder** on AI2-THOR frames (~50–100 lines,
   ~30 min compute). Closes the domain gap between natural photographs
   and Unity scenes.
2. **SD-UNet as actual diffusion-as-renderer.** The current v3 SD-VAE
   integration uses only the VAE decoder, not a diffusion sampling step.
   A full diffusion-as-renderer pipeline would: predict an `(h, z)` →
   text-conditioning-style adapter → SD-UNet sample → SD-VAE decode → RGB.
   Significantly bigger code lift; v2 of the paper.
3. **Larger render bottleneck** (`render_channels=256`, `feat_hw=16×16`
   from a deeper decoder). Independently of any pretrained component,
   simply increasing decoder capacity moves the per-frame L1 floor below
   the current ~0.05 ceiling.

## Verification protocol summary

Three layers of safety before any v3 run:

| Layer | What | When |
|---|---|---|
| `scripts/verify_v3_local.sh` | Python `ast.parse` + `bash -n` on every modified file | Laptop, before `git push` |
| `verify_v2_arch.py` | 11 architectural correctness checks: shape, gradient flow, ckpt roundtrip, throughput | Instance, after `git pull` + `uv sync`, before `run_v3.sh` |
| `notebooks/live_metrics.py` | Real-time loss curves from `metrics.jsonl`; spot-check `loss_kl > 1.0` after step 500 | Instance, while v3 training runs |

If any layer fails, abort. The cost of a 4-h H100 training run wasted on a
broken architecture is much higher than the cost of an extra debugging cycle.
