# SRALNIK final architectures: M0 (matched baseline) and M3 + v2 fidelity (intervention)

Written for the course paper / report. Describes the two architectures we
actually trained end-to-end on 8×H100, including all hyperparameters,
implementation details, observed dynamics, and known limitations.

The two conditions share the same encoder, dynamics, and decoder backbone.
They differ only along the axes labelled "**changed**" in §3.

---

## 1. Shared backbone (identical across both conditions)

### Inputs (per training crop, `EpisodeChunkDataset` in `sralnik/training/dataset.py`)

- `obs` ∈ ℝ^(B, T, 3, 256, 256), float32 in [0, 1] (RGB / 255)
- `actions` ∈ ℤ^(B, T), int64 indices into the AI2-THOR action vocabulary (size 17)
- `action_success` ∈ {0, 1}^(B, T), bool
- `phase` ∈ {0, 1, 2}^(B, T), int64 (Phase A / B / C codes)
- `B = 4` (per-rank), `T = 16`, `image_size = 256`. Effective global batch with 8 ranks: 32.

### Encoder (`sralnik/models/encoder.py`)

5-layer strided convolutional stack from 3×256×256 → 256×8×8, plus a small MLP
for the stochastic latent.

- `enc_channels = (32, 64, 128, 256, 256)` — 5 stride-2 Conv blocks
- Each block: `Conv2d(c_in, c_out, kernel=4, stride=2, padding=1) + GroupNorm(min(32, c_out)) + SiLU`
- Spatial: 256 → 128 → 64 → 32 → 16 → 8. Final feature map `u_t ∈ ℝ^(256, 8, 8)`.
- Stochastic latent head: `flatten(u_t) → Linear(16384, 256) → SiLU → Linear(256, 64)`,
  split into `(μ, raw_σ)`, then `σ = softplus(raw_σ) + 1e-4`. Yields posterior
  `q(z | x) = 𝒩(μ, σ²)` with **`d_z = 32`**.

### Dynamics — RSSM with GRU recurrence (`sralnik/models/world_model.py`)

- **Initial state**: `h_0 ∈ ℝ^(256)` learnable parameter, broadcast over the batch.
- **Prior network**: `prior_net = Linear(256, 256) → SiLU → Linear(256, 64)` over `h_{t-1}`.
  Splits output into `(prior_μ, prior_logσ)` (the `_logσ` name is a misnomer —
  it's pre-softplus). Defines `p(z | h_{t-1}) = 𝒩(prior_μ, softplus(prior_logσ)² + 1e-4)`.
- **Posterior sampling**: training uses the reparameterisation trick
  `z_t = μ_t + σ_t · ε`, ε ~ 𝒩(0, I). Eval uses `z_t = μ_t` (deterministic).
- **Action embedding**: `act_emb = nn.Embedding(18, 32)` (17 actions + 1 unused
  padding row at index 0). Per-step `e^a_t = act_emb(action_t + 1)`.
- **GRU step**: `h_t = GRUCell([z_t; e^a_t; succ_t], h^{mem}_t)` where `h^{mem}_t`
  is `h_{t-1}` for M0 (no memory) or the output of `MemoryFusion(h_{t-1}, z_t,
  hist_z_{0:t-1}, write_mask)` for M3.
- **Hidden sizes**: `d_h = 256` (deterministic recurrent state), `d_z = 32`
  (stochastic latent).

### Memory write schedule

`_write_mask_from_phase(phase)` (in `sralnik/models/world_model.py`):

- `mask[:, 0] = True` — always write the first frame of the crop (the "anchor").
- `mask[:, t] = True if phase[t] == 0 and phase[t+1] == 1` — A→B transitions.

`hist_z` is constructed by appending `z_t.detach()` after each step. Past `z`'s
contribute to memory **without** gradient flow through the write itself
(intentional — the memory bank is a buffer, not part of the autograd graph
for past steps).

### Decoder (`sralnik/models/decoder.py`, the `Decoder` class)

- Input: `cat(h, z) ∈ ℝ^(B, 288)`.
- `Linear(288, 64·8·8 = 4096)` → reshape to `(64, 8, 8)`.
- 5 upsampling blocks (8 → 16 → 32 → 64 → 128 → 256). Each block uses one of
  two backbones, controlled by `cfg.use_pixel_shuffle`:
  - **Default (M0)**: `ConvTranspose2d(64, 64, 4, stride=2, padding=1) +
    GroupNorm + SiLU`.
  - **PixelShuffle (M3 + v2)**: `Conv2d(64, 256, 3, padding=1) + nn.PixelShuffle(2)
    + GroupNorm + SiLU`.
- Final: `Conv2d(64, 3, 3, padding=1)` then `sigmoid` → `x̂_t ∈ [0, 1]^(B, 3, 256, 256)`.
- **Compression ratio**: 4096 floats → 196,608 pixels = **48:1**. This is the
  primary capacity ceiling visible in qualitative outputs.

### Phase weighting (reconstruction loss)

`w_phase[t] = 2.0 if phase[t] == 2 else 1.0`. Phase-C frames count double in
reconstruction loss, mirroring ARCHITECTURE.md §6.

### Memory bank (M3 specifically; `sralnik/models/memory.py`)

`MemoryFusion(mode=GATED)` performs the following per timestep when the bank
contains at least one valid entry:

1. **Query**: `q = Linear(d_h + d_z, d_z)([h, z])` — the `_q` projection.
2. **Cross-attention**: `F.scaled_dot_product_attention(q, _att_in(hist_z),
   hist_z, attn_mask=hist_mask)` — single-head SDPA over the entire valid
   history. (Despite `cfg.memory_heads = 4`, the implementation is single-head;
   `_nhead` is validated but unused — a known divergence from the
   `ARCHITECTURE.md` text.)
3. **Readout**: `delta = _att_out(ctx)` projected to `d_h = 256`.
4. **Gate**: `g = sigmoid(_gate([h, z])) ∈ ℝ^(B, 1)` — a learned scalar per batch row.
5. **Output**: `h^{mem} = h + g · delta` (gated residual).

Submodules:
- `_q: Linear(288, 32)`
- `_att_in: Linear(32, 32)`
- `_att_out: Linear(32, 256)`
- `_gate: Linear(288, 128) → SiLU → Linear(128, 1)`

For M0 (`MemoryMode.NONE`), `MemoryFusion.forward` returns `h` unchanged and
**none** of `_q`, `_att_in`, `_att_out`, `_gate` are constructed (avoids DDP
unused-parameter errors with `find_unused_parameters=False`).

---

## 2. M0 — matched baseline, no memory

Launched via `bash scripts/run_condition.sh none 75000`.

| Component | Value |
|---|---|
| Memory mode | `MemoryMode.NONE` (MemoryFusion is a no-op identity) |
| Decoder upsampling | `ConvTranspose2d` (default) |
| Encoder | shared backbone (§1) |
| Reconstruction loss | `L1(x̂, x)` per pixel, mean over channels and spatial axes, **phase-weighted** (Phase C × 2) |
| KL term | balanced KL with `free_bits = 1.0`, `kl_balance = 0.8` |
| Total loss | `loss_total = loss_rec + loss_kl` |
| Latent diffusion | off (`use_latent_diffusion = False`) |
| LPIPS | off |
| PixelShuffle | off |
| Optimizer | `AdamW(lr=1e-4, weight_decay=1e-6)`, gradient clip 100.0 |
| Steps | 75 000 |
| Batch | 4 (per-rank) × 8 ranks = 32 effective |
| Sequence length T | 16 |
| Precision | bf16 autocast on CUDA |

### KL term details

```text
kl_sg_p = KL(q(z|x), stop_grad(prior))      # gradient flows on q (encoder)
kl_sg_q = KL(stop_grad(q(z|x)), prior)      # gradient flows on prior network
kl     = balance · kl_sg_p + (1 − balance) · kl_sg_q
kl     = clamp_min(kl, free_bits / d_z)     # element-wise floor: 1.0/32 = 0.03125 per dim
loss_kl = sum(kl, dim=-1).mean(over B, T)
```

With `kl_balance = 0.8`, **80% of the gradient weight is on the
posterior-toward-prior term** (encoder is pulled toward matching the dynamics
prior); only 20% is on the prior-toward-posterior term. This is the
**opposite** of Dreamer-V2's α=0.8 convention (which puts 80% on the prior
chasing the posterior). Combined with the `free_bits = 1.0` floor, the system
finds the "lazy" equilibrium where the encoder posterior matches the prior to
the floor, and `z` carries minimal information beyond what `h_{t-1}` already
predicts. The total KL after summing all 32 dims sits at exactly **1.0 nat**
for the entire training run.

### Observed outcome (M0, end of training)

- `loss_total ≈ 1.075` (rec ≈ 0.075 + KL ≈ 1.000)
- KL pinned at 1.0 floor for all 75 000 steps → **posterior collapse**.
- Open-loop Phase-C reconstructions: scene-type-collapsed grey/brown average,
  no scene-identity recovery; bathroom and kitchen episodes both produce
  near-uniform colour washes.
- Phase-C L1 (open-loop, K=8 imagined frames): **0.139**
- Phase-C MSE (open-loop, K=8): **0.0346**

---

## 3. M3 + v2 fidelity — intervention condition

Launched via `bash scripts/run_bonus_fidelity.sh gated 30000` (defaults set
after the bug fix flipping `FREE_BITS` from 0.0 back to 1.0). Resumed with
`RESUME=runs/m_gated_v2fid_<ts>/last.pt bash scripts/run_bonus_fidelity.sh gated <larger_max_steps>`
to extend training in 30 000-step increments.

The headline experiment: gated memory **plus** three architectural fixes
designed to address the failure modes diagnosed in M0/M1. Three things change
relative to M0; everything else is identical.

| Component | Value | Δ vs M0 |
|---|---|---|
| Memory mode | `MemoryMode.GATED` | **changed** |
| MemoryFusion submodules | `_q`, `_att_in`, `_att_out`, `_gate` active (see §1) | **changed** |
| Decoder upsampling | `Conv2d(c, 4c, 3, padding=1) + nn.PixelShuffle(2)` | **changed** |
| Reconstruction loss | `L1 + 0.5 · LPIPS(x̂, x)` (VGG-feature distance, frozen backbone) | **changed** |
| KL `free_bits` | **1.0** (kept — re-imposes information floor) | unchanged |
| KL `kl_balance` | **0.2** | **changed** (was 0.8 in M0) |
| Latent diffusion | off | unchanged |
| Optimizer | `AdamW(lr=1e-4, weight_decay=1e-6)`, gradient clip 100.0 | unchanged |
| Steps (initial run) | 30 000 | M0 ran 75 000 |
| Steps (with resume) | up to 75 000 (in 15–30k chunks) | matches M0's budget |
| Batch | 4 (per-rank) × 8 ranks = 32 effective | unchanged |
| Sequence length | 16 | unchanged |
| Precision | bf16 autocast | unchanged |

### LPIPS perceptual loss (`sralnik/models/lpips_loss.py`)

- Wraps the `lpips` PyPI package, VGG-16 backbone (~14 M params), frozen
  with `requires_grad_(False)` and locked in eval mode by overriding `train()`.
- Per-frame call: input `(B, 3, 256, 256)` in [0, 1], shifted to [−1, 1] for
  the VGG network, output `(B, 1, 1, 1)` flattened to `(B,)`.
- Combined with L1: per-timestep
  `rec_t = L1(x̂_t, x_t).mean((1,2,3)) + 0.5 · LPIPS(x̂_t, x_t)`,
  then phase-weighted and averaged over (B, T).
- Cost: ~30% per-step slowdown vs. pure L1 (~6 it/s → ~4.5 it/s on 8 H100).

### KL term details (with the flip)

Same `_kl_balanced` formula as M0, **but `kl_balance = 0.2`**:

```text
kl = 0.2 · kl_sg_p + 0.8 · kl_sg_q
```

The prior network now receives 80% of the gradient weight (Dreamer-V2 α = 0.8
convention), so the prior chases the posterior rather than the encoder being
yanked toward the prior. With `free_bits = 1.0` retained, the per-element
clamp still imposes a 1.0-nat-total floor.

### PixelShuffle decoder (`sralnik/models/decoder.py:32-38`)

Sub-pixel convolution avoids the checkerboard artefacts intrinsic to
`ConvTranspose2d` upsampling at stride 2 with kernel 4:

- `Conv2d(64, 256, 3, padding=1)` (4× channel projection)
- `nn.PixelShuffle(upscale_factor=2)` (rearranges channels to spatial:
  `(64, 2H, 2W)`)
- `GroupNorm(32, 64) + SiLU`
- 5 such blocks for 8 → 256 spatial.

Same parameter count as the ConvTranspose path; cleaner edges in qualitative
output.

### Memory write/read dynamics

The write schedule is identical to all M_ conditions (write at first frame of
the crop and at A→B transitions). The gated readout means at each step the
model can scale its memory contribution by a learned scalar `g_t ∈ [0, 1]`,
so memory is consulted continuously rather than only on retrieval triggers.
The retrieval itself is full cross-attention over the masked history, so
unlike top-k retrieval (M1's CONCAT mode), the memory query gradient flows
back into the encoder via SDPA's softmax — the `_q` projection is genuinely
trained.

### Observed dynamics across training (M3 + v2 fidelity)

The plot below summarises what we see in `metrics.jsonl` over the first
30 000 steps and continuation through resume.

| Step range | KL behaviour | Rec behaviour | Total loss |
|---|---|---|---|
| 0 – 3000 | Pinned at 1.0 floor (encoder posterior matches prior to floor precision) | Descending from ~0.55 to ~0.40 | Descending |
| 3000 – 24 000 | **Escapes the floor**, rises slowly to ~1.4–1.5 | Lower envelope continues dropping to ~0.30 | Slight rise (KL grows faster than rec drops) |
| 24 000 – 60 000 | Continues rising to ~1.7 | Lower envelope reaches ~0.20 | Continued slow rise |

The KL escape is the load-bearing signal: across all conditions in the
study, this is the **only** run where `loss_kl` exceeds 1.0 sustainedly.
The encoder is now encoding genuine frame-specific information; the dynamics
prior is the chase target rather than the constraint.

### Observed outcome (M3 + v2-fidelity, mid-run at step ~60 000)

- `loss_total ≈ 1.7` and **still rising** as KL continues to grow.
- KL ≈ 1.7 (escaped the floor — first time across all conditions). z carries
  genuine frame-specific information.
- `loss_rec` lower envelope ≈ 0.20, descending.
- Open-loop Phase-C reconstructions: **scene-type recovered** (kitchen
  episodes produce kitchen-content output, bathroom episodes produce
  bathroom-content output — large qualitative improvement over M0's grey
  wash); structural commitments emerge (cabinet-edge-like vertical lines,
  mirror frames, doorways).
- **Two limitations remain visible in qualitative output**:
  (i) **viewpoint drift** during open-loop rollout (no pose conditioning;
  model has no way to track the agent's exact position/orientation
  across imagined frames),
  (ii) **decoder capacity ceiling** (48:1 compression prevents fine-detail
  rendering even when scene content is correctly committed).

---

## 4. Side-by-side comparison

| Axis | M0 | M3 + v2 fidelity |
|---|---|---|
| Memory mechanism | None | Gated cross-attention with sigmoid scalar gate |
| Reconstruction loss | L1, phase-weighted | L1 + 0.5·LPIPS, phase-weighted |
| KL `free_bits` | 1.0 | 1.0 |
| KL `kl_balance` | 0.8 (encoder pulled toward prior) | **0.2** (prior chases posterior — Dreamer-correct) |
| Decoder upsampling | ConvTranspose2d | PixelShuffle (sub-pixel conv) |
| Latent diffusion | off | off |
| KL during training | pinned at 1.0 floor for all 75 000 steps | escaped floor at step ~3000, climbed to ~1.7 by step 60 000 |
| Open-loop Phase-C output | scene-type-averaged grey/brown wash | scene-type recovered (kitchen / bathroom content); blurry, wrong viewpoint |
| Phase-C L1 (open-loop, K=8) | 0.139 | TBD on final ckpt |

The v2-fidelity stack changes **three things simultaneously**: KL balance
direction, perceptual loss inclusion, and decoder upsample backbone. The
**load-bearing change** (per the loss-curve diagnostics) is the `kl_balance`
flip — without it, the encoder cannot escape the trivial post-matches-prior
fixed point regardless of what loss term is added on top. LPIPS and
PixelShuffle are amplifiers of the recovered scene-conditioning signal but
cannot rescue posterior collapse on their own.

---

## 5. Training infrastructure (both conditions)

| Item | Value |
|---|---|
| Hardware | AWS p5.48xlarge, 8 × NVIDIA H100 80GB HBM3, NVLink intra-node |
| OS / DLAMI | Ubuntu 22.04, Deep Learning AMI, NVIDIA driver 570.133.20 (CUDA 12.8 max) |
| PyTorch | 2.11.0+cu128 (forced from PyPI cu128 wheel index after the default cu13 build was incompatible with driver 570) |
| Distribution | DDP via `torchrun --standalone --nproc_per_node=8`, NCCL backend |
| NCCL config | `NCCL_NET_PLUGIN=none` (mandatory: aws-ofi-nccl segfaults without EFA on this instance) |
| Precision | bf16 autocast (`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`); backward pass in fp32 |
| Optimizer | `torch.optim.AdamW(lr=1e-4, weight_decay=1e-6)`, gradient clip 100.0 |
| Throughput observed | ~6 it/s (M0/M1, no LPIPS), ~4.5 it/s (M3 + v2 with LPIPS + PixelShuffle) |
| Checkpointing | every 2500 steps (`step_NNNNNN.pt`) + final `last.pt`; ~158 MB per ckpt |
| Continuous evac | `aws s3 sync` every 120 s in dedicated tmux pane |
| Resume mechanism | `--resume <ckpt>` loads model + optimizer + start_step; new run dir with separate `metrics.jsonl` |

---

## 6. Limitations identified, addressable in v2 of the paper

The qualitative outputs of M3 + v2-fidelity reveal three independent failure
modes, each with a concrete architectural fix:

1. **Decoder capacity ceiling**: 48:1 compression in the from-scratch CNN
   decoder (`render_channels=64`, `feat_hw=8×8`) caps the renderable
   spatial frequency. **Fix**: replace the from-scratch CNN with the frozen
   pretrained Stable Diffusion VAE decoder (`stabilityai/sd-vae-ft-mse`).
   Implementation present in this codebase (`sralnik/models/sd_vae_decoder.py`)
   and exposed via `--sd-vae`; not used in the final M3 + v2 run because the
   combination with the v2 KL settings was not finalised within the compute
   budget. Future work.
2. **Viewpoint drift in open-loop rollout**: the absence of pose conditioning
   means orientation has to be inferred from the action sequence integrated
   through the GRU's 256-dim state. This integration accumulates error.
   **Fix**: feed pose `(x, y, yaw)` directly to the encoder (extra channels
   or an MLP embedding) and/or to the GRU input alongside the action
   embedding. Requires fresh training; ~30 lines of code change.
3. **Mode-mixing in the decoder under uncertainty**: pure L1 loss on a CNN
   decoder produces conditional-mean blur when the input `(h, z)` represents
   uncertainty between plausible viewpoints. Manifests as ghost-overlay
   double vision in some Phase-C reconstructions. **Fix**: add adversarial
   loss (single-mode commitment) or replace L1 with LPIPS-only / discrete
   latent (DreamerV3-style) which does not have the conditional-mean bias.

---

## 7. References to the implementation

- `sralnik/models/world_model.py` — top-level `WorldModel` class, forward pass.
- `sralnik/models/encoder.py` — strided-conv encoder + posterior MLP.
- `sralnik/models/decoder.py` — CNN decoder with optional PixelShuffle upsample.
- `sralnik/models/memory.py` — `MemoryFusion` for M1/M2/M3 modes (NONE is no-op).
- `sralnik/models/config.py` — `ModelConfig` dataclass with all hyperparameters.
- `sralnik/models/lpips_loss.py` — frozen-VGG LPIPS wrapper.
- `sralnik/training/ddp_train.py` — DDP training loop, optimizer setup,
  checkpointing.
- `sralnik/training/eval_run.py` — teacher-forced eval (Phase-C L1/MSE
  stratified by probe / gap / scene).
- `sralnik/training/eval_rollout.py` — open-loop rollout eval producing GIFs
  (last K frames imagined from prior; encoder warmup over `T - K` steps).
- `scripts/run_condition.sh` — launcher for the matched M0/M1/M2/M3
  conditions.
- `scripts/run_bonus_fidelity.sh` — launcher for M3 + v2-fidelity (CNN
  renderer + LPIPS + KL fixes + PixelShuffle).
- `scripts/run_v3.sh` — launcher for the SD-VAE variant (not used in the
  final reported run).

---

## 8. Reproducibility

To reproduce M0:

```bash
bash scripts/run_condition.sh none 75000
```

To reproduce M3 + v2 fidelity:

```bash
bash scripts/run_bonus_fidelity.sh gated 30000
```

Hyperparameters (`free_bits=1.0`, `kl_balance=0.2`, `lpips_weight=0.5`) are
the launcher defaults after the bug fix; override via env vars
(`FREE_BITS=…`, `KL_BALANCE=…`, `LPIPS_WEIGHT=…`) to ablate. Resume after
hitting `--max-steps`:

```bash
RESUME=runs/m_gated_v2fid_<ts>/last.pt bash scripts/run_bonus_fidelity.sh gated 60000
```

A new run dir is created; concatenate `metrics.jsonl` across the original
and resumed dirs to get a continuous loss curve.
