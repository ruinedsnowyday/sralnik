# SRALNIK: model architecture (concrete specification)

This document defines the **actual model** we train and use to **compare memory
augmentation** under long-gap revisit. It is consistent with the problem statement
in `write-up/main.tex`: a **compact latent world model** for AI2-THOR /
ProcTHOR-style indoor trajectories, where Phase **B** distracts the agent and
Phase **C** tests whether **Phase A** state is still reflected in prediction.

**Data contract** (shapes, splits, probes) is in `README.md`. Here we fix the
**learning architecture** and the **experimental conditions** (what differs
between runs).

**Implementation scope.** The running system lives in `sralnik.models`
(`WorldModel`, `Encoder`, `MemoryFusion`, `Decoder`, optional `LatentDiffusion`)
and `sralnik.training` (HDF5 dataset, DDP `train`, `eval`). Where this doc still
mentions **extensions** (pose/seg in the encoder, LPIPS, local Transformer mixer,
multi-step latent NLL), they are **design hooks** unless explicitly marked
*implemented in-repo*.

---

## 0. Goal and hypotheses

**Goal.** Learn a world model that predicts future RGB (and optional aux
modalities) from past frames and discrete actions, and **remains faithful to the
same room** after a **long occlusion** (large `gap_length` in Phase **B**).

**Central comparison.** All runs share the **same encoder, latent geometry,
action conditioning, and decoder family**. What changes is **how (or whether) an
external memory bank** influences the **dynamics / representation**:

| ID | Condition | Role |
|----|-----------|------|
| **M0** | **No memory** | Baseline recurrent latent dynamics only. |
| **M1** | **Retrieval + concat** | Top‑\(k\) retrieved latents are concatenated (or pooled) and fused by an MLP into the state before prediction. |
| **M2** | **Retrieval + cross‑attention** | Current state queries the memory bank (keys/values); readout replaces or adds to recurrence (IRIS / WorldMem-style read without a scalar gate). |
| **M3** | **Memory‑gated RSSM** (proposed) | Same readout as M1 or M2, but a **learned gate** \(\alpha_t \in [0,1]\) decides **how much** of the memory contribution is injected into the RSSM path (sparse use when recent context suffices; stronger injection when revisiting). |

**Hypothesis** (matches the introduction): M0 degrades as **gap** increases;
**M1–M3** improve **Phase C** fidelity vs ground truth and vs M0, with **M3**
avoiding “always-on” memory that might hurt short-horizon modeling.

---

## 1. Notation and inputs

Per timestep \(t\):

- **Observation** \(x_t \in \mathbb{R}^{H \times W \times 3}\) — default \(H=W=256\).
- **Optional aux** (training only if enabled): instance segmentation
  \(s_t\) (e.g. as extra input channels or a small side tower), depth \(d_t\).
- **Discrete action** \(a_t\) — index into the fixed vocabulary in HDF5 attrs;
  embed to \(e^a_t \in \mathbb{R}^{D_a}\).
- **Action success flag** \(f_t \in \{0,1\}\) (optional embedding).
- **Proprioception** \(p_t\) — logged in episode HDF5 as `pose`, but **not** fed
  into the reference `Encoder` / dynamics (extension: \(e^p_t\) in §2.2).
- **Phase** \(\phi_t \in \{A,B,C\}\) — stored as integers `\{0,1,2\}` in HDF5; used
  for **reconstruction weighting** and the **memory write mask** (§3.2).

**Batch training** uses subsequences of length \(T\) (CLI default **16**;
`EpisodeChunkDataset`; increase when VRAM allows — e.g. **32–64** on GPU).

---

## 2. Encoder (shared across all conditions)

### 2.1 CNN image encoder (*implemented in-repo*)

The reference `Encoder` is a **stack of strided \(4\times4\) convolutions** +
**GroupNorm + SiLU**, with channel widths from `ModelConfig.enc_channels` (default
`(32,64,128,256,256)`). For default `image_size=256`, this yields a spatial grid
\(u_t \in \mathbb{R}^{8 \times 8 \times C_u}\) with \(C_u\) equal to the last
entry of `enc_channels`.

**Extensions** (not wired in the current trunk): ResNet/ConvNeXt; extra input
channels from downsampled **seg / depth**.

### 2.2 Stochastic latent (RSSM “visual embedding”) (*implemented in-repo*)

The map \(u_t \mapsto (\mu^{z}_t,\sigma^{z}_t)\) is an MLP on **flattened** \(u_t\)
(**no** pose term in the reference code):

\[
z_t \sim \mathcal{N}(\mu^{z}_t, \sigma^{z}_t), \quad
(\mu^{z}_t, \log \sigma^{z}_t) = \mathrm{Enc}_\phi(u_t).
\]

Default **\(d_z = 32\)** (`ModelConfig.stoch_dim`); **\(d_h = 256\)** for \(h_t\)
(`deter_dim`).

**Representation loss** (*implemented*): **balanced KL** with **free-bits**
(`ModelConfig.kl_balance`, `free_bits`) between posterior
\(q_\phi(z_t \mid x_t)\) and prior \(p_\theta(z_t \mid h_{t-1})\) emitted from the
**previous** recurrent state (teacher-forced training; see §4.2).

---

## 3. Memory bank (M1–M3 only)

### 3.1 What is stored

Each **write** appends a tuple to a finite bank \(\mathcal{M}\) (cap \(N_{\max}\),
e.g. 256–4096 entries per GPU batch trajectory or **per episode buffer**):

- **Value** \(v \in \mathbb{R}^{d_z}\) (the stochastic latent \(z_t\) **or** a
  dedicated **memory value** produced by a linear head from \(u_t\)).
- **Key** \(k \in \mathbb{R}^{d_k}\), default \(d_k = d_z\), \(k = W_k z_t + b\)
  or \(k = \mathrm{normalize}(W_k [z_t; e^p_t])\).
- **Metadata** (aligns with the write-up): **scalar time index** \(t\) or
  **normalized phase id**, **camera pose** \(p_t\) (for distance / room priors),
  optional **scene id** one-hot if multi-room in future.

Metadata can enter the **similarity score** (e.g. reward closeness in yaw /
position when retrieving “same place, different angle”) but the **minimal
implementation** is **cosine or dot-product on \(k\)** only.

### 3.2 Write policy (*implemented in-repo*)

The reference **write mask** (`WorldModel` / `_write_mask_from_phase`) is
**probe-aligned** but fixed to the subsequence window:

- **Always write** timestep **0** of the training crop (often Phase **A** if the
  crop starts at episode onset; in general: first frame of the chunk).
- **Also write** on every **A\(\to\)B** transition (`phase` goes \(0 \to 1\)).

This approximates “anchor after Phase A change” whenever the boundary falls inside
the window; it does **not** add Phase **B** “route” writes.

**Frozen ablation intent**: same schedule for **M1–M3**; only **read / fusion**
differs. **M0** ignores the bank regardless of mask.

---

## 4. Dynamics: RSSM + GRU (*implemented in-repo*)

We use a **deterministic recurrent state** \(h_t \in \mathbb{R}^{d_h}\) (default
**\(d_h=256\)**, `ModelConfig.deter_dim`) and the stochastic \(z_t\) from §2.

**One-step recurrence** (teacher-forced training, `WorldModel.forward`):

1. **Prior** \(p_\theta(z_t \mid h_{t-1})\): Gaussian emitted by an MLP from
   \(h_{t-1}\) (no explicit \(a_{t-1}\) in that MLP in the reference code).
2. **Posterior** \(q_\phi(z_t \mid x_t)\) from the encoder at the **current** frame
   \(x_t\). Training samples \(z_t = \mu_t + \sigma_t \odot \epsilon\) (eval can
   use \(z_t=\mu_t\)).
3. **Memory read** (M1–M3): build context from **detached** history
   \(z_{0:t-1}\) masked by the write schedule (§3.2); **fusion updates the GRU
   hidden *input*** \(h^{\mathrm{mem}}_t\) from \(h_{t-1}\) (see §4.1).
4. **GRUCell** (*implemented*):
   \[
   h_t = \mathrm{GRUCell}\big([z_t; e^a_{t}; f_t],\, h^{\mathrm{mem}}_t\big),
   \]
   with \(e^a_t\) an embedding of the discrete action (+1 offset for a padding row)
   and \(f_t\) the scalar action-success flag.

**Extension (not in reference code):** **local Transformer** mixer over the last
\(K\) \((h,z,a)\) tokens before the GRU.

### 4.1 Where memory enters (conditions M1–M3) (*implemented in-repo*)

Let **\(h_{t-1}\)** be the GRU hidden after the previous step. The core code order
is:

1. **Prior** \(p_\theta(z_t\mid h_{t-1})\); **posterior** \(q_\phi(z_t\mid x_t)\) and
   sample/deterministic \(z_t\).
2. **Memory read**: query from \(\mathrm{concat}(h_{t-1}, z_t)\); attend / top‑\(k\)
   over **detached** history latents \(z_{0:t-1}\) with mask (§3.2). Fusion returns
   a **candidate GRU hidden** \(h^{\mathrm{mem}}_t\) (identity = \(h_{t-1}\) when
   M0 or empty bank).
3. **GRUCell:** \(h_t = \mathrm{GRUCell}([z_t; e^a_t; f_t],\, h^{\mathrm{mem}}_t)\).

Concrete fusion (`MemoryFusion`):

- **M1 — Concat / residual MLP:** top‑\(k\) \(z\)’s concatenated with \(h_{t-1}\),
  MLP produces a **residual added to** \(h_{t-1}\).
- **M2 — Cross-attention:** MHA with query from \((h_{t-1}, z_t)\), keys/values from
  history \(z\); residual adds projected readout to \(h_{t-1}\).
- **M3 — Gated:** same attention readout as M2, scaled by a sigmoid gate on
  \([h_{t-1}; z_t]\) (then residual to \(h_{t-1}\)).

**M0:** `MemoryFusion` is a no-op; \(h^{\mathrm{mem}}_t = h_{t-1}\).

### 4.2 Latent prediction loss (*implemented in-repo*)

There is **no** separate “next-step \(z\)” regression term in the reference
objective. **KL** is computed at the **same timestep \(t\)** between the encoder
posterior and the **prior from \(h_{t-1}\)** (Dreamer-style balancing +
free-bits). **Rollout / imagination** sampling is not required for the current
training loop.

---

## 5. Decoder + optional latent diffusion (*implemented in-repo*)

**Motivation.** A small **CNN decoder** from \((h_t,z_t)\) reconstructs pixels;
**optional latent diffusion** on a bottleneck grid \(l_t\) adds a train-time
\(\epsilon\)-prediction loss without replacing the pixel decoder used for
\(\hat{x}_t\) in the current `WorldModel` forward pass.

### 5.1 Deterministic bottleneck for rendering

Map \((h_t, z_t)\) to a **render latent grid**
\(l_t \in \mathbb{R}^{h' \times w' \times C_\ell}\) with defaults
\(h'=w'=8\), \(C_\ell=32\)–\(64\) via a shallow deconv / MLP-on-pixels.

### 5.2 Latent diffusion (default: **light** \(\ell\)-UNet)

Implementation default in ``ModelConfig`` is tuned for **throughput**, not SD-scale
quality:

- **Train timesteps** \(T_{\mathrm{diff}}\) **32** (sufficient for an **8×8** latent;
  increase only if samples look under-diffused).
- **Channels** **32**, **one** residual block (no multi-scale pyramid).
- **Time embedding** width **32**, small MLP (**×2** hidden vs **×4** “heavy”).

For inference, keep **few** DDIM steps (**5–10**) on this bottleneck.

A heavier variant (more blocks, \(T_{\mathrm{diff}}\!\approx\!100\), wider channels)
is a quality knob if you have extra GPU-days.

### 5.3 Pixel reconstruction path (*implemented in-repo*)

**Always:** \(\hat{x}_t = \mathrm{Dec}_\eta(h_t, z_t)\) (`Decoder`) with **per-step
L1** to \(x_t\), **phase-weighted** so **Phase C** (`phase == 2`) counts more
(`phase_weight`, default **2.0**).

**Optional extension:** add **LPIPS** on RGB (not in reference loss yet).

When `use_latent_diffusion=True`, **additional** loss: \(\epsilon\)-prediction on
\(l_t = \mathrm{Bottleneck}(h_t,z_t)\) with weight `diffusion_loss_weight`
(default **0.06**); this does **not** change which tensor is named \(\hat{x}_t\)
in the training forward (still the CNN `Decoder`).

---

## 6. Losses (reference training code)

Per timestep, the **implemented** objective is:

\[
\mathcal{L} =
\mathcal{L}_{\mathrm{rec}}(x_t, \hat{x}_t)
+ \mathcal{L}_{\mathrm{KL}}\big(q_\phi(z_t\mid x_t)\,\|\,p_\theta(z_t\mid h_{t-1})\big)
+ \mathbb{1}_{\mathrm{diff}}\,\lambda_{\mathrm{diff}}\,\mathcal{L}_{\epsilon}(l_t),
\]

where \(\mathcal{L}_{\mathrm{rec}}\) is **mean L1** over spatial channels
(**up-weighted on Phase C**), \(\mathcal{L}_{\mathrm{KL}}\) uses **balanced**
gradients + optional **free-bits**, and \(\mathcal{L}_{\epsilon}\) is the latent
diffusion training loss when enabled.

**Not in reference loss (paper extensions):** next-step latent NLL/MSE, **LPIPS**,
**segmentation** auxiliary.

**Manual / expert_eval episodes:** excluded from **`EpisodeChunkDataset`** by
default (`exclude_manual=True` in training/eval loaders).

---

## 7. Evaluation (ties to probes)

Report **per condition** (M0–M3), per **scene**, per **probe_name**, binned by
**gap_length**:

- **Pixel / latent**: **Implemented in `python -m sralnik.training eval`:** per-frame **L1** and **MSE** on **Phase C** only (`phase == 2`), teacher-forced rollout, deterministic encoder posterior (**\(z=\mu\)**). Summaries are printed (and optionally written as a per-frame parquet) with breakdowns by **`probe_name`**, **`gap_bucket`** (`g20` / `g100` / `g300` / `g1000` / `other` / `na` for missing or non-sweep gaps), and **`scene`**. This matches the paper-facing stratification for pixel metrics.
- **Object / layout** (from `tracked_objects_json` / instance maps): **state**
  agreement (toggle/open, receptacle contents, displaced object still on target
  surface when segmentation available). *Not in the reference eval script yet;* add when seg + whitelists are wired.
- **Revisit identity**: embedding distance between predicted Phase **C** latent
  and stored Phase **A** anchor (cheap diagnostic). *Optional extension.*

Train / val / test follow `split` in `manifest.parquet`; hold out
`gap_length = 1000` for extrapolation as in `README.md`.

---

## 8. Training stages (what you actually run)

Training is **not** one monolithic job. Use **stages** so CPU/single-GPU catches
bugs early; use the **8× H100** block mostly for Stage 1 (and optionally 2).

### Stage 0 — Sanity (minutes)

- **CPU or 1× GPU**, tiny batch, short sequence \(T\), **M0**, **no** diffusion.
- Goal: dataloading, shapes, finite loss, backward, optimizer step
  (`smoke-synthetic` / `smoke-fit` in `sralnik.training`).

### Stage 1 — Core world model (largest compute)

- **RSSM + GRU + CNN decoder** + KL + reconstruction (Phase **C** weighting).
- Train each **memory condition** you report (**M0** + e.g. **M2/M3**) with
  matched hyperparameters except `MemoryMode`.
- Keep **latent diffusion off** until val curves on probe slices look reasonable.

### Stage 2 — Latent diffusion renderer (optional)

- Enable **ε-prediction** on bottleneck \(l_t\) (small UNet). **Warm-start** from
  Stage 1 or train jointly if stable.
- Expect **lower** steps/sec than Stage 1; shrink UNet depth / diffusion train
  steps if needed.

### Stage 3 — Eval / tables

- Checkpoints → **held-out split**, **probe-stratified** metrics: reference CLI
  reports Phase **C** **L1 / MSE** (see §7). **LPIPS** and **object/consistency**
  metrics are **future** table columns.

You may **merge** Stages 1–2 into one joint run if ablations stay stable.

---

## 9. Expected wall time on **8× NVIDIA H100**

**Ballpark only.** Real time depends on **global batch**, **\(T\)**, **epochs**,
**diffusion on/off**, and **I/O** (local NVMe vs NFS).

**Assumptions (project-scale):** ~**800** scripted episodes × ~**100–250** frames
each → **\(10^5\)–\(10^6\)** frames total (your manifest is ground truth).
**8× H100**, **DDP**, **`bf16`**, compact encoder–GRU–decoder (default
`ModelConfig`), not a huge ViT + full pixel diffusion.

| Stage | What | Indicative wall (8× H100, one run) |
|-------|------|--------------------------------------|
| **1** | RSSM + CNN decoder until val looks strong (**many** epochs / long schedules) | **~6–24 h** |
| **2** | + **light** latent diffusion (default `ModelConfig`: 32 train steps, 32 ch, 1 block) | **+~4–15 h** wall often; **+~12–36 h** only if you scale up ε-UNet / \(T_{\mathrm{diff}}\) like a “heavy” setup |

### Capsule budget: **~4–5 h wall** on **8× H100**

Eight GPUs for **5 h** ⇒ about **40 GPU-hours** of compute — enough for a **course-scale**
run, **not** for “train until marginal returns vanish.”

**Practical recipe to stay near 4–5 h total (per training job):**

1. **Stage 1 only** for the bulk: **`use_latent_diffusion=False`**, RSSM + CNN
   decoder, **`bf16`**, largest batch + \(T\) that fit VRAM without thrashing I/O.
2. **Cap updates**, don’t chase infinite epochs — e.g. **~15–30 epochs** over the
   scripted set, or a fixed **optimizer-step budget** (\(\sim\!10^4\)–\(10^5\)
   steps — tune once you know **steps/s** from your worker). Early-stop on val if
   it plateaus earlier.
3. **Stage 2 inside the same 5 h** only as a **short fine-tune** (e.g. **≤1 h**)
   with the **light** ε-UNet defaults, **or** skip diffusion for the deadline and
   rely on the CNN decoder for numbers/figures.
4. **Two memory conditions** (e.g. M0 + M3) **one after another** ⇒ **~2–2.5 h
   each** on average to stay under **5 h** total — or use two parallel 4-GPU jobs
   if your cluster allows.

### Same **~5 h** while **keeping** the small latent diffusion decoder

Use **one joint run** for the whole wall clock (no long separate “Stage 2”):

1. Set ``use_latent_diffusion=True`` with **light** defaults (32 train timesteps,
   32 channels, 1 block) — that is still your intended architecture.
2. **Cap time by capping optimizer steps** (or epochs): after a short benchmark,
   set ``max_steps ≈ 5 h × steps/s × 0.9`` (leave slack). Joint training adds a
   modest per-step cost (~**1.3–1.6×** vs diffusion-off is typical for this tiny
   UNet); reduce the step budget accordingly if you must stay under 5 h wall.
3. Keep diffusion **subordinate** in the loss: ``diffusion_loss_weight`` in
   ``ModelConfig`` (default **0.06**) so reconstruction + KL dominate when updates
   are scarce; bump slightly only if ε looks under-trained on val.
4. **Feed the GPUs**: local **NVMe** shards, enough dataloader workers, **bf16**,
   large micro-batch per GPU — I/O matters more than ε-UNet width for the 5 h goal.
5. **M0 + M3 in 5 h total**: ~**2.5 h per condition** with the **same** step cap, or
   parallelize two 4-GPU jobs if possible.

The numbers above are still **order-of-magnitude**; your measured **samples/sec**
on NVMe-backed shards is what turns a step budget into wall clock.

For **long** schedules (“train until convergence”), the table’s Stage **1**/**2**
ranges still apply. For several **sequential** conditions, multiply wall time unless
you parallelize jobs.

**Rule of thumb (unbounded training):** **~1–3 GPU-days** per condition for Stage 1
(8 GPUs ⇒ **~3–9 h** wall if saturated). **Light** Stage 2 is often **~0.5–2 extra
GPU-days**; heavy diffusion can be **~1–5 GPU-days**.

---

## 10. Module diagram (implementation view)

```text
  x_t ──► [StridedConv Encoder] ──► u_t ──► MLP ──► z_t ~ q(z|x_t)
                                                 │
  prior p(z_t|h_{t-1}) ◄── MLP(h_{t-1}) ─────────┤ KL (balanced + free-bits)
                                                 │
  history z_{0:t-1} + write_mask ──► READ / fuse (M1–M3) ──► h_gru_in
                                                 │
  (z_t, e^a_t, f_t) + h_gru_in ──► GRUCell ──► h_t
                                                 │
                                                 ├──► Dec_η(h_t,z_t) ──► x̂_t  (L1 rec)
                                                 │
                                                 └──► Bottleneck(h_t,z_t) ──► l_t
                                                                 └── optional ε-UNet train loss
```

**Eval:** \(\hat{x}_t\) for metrics is the **CNN `Decoder`** output (teacher-forced,
\(z=\mu\) option in `eval`).

---

## 11. Roadmap (*implemented vs extensions*)

| Item | Status in-repo |
|------|----------------|
| HDF5 + `manifest.parquet` loader (`EpisodeChunkDataset`) | **Done** |
| Strided-CNN encoder, GRU RSSM, balanced KL, CNN decoder | **Done** |
| Memory M1–M3 (`MemoryMode`) + write mask (§3.2) | **Done** |
| Optional latent diffusion on \(l_t\) | **Done** (`--diffusion`) |
| DDP **`train`** (`torchrun`), bf16, checkpoints | **Done** (`README`) |
| **`eval`**: Phase-C L1/MSE, probe/gap/scene tables | **Done** |
| Pose / seg in encoder; LPIPS; seg loss | **Not yet** |
| Latent rollout imag + multi-step open-loop eval | **Not yet** |
| Object-state / layout metrics from seg + `tracked_objects_json` | **Not yet** |

---

## 12. Reference implementation in-repo

- **Hyper-parameters:** `sralnik.models.ModelConfig` (\(d_h,d_z\), `enc_channels`,
  diffusion widths, `diffusion_loss_weight`, memory knobs).
- **Model:** `sralnik.models.WorldModel`.
- **Training / eval:** `sralnik.training` — module `python -m sralnik.training`:
  - `smoke-synthetic` / `smoke-fit` — fast sanity (CPU-friendly).
  - `train` — full fit; **8× GPU**: `torchrun --standalone --nproc_per_node=8 -m sralnik.training train ...` (see `README.md`).
  - `eval` — Phase-C **L1/MSE** tables + optional per-frame **parquet**.

Exact **CLI flags** and a copy-pastable **H100** command line are in **`README.md`**.
