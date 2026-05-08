# SRALNIK design evolution: from plan to shipped state

This document is the **postmortem narrative** of what changed between the
design spec (`docs/ARCHITECTURE.md`) and the original 24h runbook
(`docs/RUN_PLAN.md`) on the one hand, and the actually-shipped end-state
(`docs/PAPER_ARCHITECTURE_SUMMARY.md`) on the other. It exists so the paper's
discussion section can faithfully describe how the experimental result was
reached, including the failure modes encountered along the way.

The intended reader is the paper's discussion-section author and any future
collaborator who needs to understand why the codebase looks the way it does.

---

## Original plan (per `docs/RUN_PLAN.md`)

A 24h capacity block on 8× H100 to run the matched ablation across the four
memory conditions defined in `docs/ARCHITECTURE.md`:

1. **M0** — no memory. Baseline.
2. **M1** — concat-style retrieval (top-k learned scoring + MLP).
3. **M2** — cross-attention (MHA query/key/value over masked history).
4. **M3** — gated cross-attention (M2 + sigmoid scalar gate).

Schedule: ~3.5h per condition × 4 conditions = ~14h training, plus eval and
final S3 evacuation. Eval = teacher-forced reconstruction on `--split val` and
`--split test`, Phase-C L1/MSE stratified by `probe_name`, `gap_bucket`,
`scene`. Latent diffusion left off (Stage 1 only). LPIPS / pose conditioning /
seg auxiliary marked as design-hooks not implemented.

## What actually shipped

**Two conditions trained to completion**, plus a third intervention condition:

1. **M0** matched, 75k steps. ✅ Ran as planned.
2. **M1** matched, 75k steps. ✅ Ran after fixing several pre-flight bugs.
3. **M2 matched** — ❌ **Skipped**.
4. **M3 matched** — ❌ **Skipped**.
5. **M3 + v2 fidelity** — ✅ Substituted in place of matched M3. CNN renderer
   with LPIPS + corrected KL balancing + PixelShuffle. ~30k steps initial
   run, extended via `--resume` toward 75k.

Plus an aborted attempt:

6. **v3 SD-VAE + LPIPS + free_bits=0** — ❌ Aborted at ~3k steps after KL
   collapsed to 0 (opposite failure from M0/M1). Diagnosed and rolled back.

Plus eval methodology changed:

- Headline figure became **open-loop rollout eval** on `expert_eval` (manual
  recordings) via a new `eval_rollout.py`, **not** the planned teacher-forced
  eval on `val/test`. Reason: teacher-forced eval doesn't actually test
  memory because the encoder peeks at every frame; open-loop with `K=8` last
  frames imagined is the regime that actually exercises the memory bank.

---

## Timeline of divergences

### Phase 1 — pre-flight bugs surfaced by `verify_correctness.py`

The planned `verify_correctness.py` (6 checks) caught issues that would have
killed the matched runs:

- **CONCAT (M1) shape mismatch.** With history shorter than `memory_topk` (k=4)
  during the first few timesteps of a crop, the concat-with-h tensor had a
  smaller last dim than the `_to_delta` MLP expected. Fix: zero-pad `top_z` to
  `memory_topk * d_z` before concat. Commit `6cd2b40`.
- **`MemoryFusion._q` was unused params for `MemoryMode.NONE`.** Created in
  `__init__` but never called in forward when mode=NONE → DDP's
  `find_unused_parameters=False` errored. Fix: only construct `_q` when
  mode != NONE. Commit `662d552`.
- **`RenderBottleneck` was always constructed** but only used when
  `use_latent_diffusion=True`. Same DDP unused-params error when diffusion
  was off. Fix: only construct `bottleneck` when diffusion is on, mirroring
  the `self.diffusion = None` pattern. Commit `c216b13`.
- **`F.scaled_dot_product_attention` returned a single Tensor**, but the code
  did `ctx, _ = F.scaled_dot_product_attention(...)` — destructured along
  dim 0 (batch axis). With B=2 in the verify script this silently picked
  row 0 with broadcasting; with the training B=4 it would crash with
  "too many values to unpack." Fix: drop the unpack. Commit `a99ca16`.
- **Dataset rows with empty `relative_path`.** Failed-collection placeholders
  in `manifest.parquet` had no .h5 file backing them, but `dataset.py` tried
  to `h5py.File(path)` them anyway. Fix: filter at construction. Commit
  `fcd9cc1`.

### Phase 2 — CUDA / NCCL infrastructure

The DLAMI shipped with NVIDIA driver 570.133.20 (CUDA 12.8 max). Default
PyTorch wheel from PyPI is built for cu13, requiring driver ≥580. PyTorch
silently fell back to CPU at first launch.

- Fixed by force-installing `torch==2.11.0+cu128` from the pytorch.org cu128
  wheel index, then pinning `torchvision` to the same index after a separate
  `nms` ABI mismatch surfaced via the LPIPS dependency.
- NCCL had a separate failure mode: the AWS OFI plugin (`aws-ofi-nccl`)
  segfaulted because the Capacity Block instance had no EFA attached. Fix:
  `NCCL_NET_PLUGIN=none`, baked into the launchers. Single-node 8-GPU NCCL
  works fine via NVLink without the OFI plugin.

Total elapsed on infrastructure debugging: ~2h.

### Phase 3 — M0/M1 trained, ablation result observed

M0 reached `loss_total ≈ 1.075` (rec ≈ 0.075 + KL pinned at 1.0 floor) at
75k steps. M1 (concat with the post-fix last-k retrieval) reached
`loss_total ≈ 1.078`, **statistically indistinguishable from M0 across all
aggregates** (overall L1, Phase-C L1, both scenes — within 1–2%).

Open-loop rollout eval on `expert_eval` (4 manual episodes) confirmed
qualitatively: M0 and M1 produce nearly identical scene-tone-averaged grey/
brown outputs. Memory at this scale + this configuration was providing
zero observable benefit.

### Phase 4 — diagnosis + architectural pivot decision

Code audit identified four candidate causes (any of which could individually
explain M0=M1):

1. **Posterior collapse** — KL pinned at the `free_bits=1.0` floor for all
   75k steps means q(z|x) ≈ p(z|h). z carries no frame-specific information,
   so memory has nothing useful to retrieve.
2. **Inverted KL balance direction.** `kl_balance=0.8` puts 80% of gradient
   on encoder pull-to-prior — the opposite of Dreamer-V2's intent. Encoder
   is *actively trained* to give up information, exacerbating (1).
3. **Decoder capacity ceiling.** From-scratch CNN with 4096-float bottleneck
   compressing to 196,608 pixels (48:1 ratio) cannot render fine detail
   regardless of how informative `z` is.
4. **No pose conditioning.** During open-loop rollout, the model has no way
   to track agent orientation; integration of action sequence through the
   GRU's 256-dim hidden state accumulates orientation error.

(1)+(2) are training-side fixes. (3) is an architectural change. (4) requires
retraining with a new encoder input.

Decision: skip M2 and matched M3 in favor of a single intervention run that
addresses (1) and (2) — the load-bearing fixes — and includes a perceptual
loss to amplify whatever scene-conditioning recovery (1)+(2) buy. SD-VAE
(addressing (3)) was held in reserve as an additional bonus condition.

### Phase 5 — v3 SD-VAE attempt + collapse + rollback

First intervention attempt: SD-VAE + LPIPS + `free_bits=0` + `kl_balance=0.2`.
The thinking was that `free_bits=0` would let KL escape the floor entirely,
combined with the corrected balance direction.

**Failure mode:** with `free_bits=0`, KL collapsed in the *opposite* direction
— q(z|x) matched the prior exactly (KL → 0), encoder gave up encoding any
information at all. The free-bits floor was the only thing preventing this
trivial degenerate solution; setting it to 0 removed the constraint.

Aborted at ~3k steps. Lesson: keep `free_bits=1.0` (the floor *is* needed),
combined with `kl_balance=0.2` (the *direction* needed correcting). Don't
remove the floor — that was a misdiagnosis.

### Phase 6 — v2 CNN intervention, the shipped run

Final config:

- Memory mode `gated` (M3)
- `--lpips --lpips-weight 0.5`
- `--free-bits 1.0` (kept)
- `--kl-balance 0.2` (flipped from default 0.8)
- `--pixel-shuffle`
- No SD-VAE, no diffusion

Observed: KL escaped the floor for the first time across all conditions,
climbing from 1.0 (steps 0–3000, encoder still equilibrating) to ~1.7 at
step 60k and continuing to rise slowly. Reconstruction loss continued to
descend in parallel. Open-loop GIFs showed scene-type recovery (kitchen
content for kitchen episodes, bathroom content for bathroom) — substantially
different from M0/M1's averaged grey wash.

Two limitations remained visible in the qualitative output:
- Decoder capacity ceiling: outputs are blurry / mode-mixed.
- Pose-tracking drift: viewpoint is wrong even at K=2 rollout (suggesting
  the decoder doesn't represent viewpoint at the resolution of distinct
  facets, beyond just orientation accumulation).

Both are documented as v2-of-paper future work.

### Phase 7 — extension via `--resume`

Added `RESUME` env-var support to `run_bonus_fidelity.sh` and `run_v3.sh`
(commit `e7f9a3c`) to allow continuing past the original 30k step cap
without restarting from scratch. The resumed run continues from
`step_NNNNNN.pt` — `last.pt` is only written at clean loop exit, so a
mid-run kill leaves only numbered ckpts; both kinds are valid resume
sources.

---

## How the docs were updated

| Doc | Before run | After run | Status |
|---|---|---|---|
| `ARCHITECTURE.md` | Original design spec | Same spec + §13 errata listing implementation deltas | Spec preserved as planning record |
| `RUN_PLAN.md` | Pre-run runbook with M0/M1/M2/M3 schedule | Same body + postmortem header pointing to actual record | Plan preserved as planning record |
| `V2_FIDELITY.md` | Written mid-run when SD-VAE was the headline | Status header + free_bits=1.0 correction + tightened caveats | Now matches shipped intervention |
| `SSH_ACCESS.md` | Operational, written for teammate | Unchanged | No drift; pure ops |
| `PAPER_ARCHITECTURE_SUMMARY.md` | Did not exist | Written post-run | Authoritative final-state spec |
| `DESIGN_EVOLUTION.md` | Did not exist | This document | Narrative postmortem |

`PAPER_ARCHITECTURE_SUMMARY.md` is the **source of truth** for what was
actually trained. The other docs preserve their respective roles (design
spec, runbook, ops, intervention catalog) with errata pointers.

---

## What this means for the paper

The natural framing for the methods + discussion section is the diagnostic
arc rather than a clean ablation:

1. **Context**: compact memory-augmented world model on AI2-THOR; goal is to
   demonstrate memory utility for Phase-C recovery after long Phase-B gaps.
2. **Matched ablation result** (M0 vs M1 at 75k steps): memory provides no
   measurable benefit. M1 ≈ M0 within 1–2% on every aggregate.
3. **Diagnosis** (code audit + KL-trajectory inspection): four independent
   failure modes — posterior collapse, inverted KL balance, decoder
   capacity, absent pose conditioning.
4. **First intervention attempt** (v3, free_bits=0): the *opposite* failure
   mode (KL→0). Removing the information floor doesn't help; the floor is
   load-bearing.
5. **Corrected intervention** (v2, free_bits=1.0 + kl_balance=0.2 + LPIPS +
   PixelShuffle): KL escapes the floor; scene-type conditioning recovers
   in the qualitative output. This is the **first positive result** in the
   experimental campaign.
6. **Limitations of the corrected intervention**: decoder capacity ceiling
   and pose-tracking drift remain visible. Two clean future-work items
   (SD-VAE + pose conditioning) are identified with concrete implementation
   sketches in the codebase.

The narrative is "we ran the planned ablation, found it null, diagnosed
why, fixed one of the four failure modes cleanly, identified the others
as v2 work." That's a stronger course-paper contribution than "we built X
and it worked great" — it's an honest experimental-debugging story with
a positive scene-conditioning recovery as the load-bearing finding.

---

## File / commit pointers

For the paper's reproducibility statement:

- **Final M0 checkpoint**: `runs/m_none_20260506T131017/last.pt`
- **Final M1 checkpoint**: `runs/m_concat_20260506T173341/last.pt`
- **M3 + v2-fidelity initial run**: `runs/m_gated_v2fid_20260506T224811/last.pt`
- **M3 + v2-fidelity resume run**: `runs/m_gated_v2fid_20260507T015152/step_*.pt`
- **Code at end of run**: commit `c68c143` (`docs: paper-grade architecture summary…`)
- **Earlier load-bearing fixes**: commits `276f8d6` (last-k retrieval),
  `662d552` (`_q` conditional), `c216b13` (bottleneck conditional),
  `a99ca16` (SDPA tuple-unpack + v3 stack scaffolding), `8db467f`
  (free_bits default 0.0→1.0), `e7f9a3c` (RESUME env-var).
