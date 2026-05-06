# 24h 8×H100 Run: SRALNIK Memory-Augmented World Model

## Context

**Why this plan exists.** SRALNIK (MIT 6.S058 course project, Bohdan + Trofimov) is a memory-augmented latent world model for AI2-THOR scene reconstruction. The full ablation across memory conditions M0–M3 (none / concat / cross-attn / gated) needs 8×H100 wall-clock, which we have for exactly **24 hours** via an AWS Capacity Block that **starts in 55 minutes** and at whose end the instance is **terminated** (anything not evacuated is gone). The codebase is implemented end-to-end (DDP + bf16 + ckpt/resume + Phase-C eval), but there is **no** wandb, no Docker, no AWS scripts. This plan covers: (1) pre-flight in the next 50 minutes, (2) AWS launch + instance setup, (3) a time-blocked schedule across all four memory conditions plus eval, (4) live observability over SSH, and (5) a three-layer evacuation strategy that survives the hard 24h cutoff. Success = four trained checkpoints + Phase-C eval parquets in S3 + reproducible run metadata, ready to feed paper figures.

**Top three risks driving every decision below:**
1. 41 GB upload from a home connection is the long pole to first training step.
2. `torch.save` in `ddp_train.py:70` is non-atomic — naive 60s S3 sync can capture a half-written checkpoint.
3. The instance vanishes at hour 24:00 sharp. Anything still on local NVMe at that moment is lost forever.

---

## TL;DR Timeline

| When | What | Where |
|---|---|---|
| **T−50m → T−0** (now) | Pre-flight: S3 upload of `data/`, AWS creds check, AMI lookup, key prep, smoke locally | Laptop |
| **T+0 → T+15m** | `aws ec2 run-instances` with capacity-reservation flag; SSH in | Laptop → AWS |
| **T+15 → T+30m** | `git clone`, `uv sync`, pull data from S3 to NVMe | Instance |
| **T+30 → T+40m** | **Correctness verification** (`verify_correctness.py`) — catches bugs cheap | Instance |
| **T+40 → T+50m** | DDP smoke + steps/sec calibration → lock `--max-steps` | Instance |
| **T+0:50 → T+5:20** | M0 baseline (no memory) — full 8 GPUs | Instance |
| **T+5:20 → T+9:50** | M1 (concat) | Instance |
| **T+9:50 → T+14:20** | M2 (cross-attn) | Instance |
| **T+14:20 → T+18:50** | M3 (gated) — the headline condition | Instance |
| **T+18:50 → T+21:00** | Eval all four ckpts on `--split val` and `--split test` | Instance |
| **T+21:00 → T+23:30** | Plot generation (Jupyter), final S3 sync, sanity download to laptop | Both |
| **T+23:30 → T+24:00** | Hard cutoff buffer; second-machine `aws s3 ls` verification | Laptop |

If M0 calibration shows steps/sec is much slower than expected, drop M1 (M0 vs M2 vs M3 still tells the paper story) and keep the schedule.

---

## Phase 0 — Pre-flight (NOW, T−50m → T−0)

Run these in **parallel tmux panes** on the laptop. The S3 upload is the dominant clock.

### 0.1 Verify AWS state — do this first (1 minute)

```bash
aws sts get-caller-identity                  # confirms credentials work
aws ec2 describe-capacity-reservations \
    --filters Name=state,Values=active,scheduled    # confirms the block exists; note region + AZ
aws s3 ls                                    # any bucket? if not, create one in the same region as the reservation
```

If `aws sts` fails: this is a hard blocker — `aws configure` with an access key that has EC2 + S3 + IAM:PassRole. Cannot proceed otherwise.

If no S3 bucket exists in the reservation's region:
```bash
aws s3api create-bucket --bucket sralnik-runs-<your-initials> \
    --region <reservation-region> \
    --create-bucket-configuration LocationConstraint=<reservation-region>
aws s3api put-bucket-versioning --bucket sralnik-runs-<your-initials> \
    --versioning-configuration Status=Enabled        # critical — protects against bad-ckpt overwrite
```

### 0.2 Start the data upload — kicks off the long pole (~30–60 min)

**Context: the project arrived as an archive (`sralnik.zip` in the parent dir, 36 GB)** which contains a foreign `.venv/` from whoever packaged it — that's why the local Python is glitching. You don't need to fix it for pre-flight; nothing in §0 below requires running Python locally. The instance will build a fresh `.venv` against the lockfile.

Skip the zip itself (it's polluted with `.venv/`). Upload **both** dataset trees as separate tars and merge on the instance — this avoids mutating local `data/` with a broken environment, and the merge will replace the bad/crashed FloorPlan1 episodes in v2 with the good ones from fp1:

```bash
cd /Users/atrof002/Desktop/mit/studying/6.S058/sralnik-container/sralnik

# Tar each tree as a single multipart upload (HDF5 is already gzipped — no compression)
tar -cf - data/ithor_v2     | aws s3 cp - s3://sralnik-runs-<initials>/data/ithor_v2.tar     --expected-size 26000000000
tar -cf - data/ithor_v2_fp1 | aws s3 cp - s3://sralnik-runs-<initials>/data/ithor_v2_fp1.tar --expected-size 14000000000
```

Single-file multipart upload is **much** faster than `aws s3 sync` over thousands of small `.h5` files (per-object PUT latency dominates). On a 100 Mbps uplink expect ~50–80 minutes total for both. Watch with `aws s3 ls --human-readable s3://...` periodically.

**Fallback if total upload won't finish before launch:** start v2 first (the larger and more critical tree), launch the instance when v2 is done, kick off fp1 in parallel as M0 starts. The instance can pull fp1 once it lands and run merge between conditions — cost is small if fp1 trickles in during the first hour of training.

### 0.3 Push code to GitHub — small, fast (2 min)

```bash
git status                                   # untracked: docs/, sralnik/models/, sralnik/training/
git add docs/ sralnik/models/ sralnik/training/ pyproject.toml uv.lock README.md sralnik/__init__.py
git commit -m "Snapshot before 8xH100 run"
git push                                     # use a private repo
```

This makes `git clone` on the instance the fastest way to get code there.

### 0.4 Pick the AMI and prep launch flags — read-only (5 min)

Look up the current Deep Learning AMI ID for the reservation's region — has CUDA 12.x, drivers, NCCL, Python pre-installed:
```bash
aws ssm get-parameter \
    --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.4-ubuntu-22.04/latest/ami-id \
    --region <reservation-region> --query Parameter.Value --output text
```

Confirm SSH key pair exists in that region:
```bash
aws ec2 describe-key-pairs --region <reservation-region>
```
If not: `aws ec2 create-key-pair --key-name sralnik-h100 --query 'KeyMaterial' --output text > ~/.ssh/sralnik-h100.pem && chmod 400 ~/.ssh/sralnik-h100.pem`.

Confirm a security group allowing SSH (port 22) from your IP, and a default subnet in the reservation's AZ.

### 0.5 Prepare the IAM instance profile — gives the instance S3 access (3 min)

The instance needs to read/write S3 without long-lived keys. Either reuse an existing role with `AmazonS3FullAccess` (or scoped to the bucket), or create one. Have the role name ready for `--iam-instance-profile Name=...`.

### 0.6 Local sanity — *skip*

Originally I'd recommend a local `smoke-synthetic` to catch code bugs cheaply. **But your local `.venv/` came inside the archive** and is built for a different machine — it'll OOM-kill on import. Repairing it (`rm -rf .venv && uv sync`) takes ~2 min but adds risk of pulling different wheel versions than the lockfile expects on this Python. Skip it: the instance-side `verify_correctness.py` (§2.4) is strictly more comprehensive, runs on the actual H100s, and catches everything a CPU smoke would.

### 0.7 Decide: wandb or JSONL? (decide now)

See §"Observability" below. Default recommendation: **JSONL + S3** (zero-touch). If you have a wandb account ready and want pretty plots auto-logged, allocate 10 minutes during T+15→T+45 to wire it in.

---

## Phase 1 — AWS Launch (T+0 → T+15m)

The capacity is now live. Launch the instance into it.

```bash
# Capacity Blocks REQUIRE both --instance-market-options MarketType=capacity-block
# AND --capacity-reservation-specification. Missing the market option yields:
#   "The market type (purchasing) option is not valid."
aws ec2 run-instances \
    --instance-type p5.48xlarge \
    --image-id <DLAMI_ID> \
    --instance-market-options 'MarketType=capacity-block' \
    --capacity-reservation-specification \
        'CapacityReservationTarget={CapacityReservationId=cr-XXXXXXXX}' \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500,VolumeType=gp3,Iops=10000,Throughput=500}' \
    --iam-instance-profile Name=<your-s3-role> \
    --key-name sralnik-h100 \
    --security-group-ids sg-XXXX \
    --subnet-id subnet-XXXX \
    --region <reservation-region> \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=sralnik-h100-24h},{Key=auto-stop,Value=2026-05-07T<your-cutoff>}]' \
    --metadata-options 'HttpTokens=required'
```

Notes:
- `p5.48xlarge` = 8×H100 80GB, 192 vCPU, 2TB RAM, ~30 TB local NVMe (bind it to `/mnt/data` first thing).
- 500 GB gp3 root is enough for code + checkpoints + a buffer; the dataset goes on local NVMe.
- The `auto-stop` tag is informational — Capacity Block terminates regardless.

Wait for `running` state and grab the public IP:
```bash
aws ec2 describe-instances --instance-ids i-XXXX --query 'Reservations[].Instances[].PublicIpAddress'
ssh -i ~/.ssh/sralnik-h100.pem ubuntu@<public-ip>
```

---

## Phase 2 — Instance Setup & Smoke (T+15 → T+45m)

All commands below run **inside tmux** on the instance — `tmux new -s sralnik`. If SSH drops, training survives.

### 2.1 Mount NVMe, prepare workspace

```bash
lsblk                                                # find the local NVMe device(s)
sudo mkfs.ext4 /dev/nvme1n1                          # or use mdraid across multiple NVMes if present
sudo mkdir -p /mnt/data && sudo mount /dev/nvme1n1 /mnt/data
sudo chown ubuntu:ubuntu /mnt/data
```

If the instance has multiple NVMe drives, RAID0 them for I/O headroom:
```bash
sudo mdadm --create /dev/md0 --level=0 --raid-devices=<N> /dev/nvme[1-8]n1 \
    && sudo mkfs.ext4 /dev/md0 && sudo mount /dev/md0 /mnt/data
```

### 2.2 Pull data from S3 + run the deferred merge (3–5 min intra-region)

```bash
mkdir -p /mnt/data/sralnik && cd /mnt/data/sralnik

# Both tars in parallel (intra-region S3 → EC2 is ~5–10 GB/s)
aws s3 cp s3://sralnik-runs-<initials>/data/ithor_v2.tar     - | tar -xf - &
aws s3 cp s3://sralnik-runs-<initials>/data/ithor_v2_fp1.tar - | tar -xf - &
wait

ls data/ithor_v2/manifest.parquet data/ithor_v2_fp1/manifest.parquet   # sanity
```

Then run the merge (the local laptop never had a working venv to do this — it must happen on the instance). First, set up a *temporary* environment just to run merge — we can't `uv sync` yet because we haven't cloned the repo (next step). Quickest workaround: do step 2.3 first (`git clone` + `uv sync`), then come back here and run:

```bash
cd /mnt/data/sralnik/repo
# Backup before the destructive merge
cp /mnt/data/sralnik/data/ithor_v2/manifest.parquet \
   /mnt/data/sralnik/data/ithor_v2/manifest.parquet.pre-merge.bak

uv run python -m sralnik.data merge \
    --base /mnt/data/sralnik/data/ithor_v2 \
    --add  /mnt/data/sralnik/data/ithor_v2_fp1 \
    --scene FloorPlan1

# Quick verification that the merge took (rows should reflect fp1 replacement)
uv run python -c "import pandas as pd; \
  m = pd.read_parquet('/mnt/data/sralnik/data/ithor_v2/manifest.parquet'); \
  print('total rows:', len(m)); \
  print('FP1 by split:', m[m['scene']=='FloorPlan1']['split'].value_counts().to_dict())"
```

### 2.3 Code + env

```bash
cd /mnt/data/sralnik
git clone <your-repo-url> repo && cd repo
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv sync                                              # ~2 minutes; pulls torch 2.11+cu13 + nvidia wheels
```

The DLAMI already has CUDA 12.x + drivers; uv pulls the matching Python wheels. No system-level CUDA install needed.

### 2.4 Correctness verification — runs **before** any long run (5 min)

The training and eval scripts are implemented but never run end-to-end on a real GPU under DDP at scale. We have one shot — verify cheap before burning H100 hours. Save this as `verify_correctness.py` in the repo root:

```python
# verify_correctness.py — pre-flight checks the long run depends on
# Run: uv run python verify_correctness.py
import argparse, copy, json, tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sralnik.models import MemoryMode, ModelConfig, WorldModel
from sralnik.training.dataset import EpisodeChunkDataset, collate_fn
from sralnik.training.ddp_train import save_checkpoint, load_checkpoint

DATA = "/mnt/data/sralnik/data/ithor_v2"
DEV  = "cuda:0"

print("=== CHECK 1: All 4 memory modes give finite, *different* outputs ===")
torch.manual_seed(0)
B, T = 2, 16
obs   = torch.rand(B, T, 3, 64, 64, device=DEV)
acts  = torch.zeros(B, T, dtype=torch.long,  device=DEV)
succ  = torch.ones(B,  T, dtype=torch.bool,  device=DEV)
phase = torch.zeros(B, T, dtype=torch.long,  device=DEV)
phase[:, T//2:] = 1; phase[:, -2:] = 2  # crude A/B/C split

losses = {}
for mode in ["none", "concat", "attention", "gated"]:
    cfg = ModelConfig(image_size=64, memory_mode=MemoryMode(mode))
    torch.manual_seed(42)
    m = WorldModel(cfg).to(DEV)
    out = m(obs, acts, succ, phase=phase, posterior_sample=False)
    assert torch.isfinite(out["loss_total"]), f"{mode}: non-finite loss"
    losses[mode] = float(out["loss_total"])
    print(f"  {mode:10s} loss={losses[mode]:.4f}")
# Memory modes must differ from M0 (otherwise memory module is a no-op)
for m in ["concat", "attention", "gated"]:
    assert abs(losses["none"] - losses[m]) > 1e-4, f"{m} loss matches M0 — memory disengaged"
print("  ✓ memory modes produce distinct outputs")

print("\n=== CHECK 2: Gradients flow into MemoryFusion params (M3) ===")
cfg = ModelConfig(image_size=64, memory_mode=MemoryMode.GATED)
m = WorldModel(cfg).to(DEV)
m(obs, acts, succ, phase=phase)["loss_total"].backward()
mem_params = [(n, p) for n, p in m.named_parameters() if n.startswith("memory.")]
nz = sum(1 for _, p in mem_params if p.grad is not None and p.grad.abs().max() > 0)
print(f"  {nz}/{len(mem_params)} memory params received nonzero grad")
assert nz > 0, "Memory module never receives gradient — M1/M2/M3 will train identically to M0"
print("  ✓ memory module trains")

print("\n=== CHECK 3: Checkpoint save → load roundtrip ===")
with tempfile.NamedTemporaryFile(suffix=".pt") as f:
    p = Path(f.name)
    cfg = ModelConfig(image_size=64, memory_mode=MemoryMode.GATED, use_latent_diffusion=False)
    m1 = WorldModel(cfg).to(DEV); o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4)
    save_checkpoint(p, m1, o1, step=42, cfg=cfg)
    m2 = WorldModel(ModelConfig(image_size=64, memory_mode=MemoryMode.GATED)).to(DEV)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-4)
    step, cfg2 = load_checkpoint(p, m2, o2)
    assert step == 42 and cfg2.memory_mode == MemoryMode.GATED
    for (n1, q1), (_, q2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.allclose(q1, q2), f"param mismatch at {n1}"
print("  ✓ ckpt roundtrip preserves params + cfg")

print("\n=== CHECK 4: Phase-C frames + memory-write events present in real data ===")
ds = EpisodeChunkDataset(DATA, seq_len=16, split="train",
                         exclude_manual=True, max_rows=64, return_meta=True)
loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
phase_c = 0
write_evts = 0
for batch in loader:
    p = batch["phase"]
    phase_c += int((p == 2).sum())
    write_evts += int(p.shape[0])  # first frames
    if p.shape[1] > 1:
        write_evts += int(((p[:, :-1] == 0) & (p[:, 1:] == 1)).sum())
print(f"  Phase-C frames in 64 episodes × 16-frame crops: {phase_c}")
print(f"  Memory-write events (first-frame + A→B): {write_evts}")
assert phase_c > 0, "NO PHASE-C FRAMES — eval will be empty; check phase encoding"
assert write_evts > 0, "NO WRITE EVENTS — memory bank stays empty across all batches"
print("  ✓ phase + write-mask logic engages on real data")

print("\n=== CHECK 5: bf16 autocast doesn't NaN ===")
cfg = ModelConfig(image_size=64, memory_mode=MemoryMode.GATED)
m = WorldModel(cfg).to(DEV); o = torch.optim.AdamW(m.parameters(), lr=1e-4)
for s in range(20):
    o.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        L = m(obs, acts, succ, phase=phase)["loss_total"]
    L.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 100.0)
    o.step()
    assert torch.isfinite(L), f"loss NaN at step {s} under bf16"
print(f"  ✓ 20-step bf16 micro-train stable (final loss={float(L):.4f})")

print("\n=== CHECK 6: Eval pipeline runs end-to-end on a fresh ckpt ===")
with tempfile.NamedTemporaryFile(suffix=".pt") as f:
    p = Path(f.name)
    cfg = ModelConfig(image_size=64, memory_mode=MemoryMode.NONE)
    m = WorldModel(cfg).to(DEV); o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    save_checkpoint(p, m, o, step=0, cfg=cfg)
    from sralnik.training.eval_run import run_eval
    args = argparse.Namespace(
        checkpoint=p, data=DATA, device=DEV, split="val",
        batch=4, seq=16, num_workers=2, seed=0, max_rows=8,
        out_parquet=None, progress=False)
    run_eval(args)
print("  ✓ eval runs end-to-end")

print("\n=== ALL CHECKS PASSED ===")
```

Run it on the instance:
```bash
uv run python verify_correctness.py 2>&1 | tee runs/verify_correctness.log
aws s3 cp runs/verify_correctness.log s3://sralnik-runs-<initials>/runs/   # save proof
```

If any check fails, **do not start M0**. Debug on the instance via the workflow in §"Debugging on the instance" below, then re-run.

### 2.5 DDP smoke + steps/sec calibration (5–10 min)

```bash
# 200-step DDP smoke on full 8 GPUs to measure steps/sec
mkdir -p runs/calib
uv run torchrun --standalone --nproc_per_node=8 -m sralnik.training train \
    --data /mnt/data/sralnik/data/ithor_v2 \
    --batch 4 --seq 16 --max-steps 200 \
    --memory none --bf16 --num-workers 4 \
    --ckpt-dir runs/calib --ckpt-every 200 \
    2>&1 | tee runs/calib/stdout.log
```

When this finishes, parse `dt` and `global_step` from the trailing line. Compute `steps_per_sec`. Then **lock the per-condition `--max-steps` budget**:

```
max_steps = steps_per_sec × 4.5h × 3600s × 0.85 (slack)   # ~0.85 to leave room for ckpt I/O
```

If `steps_per_sec ≈ 1.5`, that's ~20,600 steps per condition — well below the 50,000 the README example uses. **Use this number, not 50,000.** Comparing M0–M3 at *matched* step counts is what the paper needs; chasing absolute SOTA isn't the goal.

**Expected first-50-steps numerical bounds for the M0 run** (use these as live sanity during launch):
- `loss_total`: starts ~0.5–0.8, drops monotonically (with noise) over the first 200 steps.
- `loss_rec` (L1 reconstruction): starts ~0.30–0.45 (random init vs targets in [0,1]).
- `loss_kl`: starts ~1.0 (free_bits=1.0 nat per dim, summed clamp), drops slowly.
- All losses **finite** (no NaN/Inf) every single step.
- **All 8 GPUs >50% utilization** in `nvidia-smi`. If only one GPU is busy, DDP didn't initialize.
- **Memory: 30–60 GB / 80 GB per GPU** at batch=4 image_size=256 bf16. If pushing 75+ GB, drop `--seq` to 12.
- Step time: roughly stable from step 50 onward (warmup is the first ~30).

If smoke shows `num_workers 8` is causing `iowait` to climb (use `iostat -xm 5` in another tmux pane), drop to `--num-workers 4`. With 8 procs × 8 workers you have 64 dataloader workers per node — overkill on this dataset.

### 2.6 Start the evacuation watcher (now, before any real training)

In a dedicated tmux pane, see §"Data Evacuation" below for the exact loop. Start it **before the first real run** so even calibration artifacts are saved.

---

## Phase 3 — Training Schedule (T+0:45 → T+18:45)

Sequential 8-GPU runs across four memory conditions. Diffusion stays **off** — it adds ~1.3–1.6× per-step cost (`find_unused_parameters=True` at `ddp_train.py:118` when diffusion is on) and isn't required for the headline ablation.

### 3.1 Per-condition launch template

For `<COND>` in `none`, `concat`, `attention`, `gated`:

```bash
RUN=runs/m_<COND>_$(date -u +%Y%m%dT%H%M%S)
mkdir -p $RUN
# Capture run metadata for reproducibility
{
  echo "git_sha=$(git rev-parse HEAD)"
  echo "memory=<COND>"
  echo "max_steps=<calibrated-from-2.4>"
  echo "started_utc=$(date -u --iso-8601=seconds)"
  uv pip freeze
} > $RUN/run_meta.txt

uv run torchrun --standalone --nproc_per_node=8 -m sralnik.training train \
    --data /mnt/data/sralnik/data/ithor_v2 \
    --batch 4 --seq 16 \
    --max-steps <calibrated> \
    --memory <COND> --bf16 --num-workers 4 \
    --ckpt-dir $RUN --ckpt-every 1000 \
    2>&1 | tee $RUN/stdout.log
```

Notes:
- `--ckpt-every 1000`, **not 500.** Each ckpt is the full optimizer state (≈hundreds of MB); at 500 you torture S3 and accumulate ~50 ckpts/condition. There is no rotation in `save_checkpoint` (`ddp_train.py:61`) — they all stay on disk. Prune all but `last.pt` and the final numbered ckpt before starting the next condition: `find $RUN -name 'step_*.pt' | head -n -1 | xargs rm -f`.
- `tee` to `stdout.log` so the S3 sync also captures training logs and the final `done <N> steps in <T>s` line you'll need for paper timing.
- Each condition writes to its own `$RUN` directory so there is zero cross-talk.

### 3.2 Schedule (sequential)

| Slot | Condition | Notes |
|---|---|---|
| T+0:45 → 5:15 | `--memory none` (M0) | Baseline. First real number for the paper. |
| T+5:15 → 9:45 | `--memory concat` (M1) | If you fall behind, this is the first to drop. |
| T+9:45 → 14:15 | `--memory attention` (M2) | |
| T+14:15 → 18:45 | `--memory gated` (M3) | The headline condition. |

If you finish early on any condition, **do not** retrain — pick that time up at evaluation/plotting.

### 3.3 Why sequential and not parallel 4-GPU jobs

Parallel 4-GPU jobs would compete on the same NVMe-backed HDF5 episodes and the same dataloader CPU pool, while halving per-job throughput on a near-linear-scaling RSSM. Net result is worse than sequential, with more failure modes (mid-run OOM on one job kills both).

### 3.4 If something goes wrong mid-condition

- **Loss → NaN**: kill, drop `--lr` to 5e-5 or `--grad-clip` to 10, restart. The current 100.0 grad clip + 1e-4 LR should be safe but bf16 + diffusion-off is mostly untested at scale.
- **GPU goes offline**: `nvidia-smi -L` to confirm; if real, `--resume runs/m_<cond>/step_NNNNNN.pt` from the latest good ckpt and restart with the survivors via `--nproc_per_node=<remaining>`. Note: `--resume` does **not** restore `DistributedSampler.set_epoch()`, so the recovered run re-walks crops; acceptable for a 24h budget.
- **Loss plateaus visibly early**: don't worry. With matched `--max-steps` across conditions the comparison is still apples-to-apples; that's what the paper wants.

---

## Phase 4 — Evaluation (T+18:45 → T+21:00)

Eval is fast (forward pass only) compared to training. Run all four conditions on **both `val` and `test`** splits — the README example only does `val`, but ARCHITECTURE.md §7 says `gap_length=1000` is held out in `test` for the long-gap extrapolation claim that the paper needs.

```bash
mkdir -p eval
for COND in none concat attention gated; do
  CKPT=$(ls -1 runs/m_${COND}_*/last.pt | tail -1)
  for SPLIT in val test; do
    uv run python -m sralnik.training eval \
      --checkpoint $CKPT --data /mnt/data/sralnik/data/ithor_v2 \
      --split $SPLIT --batch 8 \
      --out-parquet eval/${COND}_${SPLIT}_phase_c.parquet \
      2>&1 | tee eval/${COND}_${SPLIT}.log
  done
done
```

The eval run prints overall L1/MSE then breakdowns by `probe_name`, `gap_bucket`, `scene`, and probe×gap. The parquet files are small (KB–MB) — sync them to S3 the moment they're written (the watcher in §Evacuation already covers this).

---

## Phase 5 — Plotting + Final Evacuation (T+21:00 → T+23:30)

### 5.1 Plotting

Open VSCode Remote-SSH against the instance (`Remote-SSH: Connect to Host…` with `~/.ssh/sralnik-h100.pem`). Open a notebook in `repo/notebooks/` (create the dir):

- Load each `eval/<cond>_<split>_phase_c.parquet` with `pandas`.
- Pivot by `gap_bucket` × `probe_name` × `condition` → seaborn heatmaps for the paper.
- Optional: load each `runs/.../last.pt`, run a forward pass on 1–2 expert-eval episodes (`split == "expert_eval"` in manifest), save reconstruction GIFs to `eval/qual/`.

Save notebooks **in the repo**, commit + push from the instance — they sync to GitHub independently of S3.

### 5.2 Final hard sync (T+23:00, set a tmux timer)

```bash
# Bullet-proof final sync — no exclusions
aws s3 sync runs/ s3://sralnik-runs-<initials>/runs/  --size-only
aws s3 sync eval/ s3://sralnik-runs-<initials>/eval/ --size-only
# Plus the notebook outputs
aws s3 cp notebooks/ s3://sralnik-runs-<initials>/notebooks/ --recursive
# Plus the meta files in case any didn't get caught
aws s3 sync repo/runs/ s3://sralnik-runs-<initials>/runs/ --size-only
aws s3 ls --recursive --human-readable s3://sralnik-runs-<initials>/ > final_inventory.txt
aws s3 cp final_inventory.txt s3://sralnik-runs-<initials>/
```

### 5.3 Verify from a *second* machine (T+23:30)

On your laptop — **not** the dying instance:

```bash
aws s3 ls --recursive s3://sralnik-runs-<initials>/runs/ | grep last.pt
aws s3 ls --recursive s3://sralnik-runs-<initials>/eval/ | grep parquet
```

You should see four `last.pt` and at least 8 `*.parquet` (4 conditions × 2 splits). If anything is missing, you have ~30 minutes to re-sync from the instance before termination.

Pull the small artifacts to the laptop:
```bash
aws s3 sync s3://sralnik-runs-<initials>/eval/ ./eval-final/
aws s3 sync s3://sralnik-runs-<initials>/notebooks/ ./notebooks-final/
# Don't pull the .pt files now — they'll cost egress and you can pull later from S3
```

---

## Observability Strategy

**Stack: tmux (cockpit) + JSONL on disk + S3 sync (metrics travel with checkpoints) + VSCode Remote-SSH (interactive).**

### tmux cockpit — 4 panes (set up at T+15m, before training)

```
┌─────────────────────────┬──────────────────────┐
│ pane 0: training stdout │ pane 1: nvidia-smi   │
│  tee runs/.../stdout.log│  watch -n 5 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv │
├─────────────────────────┼──────────────────────┤
│ pane 2: iostat / top    │ pane 3: aws s3 sync  │
│  iostat -xm 5           │  loop (see §Evacuation) │
└─────────────────────────┴──────────────────────┘
```

`tmux ls` from a fresh SSH session reattaches if the connection drops — this is non-negotiable, never run training in a foreground terminal that dies with SSH.

### JSONL metrics (15 min to wire in, do during 2.4 calibration)

Add to `ddp_train.py` after the `it.set_postfix(...)` call (around line 180), gated on `is_main`:

```python
if is_main:
    import json, time
    with open(Path(args.ckpt_dir) / "metrics.jsonl", "a") as f:
        f.write(json.dumps({
            "step": global_step,
            "loss_total": float(losses["loss_total"].detach()),
            "loss_rec": float(losses.get("loss_rec", 0.0)),
            "loss_kl": float(losses.get("loss_kl", 0.0)),
            "loss_diff": float(losses.get("loss_diff", 0.0)),
            "t_wall": time.time(),
        }) + "\n")
```

JSONL is append-only and atomic per `write` for small lines on Linux — survives instance death and partial sync. Plot from the laptop with `pd.read_json(... lines=True)`.

### Why not wandb (default off)

Login flow + project setup + API key on the instance + network blips that hang `wandb.log()` add ~15 minutes of risk for plots you can produce in 2 minutes from JSONL + matplotlib. **Wire in wandb only if you have an account ready and a strong personal preference** — the patch is similar (4 lines: `wandb.init(...)`, `wandb.config.update(vars(args))`, `wandb.log(...)`, `wandb.finish()`). If you go this route, do it during phase 2.4 (calibration) so it's tested before the long M0 run.

### VSCode Remote-SSH for interactive work

For mid-run reconstructions / sanity plots:
```
~/.ssh/config
Host sralnik
  HostName <public-ip>
  User ubuntu
  IdentityFile ~/.ssh/sralnik-h100.pem
```
Then in VSCode: `Remote-SSH: Connect to Host… → sralnik`. You can edit code, open Jupyter notebooks (uv pulls ipykernel already), tail logs, and view checkpoint introspection. Notebooks should live in `repo/notebooks/` so they're git-versioned.

---

## Data Evacuation Strategy

**Three layers + verification on a second machine.** The instance terminates at T+24:00; assume you're not awake.

### Layer 1 — Continuous S3 sync (start at T+15m, in tmux pane 3)

```bash
# In repo/ on the instance, in a dedicated tmux pane:
mkdir -p runs eval
while true; do
  aws s3 sync runs/ s3://sralnik-runs-<initials>/runs/ \
      --size-only --exclude "*.tmp" --exclude "*.partial" \
      --only-show-errors
  aws s3 sync eval/ s3://sralnik-runs-<initials>/eval/ \
      --size-only --only-show-errors
  sleep 120
done
```

Why `--size-only` and `sleep 120` (not `sleep 60`):
- `torch.save` in `save_checkpoint` is non-atomic (`ddp_train.py:70` writes directly to `path`). At 60s polling, a 10–60s ckpt write has a real chance of being captured mid-flight. At 120s polling and `--size-only`, the file is either fully written (and stable size) or not yet started.
- `--size-only` avoids re-uploads on clock skew. Don't use `--exact-timestamps` here.

### Layer 2 — Per-condition explicit upload (start of next condition)

Between conditions, after the previous run wrote `last.pt`:

```bash
aws s3 cp $RUN/last.pt s3://sralnik-runs-<initials>/runs/$(basename $RUN)/last.pt
aws s3 cp $RUN/run_meta.txt s3://sralnik-runs-<initials>/runs/$(basename $RUN)/run_meta.txt
aws s3 cp $RUN/stdout.log s3://sralnik-runs-<initials>/runs/$(basename $RUN)/stdout.log
aws s3 cp $RUN/metrics.jsonl s3://sralnik-runs-<initials>/runs/$(basename $RUN)/metrics.jsonl
# prune old numbered ckpts on the instance to free NVMe for the next run
find $RUN -name 'step_*.pt' | head -n -1 | xargs -r rm -f
```

### Layer 3 — Hard-cutoff final sync (T+23:00)

A dedicated tmux pane runs `at now + 22h45m` from T+15m so this fires automatically even if you're disconnected:
```bash
echo 'cd /mnt/data/sralnik/repo && \
      aws s3 sync runs/ s3://sralnik-runs-<initials>/runs/ --size-only && \
      aws s3 sync eval/ s3://sralnik-runs-<initials>/eval/ --size-only && \
      aws s3 cp notebooks/ s3://sralnik-runs-<initials>/notebooks/ --recursive && \
      aws s3 ls --recursive --human-readable s3://sralnik-runs-<initials>/ \
        > /tmp/final_inventory.txt && \
      aws s3 cp /tmp/final_inventory.txt s3://sralnik-runs-<initials>/' \
  | at now + 22h45min
```

### Verification (T+23:30, from laptop)

```bash
aws s3 ls --recursive s3://sralnik-runs-<initials>/ | wc -l
aws s3 ls --recursive s3://sralnik-runs-<initials>/runs/ | grep -c last.pt    # expect 4
aws s3 ls --recursive s3://sralnik-runs-<initials>/eval/ | grep -c parquet    # expect ≥8
```

Bucket has versioning enabled (from §0.1), so a bad sync overwriting a good ckpt is recoverable via `aws s3api list-object-versions` if needed.

---

## Code Audit Findings (from reading every script we ship to the instance)

These are the issues I found by reading `sralnik/training/{train,ddp_train,dataset,eval_run}.py` and all of `sralnik/models/*.py`. None are blockers. They explain *why* the verification script in §2.4 checks what it does, and what to watch for during M0 startup.

**Correctness — no blocking bugs, but some quirks worth knowing:**

- `ddp_train.py:148` — `epochs = (max_steps + len(loader)) // len(loader)`. `len(loader)` is per-rank (post-`DistributedSampler`), so the inner loop relies on `if global_step >= max_steps: break` to exit cleanly. Correct, but means `epochs` is *upper-bound only*; the actual run terminates on step count, not epoch count.
- `ddp_train.py:101–112` — when `--resume`, the ckpt is `torch.load`-ed twice (once for cfg at line 101, once via `load_checkpoint` at line 112). Slow on big ckpts but correct.
- `ddp_train.py:181` — checkpoints are written as `step_NNNNNN.pt`, **never rotated**. After a 25k-step run with `--ckpt-every 1000` you'll have ~25 ckpts. Prune between conditions: `find $RUN -name 'step_*.pt' | head -n -1 | xargs -r rm -f`.
- `ddp_train.py:70` — `save_checkpoint` writes `torch.save` directly to the final path (non-atomic). The S3 sync uses `--size-only` + a 120s interval (§"Data Evacuation") to avoid uploading half-written files.
- `world_model.py:146` — `z_hist.append(z.detach())` is intentional: gradients don't flow through past memory cells. This means the encoder is *not* trained to write better keys — keys are whatever the encoder produces for the current frame. Acceptable per the architecture spec; the proposed paper claims don't depend on learned-write keys.
- `memory.py:34–36` — `cfg.memory_heads` is validated to divide `stoch_dim` but **the attention path uses single-head SDPA**, not multi-head. M2/M3 are effectively single-head despite the config knob. Doesn't affect correctness or the headline claim, but the architecture doc's mention of "MHA" is aspirational here.
- `memory.py:84–91` — `F.scaled_dot_product_attention` with `attn_mask=key_keep.unsqueeze(1)` (boolean). PyTorch ≥2.1 convention is **True = participate**; `key_keep` follows that. Verified by check 1 in `verify_correctness.py` (memory modes diverge from M0).
- `memory.py:67` — CONCAT mode's top-k score is `hist_z @ _q([h, z])`, dot-product (not scaled). With `dz=32` and reasonable z magnitudes, scores stay O(10); no overflow/underflow risk.
- `dataset.py:62` — `pd.read_parquet` runs once per worker init. With 8 procs × 4 workers = 32 dataloader workers each holding their own ~800-row DataFrame; trivial RAM.
- `dataset.py:102` — `np.asarray(f["rgb"][sl], dtype=np.float32) / 255.0` divides on CPU per worker. With many workers it's the dominant CPU-side cost. If `iostat`/`top` shows CPU saturation, dropping `--num-workers` from 8 to 4 helps more than raising it.
- `eval_run.py:65` — eval uses `torch.no_grad()` only, no autocast. L1/MSE are deterministic regardless of dtype, but for ~10–20% eval speedup you can wrap the forward pass in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Not worth patching for this run.
- `eval_run.py:40` — `torch.load(..., weights_only=False)`. Needed because the ckpt contains a `ModelConfig` object (`to_checkpoint_dict()` returns a dict, but pickle still applies). For our own ckpts this is fine.

**CLI flags that exist but you should *not* pass during the long runs:**
- `--max-rows N` — silently truncates the dataset to N episodes. Useful for smoke; lethal if accidentally passed to the real run.
- `--device cpu` — works but kills throughput. Distributed launch ignores this; single-launch (no torchrun) honors it.

**CLI knobs not exposed (would need a code edit if needed mid-run):** `deter_dim`, `stoch_dim`, `memory_topk`, `memory_heads`, `gate_hidden`, `free_bits`, `kl_balance`, `phase_weight`, `diffusion_loss_weight`. Defaults in `ModelConfig` are H100-tuned and match the architecture doc — leave them.

## Debugging on the instance — when something breaks

**The pattern is: VSCode Remote-SSH for editing, Claude Code on the instance for triage, and a separate `git worktree` for debugging so the running training is never disturbed.** This is faster and safer than debugging locally because most failure modes (NCCL hangs, bf16 overflow, OOM under real batch, kernel mismatches, dataloader stalls under real I/O) only manifest on the H100 hardware itself.

### Setup (do during T+15→T+30m, before training)

1. **Install Claude CLI on the instance** (one-time, ~30s):
    ```bash
    npm install -g @anthropic-ai/claude-code   # or curl install per https://claude.com/claude-code
    export ANTHROPIC_API_KEY=<your-key>        # add to ~/.bashrc
    ```
    DLAMI ships with Node; if missing: `sudo apt-get install -y nodejs npm`.
2. **VSCode: Remote-SSH → connect to `sralnik`** (config in §"Observability"). Open the folder `/mnt/data/sralnik/repo`.
3. **Pre-create a debug worktree** so any code edits during debugging happen in a separate dir from the running training:
    ```bash
    cd /mnt/data/sralnik/repo
    git worktree add /mnt/data/sralnik/debug debug
    ```
    Now `/mnt/data/sralnik/repo` (branch `main`) is the *frozen* tree the training runs from; `/mnt/data/sralnik/debug` (branch `debug`) is the *mutable* tree for triage.

### Workflow when something goes wrong

A 5-step runbook, **do not improvise** — under deadline pressure this is the cheapest path:

1. **Capture state first, fix later.** Don't kill the training (yet) and don't restart the instance:
    ```bash
    # in a fresh tmux pane:
    nvidia-smi > /tmp/nvidia.log
    ps aux | grep torchrun > /tmp/procs.log
    cp -r runs/<failing-cond>/ /tmp/snapshot/
    aws s3 sync /tmp/ s3://sralnik-runs-<initials>/debug/$(date -u +%Y%m%dT%H%M%S)/
    ```
2. **Reproduce small.** In the debug worktree, drive the smallest failing case:
    ```bash
    cd /mnt/data/sralnik/debug
    uv run python -m sralnik.training smoke-fit --data /mnt/data/sralnik/data/ithor_v2 \
        --steps 5 --memory <failing-mode> --max-rows 8 \
        2>&1 | tee /tmp/smoke_repro.log
    ```
    If it reproduces: now it's a 5-step iteration loop, not a 4-hour-per-iteration loop.
3. **Triage with Claude on the instance.** From `/mnt/data/sralnik/debug`:
    ```bash
    claude   # interactive; has Read/Edit/Bash on the project tree
    ```
    Hand it the stack trace + the smoke log. Claude can read the relevant file, propose a patch, and run the smoke command to verify.
4. **Decide: hot-fix, or kill+restart?**
    - If the bug is in code only the failing condition triggers (e.g., M3 gate explodes but M0–M2 are fine): let M0/M1/M2 finish, fix M3 in the debug worktree, push to main, then run M3.
    - If it's a hard hang or NaN that affects all conditions: kill the running torchrun, fix, push, restart with `--resume <latest-good-ckpt>`. **Never restart from step 0 if you have a checkpoint** — the ckpt has hours of compute baked in.
5. **Always commit + push the fix before resuming.** The instance dies at T+24h. An uncommitted hot-fix is a fix that doesn't exist tomorrow.
    ```bash
    cd /mnt/data/sralnik/debug
    git add -A && git commit -m "fix: <one-liner>"
    git push origin debug
    cd /mnt/data/sralnik/repo
    git fetch && git merge origin/debug   # bring fix into the main worktree
    git push
    ```

### Safety rules — non-negotiable under deadline

- **The training tmux session is sacred.** Never `Ctrl+C` it to "just check something" — your "just" might cost an hour of compute.
- **Edit only in the debug worktree.** Editing files in `/mnt/data/sralnik/repo` while training reads them is undefined behavior (and on Linux Python keeps file handles to imported modules — usually OK for already-imported code, but absolutely not for Python files re-imported at checkpoint load).
- **Never `rm -rf` `runs/`.** If you must reclaim NVMe space, prune `step_*.pt` between conditions; never touch `last.pt` or `metrics.jsonl`.
- **Claude on the instance has full Bash access.** It will run `aws s3 cp`, `nvidia-smi`, `pkill`, etc. when asked. Read its proposed commands before you let it execute destructive ones (rm, kill, force-push, drop).
- **If you absolutely must reboot:** save everything to S3 first (`aws s3 sync runs/ s3://...`), then `sudo reboot`. Capacity-Block instances retain their EBS root after reboot but **lose all NVMe contents** (the `data/` and `runs/` you put on `/mnt/data/`). Don't reboot unless desperate.

### When to *not* try to fix on the instance

- **Bug in `dataset.py` that requires re-collecting data:** that takes hours; cut your losses, drop the affected probe/scene from eval.
- **Hardware fault** (a GPU disappears from `nvidia-smi`): not your problem to fix; resume from latest ckpt with `--nproc_per_node=<remaining>` and accept reduced parallelism.
- **AWS API failure** (S3 throttling, instance metadata service stalls): retry with exponential backoff; don't restructure the run.

## Risk Register

| Risk | Mitigation |
|---|---|
| 41 GB upload too slow | Tar to single-file multipart upload, started at T−50m. Skip `ithor_v1` and `sralnik.zip`. Fall back to direct rsync after launch only if S3 path stalls. |
| Non-atomic ckpt writes captured mid-write | 120s sync interval + `--size-only` + `--exclude "*.tmp"`. S3 versioning protects against bad overwrite. |
| SSH dies mid-training | All training in tmux. Reattach with `tmux attach -t sralnik`. |
| Instance hard-terminates at T+24h | Layered sync (continuous + per-condition + `at`-scheduled final). Verify from second machine at T+23:30. |
| Loss → NaN | Lower `--lr` to 5e-5 or `--grad-clip` to 10; resume from latest good ckpt. |
| One GPU dies | Resume with `--nproc_per_node=<remaining>` from latest ckpt. Sampler offset isn't restored — accept non-bit-exact resume. |
| Numbered ckpts fill NVMe | Prune `step_*.pt` between conditions, keep only `last.pt` and final-step. |
| Calibration shows steps/sec slower than budget | Drop M1 (paper still works with M0/M2/M3). Don't drop M3. |
| Eval forgotten / runs out of time | Schedule reserves 2h15m for eval + 2h30m for plotting/evac. Eval is fast — even 1h is enough for 4 ckpts × 2 splits if needed. |
| Capacity Block wasted on debugging at start | Local smoke (§0.6) catches code bugs cheaply. DLAMI handles CUDA/drivers. |
| Both collaborators try to drive at once | Decide *now* who runs the cluster and who watches metrics. One SSH session, one cockpit. |
| Bucket region != reservation region | Egress costs + slow transfers. §0.1 explicitly puts the bucket in the reservation's region. |
| `find_unused_parameters=True` slowness | Diffusion stays **off** for the ablation. Only enable if you have surplus time at T+18:45 and want a stretch M3+diff run. |
| `manifest.parquet` re-read by 64 dataloader workers | If smoke shows memory pressure, drop to `--num-workers 4`. |
| Code bug surfaces during a long run (NaN, hang, OOM under real batch) | `verify_correctness.py` (§2.4) catches the common ones at T+30m. For runtime issues: capture state to S3 first, reproduce via `smoke-fit` in debug worktree, fix-and-push, resume from latest ckpt. See §"Debugging on the instance". |
| Hot-fix applied on instance, never committed → lost when instance terminates | Always `git commit && git push` *before* resuming training. The 5-step runbook in §"Debugging" enforces this order. |
| Editing repo files while training reads them | All debug edits go to `/mnt/data/sralnik/debug` (separate `git worktree`), never the running tree at `/mnt/data/sralnik/repo`. |
| Single-head SDPA in `MemoryFusion` (despite `memory_heads=4` config) | Documented; doesn't change paper claims. M2/M3 are single-head attention. |
| Cache-effect from GPU warmup contaminates first eval | All eval is teacher-forced and the model is in `eval()` mode; no statefulness across batches. Safe. |

---

## Critical Files & Paths

In-repo (don't redesign, just use):
- `sralnik/training/ddp_train.py:61` — `save_checkpoint` (non-atomic; mitigated via sync `--size-only`)
- `sralnik/training/ddp_train.py:118` — `find_unused_parameters` toggles with `--diffusion`
- `sralnik/training/ddp_train.py:148` — `epochs` derived from `--max-steps`; `--max-steps` is the real budget
- `sralnik/training/ddp_train.py:181` — `--ckpt-every` writes `step_NNNNNN.pt` (no rotation; prune manually)
- `sralnik/training/train.py` — CLI dispatch for `train|eval|smoke-fit|smoke-synthetic`
- `sralnik/training/eval_run.py` — Phase-C L1/MSE eval; pass `--split test` for long-gap extrapolation
- `sralnik/models/config.py` — `ModelConfig` dataclass; default knobs are already H100-tuned
- `docs/ARCHITECTURE.md` §7 (eval) and §9 (wall-time budget)
- `README.md` — copy-pastable launch command (line 47)

Cluster paths (target layout):
- `/mnt/data/sralnik/data/ithor_v2/` — dataset on local NVMe
- `/mnt/data/sralnik/repo/` — git clone
- `/mnt/data/sralnik/repo/runs/m_<cond>_<ts>/` — one dir per condition
- `s3://sralnik-runs-<initials>/{data,runs,eval,notebooks}/` — destination

---

## Verification (end-to-end success criteria)

Success means **all** of these are true at T+24h on the laptop, with the instance gone:

1. `aws s3 ls --recursive s3://sralnik-runs-<initials>/runs/ | grep last.pt` returns ≥3 (4 ideally) `last.pt` files.
2. `aws s3 ls --recursive s3://sralnik-runs-<initials>/eval/ | grep parquet` returns ≥6 (8 ideally) Phase-C parquet files.
3. Each `runs/m_<cond>_*/run_meta.txt` is in S3 — proves reproducibility (git SHA, max-steps, env).
4. Each `runs/m_<cond>_*/metrics.jsonl` is in S3 — enables loss curves for paper figures.
5. Loading a `last.pt` locally with `torch.load` succeeds and yields a state dict matching `ModelConfig`.
6. A loaded ckpt + a small batch through the eval CLI on the laptop reproduces (within numerical noise) one row of the corresponding eval parquet.

---

## Decisions locked in

- **AWS state:** reservation ID + region are ready; user has them at hand.
- **Ablation scope:** all four memory conditions M0/M1/M2/M3 (full ablation table).
- **Observability:** JSONL on disk + S3 sync, plotted locally with pandas+matplotlib. No wandb (avoids login/key/network-stall risk).
- **Data:** ship both `ithor_v2` and `ithor_v2_fp1` as separate tars; merge on instance after `uv sync`. Skip local merge (foreign `.venv/` from archive cannot be trusted).
- **Local Python:** not used for pre-flight; instance-side `verify_correctness.py` replaces local smoke.
- **Hyperparameters:** only `--memory` varies across the four runs; all other knobs locked at `ModelConfig` defaults (image_size=256, batch=4, seq=16, bf16, lr=1e-4). `--max-steps` calibrated from a 200-step measurement at T+40m.
