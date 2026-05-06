# SRALNIK

**S**cene **R**econstruction with **A**ugmented **L**ong-horizon **N**eural **I**ndexed **K**nowledge

Course project for MIT 6.S058 (Computer Vision).

A retrieval-augmented latent world model for long-horizon scene consistency in indoor
embodied environments. We study whether an external memory bank of past latent states
can help a compact world model preserve room identity across revisits in AI2-THOR /
ProcTHOR scenes.

## Authors

- Maryna Bohdan (`mbohdan@mit.edu`)
- Aleksandr Trofimov (`atrof002@mit.edu`)

## Repository layout

- `sralnik/` — Python package.
  - `sralnik/data/` — AI2-THOR data collection (controller wrapper, exploration policy,
    five memory probes, HDF5 writer, parquet manifest, CLI).
- `write-up/` — LaTeX source (`main.tex`, `cvpr.sty`) and compiled PDF for the project
  introduction.
- `docs/ARCHITECTURE.md` — **concrete model spec**: shared encoder + RSSM core,
  **memory conditions M0–M3** (none / concat / cross-attn / **memory-gated**),
  **latent diffusion decoder**, losses, probe-aligned eval; CPU smoke vs 8× H100.
- **`sralnik.models` / `sralnik.training`** — PyTorch **RSSM** (`WorldModel`): encoder,
  GRU dynamics, **memory modes** ``none|concat|attention|gated``, CNN decoder, optional
  **latent diffusion** head; HDF5 **chunk dataset** + CPU smoke CLI.

See `write-up/main.pdf` for the current introduction draft.

### World model sanity (CPU)

```bash
uv sync
# No dataset: random tensors, one optimizer step
uv run python -m sralnik.training smoke-synthetic --image-size 64
# Real HDF5 under manifest.parquet (tiny subset)
uv run python -m sralnik.training smoke-fit --data data/ithor_v2 --steps 2 --max-rows 8
```

Use `--memory gated` / `--diffusion` to exercise M3 and the diffusion loss.

**8× H100 (single node):** NCCL + `torchrun` scales batch across GPUs; bf16 autocast and TF32 are enabled in the training entry point.

```bash
uv run torchrun --standalone --nproc_per_node=8 -m sralnik.training train \
  --data data/ithor_v2 \
  --batch 4 --seq 16 --max-steps 50000 \
  --memory gated --bf16 --num-workers 8 \
  --ckpt-dir runs/wm_m3 --ckpt-every 1000
```

**Evaluation** (Phase **C** reconstruction, teacher-forced, posterior mean \(z=\mu\)):

```bash
uv run python -m sralnik.training eval \
  --checkpoint runs/wm_m3/last.pt --data data/ithor_v2 --split val \
  --batch 8 --out-parquet eval/val_phase_c.parquet
```

Printed tables: overall L1/MSE, then groupby `probe_name`, `gap_bucket` (20/100/300/1000 vs other/na), `scene`, and a probe×gap slice. See `docs/ARCHITECTURE.md` §7 for the full eval plan (LPIPS, object-state checks as optional extensions).

## Data collection

We collect AI2-THOR iTHOR trajectories designed specifically to test
**memory augmentation**: each probe episode has a *Phase A* where the agent sets some
state, a *Phase B* (distractor walking) of variable length, and a *Phase C* where the
agent returns from a different angle and we test whether the model remembers what
happened in Phase A.

### Probes

| # | Probe | Phase A | Phase C | What it tests |
|---|---|---|---|---|
| 1 | `fridge` | Open receptacle, place item inside, close it | Re-open from new bearing | Receptacle-content memory |
| 2 | `toggle` | Toggle a lamp / appliance on | Look at it again from a new angle | Toggle-state persistence |
| 3 | `displacement` | Pick item from one surface, drop on another | Look at the destination surface | Object-location memory |
| 4 | `layout` | 360° pan from anchor pose | Re-observe from opposite anchor | Pure geometric memory (no state change) |
| 5 | `rearrangement` | Move 2-3 items onto one surface | Look at that surface again | Multi-object arrangement memory |

`gap_length` (Phase B steps) is the key independent variable. Default sweep is
`{20, 100, 300, 1000}`. Episodes with `gap_length=1000` are held out for the test
split so we can measure long-gap extrapolation.

### Scenes

Default pilot uses two iTHOR FloorPlans: `FloorPlan1` (kitchen) and `FloorPlan403`
(bathroom). Per-scene probe targets and tracked-object whitelists live in
`sralnik/data/scenes.py`.

| Scene | Receptacle probe | Toggle probe | Pickup / displacement |
|-------|-------------------|--------------|------------------------|
| `FloorPlan1` (kitchen) | `Fridge` ← `Apple` | `Microwave` | `Mug` → `CounterTop` |
| `FloorPlan403` (bathroom) | `Cabinet` ← `SoapBar` | `Faucet` | `SoapBar` → `CounterTop` |

### Output format

One HDF5 file per episode, plus a `manifest.parquet` at the dataset root:

```
data/ithor_v1/
  manifest.parquet
  episodes/
    FloorPlan1/
      exploration/FloorPlan1_explore_0000.h5
      probe_fridge/gap20/FloorPlan1_fridge_g20_0000.h5
      probe_fridge/gap100/...
      ...
```

Each HDF5 file contains:

- `rgb` — `(T, H, W, 3)` `uint8` (gzip)
- `instance_seg` — `(T, H, W)` `uint16` (gzip), if segmentation enabled
- `depth` — `(T, H, W)` `float16` (gzip), if depth enabled
- `pose` — `(T, 5)` `float32` (`x, y, z, yaw, pitch`)
- `action_id` — `(T,)` `int16`
- `action_success` — `(T,)` `bool`
- `action_object_id` — `(T,)` fixed-length bytes
- `phase` — `(T,)` `int8` with codes `{A: 0, B: 1, C: 2}`
- `tracked_objects_json` — `(T,)` JSON-encoded snapshot of whitelisted objects
- File-level attrs: `episode_id`, `scene`, `episode_type`, `probe_name`,
  `gap_length`, `seed`, `split`, `target_object_id`, `target_receptacle_id`,
  `action_names` (canonical action vocabulary).

The manifest has one row per episode with the same metadata, suitable for fast
filtering/sharding by `pandas` / `pyarrow`.

### Running collection

Install (uses [`uv`](https://github.com/astral-sh/uv)):

```bash
uv sync
```

Smoke test (single scene, 10 walker steps + one short fridge probe, nothing saved):

```bash
uv run python -m sralnik.data smoke
```

The first run downloads the AI2-THOR Unity build (~hundreds of MB).

Pilot collection (default: 2 scenes × 400 episodes, 256×256 RGB + segmentation,
~10–12 GB on disk without depth, ~17–20 GB with depth, ~55–70 min wall time
on Apple Silicon):

```bash
uv run python -m sralnik.data collect --output data/ithor_v2 --with-depth
```

Object layout **is deterministic per scene by default** (same pickupable
positions every episode). This is the stable setting for long runs on macOS.
To shuffle object placements between episodes, pass ``--randomize-layout`` —
experimental; can destabilise Unity.

#### If one scene needs a re-collect (merge workflow)

If a multi-scene run finishes with one scene corrupted (Unity crash) but the
other is healthy, re-collect **only the bad scene** into a fresh directory,
then merge it into the main dataset:

```bash
uv run python -m sralnik.data collect \
    --output data/ithor_v2_fp1 \
    --scenes FloorPlan1 \
    --episodes 400 \
    --with-depth

uv run python -m sralnik.data merge \
    --base data/ithor_v2 \
    --add data/ithor_v2_fp1 \
    --scene FloorPlan1
```

The ``merge`` command copies successful ``.h5`` files from ``add`` into
``base``, replaces any stale manifest rows with the same ``episode_id``,
and rewrites ``base/manifest.parquet``.

Smaller subset for quick iteration:

```bash
uv run python -m sralnik.data collect \
    --output data/quick \
    --scenes FloorPlan1 \
    --episodes 20 \
    --gap-lengths 20 100
```

Common flags:

- `--width / --height` — frame resolution (default 256).
- `--episodes` — episodes per scene (default 400).
- `--with-depth` / `--no-segmentation` — toggle modalities.
- `--randomize-layout` — enable per-episode `InitialRandomSpawn` (experimental).
- `--platform CloudRendering` — for headless Linux servers.
- `--exploration-fraction 0.5` — split between exploration and probe episodes.
- `--gap-lengths 20 100 300 1000` and `--held-out-gap 1000` — memory-probe schedule.

### Manual / expert recording

For pre-flight scene checks and a small "expert eval" set with cleaner Phase C
framing, drive the agent yourself with the keyboard:

```bash
uv run python -m sralnik.data record \
    --output data/ithor_v1 \
    --scene FloorPlan1 \
    --note "fridge-noput-baseline"
```

This opens a pygame window showing the live AI2-THOR frame plus a side panel
with visible interactables (numbered hotkeys), the held object, and current
phase. Episodes are written to `episodes/<scene>/manual/` and appended to the
same `manifest.parquet` with `episode_type="manual"` and `split="expert_eval"`
by default.

| Key       | Action |
|-----------|--------|
| W / S     | MoveAhead / MoveBack |
| A / D     | RotateLeft / RotateRight |
| Q / E     | Strafe Left / Right |
| R / F     | LookUp / LookDown |
| Space     | Hold pose (no THOR step, duplicate frame) |
| 1..9 / 0  | Select / clear visible interactable shown in side panel |
| O / C     | OpenObject / CloseObject (on selection) |
| P         | PickupObject (on selection) |
| V         | PutObject (selection = receptacle, held object goes in) |
| T / Y     | ToggleObjectOn / ToggleObjectOff (on selection) |
| TAB       | Cycle phase A → B → C |
| Z         | Undo last logged step (does not roll back THOR state) |
| Enter     | Save current episode, start a new one in the same scene |
| Backspace | Discard current episode, start a new one |
| Esc       | Save (if non-empty) and quit |

Run one session per scene. ~10 episodes per scene (≈30 min) is plenty for the
expert eval set described above.

### Memory-augmentation evaluation (downstream)

The dataset is designed so the same model checkpoint can be evaluated on:

1. **Train-distribution gaps** (gap ∈ `{20, 100, 300}`, `val` split): does memory
   help even at moderate horizons?
2. **Held-out long gaps** (gap = `1000`, `test` split): does memory generalise to
   horizons unseen during training?
3. **Per-probe accuracy**, computed at Phase C frames using `tracked_objects_json`
   ground truth and `instance_seg` masks.
