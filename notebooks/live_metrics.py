# %% [markdown]
# # Live training-loss viewer
#
# Run cell-by-cell in VSCode (Remote-SSH'd into `sralnik`) with the Python+Jupyter
# extensions installed. Each `# %%` is a Jupyter cell (Shift+Enter to run, like a normal notebook).
#
# Designed to be re-run while training is in progress. The DDP loop appends one
# JSON line per step to `runs/<COND>/metrics.jsonl`; `pd.read_json(... lines=True)`
# re-reads it from disk, so subsequent re-runs of the plotting cell pick up new data.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path("/mnt/data/sralnik/repo")

# %% Find all run dirs
runs = sorted(REPO.glob("runs/m_*"))
for r in runs:
    j = r / "metrics.jsonl"
    n = sum(1 for _ in j.open()) if j.exists() else 0
    print(f"{r.name:50s} steps logged: {n}")

# %% Load + plot the most recent run's metrics
run_dir = max(runs, key=lambda p: p.stat().st_mtime)
print(f"plotting: {run_dir}")
df = pd.read_json(run_dir / "metrics.jsonl", lines=True)
print(f"  {len(df)} rows; latest step: {df['step'].iloc[-1]}")

fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
df.set_index("step")[["loss_total", "loss_rec", "loss_kl"]].plot(ax=axes[0, 0], title="losses")
df.set_index("step")[["loss_total"]].rolling(50).mean().plot(ax=axes[0, 1], title="loss_total (50-step moving avg)")
df.set_index("step")[["loss_rec"]].plot(ax=axes[1, 0], title="loss_rec (L1 reconstruction)")
df.set_index("step")[["loss_kl"]].plot(ax=axes[1, 1], title="loss_kl (balanced KL + free-bits)")
for ax in axes.flat:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% Throughput check: steps/sec from t_wall
df["dt"] = df["t_wall"].diff()
sps = (1 / df["dt"]).rolling(50).mean()
sps.index = df["step"]
sps.plot(figsize=(11, 3), title="steps/sec (50-step moving avg)")
plt.axhline(7.0, color="g", ls="--", label="calibration baseline (7 steps/sec)")
plt.grid(alpha=0.3); plt.legend(); plt.show()
print(f"median steps/sec (last 200 steps): {sps.iloc[-200:].median():.2f}")

# %% Compare across conditions (run after multiple conditions have finished)
fig, ax = plt.subplots(figsize=(11, 4))
for r in runs:
    j = r / "metrics.jsonl"
    if not j.exists():
        continue
    d = pd.read_json(j, lines=True)
    if len(d) == 0:
        continue
    d.set_index("step")["loss_total"].rolling(100).mean().plot(ax=ax, label=r.name)
ax.set_title("loss_total (100-step MA) across conditions"); ax.grid(alpha=0.3); ax.legend()
plt.show()

# %% Quick visual check on a checkpoint (latest M0 step_*.pt)
# Heavy: loads a ckpt + runs forward on a small batch. Skip if training is still hot.
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(REPO))
from sralnik.models import ModelConfig, WorldModel
from sralnik.training.dataset import EpisodeChunkDataset, collate_fn

ckpts = sorted(run_dir.glob("step_*.pt"))
if ckpts:
    ck_path = ckpts[-1]
    print(f"loading {ck_path.name}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_checkpoint_dict(ck["model_cfg"])
    model = WorldModel(cfg).cuda().eval()
    model.load_state_dict(ck["model_state"])

    ds = EpisodeChunkDataset(
        "/mnt/data/sralnik/data/ithor_v2",
        seq_len=16, split="val", exclude_manual=True, max_rows=4, return_meta=True,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    obs = batch["obs"].cuda()
    with torch.no_grad():
        out = model(obs, batch["actions"].cuda(), batch["action_success"].cuda(),
                    phase=batch["phase"].cuda(), posterior_sample=False, return_reconstructions=True)
    x_hat = out["x_hat"]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for col, t in enumerate([0, 5, 10, 15]):
        axes[0, col].imshow(obs[0, t].cpu().permute(1, 2, 0).numpy()); axes[0, col].set_title(f"truth t={t}")
        axes[1, col].imshow(x_hat[0, t].cpu().permute(1, 2, 0).numpy().clip(0, 1)); axes[1, col].set_title(f"recon t={t}")
        for ax in axes[:, col]:
            ax.axis("off")
    plt.tight_layout(); plt.show()
else:
    print(f"no step_*.pt yet in {run_dir} — let M0 run a bit longer")
