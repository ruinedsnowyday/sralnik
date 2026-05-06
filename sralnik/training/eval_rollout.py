"""Open-loop Phase-C rollout eval with paper-grade GIF output.

Given a trained checkpoint, for each episode in the chosen split:
  1. Run the encoder normally as warmup over the first ``T - K`` frames
     (so the recurrent state h has digested Phase A + most of Phase B/C).
  2. At step ``T - K``, stop feeding the encoder. From then on, the model
     ``imagines`` the next ``K`` frames using only:
         - the dynamics prior mean ``z = prior_mu`` (deterministic),
         - the actual action sequence (teacher-forced actions, that's a
           given since the agent already moved),
         - the memory bank built up during warmup (modes M1/M2/M3 only).
  3. Decode each imagined ``(h, z)`` to RGB and save as a side-by-side GIF
     against the ground-truth Phase-C frames.

This is the test the paper actually wants: ``does the model render Phase C
faithfully when it can no longer see the frame?`` The teacher-forced
``eval_run.py`` doesn't answer that question because the encoder peeks at
every step.

Default: ``--split expert_eval`` (the manually-recorded episodes), ``K=8``.

Outputs (under ``--out-dir``):
  - ``rollout_metrics.parquet``  per-frame L1/MSE of the imagined window,
                                 stratified by ``probe_name`` / ``gap_bucket`` / ``scene``.
  - ``gifs/<episode_id>.gif``    side-by-side GT vs prediction, K frames.
  - ``rollout_summary.txt``      printed tables (overall, by probe, by gap_bucket).
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from sralnik.models import MemoryMode, ModelConfig, WorldModel

_GAP_EXACT = frozenset({20, 100, 300, 1000})


def _gap_bucket(gl: int) -> str:
    if gl < 0:
        return "na"
    if gl in _GAP_EXACT:
        return f"g{gl}"
    return "other"


def _load_full_episode(root: Path, row: pd.Series) -> dict[str, Any]:
    """Read every frame of one episode (no chunking, no random crop)."""
    path = root / str(row["relative_path"])
    with h5py.File(path, "r") as f:
        rgb = np.asarray(f["rgb"][:], dtype=np.float32) / 255.0  # (T,H,W,3)
        actions = np.asarray(f["action_id"][:], dtype=np.int64)
        succ = np.asarray(f["action_success"][:], dtype=np.bool_)
        phase = np.asarray(f["phase"][:], dtype=np.int64)
    rgb_t = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()  # (T,3,H,W)
    gap = row.get("gap_length")
    if gap is None or (isinstance(gap, float) and np.isnan(gap)):
        gap = -1
    return {
        "obs": rgb_t,
        "actions": torch.from_numpy(actions).long(),
        "action_success": torch.from_numpy(succ),
        "phase": torch.from_numpy(phase).long(),
        "episode_id": str(row["episode_id"]),
        "scene": str(row.get("scene", "") or ""),
        "probe_name": str(row.get("probe_name") or row.get("episode_type") or "unknown"),
        "gap_length": int(gap),
        "episode_type": str(row.get("episode_type", "") or ""),
    }


def _write_mask_from_phase(phase: torch.Tensor) -> torch.Tensor:
    """Mirrors WorldModel._write_mask_from_phase. (B,T) -> (B,T) bool."""
    B, T = phase.shape
    m = torch.zeros(B, T, dtype=torch.bool, device=phase.device)
    m[:, 0] = True
    if T > 1:
        m[:, :-1] |= (phase[:, :-1] == 0) & (phase[:, 1:] == 1)
    return m


@torch.no_grad()
def _rollout_one(
    model: WorldModel,
    cfg: ModelConfig,
    obs: torch.Tensor,
    actions: torch.Tensor,
    succ: torch.Tensor,
    phase: torch.Tensor,
    k_imagine: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Manually replicates WorldModel.forward but with imagination after t_imag.

    obs/actions/succ/phase: (1, T, ...) tensors already on ``device``.
    Returns (1, T, 3, H, W) decoded frames + the imagine boundary index.
    """
    _, T, _, H, W = obs.shape
    actions = actions.clamp(0, cfg.action_vocab_size - 1)
    write_mask = _write_mask_from_phase(phase)
    t_imag = max(1, T - int(k_imagine))  # always at least 1 warmup step

    h = model.h0.expand(1, -1).contiguous()
    z_hist: list[torch.Tensor] = []
    rec: list[torch.Tensor] = []

    for t in range(T):
        prior_raw = model.prior_net(h)
        prior_mu, prior_logstd = torch.chunk(prior_raw, 2, dim=-1)
        prior_std = F.softplus(prior_logstd) + 1e-4  # noqa: F841 (kept for parity / future use)

        if t < t_imag:
            _, post_mu, _ = model.encoder(obs[:, t])
            z = post_mu
        else:
            z = prior_mu

        if z_hist and cfg.memory_mode is not MemoryMode.NONE:
            hist = torch.stack(z_hist, dim=1)
            mem_mask = write_mask[:, : hist.shape[1]].bool()
            h_mem = model.memory(h, z, hist, mem_mask)
        else:
            h_mem = h

        act_e = model.act_emb(actions[:, t] + 1)
        s = succ[:, t].float().unsqueeze(-1)
        gru_in = torch.cat([z, act_e, s], dim=-1)
        h = model.gru(gru_in, h_mem)
        z_hist.append(z.detach())
        rec.append(model.decoder(h, z))

    return {
        "x_hat": torch.stack(rec, dim=1),  # (1, T, 3, H, W)
        "t_imag": t_imag,
    }


def _to_uint8(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) float[0,1] -> (H,W,3) uint8."""
    x = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255.0).round().astype(np.uint8)


def _save_comparison_gif(
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    out_path: Path,
    duration_ms: int = 250,
    label_top: str | None = None,
) -> None:
    """Stack ground-truth | prediction side-by-side, write an animated GIF.

    Each frame is a (H, 2*W + gap, 3) uint8 panel: GT on the left, prediction
    on the right, separated by a 4-pixel light-grey gap.
    """
    assert len(gt_frames) == len(pred_frames), "frame count mismatch"
    H, W, _ = gt_frames[0].shape
    gap = 4
    sep = np.full((H, gap, 3), 200, dtype=np.uint8)

    composite = []
    for gt, pred in zip(gt_frames, pred_frames):
        row = np.concatenate([gt, sep, pred], axis=1)
        composite.append(Image.fromarray(row))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite[0].save(
        out_path,
        save_all=True,
        append_images=composite[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def run_eval_rollout(args: Namespace) -> None:
    device = torch.device(args.device)
    ck_path = Path(args.checkpoint)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_checkpoint_dict(ck["model_cfg"])
    model = WorldModel(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    data_root = Path(args.data)
    out_dir = Path(args.out_dir)
    gif_dir = out_dir / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_root / "manifest.parquet")
    df = df[df["split"] == args.split]
    df = df[df["relative_path"].fillna("").str.endswith(".h5")]
    if args.max_rows is not None:
        df = df.head(args.max_rows)
    if len(df) == 0:
        print(f"no episodes for split={args.split!r}; nothing to do.")
        return

    rows: list[dict] = []
    K = int(args.k_imagine)
    print(f"\n=== open-loop rollout eval ===")
    print(f"  ckpt:   {ck_path}")
    print(f"  cfg:    image_size={cfg.image_size}, memory_mode={cfg.memory_mode.value}, "
          f"diffusion={cfg.use_latent_diffusion}")
    print(f"  split:  {args.split} ({len(df)} episodes)")
    print(f"  K:      {K} (last frames imagined open-loop)")
    print(f"  out:    {out_dir}")
    print()

    for _, row in df.iterrows():
        ep = _load_full_episode(data_root, row)
        T = ep["obs"].shape[0]
        if T < K + 2:
            print(f"  skip {ep['episode_id']}: too short ({T} < K+2={K+2})")
            continue

        obs = ep["obs"].unsqueeze(0).to(device)
        actions = ep["actions"].unsqueeze(0).to(device)
        succ = ep["action_success"].unsqueeze(0).to(device)
        phase = ep["phase"].unsqueeze(0).to(device)

        out = _rollout_one(model, cfg, obs, actions, succ, phase,
                           k_imagine=K, device=device)
        x_hat = out["x_hat"]              # (1, T, 3, H, W)
        t_imag = out["t_imag"]

        # Per-frame metrics, only over the imagined window
        for t in range(t_imag, T):
            l1 = F.l1_loss(x_hat[0, t], obs[0, t], reduction="mean").item()
            mse = F.mse_loss(x_hat[0, t], obs[0, t], reduction="mean").item()
            phase_t = int(phase[0, t].item())
            rows.append({
                "episode_id": ep["episode_id"],
                "scene": ep["scene"],
                "probe_name": ep["probe_name"],
                "episode_type": ep["episode_type"],
                "gap_length": ep["gap_length"],
                "gap_bucket": _gap_bucket(ep["gap_length"]),
                "frame_index": t,
                "phase": phase_t,
                "l1": l1,
                "mse": mse,
                "is_phase_c": phase_t == 2,
            })

        # GIF: imagined window only, GT | pred side by side
        gt_frames = [_to_uint8(obs[0, t]) for t in range(t_imag, T)]
        pred_frames = [_to_uint8(x_hat[0, t]) for t in range(t_imag, T)]
        gif_path = gif_dir / f"{ep['episode_id']}.gif"
        _save_comparison_gif(gt_frames, pred_frames, gif_path,
                             duration_ms=int(args.gif_frame_ms))

        last_l1 = rows[-1]["l1"]
        print(f"  {ep['episode_id']:42s} T={T:3d} t_imag={t_imag:3d} "
              f"final-frame L1={last_l1:.4f}  -> {gif_path.name}")

    if not rows:
        print("no usable episodes; check phase/length filters.")
        return

    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "rollout_metrics.parquet"
    metrics_df.to_parquet(metrics_path, index=False)

    # Print summary tables (also written to rollout_summary.txt)
    summary_lines: list[str] = []

    def _emit(s: str) -> None:
        summary_lines.append(s)
        print(s)

    _emit(f"\nwrote {len(metrics_df)} per-frame rows to {metrics_path}")
    overall_pc = metrics_df[metrics_df["is_phase_c"]]
    _emit("\n=== overall (imagined window) ===")
    _emit(metrics_df.agg({"l1": "mean", "mse": "mean"}).to_string())
    if len(overall_pc) > 0:
        _emit("\n=== Phase-C frames within imagined window ===")
        _emit(overall_pc.agg({"l1": "mean", "mse": "mean"}).to_string())

    for key in ["probe_name", "gap_bucket", "scene"]:
        if metrics_df[key].nunique() <= 1:
            continue
        _emit(f"\n=== by {key} ===")
        agg = (metrics_df.groupby(key, dropna=False)
                          .agg(n=("l1", "size"), l1_mean=("l1", "mean"),
                               mse_mean=("mse", "mean"))
                          .sort_values("l1_mean"))
        _emit(agg.to_string())

    summary_path = out_dir / "rollout_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\nsummary -> {summary_path}")
    print(f"gifs    -> {gif_dir} ({len(metrics_df['episode_id'].unique())} files)")
