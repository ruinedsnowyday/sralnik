"""One-GPU memory diagnostics, interventions, and latent probe evals."""

from __future__ import annotations

import json
import math
import os
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


def _get_plt():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/sralnik-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sralnik-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _gap_bucket(gl: int) -> str:
    if gl < 0:
        return "na"
    if gl in _GAP_EXACT:
        return f"g{gl}"
    return "other"


def _load_model(checkpoint: Path, device: torch.device) -> tuple[WorldModel, ModelConfig]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_checkpoint_dict(ck["model_cfg"])
    model = WorldModel(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg


def _load_manifest(data_root: Path, split: str, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_parquet(data_root / "manifest.parquet")
    df = df[df["split"] == split]
    df = df[df["relative_path"].fillna("").str.endswith(".h5")]
    if max_rows is not None:
        df = df.head(max_rows)
    return df.reset_index(drop=True)


def _decode_json_cell(x: Any) -> dict[str, Any]:
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    if not x:
        return {}
    try:
        return json.loads(str(x))
    except json.JSONDecodeError:
        return {}


def _load_full_episode(root: Path, row: pd.Series) -> dict[str, Any]:
    path = root / str(row["relative_path"])
    with h5py.File(path, "r") as f:
        rgb = np.asarray(f["rgb"][:], dtype=np.float32) / 255.0
        actions = np.asarray(f["action_id"][:], dtype=np.int64)
        succ = np.asarray(f["action_success"][:], dtype=np.bool_)
        phase = np.asarray(f["phase"][:], dtype=np.int64)
        tracked = [_decode_json_cell(x) for x in f["tracked_objects_json"][:]]
    rgb_t = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()
    gap = row.get("gap_length")
    if gap is None or (isinstance(gap, float) and np.isnan(gap)):
        gap = -1
    return {
        "obs": rgb_t,
        "actions": torch.from_numpy(actions).long(),
        "action_success": torch.from_numpy(succ),
        "phase": torch.from_numpy(phase).long(),
        "tracked": tracked,
        "episode_id": str(row["episode_id"]),
        "scene": str(row.get("scene", "") or ""),
        "probe_name": str(row.get("probe_name") or row.get("episode_type") or "unknown"),
        "episode_type": str(row.get("episode_type", "") or ""),
        "gap_length": int(gap),
        "target_object_id": str(row.get("target_object_id") or ""),
        "target_receptacle_id": str(row.get("target_receptacle_id") or ""),
    }


def _write_mask_from_phase(phase: torch.Tensor) -> torch.Tensor:
    B, T = phase.shape
    m = torch.zeros(B, T, dtype=torch.bool, device=phase.device)
    m[:, 0] = True
    if T > 1:
        m[:, :-1] |= (phase[:, :-1] == 0) & (phase[:, 1:] == 1)
    return m


def _decode_frame(model: WorldModel, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    if getattr(model, "sd_vae", None) is not None:
        return model.sd_vae(h, z)
    return model.decoder(h, z)


def _safe_obj(tracked: dict[str, Any], oid: str) -> dict[str, Any] | None:
    if oid and oid in tracked:
        return tracked[oid]
    if oid:
        prefix = oid.split("|", 1)[0]
        for k, v in tracked.items():
            if k.split("|", 1)[0] == prefix:
                return v
    return None


def _labels_from_tracked(ep: dict[str, Any], t: int) -> dict[str, Any]:
    tracked = ep["tracked"][t] if t < len(ep["tracked"]) else {}
    target_id = ep["target_object_id"]
    recept_id = ep["target_receptacle_id"]
    target = _safe_obj(tracked, target_id)
    recept = _safe_obj(tracked, recept_id)
    target_type = target.get("type") if target else (target_id.split("|", 1)[0] if target_id else "")
    parents = target.get("parentReceptacles") if target else None
    recept_contents = recept.get("receptacleObjectIds") if recept else None
    parents = parents if isinstance(parents, list) else []
    recept_contents = recept_contents if isinstance(recept_contents, list) else []
    target_on_recept = False
    if recept_id:
        rid_prefix = recept_id.split("|", 1)[0]
        target_on_recept = any((p == recept_id or str(p).split("|", 1)[0] == rid_prefix) for p in parents)
    recept_contains_target = False
    if target_id:
        tid_prefix = target_id.split("|", 1)[0]
        recept_contains_target = any((o == target_id or str(o).split("|", 1)[0] == tid_prefix) for o in recept_contents)
    n_on_recept = 0
    if recept_id:
        rid_prefix = recept_id.split("|", 1)[0]
        for obj in tracked.values():
            ps = obj.get("parentReceptacles")
            ps = ps if isinstance(ps, list) else []
            if any((p == recept_id or str(p).split("|", 1)[0] == rid_prefix) for p in ps):
                n_on_recept += 1
    return {
        "target_present": target is not None,
        "target_type": target_type,
        "target_visible": bool(target.get("visible")) if target else None,
        "target_is_toggled": target.get("isToggled") if target else None,
        "target_is_open": target.get("isOpen") if target else None,
        "target_on_receptacle": target_on_recept,
        "receptacle_contains_target": recept_contains_target,
        "tracked_count_on_receptacle": n_on_recept,
    }


def _memory_stats(
    model: WorldModel,
    cfg: ModelConfig,
    h: torch.Tensor,
    z: torch.Tensor,
    hist: torch.Tensor | None,
    mask: torch.Tensor | None,
    phases: torch.Tensor | None,
) -> dict[str, Any]:
    if cfg.memory_mode is MemoryMode.NONE or hist is None or mask is None or hist.shape[1] == 0 or not mask.any():
        return {
            "gate": np.nan,
            "attn_entropy": np.nan,
            "top1_index": -1,
            "top1_phase": -1,
            "top1_weight": np.nan,
            "phase_a_mass": np.nan,
            "phase_b_mass": np.nan,
            "phase_c_mass": np.nan,
        }
    if cfg.memory_mode is MemoryMode.CONCAT:
        valid = torch.nonzero(mask[0], as_tuple=False).flatten()
        top = int(valid[-1].item()) if valid.numel() else -1
        ph = int(phases[top].item()) if phases is not None and top >= 0 else -1
        return {
            "gate": np.nan,
            "attn_entropy": np.nan,
            "top1_index": top,
            "top1_phase": ph,
            "top1_weight": np.nan,
            "phase_a_mass": np.nan,
            "phase_b_mass": np.nan,
            "phase_c_mass": np.nan,
        }

    mem = model.memory
    q = mem._q(torch.cat([h, z], dim=-1)).view(1, -1)
    k = mem._att_in(hist[0])
    scores = (q @ k.T).squeeze(0) / math.sqrt(float(k.shape[-1]))
    scores = scores.masked_fill(~mask[0].bool(), -torch.inf)
    w = torch.softmax(scores, dim=0)
    top = int(torch.argmax(w).item())
    entropy = float((-(w[w > 0] * torch.log(w[w > 0])).sum()).detach().cpu())
    gate = np.nan
    if cfg.memory_mode is MemoryMode.GATED:
        gate = float(torch.sigmoid(mem._gate(torch.cat([h, z], dim=-1))).squeeze().detach().cpu())
    phase_vals = phases.to(w.device) if phases is not None else torch.full_like(w, -1, dtype=torch.long)
    def mass(p: int) -> float:
        return float(w[phase_vals == p].sum().detach().cpu()) if (phase_vals == p).any() else 0.0
    return {
        "gate": gate,
        "attn_entropy": entropy,
        "top1_index": top,
        "top1_phase": int(phase_vals[top].item()) if top >= 0 else -1,
        "top1_weight": float(w[top].detach().cpu()) if top >= 0 else np.nan,
        "phase_a_mass": mass(0),
        "phase_b_mass": mass(1),
        "phase_c_mass": mass(2),
    }


@torch.no_grad()
def _roll_episode(
    model: WorldModel,
    cfg: ModelConfig,
    ep: dict[str, Any],
    device: torch.device,
    *,
    intervention: str = "normal",
    donor_memory: dict[str, torch.Tensor] | None = None,
    k_imagine: int | None = None,
    collect_frames: bool = False,
) -> dict[str, Any]:
    obs = ep["obs"].unsqueeze(0).to(device)
    actions = ep["actions"].unsqueeze(0).to(device).clamp(0, cfg.action_vocab_size - 1)
    succ = ep["action_success"].unsqueeze(0).to(device)
    phase = ep["phase"].unsqueeze(0).to(device)
    _, T, _, _, _ = obs.shape
    write_mask = _write_mask_from_phase(phase)
    if intervention == "all_history":
        write_mask = torch.ones_like(write_mask)
    t_imag = T if k_imagine is None else max(1, T - int(k_imagine))

    h = model.h0.expand(1, -1).contiguous()
    z_hist: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    h_rows: list[np.ndarray] = []
    z_rows: list[np.ndarray] = []
    prior_rows: list[np.ndarray] = []
    pred_frames: list[np.ndarray] = []

    for t in range(T):
        prior_raw = model.prior_net(h)
        prior_mu, prior_logstd = torch.chunk(prior_raw, 2, dim=-1)
        if t < t_imag:
            _, post_mu, _ = model.encoder(obs[:, t])
            z = post_mu
        else:
            z = prior_mu

        hist = torch.stack(z_hist, dim=1) if z_hist else None
        mem_mask = write_mask[:, : hist.shape[1]].bool() if hist is not None else None
        mem_phases = phase[0, : hist.shape[1]].detach() if hist is not None else None
        stats = _memory_stats(model, cfg, h, z, hist, mem_mask, mem_phases)

        use_memory = hist is not None and cfg.memory_mode is not MemoryMode.NONE
        if intervention == "disabled_all":
            use_memory = False
        if intervention == "disabled_phase_c" and int(phase[0, t].item()) == 2:
            use_memory = False
        if intervention == "shuffled_memory" and donor_memory is not None:
            hist = donor_memory["hist"].to(device)
            mem_mask = donor_memory["mask"].to(device)
            mem_phases = donor_memory["phases"].to(device)
            stats = _memory_stats(model, cfg, h, z, hist, mem_mask, mem_phases)
            use_memory = cfg.memory_mode is not MemoryMode.NONE and hist.shape[1] > 0

        if use_memory and hist is not None and mem_mask is not None:
            h_mem = model.memory(h, z, hist, mem_mask)
        else:
            h_mem = h

        act_e = model.act_emb(actions[:, t] + 1)
        gru_in = torch.cat([z, act_e, succ[:, t].float().unsqueeze(-1)], dim=-1)
        h = model.gru(gru_in, h_mem)
        z_hist.append(z.detach())
        x_hat = _decode_frame(model, h, z)

        l1 = F.l1_loss(x_hat[0], obs[0, t], reduction="mean").item()
        mse = F.mse_loss(x_hat[0], obs[0, t], reduction="mean").item()
        labels = _labels_from_tracked(ep, t)
        row = {
            "episode_id": ep["episode_id"],
            "scene": ep["scene"],
            "probe_name": ep["probe_name"],
            "episode_type": ep["episode_type"],
            "gap_length": ep["gap_length"],
            "gap_bucket": _gap_bucket(ep["gap_length"]),
            "target_object_id": ep["target_object_id"],
            "target_receptacle_id": ep["target_receptacle_id"],
            "frame_index": t,
            "phase": int(phase[0, t].item()),
            "is_phase_c": int(phase[0, t].item()) == 2,
            "is_imagined": t >= t_imag,
            "write_mask": bool(write_mask[0, t].item()),
            "intervention": intervention,
            "l1": l1,
            "mse": mse,
            "h_norm": float(h.norm(dim=-1).item()),
            "z_norm": float(z.norm(dim=-1).item()),
            "prior_norm": float(prior_mu.norm(dim=-1).item()),
            **stats,
            **labels,
        }
        rows.append(row)
        h_rows.append(h.squeeze(0).detach().cpu().float().numpy())
        z_rows.append(z.squeeze(0).detach().cpu().float().numpy())
        prior_rows.append(prior_mu.squeeze(0).detach().cpu().float().numpy())
        if collect_frames and t >= t_imag:
            pred_frames.append(_to_uint8(x_hat[0]))

    hist_tensor = torch.stack(z_hist, dim=1).detach()
    mask_tensor = write_mask[:, : hist_tensor.shape[1]].bool().detach()
    return {
        "rows": rows,
        "h": np.stack(h_rows, axis=0),
        "z": np.stack(z_rows, axis=0),
        "prior": np.stack(prior_rows, axis=0),
        "memory": {
            "hist": hist_tensor,
            "mask": mask_tensor,
            "phases": phase[0, : hist_tensor.shape[1]].detach(),
        },
        "pred_frames": pred_frames,
        "t_imag": t_imag,
    }


def _to_uint8(img: torch.Tensor) -> np.ndarray:
    x = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255.0).round().astype(np.uint8)


def _save_panel_gif(frames: list[list[np.ndarray]], labels: list[str], out: Path, duration_ms: int = 250) -> None:
    frames = [f for f in frames if f]
    if not frames or not frames[0]:
        return
    n = min(len(f) for f in frames)
    if n <= 0:
        return
    H, W, _ = frames[0][0].shape
    sep = np.full((H, 4, 3), 210, dtype=np.uint8)
    imgs = []
    for i in range(n):
        parts = []
        for j, seq in enumerate(frames):
            if j:
                parts.append(sep)
            parts.append(seq[i])
        imgs.append(Image.fromarray(np.concatenate(parts, axis=1)))
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0, optimize=False)


def _write_summary(df: pd.DataFrame, out: Path, metric: str = "l1") -> None:
    lines = []
    lines.append("=== overall ===")
    lines.append(df.agg({metric: "mean", "mse": "mean"}).to_string())
    for key in ["intervention", "probe_name", "gap_bucket", "scene", "phase"]:
        if key in df and df[key].nunique(dropna=False) > 1:
            lines.append(f"\n=== by {key} ===")
            lines.append(df.groupby(key, dropna=False).agg(n=(metric, "size"), l1=(metric, "mean"), mse=("mse", "mean")).to_string())
    out.write_text("\n".join(lines))


def _plot_metric_bars(df: pd.DataFrame, out_dir: Path, *, x: str, hue: str, y: str, title: str, name: str) -> None:
    if df.empty or x not in df or hue not in df:
        return
    plt = _get_plt()
    agg = df.groupby([x, hue], dropna=False)[y].mean().reset_index()
    piv = agg.pivot(index=x, columns=hue, values=y)
    ax = piv.plot(kind="bar", figsize=(max(6, 1.2 * len(piv)), 4), rot=30)
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{name}.png", dpi=180)
    plt.savefig(out_dir / f"{name}.pdf")
    plt.close()


def _plot_trace(df: pd.DataFrame, out_dir: Path) -> None:
    plt = _get_plt()
    out_dir.mkdir(parents=True, exist_ok=True)
    if "gate" in df and df["gate"].notna().any():
        pc = df[df["gate"].notna()]
        _plot_metric_bars(pc, out_dir, x="phase", hue="gap_bucket", y="gate", title="M3 gate by phase and gap", name="gate_by_phase_gap")
        _plot_metric_bars(pc, out_dir, x="probe_name", hue="phase", y="gate", title="M3 gate by probe and phase", name="gate_by_probe_phase")
        for ep_id, g in pc.groupby("episode_id"):
            if len(g) < 3:
                continue
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(g["frame_index"], g["gate"], lw=2)
            for phase_code, color, label in [(0, "#e6f2ff", "A"), (1, "#f7f7f7", "B"), (2, "#fff0e6", "C")]:
                gg = g[g["phase"] == phase_code]
                if len(gg):
                    ax.axvspan(gg["frame_index"].min(), gg["frame_index"].max(), color=color, alpha=0.8, label=label)
            ax.plot(g["frame_index"], g["gate"], color="#1f77b4", lw=2)
            ax.set_title(f"Gate over time: {ep_id}")
            ax.set_xlabel("frame")
            ax.set_ylabel("gate")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            plt.tight_layout()
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(ep_id))[:80]
            plt.savefig(out_dir / f"gate_timeline_{safe}.png", dpi=180)
            plt.close()
            break
    if "top1_phase" in df and (df["top1_phase"] >= 0).any():
        tmp = df[df["top1_phase"] >= 0].copy()
        tmp["retrieved_phase"] = tmp["top1_phase"].map({0: "A", 1: "B", 2: "C"}).fillna("other")
        counts = tmp.groupby(["phase", "retrieved_phase"]).size().reset_index(name="n")
        piv = counts.pivot(index="phase", columns="retrieved_phase", values="n").fillna(0)
        ax = piv.plot(kind="bar", stacked=True, figsize=(6, 4), rot=0)
        ax.set_title("Top retrieved memory phase")
        ax.set_xlabel("query phase")
        ax.set_ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "retrieved_phase_hist.png", dpi=180)
        plt.savefig(out_dir / "retrieved_phase_hist.pdf")
        plt.close()


def run_eval_memory_trace(args: Namespace) -> None:
    device = torch.device(args.device)
    model, cfg = _load_model(Path(args.checkpoint), device)
    data_root = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    df = _load_manifest(data_root, args.split, args.max_rows)
    for _, row in df.iterrows():
        ep = _load_full_episode(data_root, row)
        rolled = _roll_episode(model, cfg, ep, device)
        rows.extend(rolled["rows"])
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_dir / "memory_trace.parquet", index=False)
    _write_summary(out_df, out_dir / "memory_trace_summary.txt")
    _plot_trace(out_df, out_dir / "plots")
    print(f"wrote memory trace: {out_dir}")


def run_eval_memory_intervention(args: Namespace) -> None:
    device = torch.device(args.device)
    model, cfg = _load_model(Path(args.checkpoint), device)
    data_root = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_manifest(data_root, args.split, args.max_rows)
    interventions = [s.strip() for s in args.interventions.split(",") if s.strip()]
    rows: list[dict[str, Any]] = []
    prev_memory: dict[str, torch.Tensor] | None = None
    gif_count = 0
    for _, row in df.iterrows():
        ep = _load_full_episode(data_root, row)
        normal = _roll_episode(model, cfg, ep, device, intervention="normal", k_imagine=args.k_imagine)
        if prev_memory is None:
            prev_memory = normal["memory"]
        frames_for_gif: list[list[np.ndarray]] = []
        gif_labels: list[str] = []
        gt_frames = [_to_uint8(ep["obs"][t]) for t in range(normal["t_imag"], ep["obs"].shape[0])]
        for intervention in interventions:
            donor = prev_memory if intervention == "shuffled_memory" else None
            collect = gif_count < int(args.max_gifs) and intervention in {"normal", "disabled_all", "shuffled_memory"}
            if intervention == "normal" and not collect:
                rolled = normal
            else:
                rolled = _roll_episode(
                    model,
                    cfg,
                    ep,
                    device,
                    intervention=intervention,
                    donor_memory=donor,
                    k_imagine=args.k_imagine,
                    collect_frames=collect,
                )
            rows.extend(rolled["rows"])
            if collect:
                frames_for_gif.append(rolled["pred_frames"])
                gif_labels.append(intervention)
        if gif_count < int(args.max_gifs) and frames_for_gif:
            _save_panel_gif([gt_frames] + frames_for_gif, ["gt"] + gif_labels, out_dir / "gifs" / f"{ep['episode_id']}.gif")
            gif_count += 1
        prev_memory = normal["memory"]
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_dir / "memory_interventions.parquet", index=False)
    _write_summary(out_df, out_dir / "memory_interventions_summary.txt")
    pc = out_df[out_df["is_phase_c"]]
    _plot_metric_bars(pc, out_dir / "plots", x="intervention", hue="gap_bucket", y="l1", title="Phase-C L1 by memory intervention", name="intervention_l1_by_gap")
    _plot_metric_bars(pc, out_dir / "plots", x="intervention", hue="probe_name", y="l1", title="Phase-C L1 by intervention and probe", name="intervention_l1_by_probe")
    print(f"wrote memory interventions: {out_dir}")


def run_eval_latent_cache(args: Namespace) -> None:
    device = torch.device(args.device)
    model, cfg = _load_model(Path(args.checkpoint), device)
    data_root = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_manifest(data_root, args.split, args.max_rows)
    rows: list[dict[str, Any]] = []
    hs: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    offset = 0
    for _, row in df.iterrows():
        ep = _load_full_episode(data_root, row)
        rolled = _roll_episode(model, cfg, ep, device)
        n = len(rolled["rows"])
        for i, r in enumerate(rolled["rows"]):
            rr = dict(r)
            rr["feature_index"] = offset + i
            rows.append(rr)
        hs.append(rolled["h"])
        zs.append(rolled["z"])
        priors.append(rolled["prior"])
        offset += n
    index = pd.DataFrame(rows)
    index.to_parquet(out_dir / "index.parquet", index=False)
    np.savez_compressed(
        out_dir / "features.npz",
        h=np.concatenate(hs, axis=0) if hs else np.zeros((0, cfg.deter_dim), dtype=np.float32),
        z=np.concatenate(zs, axis=0) if zs else np.zeros((0, cfg.stoch_dim), dtype=np.float32),
        prior=np.concatenate(priors, axis=0) if priors else np.zeros((0, cfg.stoch_dim), dtype=np.float32),
    )
    (out_dir / "cache_meta.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "split": args.split}, indent=2))
    print(f"wrote latent cache: {out_dir}")


def _binary_probe_score(X: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    ok = ~np.isnan(y)
    X = X[ok]
    y = y[ok].astype(np.float32)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return {"n": float(len(y)), "acc": np.nan, "majority": np.nan}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_train = max(1, int(0.7 * len(y)))
    tr, te = perm[:n_train], perm[n_train:]
    if len(te) == 0 or len(np.unique(y[tr])) < 2:
        return {"n": float(len(y)), "acc": np.nan, "majority": np.nan}
    mu = X[tr].mean(axis=0, keepdims=True)
    sd = X[tr].std(axis=0, keepdims=True) + 1e-6
    Xt = torch.from_numpy(((X[tr] - mu) / sd).astype(np.float32))
    yt = torch.from_numpy(y[tr, None].astype(np.float32))
    Xv = torch.from_numpy(((X[te] - mu) / sd).astype(np.float32))
    yv = torch.from_numpy(y[te].astype(np.float32))
    clf = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 1))
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(clf(Xt), yt)
        loss.backward()
        opt.step()
    pred = (torch.sigmoid(clf(Xv)).squeeze(-1) >= 0.5).float()
    acc = float((pred == yv).float().mean().item())
    maj = float(max(y[te].mean(), 1.0 - y[te].mean()))
    return {"n": float(len(y)), "acc": acc, "majority": maj}


def _load_cache(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    idx = pd.read_parquet(path / "index.parquet")
    arr = np.load(path / "features.npz")
    return idx, arr["h"]


def run_eval_latent_probes(args: Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caches = [("m0", Path(args.m0_cache)), ("m3", Path(args.m3_cache))]
    targets = [
        "target_visible",
        "target_is_toggled",
        "target_is_open",
        "target_on_receptacle",
        "receptacle_contains_target",
    ]
    rows = []
    for cond, path in caches:
        idx, h = _load_cache(path)
        idx = idx.reset_index(drop=True)
        for subset_name, mask in {
            "all": np.ones(len(idx), dtype=bool),
            "phase_c": idx["is_phase_c"].to_numpy(dtype=bool),
        }.items():
            for target in targets:
                if target not in idx:
                    continue
                vals = idx[target].map(lambda x: np.nan if pd.isna(x) else float(bool(x))).to_numpy(dtype=np.float32)
                score = _binary_probe_score(h[mask], vals[mask], int(args.seed))
                rows.append({"condition": cond, "subset": subset_name, "target": target, **score})
            for key in ["probe_name", "gap_bucket"]:
                if key not in idx:
                    continue
                for value in sorted(idx.loc[mask, key].dropna().unique()):
                    mm = mask & (idx[key] == value).to_numpy()
                    vals = idx["receptacle_contains_target"].map(lambda x: np.nan if pd.isna(x) else float(bool(x))).to_numpy(dtype=np.float32)
                    score = _binary_probe_score(h[mm], vals[mm], int(args.seed))
                    rows.append({"condition": cond, "subset": f"phase_c_{key}={value}", "target": "receptacle_contains_target", **score})
    res = pd.DataFrame(rows)
    res.to_parquet(out_dir / "latent_probe_results.parquet", index=False)
    lines = ["=== latent probe results ===", res.to_string(index=False)]
    (out_dir / "latent_probe_summary.txt").write_text("\n".join(lines))
    if not res.empty:
        plot_df = res[(res["subset"] == "phase_c") & res["acc"].notna()]
        _plot_metric_bars(plot_df, out_dir / "plots", x="target", hue="condition", y="acc", title="Phase-C latent probe accuracy", name="phase_c_probe_accuracy")
    print(f"wrote latent probe eval: {out_dir}")
