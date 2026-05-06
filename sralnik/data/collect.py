"""Top-level data collection orchestration.

Usage example::

    from pathlib import Path
    from sralnik.data.collect import run_collection
    from sralnik.data.config import CollectConfig

    cfg = CollectConfig(output_dir=Path("data/ithor_v1"))
    run_collection(cfg)

The default budget per scene is laid out as:

* 50% **exploration** episodes (waypoint random walker).
* 50% **probe** episodes, distributed evenly across the 5 probe types and the
  configured ``gap_lengths``. Probes whose ``gap_length`` equals
  ``probe.held_out_gap`` are forced into the ``test`` split (held-out long-gap
  evaluation). Other episodes follow ``val_fraction`` / ``test_fraction``.
"""

from __future__ import annotations

import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .config import CollectConfig
from .controller import StepResult, ThorEnv
from .policies import WaypointWalker
from .probes import PROBE_REGISTRY
from .probes.base import ProbeOutcome
from .scenes import get_scene_config
from .storage import EpisodeRecord, EpisodeWriter, ManifestBuilder


PROBE_NAMES: tuple[str, ...] = (
    "fridge",
    "toggle",
    "displacement",
    "layout",
    "rearrangement",
)


@dataclass
class _Plan:
    """Schedule item describing one episode to collect."""

    episode_id: str
    scene: str
    episode_type: str  # "exploration" | probe name
    probe_name: str | None
    gap_length: int | None
    seed: int
    split: str


def _plan_episodes(cfg: CollectConfig) -> list[_Plan]:
    rng = random.Random(cfg.seed)
    plans: list[_Plan] = []

    n_total = cfg.episodes_per_scene
    n_explore = int(round(n_total * cfg.exploration_fraction))
    n_probe = n_total - n_explore

    gap_lengths = list(cfg.probe.gap_lengths)
    held_out = cfg.probe.held_out_gap

    for scene in cfg.scenes:
        # Exploration episodes.
        for i in range(n_explore):
            split = _split_for_index(i, n_explore, cfg, rng)
            plans.append(
                _Plan(
                    episode_id=f"{scene}_explore_{i:04d}",
                    scene=scene,
                    episode_type="exploration",
                    probe_name=None,
                    gap_length=None,
                    seed=rng.randrange(2**31),
                    split=split,
                )
            )

        # Probe episodes: cycle (probe, gap) pairs.
        per_probe = n_probe // len(PROBE_NAMES)
        remainder = n_probe - per_probe * len(PROBE_NAMES)
        idx = 0
        for pi, probe_name in enumerate(PROBE_NAMES):
            count = per_probe + (1 if pi < remainder else 0)
            for k in range(count):
                gap = gap_lengths[k % len(gap_lengths)]
                if gap == held_out:
                    split = "test"
                else:
                    split = _split_for_index(idx, n_probe, cfg, rng)
                plans.append(
                    _Plan(
                        episode_id=f"{scene}_{probe_name}_g{gap}_{k:04d}",
                        scene=scene,
                        episode_type=probe_name,
                        probe_name=probe_name,
                        gap_length=gap,
                        seed=rng.randrange(2**31),
                        split=split,
                    )
                )
                idx += 1

    return plans


def _split_for_index(i: int, total: int, cfg: CollectConfig, rng: random.Random) -> str:
    n_test = int(round(total * cfg.test_fraction))
    n_val = int(round(total * cfg.val_fraction))
    if i < n_test:
        return "test"
    if i < n_test + n_val:
        return "val"
    return "train"


def _output_path(root: Path, plan: _Plan) -> Path:
    if plan.episode_type == "exploration":
        return root / "episodes" / plan.scene / "exploration" / f"{plan.episode_id}.h5"
    return (
        root
        / "episodes"
        / plan.scene
        / f"probe_{plan.probe_name}"
        / f"gap{plan.gap_length}"
        / f"{plan.episode_id}.h5"
    )


def _run_exploration(env: ThorEnv, cfg: CollectConfig, seed: int) -> ProbeOutcome:
    rng = random.Random(seed)
    walker = WaypointWalker(env, cfg.exploration, rng)
    outcome = ProbeOutcome()
    last = env.reset(randomize_seed=seed if cfg.randomize_object_layout else None)
    outcome.append(last, phase="B")
    for _ in range(cfg.exploration.episode_length):
        spec = walker.next_action(last)
        last = env.step(spec)
        outcome.append(last, phase="B")
    outcome.success = True
    return outcome


def _run_probe(env: ThorEnv, plan: _Plan, cfg: CollectConfig, scene_cfg) -> ProbeOutcome:
    cls = PROBE_REGISTRY[plan.probe_name]  # type: ignore[index]
    rng = random.Random(plan.seed)
    probe = cls(
        env=env,
        scene_cfg=scene_cfg,
        probe_cfg=cfg.probe,
        exp_cfg=cfg.exploration,
        rng=rng,
        gap_length=plan.gap_length or 0,
    )
    return probe.run(randomize_seed=plan.seed if cfg.randomize_object_layout else None)


def _needs_unity_restart(exc: BaseException) -> bool:
    """Heuristic: backend hung or subprocess died — recycle the controller."""

    if isinstance(exc, TimeoutError):
        return True
    msg = str(exc).lower()
    if "write to closed file" in msg or "broken pipe" in msg:
        return True
    if "timed out" in msg and "ai2-thor" in msg:
        return True
    if "error encountered when running action" in msg and "initialize" in msg:
        return True
    return False


def run_collection(cfg: CollectConfig) -> Path:
    """Execute the full collection plan and return the manifest path."""

    plans = _plan_episodes(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ManifestBuilder(cfg.output_dir)

    # Group by scene so we re-init the THOR controller as little as possible.
    plans_by_scene: dict[str, list[_Plan]] = {}
    for p in plans:
        plans_by_scene.setdefault(p.scene, []).append(p)

    for scene, scene_plans in plans_by_scene.items():
        scene_cfg = get_scene_config(scene)
        env = ThorEnv(cfg.controller, scene, scene_cfg.tracked_types)
        episodes_since_restart = 0
        try:
            for plan in tqdm(scene_plans, desc=scene):
                # Periodically recycle the AI2-THOR controller to prevent
                # Unity memory leaks (especially when ``InitialRandomSpawn``
                # is called every episode).
                if (
                    cfg.controller_restart_every > 0
                    and episodes_since_restart >= cfg.controller_restart_every
                ):
                    env.stop()
                    env = ThorEnv(cfg.controller, scene, scene_cfg.tracked_types)
                    episodes_since_restart = 0
                episodes_since_restart += 1

                t0 = time.time()
                try:
                    if plan.episode_type == "exploration":
                        outcome = _run_exploration(env, cfg, plan.seed)
                    else:
                        outcome = _run_probe(env, plan, cfg, scene_cfg)
                except Exception as exc:  # noqa: BLE001
                    print(f"[error] {plan.episode_id}: {exc}\n{traceback.format_exc()}")
                    # If the underlying Unity process died, every subsequent
                    # call will fail with "write to closed file" until we
                    # restart the controller. Detect that and recover.
                    if _needs_unity_restart(exc):
                        try:
                            env.stop()
                        except Exception:
                            pass
                        try:
                            env = ThorEnv(cfg.controller, scene, scene_cfg.tracked_types)
                            episodes_since_restart = 0
                            print(
                                f"[recover] restarted AI2-THOR for {scene} "
                                f"after backend error: {type(exc).__name__}"
                            )
                        except Exception as restart_exc:
                            print(f"[recover] failed to restart: {restart_exc}")
                    record = EpisodeRecord(
                        episode_id=plan.episode_id,
                        scene=plan.scene,
                        episode_type=plan.episode_type,
                        probe_name=plan.probe_name,
                        gap_length=plan.gap_length,
                        seed=plan.seed,
                        split=plan.split,
                        num_steps=0,
                        success=False,
                        failure_reason=f"exception: {exc}",
                        target_object_id=None,
                        target_receptacle_id=None,
                        relative_path="",
                    )
                    manifest.add(record)
                    continue

                if not outcome.success or not outcome.steps:
                    record = EpisodeRecord(
                        episode_id=plan.episode_id,
                        scene=plan.scene,
                        episode_type=plan.episode_type,
                        probe_name=plan.probe_name,
                        gap_length=plan.gap_length,
                        seed=plan.seed,
                        split=plan.split,
                        num_steps=len(outcome.steps),
                        success=False,
                        failure_reason=outcome.failure_reason,
                        target_object_id=outcome.target_object_id,
                        target_receptacle_id=outcome.target_receptacle_id,
                        relative_path="",
                    )
                    manifest.add(record)
                    continue

                out_path = _output_path(cfg.output_dir, plan)
                writer = EpisodeWriter(out_path)
                writer.write(
                    steps=outcome.steps,
                    phases=outcome.phases,
                    attrs={
                        "episode_id": plan.episode_id,
                        "scene": plan.scene,
                        "episode_type": plan.episode_type,
                        "probe_name": plan.probe_name,
                        "gap_length": plan.gap_length or -1,
                        "seed": plan.seed,
                        "split": plan.split,
                        "target_object_id": outcome.target_object_id,
                        "target_receptacle_id": outcome.target_receptacle_id,
                        "wallclock_seconds": time.time() - t0,
                    },
                )
                rel = out_path.relative_to(cfg.output_dir).as_posix()
                manifest.add(
                    EpisodeRecord(
                        episode_id=plan.episode_id,
                        scene=plan.scene,
                        episode_type=plan.episode_type,
                        probe_name=plan.probe_name,
                        gap_length=plan.gap_length,
                        seed=plan.seed,
                        split=plan.split,
                        num_steps=len(outcome.steps),
                        success=True,
                        failure_reason=None,
                        target_object_id=outcome.target_object_id,
                        target_receptacle_id=outcome.target_receptacle_id,
                        relative_path=rel,
                    )
                )
        finally:
            env.stop()

    return manifest.write()
