"""CLI for data collection.

Examples::

    # Tiny smoke test: 1 exploration episode in FloorPlan1, no probes, no save.
    python -m sralnik.data smoke

    # Pilot collection (200 episodes/scene, 3 scenes).
    python -m sralnik.data collect --output data/ithor_v1

    # Custom subset.
    python -m sralnik.data collect --output data/quick \\
        --scenes FloorPlan1 --episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .collect import run_collection
from .config import (
    CollectConfig,
    ControllerConfig,
    ExplorationConfig,
    ProbeConfig,
)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scenes", nargs="+",
                   default=("FloorPlan1", "FloorPlan403"))
    p.add_argument("--episodes", type=int, default=400,
                   help="Episodes per scene")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-segmentation", action="store_true")
    p.add_argument("--with-depth", action="store_true",
                   help="Also render and save per-step depth maps.")
    p.add_argument("--randomize-layout", action="store_true",
                   help="Enable per-episode InitialRandomSpawn (object shuffling). "
                        "Experimental; can destabilize Unity on long macOS runs.")
    p.add_argument("--platform", type=str, default=None,
                   help="THOR platform override: e.g. CloudRendering, Linux64, OSXIntel64.")
    p.add_argument("--unity-timeout", type=float, default=600.0,
                   help="Seconds to wait for Unity to respond (default 600; bump if cold start times out).")
    p.add_argument("--exploration-fraction", type=float, default=0.5)
    p.add_argument("--gap-lengths", nargs="+", type=int, default=(20, 100, 300, 1000))
    p.add_argument("--held-out-gap", type=int, default=1000)
    p.add_argument("--episode-length", type=int, default=200,
                   help="Steps for exploration episodes")


def _build_config(args: argparse.Namespace, output_dir: Path) -> CollectConfig:
    return CollectConfig(
        output_dir=output_dir,
        scenes=tuple(args.scenes),
        episodes_per_scene=args.episodes,
        exploration_fraction=args.exploration_fraction,
        seed=args.seed,
        controller=ControllerConfig(
            width=args.width,
            height=args.height,
            render_instance_segmentation=not args.no_segmentation,
            render_depth=args.with_depth,
            platform=args.platform,
            server_timeout=args.unity_timeout,
            server_start_timeout=args.unity_timeout,
        ),
        exploration=ExplorationConfig(episode_length=args.episode_length),
        probe=ProbeConfig(
            gap_lengths=tuple(args.gap_lengths),
            held_out_gap=args.held_out_gap,
        ),
        randomize_object_layout=args.randomize_layout,
    )


def cmd_collect(args: argparse.Namespace) -> int:
    output = Path(args.output)
    cfg = _build_config(args, output)
    print(f"[collect] writing to {output} :: scenes={cfg.scenes} "
          f"episodes/scene={cfg.episodes_per_scene}")
    manifest = run_collection(cfg)
    print(f"[collect] done. manifest: {manifest}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    from .manual_recorder import run_recorder

    run_recorder(
        output_dir=Path(args.output),
        scene=args.scene,
        width=args.width,
        height=args.height,
        display_size=args.display_size,
        split=args.split,
        note=args.note,
    )
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Run a short end-to-end check without saving anything large.

    Spins up the controller, runs ~10 walker steps, and runs the fridge probe
    with gap=5 in FloorPlan1. Useful for sanity-checking the AI2-THOR install.
    """

    from .config import CollectConfig, ControllerConfig, ExplorationConfig, ProbeConfig
    from .controller import ThorEnv
    from .policies.waypoint_walker import WaypointWalker
    from .probes import FridgeProbe
    from .scenes import get_scene_config
    import random

    print("[smoke] starting AI2-THOR controller (this downloads the Unity build the first time)...")
    scene = "FloorPlan1"
    scene_cfg = get_scene_config(scene)
    controller_cfg = ControllerConfig(
        width=args.width,
        height=args.height,
        render_instance_segmentation=not args.no_segmentation,
        render_depth=args.with_depth,
        platform=args.platform,
        server_timeout=args.unity_timeout,
        server_start_timeout=args.unity_timeout,
    )
    env = ThorEnv(controller_cfg, scene, scene_cfg.tracked_types)
    try:
        rng = random.Random(0)
        last = env.reset()
        print(f"[smoke] reset OK. agent pose={last.pose.tolist()}")
        walker = WaypointWalker(env, ExplorationConfig(), rng)
        for i in range(10):
            spec = walker.next_action(last)
            last = env.step(spec)
            print(f"[smoke] step {i}: {spec.name} success={last.action_success}")

        print("[smoke] running fridge probe with gap=5 ...")
        probe = FridgeProbe(
            env=env,
            scene_cfg=scene_cfg,
            probe_cfg=ProbeConfig(),
            exp_cfg=ExplorationConfig(),
            rng=random.Random(1),
            gap_length=5,
        )
        outcome = probe.run()
        print(
            f"[smoke] probe success={outcome.success} steps={len(outcome.steps)} "
            f"failure_reason={outcome.failure_reason}"
        )
        return 0 if outcome.success else 1
    finally:
        env.stop()


def cmd_merge_manifest(args: argparse.Namespace) -> int:
    from .merge import merge_datasets

    scenes = tuple(args.scene) if args.scene else None
    path = merge_datasets(args.base, args.add, scenes=scenes, dry_run=args.dry_run)
    print(
        f"[merge] {'(dry-run) ' if args.dry_run else ''}"
        f"updated {path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sralnik.data")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Run a full collection pass.")
    p_collect.add_argument("--output", required=True, help="Output directory (e.g. data/ithor_v1)")
    _add_common_args(p_collect)
    p_collect.set_defaults(func=cmd_collect)

    p_smoke = sub.add_parser("smoke", help="Run a tiny smoke test, no files written.")
    _add_common_args(p_smoke)
    p_smoke.set_defaults(func=cmd_smoke)

    p_rec = sub.add_parser("record", help="Drive the agent manually with the keyboard and save trajectories.")
    p_rec.add_argument("--output", required=True, help="Dataset root (e.g. data/ithor_v1)")
    p_rec.add_argument("--scene", default="FloorPlan1")
    p_rec.add_argument("--width", type=int, default=256)
    p_rec.add_argument("--height", type=int, default=256)
    p_rec.add_argument("--display-size", type=int, default=768,
                       help="Pixel size of the on-screen frame (THOR frame is upscaled).")
    p_rec.add_argument("--split", default="expert_eval",
                       help="Split label assigned to recorded episodes (default: expert_eval).")
    p_rec.add_argument("--note", default=None,
                       help="Free-form note attached to every saved episode.")
    p_rec.set_defaults(func=cmd_record)

    p_merge = sub.add_parser(
        "merge",
        help="Merge successful episodes from a supplemental root into a base root.",
    )
    p_merge.add_argument("--base", required=True, type=Path)
    p_merge.add_argument("--add", required=True, type=Path)
    p_merge.add_argument(
        "--scene",
        action="append",
        help="Only import this scene from ADD (repeatable). Omit = all.",
    )
    p_merge.add_argument("--dry-run", action="store_true")
    p_merge.set_defaults(func=cmd_merge_manifest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
