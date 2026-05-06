"""Dataclasses describing collection configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ControllerConfig:
    """AI2-THOR controller settings."""

    width: int = 256
    height: int = 256
    grid_size: float = 0.25
    # AI2-THOR requires rotateStepDegrees in {0, 90, 180, 270, 360} when
    # snapToGrid is True. 90 keeps the agent on a clean 4-heading grid which is
    # easier for the world model to learn; bump to 30 *and* set
    # snap_to_grid=False if you want finer rotations.
    rotate_step_degrees: float = 90.0
    visibility_distance: float = 1.5
    field_of_view: float = 90.0
    snap_to_grid: bool = True
    render_depth: bool = False
    render_instance_segmentation: bool = True
    agent_mode: str = "default"
    platform: str | None = None
    # Cold-start Unity launch on Apple Silicon (under Rosetta) can exceed the
    # ai2thor default of 100s. We bump it generously; warm restarts are fast.
    server_timeout: float = 600.0
    server_start_timeout: float = 600.0


@dataclass(frozen=True)
class ExplorationConfig:
    """Random + reachable-position-waypoint walker settings."""

    episode_length: int = 200
    waypoint_replan_every: int = 12
    interaction_attempt_every: int = 30
    rotate_after_block_attempts: int = 4


@dataclass(frozen=True)
class ProbeConfig:
    """Memory-probe episode settings."""

    gap_lengths: tuple[int, ...] = (20, 100, 300, 1000)
    held_out_gap: int = 1000  # used only at test time
    phase_a_max_steps: int = 60
    phase_c_max_steps: int = 30
    hold_frames_at_query: int = 5


@dataclass
class CollectConfig:
    """Top-level config for one collection run."""

    output_dir: Path
    scenes: tuple[str, ...] = ("FloorPlan1", "FloorPlan403")
    episodes_per_scene: int = 400
    exploration_fraction: float = 0.5
    seed: int = 0
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    # Re-randomise object placements between episodes via THOR's
    # ``InitialRandomSpawn``. **Disabled by default** — in long runs on
    # macOS we observed Unity instability. Enable only if you know you
    # need cross-episode object diversity and can tolerate occasional
    # crashes.
    randomize_object_layout: bool = False
    # Tear down and recreate the AI2-THOR controller every N episodes
    # within a scene. In theory this would prevent Unity memory leaks
    # from accumulating across many ``InitialRandomSpawn`` calls, but
    # in practice the proactive recycle is itself unreliable on macOS
    # (Unity FIFO pipe doesn't always release in time for the next
    # process). Disabled by default; the reactive recovery in
    # ``collect.py`` still runs if Unity dies anyway.
    controller_restart_every: int = 0

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.scenes = tuple(self.scenes)
