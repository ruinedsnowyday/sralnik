"""Probe base class and shared helpers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from ..actions import ActionSpec
from ..config import ExplorationConfig, ProbeConfig
from ..controller import StepResult, ThorEnv
from ..policies.waypoint_walker import WaypointWalker
from ..scenes import SceneConfig


PROBE_REGISTRY: dict[str, type["Probe"]] = {}


def register_probe(name: str):
    def deco(cls: type["Probe"]) -> type["Probe"]:
        PROBE_REGISTRY[name] = cls
        cls.probe_name = name  # type: ignore[attr-defined]
        return cls
    return deco


@dataclass
class ProbeOutcome:
    """Result of running a probe episode."""

    steps: list[StepResult] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    success: bool = False
    failure_reason: str | None = None
    target_object_id: str | None = None
    target_receptacle_id: str | None = None

    def append(self, step: StepResult, phase: str) -> None:
        self.steps.append(step)
        self.phases.append(phase)


class Probe:
    """Base class for memory probes.

    Subclasses implement ``phase_a`` (set state, pre-gap) and ``phase_c``
    (query, post-gap). The base class drives the structure and runs the gap.
    """

    probe_name: str = "base"

    def __init__(
        self,
        env: ThorEnv,
        scene_cfg: SceneConfig,
        probe_cfg: ProbeConfig,
        exp_cfg: ExplorationConfig,
        rng: random.Random,
        gap_length: int,
    ) -> None:
        self.env = env
        self.scene_cfg = scene_cfg
        self.probe_cfg = probe_cfg
        self.exp_cfg = exp_cfg
        self.rng = rng
        self.gap_length = gap_length

    # -------------------------------------------------------- public entry pt
    def run(self, randomize_seed: int | None = None) -> ProbeOutcome:
        outcome = ProbeOutcome()
        last = self.env.reset(randomize_seed=randomize_seed)
        outcome.append(last, phase="A")

        try:
            last = self.phase_a(outcome, last)
            last = self.phase_b(outcome, last)
            last = self.phase_c(outcome, last)
            outcome.success = True
        except ProbeError as exc:
            outcome.failure_reason = str(exc)
            outcome.success = False
        return outcome

    # -------------------------------------------------------- phases (abstract)
    def phase_a(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        raise NotImplementedError

    def phase_c(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        raise NotImplementedError

    # ------------------------------------------------------------- distractor
    def phase_b(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        walker = WaypointWalker(self.env, self.exp_cfg, self.rng)
        for _ in range(self.gap_length):
            spec = walker.next_action(last)
            last = self.env.step(spec)
            outcome.append(last, phase="B")
        return last

    # ---------------------------------------------------------------- helpers
    def _step(self, outcome: ProbeOutcome, spec: ActionSpec, phase: str) -> StepResult:
        result = self.env.step(spec)
        outcome.append(result, phase=phase)
        return result

    def _teleport(
        self,
        outcome: ProbeOutcome,
        position: dict,
        rotation: float,
        horizon: float,
        phase: str,
    ) -> StepResult:
        result = self.env.teleport(position=position, rotation=rotation, horizon=horizon)
        outcome.append(result, phase=phase)
        if not result.action_success:
            raise ProbeError(f"Teleport failed: {result.error_message}")
        return result

    def _approach_and_teleport(
        self,
        outcome: ProbeOutcome,
        target_obj: dict,
        bearing_deg: float,
        phase: str,
    ) -> StepResult:
        """Try the requested bearing first, then jittered alternatives.

        AI2-THOR rejects teleports that would clip the agent's capsule into
        another collider; some chosen approach poses are physically invalid
        even though the underlying grid position is "reachable". We try a
        ring of bearings before giving up.
        """

        bearings = [
            bearing_deg,
            (bearing_deg + 45) % 360,
            (bearing_deg - 45) % 360,
            (bearing_deg + 90) % 360,
            (bearing_deg - 90) % 360,
            (bearing_deg + 135) % 360,
        ]
        last_err = "no candidates"
        for b in bearings:
            try:
                position, yaw, horizon = self._approach_pose(target_obj, b)
            except ProbeError as exc:
                last_err = str(exc)
                continue
            result = self.env.teleport(position=position, rotation=yaw, horizon=horizon)
            outcome.append(result, phase=phase)
            if result.action_success:
                return result
            last_err = result.error_message
        raise ProbeError(f"All approach poses failed for {target_obj.get('objectId')}: {last_err[:140]}")

    def _find_object(self, type_name: str) -> dict:
        objs = self.env.find_objects_by_type(type_name)
        if not objs:
            raise ProbeError(f"No object of type {type_name!r} in scene {self.env.scene}")
        return objs[0]

    def _approach_pose(
        self,
        target_obj: dict,
        bearing_deg: float,
        min_dist: float = 0.85,
        max_dist: float = 1.6,
    ) -> tuple[dict, float, float]:
        """Return (position, yaw, horizon) such that the agent looks at the
        object from the requested bearing (degrees clockwise from +z).

        ``min_dist`` of 0.85 keeps the agent's collider clear of the target;
        we previously saw teleport collisions at 0.6 against tall objects
        (Stove, tall Cabinet, etc.).
        """

        positions = self.env.reachable_positions()
        if positions.size == 0:
            raise ProbeError("No reachable positions in scene")

        tgt = target_obj["position"]
        tgt_xz = np.asarray([tgt["x"], tgt["z"]], dtype=np.float32)

        dx = positions[:, 0] - tgt_xz[0]
        dz = positions[:, 2] - tgt_xz[1]
        dist = np.sqrt(dx * dx + dz * dz)
        bearings = (np.degrees(np.arctan2(-dx, -dz)) + 360.0) % 360.0
        # bearings[i] = direction from object to position i.
        # We want positions roughly *opposite* to the look direction we want.

        target_bearing = bearing_deg % 360.0
        diff = np.abs(((bearings - target_bearing + 540.0) % 360.0) - 180.0)
        mask = (dist >= min_dist) & (dist <= max_dist)
        if not mask.any():
            mask = (dist >= 0.6) & (dist <= max_dist + 0.5)
        if not mask.any():
            raise ProbeError("No suitable approach pose for object")
        cand_idx = np.where(mask)[0]
        best = cand_idx[np.argmin(diff[cand_idx])]
        pos = positions[best]
        # Yaw points back toward the object.
        yaw = (math.degrees(math.atan2(tgt_xz[0] - pos[0], tgt_xz[1] - pos[2])) + 360.0) % 360.0
        # We deliberately keep the camera level (no pitch) so the entire
        # dataset has a consistent horizontal viewpoint distribution. Objects
        # above or below eye level may be partially clipped; for very low
        # receptacles, swap them in scenes.py to a taller alternative.
        horizon = 0.0
        return (
            {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            float(yaw),
            float(horizon),
        )

    def _walk_in(
        self,
        outcome: ProbeOutcome,
        last: StepResult,
        target_obj: dict,
        max_steps: int = 6,
        phase: str = "A",
    ) -> StepResult:
        """A few realistic forward steps so we don't only have a teleport frame."""

        for _ in range(max_steps):
            obj = self.env.find_object_by_id(target_obj["objectId"])
            if obj and obj.get("visible"):
                break
            last = self._step(outcome, ActionSpec("MoveAhead"), phase)
            if not last.action_success:
                last = self._step(
                    outcome,
                    ActionSpec(self.rng.choice(("RotateLeft", "RotateRight"))),
                    phase,
                )
        return last

    def _hold_pose(
        self,
        outcome: ProbeOutcome,
        last: StepResult,
        frames: int,
        phase: str,
    ) -> StepResult:
        """Duplicate the last frame ``frames`` times under the ``Done`` action.

        We do not actually advance the simulation; this gives the evaluator
        multiple frames at the query pose for stable metrics.
        """

        from dataclasses import replace

        from ..actions import ACTION_TO_ID

        for _ in range(frames):
            held = replace(
                last,
                action_name="Done",
                action_id=ACTION_TO_ID["Done"],
                action_object_id=None,
                action_success=True,
                error_message="",
            )
            outcome.append(held, phase=phase)
            last = held
        return last


class ProbeError(RuntimeError):
    """Raised when a probe cannot complete (missing target objects, etc.)."""
