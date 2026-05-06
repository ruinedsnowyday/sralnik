"""Random walker that navigates toward sampled reachable positions.

Strategy:
1. Sample a target from ``GetReachablePositions`` (optionally filtered by an
   exclusion radius around the agent so we actually move).
2. Greedily align the agent's heading toward the target and step forward.
3. If a forward step fails (collision/blocked), rotate randomly and retry.
4. Re-sample a new target every ``waypoint_replan_every`` steps, or when within
   one grid cell of the current target.
5. Every ``interaction_attempt_every`` steps, attempt a random open/toggle on a
   visible interactable object so the renderer sees state changes too.
"""

from __future__ import annotations

import math
import random

import numpy as np

from ..actions import ActionSpec
from ..config import ExplorationConfig
from ..controller import StepResult, ThorEnv


_INTERACTABLE_FILTERS: tuple[tuple[str, str], ...] = (
    ("openable", "OpenObject"),
    ("toggleable", "ToggleObjectOn"),
)


class WaypointWalker:
    def __init__(self, env: ThorEnv, cfg: ExplorationConfig, rng: random.Random):
        self._env = env
        self._cfg = cfg
        self._rng = rng
        self._target: np.ndarray | None = None
        self._steps_since_replan = 0
        self._steps_since_interaction = 0

    # ------------------------------------------------------------------ main
    def next_action(self, last: StepResult) -> ActionSpec:
        self._steps_since_interaction += 1
        if self._steps_since_interaction >= self._cfg.interaction_attempt_every:
            interaction = self._maybe_interact()
            if interaction is not None:
                self._steps_since_interaction = 0
                return interaction

        if self._needs_new_target(last):
            self._sample_target(last)
            self._steps_since_replan = 0

        self._steps_since_replan += 1
        return self._navigation_step(last)

    # ---------------------------------------------------------------- target
    def _needs_new_target(self, last: StepResult) -> bool:
        if self._target is None:
            return True
        if self._steps_since_replan >= self._cfg.waypoint_replan_every:
            return True
        pos = last.pose[:3]
        dist = float(np.linalg.norm(pos[[0, 2]] - self._target[[0, 2]]))
        return dist < 0.4

    def _sample_target(self, last: StepResult) -> None:
        positions = self._env.reachable_positions()
        if positions.size == 0:
            self._target = None
            return
        agent_xz = last.pose[[0, 2]]
        dists = np.linalg.norm(positions[:, [0, 2]] - agent_xz, axis=1)
        far = positions[dists > 1.0]
        pool = far if len(far) > 0 else positions
        idx = self._rng.randrange(len(pool))
        self._target = pool[idx].astype(np.float32)

    # ------------------------------------------------------------- navigation
    def _navigation_step(self, last: StepResult) -> ActionSpec:
        if self._target is None:
            return ActionSpec(self._rng.choice(("RotateLeft", "RotateRight")))

        if not last.action_success and last.action_name == "MoveAhead":
            return ActionSpec(self._rng.choice(("RotateLeft", "RotateRight")))

        x, _, z, yaw, _ = last.pose
        dx = self._target[0] - x
        dz = self._target[2] - z
        # AI2-THOR yaw=0 faces +z, increases clockwise (left-handed in 2D).
        target_yaw = (math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0
        cur_yaw = yaw % 360.0
        diff = (target_yaw - cur_yaw + 540.0) % 360.0 - 180.0  # in [-180, 180]

        if abs(diff) > self._env._cfg.rotate_step_degrees / 2.0:
            return ActionSpec("RotateLeft" if diff < 0 else "RotateRight")
        return ActionSpec("MoveAhead")

    # ------------------------------------------------------------- interaction
    def _maybe_interact(self) -> ActionSpec | None:
        ev = self._env.last_event
        objects = ev.metadata.get("objects", [])
        candidates: list[ActionSpec] = []

        for obj in objects:
            if not obj.get("visible"):
                continue
            for flag, action in _INTERACTABLE_FILTERS:
                if not obj.get(flag):
                    continue
                if action == "OpenObject" and obj.get("isOpen"):
                    candidates.append(ActionSpec("CloseObject", object_id=obj["objectId"]))
                elif action == "OpenObject":
                    candidates.append(ActionSpec("OpenObject", object_id=obj["objectId"]))
                elif action == "ToggleObjectOn" and obj.get("isToggled"):
                    candidates.append(ActionSpec("ToggleObjectOff", object_id=obj["objectId"]))
                elif action == "ToggleObjectOn":
                    candidates.append(ActionSpec("ToggleObjectOn", object_id=obj["objectId"]))

        if not candidates:
            return None
        return self._rng.choice(candidates)
