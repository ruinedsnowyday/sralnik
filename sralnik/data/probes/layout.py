"""Probe #4: layout consistency from a new viewpoint.

No state changes -- this is a pure geometric memory test. Phase A does a slow
360 pan from a chosen anchor pose. Phase C teleports to a *different* anchor on
the opposite side of the room and looks back, so the same objects should be
visible from a new angle.
"""

from __future__ import annotations

import math

import numpy as np

from ..actions import ActionSpec
from ..controller import StepResult
from .base import Probe, ProbeError, ProbeOutcome, register_probe


def _pick_anchor(positions: np.ndarray, *, bias: str, rng) -> np.ndarray:
    if positions.size == 0:
        raise ProbeError("No reachable positions for layout probe")
    centroid = positions.mean(axis=0)
    if bias == "near":
        diffs = positions - centroid
        dists = np.linalg.norm(diffs[:, [0, 2]], axis=1)
        order = np.argsort(dists)
        idx = order[rng.randrange(min(5, len(order)))]
    else:
        diffs = positions - centroid
        dists = np.linalg.norm(diffs[:, [0, 2]], axis=1)
        order = np.argsort(-dists)
        idx = order[rng.randrange(min(5, len(order)))]
    return positions[idx]


@register_probe("layout")
class LayoutConsistencyProbe(Probe):
    def phase_a(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        positions = self.env.reachable_positions()
        anchor = _pick_anchor(positions, bias="far", rng=self.rng)
        yaw = float(self.rng.uniform(0.0, 360.0))
        last = self._teleport(
            outcome,
            position={"x": float(anchor[0]), "y": float(anchor[1]), "z": float(anchor[2])},
            rotation=yaw,
            horizon=0.0,
            phase="A",
        )
        # 12 RotateRight steps = 360 degrees at 30° per rotation.
        rot_step = self.env._cfg.rotate_step_degrees
        n = max(4, int(round(360.0 / rot_step)))
        for _ in range(n):
            last = self._step(outcome, ActionSpec("RotateRight"), "A")
        outcome.target_object_id = None  # no specific object
        outcome.target_receptacle_id = None
        # Cache the anchor so phase_c can pick the opposite side.
        self._anchor_a = anchor
        return last

    def phase_c(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        positions = self.env.reachable_positions()
        if positions.size == 0:
            raise ProbeError("No reachable positions for layout probe phase C")

        centroid = positions.mean(axis=0)
        # Pick the position whose vector from centroid is closest to opposite of anchor_a's.
        anchor_a = getattr(self, "_anchor_a", centroid)
        v = anchor_a - centroid
        angle_a = math.atan2(v[0], v[2])
        target_angle = angle_a + math.pi
        rel = positions - centroid
        rel_angles = np.arctan2(rel[:, 0], rel[:, 2])
        diff = np.abs(((rel_angles - target_angle + 3 * math.pi) % (2 * math.pi)) - math.pi)
        # Prefer points far from centroid AND close to target angle.
        dist = np.linalg.norm(rel[:, [0, 2]], axis=1)
        score = diff - 0.3 * dist
        anchor = positions[int(np.argmin(score))]
        # Look back roughly toward Phase A anchor so we re-observe the room.
        target_xz = anchor_a[[0, 2]]
        yaw = (math.degrees(math.atan2(target_xz[0] - anchor[0], target_xz[1] - anchor[2])) + 360.0) % 360.0
        last = self._teleport(
            outcome,
            position={"x": float(anchor[0]), "y": float(anchor[1]), "z": float(anchor[2])},
            rotation=float(yaw),
            horizon=0.0,
            phase="C",
        )
        last = self._hold_pose(outcome, last, frames=self.probe_cfg.hold_frames_at_query, phase="C")
        return last
