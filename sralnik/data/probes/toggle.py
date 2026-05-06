"""Probe #2: toggle persistence (lamp / microwave on or off after a gap)."""

from __future__ import annotations

from ..actions import ActionSpec
from ..controller import StepResult
from .base import Probe, ProbeError, ProbeOutcome, register_probe


@register_probe("toggle")
class TogglePersistenceProbe(Probe):
    def phase_a(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        target_type = self.scene_cfg.probes.toggleable
        if target_type is None:
            raise ProbeError("Scene has no toggleable target configured")
        obj = self._find_object(target_type)
        outcome.target_object_id = obj["objectId"]

        last = self._approach_and_teleport(outcome, obj, bearing_deg=0.0, phase="A")
        last = self._walk_in(outcome, last, obj, max_steps=4, phase="A")

        if obj.get("isToggled"):
            last = self._step(outcome, ActionSpec("ToggleObjectOff", object_id=obj["objectId"]), "A")
        last = self._step(outcome, ActionSpec("ToggleObjectOn", object_id=obj["objectId"]), "A")
        last = self._hold_pose(outcome, last, frames=2, phase="A")
        return last

    def phase_c(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        obj_id = outcome.target_object_id
        if obj_id is None:
            raise ProbeError("Phase A did not record toggle target")
        obj = self.env.find_object_by_id(obj_id)
        if obj is None:
            raise ProbeError("Toggle target missing at phase C")

        last = self._approach_and_teleport(outcome, obj, bearing_deg=180.0, phase="C")
        last = self._walk_in(outcome, last, obj, max_steps=4, phase="C")
        last = self._hold_pose(outcome, last, frames=self.probe_cfg.hold_frames_at_query, phase="C")
        return last
