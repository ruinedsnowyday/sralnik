"""Probe #3: object displacement.

Phase A: pick up an object from its original location, carry it across the
room, and place it on a different surface. Phase C: revisit both the original
location (item should be gone) and the destination surface (item should be
there).
"""

from __future__ import annotations

from ..actions import ActionSpec
from ..controller import StepResult
from .base import Probe, ProbeError, ProbeOutcome, register_probe


@register_probe("displacement")
class ObjectDisplacementProbe(Probe):
    def phase_a(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        item_type = self.scene_cfg.probes.pickupable
        dest_type = self.scene_cfg.probes.displacement_dest
        if item_type is None or dest_type is None:
            raise ProbeError("Scene has no displacement targets configured")

        item = self._find_object(item_type)
        dest = self._find_object(dest_type)
        outcome.target_object_id = item["objectId"]
        outcome.target_receptacle_id = dest["objectId"]

        # Approach item, pick it up.
        last = self._approach_and_teleport(outcome, item, bearing_deg=0.0, phase="A")
        last = self._walk_in(outcome, last, item, max_steps=4, phase="A")
        last = self._step(outcome, ActionSpec("PickupObject", object_id=item["objectId"]), "A")

        # Carry to destination.
        last = self._approach_and_teleport(outcome, dest, bearing_deg=0.0, phase="A")
        last = self._walk_in(outcome, last, dest, max_steps=4, phase="A")
        last = self._step(
            outcome,
            ActionSpec("PutObject", receptacle_id=dest["objectId"], object_id=item["objectId"]),
            "A",
        )
        last = self._hold_pose(outcome, last, frames=2, phase="A")
        return last

    def phase_c(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        dest_id = outcome.target_receptacle_id
        if dest_id is None:
            raise ProbeError("Phase A did not record destination")
        dest = self.env.find_object_by_id(dest_id)
        if dest is None:
            raise ProbeError("Destination missing at phase C")

        last = self._approach_and_teleport(outcome, dest, bearing_deg=180.0, phase="C")
        last = self._walk_in(outcome, last, dest, max_steps=4, phase="C")
        last = self._hold_pose(outcome, last, frames=self.probe_cfg.hold_frames_at_query, phase="C")
        return last
