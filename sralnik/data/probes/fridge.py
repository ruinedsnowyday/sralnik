"""Probe #1: receptacle memory ("fridge test").

Phase A: agent approaches a receptacle, opens it, places an item inside, closes
it. Phase B: distractor walking. Phase C: agent re-approaches from a *different*
bearing and opens the receptacle again. The model must remember that the item
is inside.
"""

from __future__ import annotations

from ..actions import ActionSpec
from ..controller import StepResult
from .base import Probe, ProbeError, ProbeOutcome, register_probe


@register_probe("fridge")
class FridgeProbe(Probe):
    def phase_a(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        targets = self.scene_cfg.probes
        recep_type = targets.receptacle_target or targets.fallback_receptacle
        item_type = targets.receptacle_item
        if recep_type is None or item_type is None:
            raise ProbeError("Scene has no configured receptacle/item targets")

        recep = self._find_object(recep_type)
        item = self._find_object(item_type)
        outcome.target_object_id = item["objectId"]
        outcome.target_receptacle_id = recep["objectId"]

        last = self._approach_and_teleport(outcome, recep, bearing_deg=0.0, phase="A")
        last = self._walk_in(outcome, last, recep, max_steps=4, phase="A")

        last = self._step(outcome, ActionSpec("OpenObject", object_id=recep["objectId"]), "A")
        last = self._step(outcome, ActionSpec("PickupObject", object_id=item["objectId"]), "A")
        last = self._step(
            outcome,
            ActionSpec("PutObject", receptacle_id=recep["objectId"], object_id=item["objectId"]),
            "A",
        )
        last = self._step(outcome, ActionSpec("CloseObject", object_id=recep["objectId"]), "A")
        last = self._hold_pose(outcome, last, frames=2, phase="A")
        return last

    def phase_c(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        recep_id = outcome.target_receptacle_id
        if recep_id is None:
            raise ProbeError("Phase A did not record receptacle id")
        recep = self.env.find_object_by_id(recep_id)
        if recep is None:
            raise ProbeError("Receptacle no longer present at phase C")

        # Approach from the opposite side this time.
        last = self._approach_and_teleport(outcome, recep, bearing_deg=180.0, phase="C")
        last = self._walk_in(outcome, last, recep, max_steps=4, phase="C")
        last = self._step(outcome, ActionSpec("OpenObject", object_id=recep_id), "C")
        last = self._hold_pose(outcome, last, frames=self.probe_cfg.hold_frames_at_query, phase="C")
        return last
