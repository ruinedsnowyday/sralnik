"""Probe #5: multi-step rearrangement.

Phase A: move 2-3 objects between surfaces. Phase C: revisit and check that
the new arrangement is preserved.
"""

from __future__ import annotations

import math

import numpy as np

from ..actions import ActionSpec
from ..controller import StepResult
from .base import Probe, ProbeError, ProbeOutcome, register_probe


@register_probe("rearrangement")
class RearrangementProbe(Probe):
    def phase_a(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        items = self.scene_cfg.probes.rearrangement_items
        surface_type = self.scene_cfg.probes.rearrangement_surface
        if not items or surface_type is None:
            raise ProbeError("Scene has no rearrangement targets configured")

        surface = self._find_object(surface_type)
        outcome.target_receptacle_id = surface["objectId"]
        moved_ids: list[str] = []

        for item_type in items[:3]:
            try:
                item = self._find_object(item_type)
            except ProbeError:
                continue
            try:
                last = self._approach_and_teleport(outcome, item, bearing_deg=0.0, phase="A")
            except ProbeError:
                continue
            last = self._walk_in(outcome, last, item, max_steps=3, phase="A")
            pickup = self._step(
                outcome, ActionSpec("PickupObject", object_id=item["objectId"]), "A"
            )
            if not pickup.action_success:
                continue

            try:
                last = self._approach_and_teleport(outcome, surface, bearing_deg=0.0, phase="A")
            except ProbeError:
                continue
            last = self._walk_in(outcome, last, surface, max_steps=3, phase="A")
            put = self._step(
                outcome,
                ActionSpec(
                    "PutObject",
                    receptacle_id=surface["objectId"],
                    object_id=item["objectId"],
                ),
                "A",
            )
            if put.action_success:
                moved_ids.append(item["objectId"])

        if not moved_ids:
            raise ProbeError("Failed to rearrange any items")
        outcome.target_object_id = ",".join(moved_ids)
        last = self._hold_pose(outcome, last, frames=2, phase="A")
        return last

    def phase_c(self, outcome: ProbeOutcome, last: StepResult) -> StepResult:
        surface_id = outcome.target_receptacle_id
        if surface_id is None:
            raise ProbeError("Phase A did not record surface")
        surface = self.env.find_object_by_id(surface_id)
        if surface is None:
            raise ProbeError("Surface missing at phase C")

        last = self._approach_and_teleport(outcome, surface, bearing_deg=180.0, phase="C")
        last = self._walk_in(outcome, last, surface, max_steps=4, phase="C")
        last = self._hold_pose(outcome, last, frames=self.probe_cfg.hold_frames_at_query, phase="C")
        return last
