"""Thin wrapper around ai2thor.controller.Controller.

Handles platform autodetection and exposes a typed ``StepResult`` so the rest of
the code base does not deal with raw THOR ``Event`` objects directly.
"""

from __future__ import annotations

import os
import platform as py_platform
from dataclasses import dataclass
from typing import Any

import numpy as np

from .actions import ActionSpec, ACTION_TO_ID
from .config import ControllerConfig


@dataclass
class StepResult:
    """Single environment step record."""

    rgb: np.ndarray  # (H, W, 3) uint8
    instance_seg: np.ndarray | None  # (H, W) uint16 or None
    depth: np.ndarray | None  # (H, W) float16 or None
    pose: np.ndarray  # (5,) float32: x, y, z, yaw, pitch
    action_name: str
    action_id: int
    action_object_id: str | None
    action_success: bool
    error_message: str
    tracked_object_states: dict[str, dict[str, Any]]


def _autodetect_platform() -> str | None:
    """Pick a sensible AI2-THOR Unity platform string.

    On macOS we let ai2thor decide (native build).
    On Linux with $DISPLAY we use the standard build, otherwise CloudRendering.
    """

    system = py_platform.system()
    if system == "Darwin":
        return None
    if system == "Linux":
        if os.environ.get("DISPLAY"):
            return None
        return "CloudRendering"
    return None


def _extract_pose(metadata: dict) -> np.ndarray:
    a = metadata["agent"]
    pos = a["position"]
    rot = a["rotation"]
    horizon = a.get("cameraHorizon", 0.0)
    return np.asarray(
        [pos["x"], pos["y"], pos["z"], rot["y"], horizon], dtype=np.float32
    )


def _filter_objects(metadata: dict, tracked_types: tuple[str, ...]) -> dict:
    """Keep only objects whose type is in the whitelist."""

    types = set(tracked_types)
    out: dict[str, dict[str, Any]] = {}
    for obj in metadata.get("objects", []):
        otype = obj.get("objectType", "")
        if otype not in types:
            continue
        out[obj["objectId"]] = {
            "type": otype,
            "position": obj.get("position"),
            "rotation": obj.get("rotation"),
            "isOpen": obj.get("isOpen"),
            "isToggled": obj.get("isToggled"),
            "isPickedUp": obj.get("isPickedUp"),
            "parentReceptacles": obj.get("parentReceptacles"),
            "receptacleObjectIds": obj.get("receptacleObjectIds"),
            "visible": obj.get("visible"),
        }
    return out


class ThorEnv:
    """Lightweight wrapper around AI2-THOR's Controller."""

    def __init__(self, cfg: ControllerConfig, scene: str, tracked_types: tuple[str, ...]):
        from ai2thor.controller import Controller  # local import: heavy dep

        self._cfg = cfg
        self._scene = scene
        self._tracked_types = tracked_types
        kwargs: dict[str, Any] = dict(
            scene=scene,
            width=cfg.width,
            height=cfg.height,
            gridSize=cfg.grid_size,
            rotateStepDegrees=cfg.rotate_step_degrees,
            visibilityDistance=cfg.visibility_distance,
            fieldOfView=cfg.field_of_view,
            snapToGrid=cfg.snap_to_grid,
            renderDepthImage=cfg.render_depth,
            renderInstanceSegmentation=cfg.render_instance_segmentation,
            agentMode=cfg.agent_mode,
            server_timeout=cfg.server_timeout,
            server_start_timeout=cfg.server_start_timeout,
        )
        plat = cfg.platform or _autodetect_platform()
        if plat is not None:
            kwargs["platform"] = plat
        self._controller = Controller(**kwargs)
        self._last_event = self._controller.last_event

    @property
    def scene(self) -> str:
        return self._scene

    @property
    def last_event(self):
        return self._last_event

    def reset(self, scene: str | None = None, randomize_seed: int | None = None) -> StepResult:
        target = scene or self._scene
        self._controller.reset(scene=target)
        self._scene = target
        if randomize_seed is not None:
            # InitialRandomSpawn shuffles object placements among physically
            # valid spots (keeping object types/quantities). Per-episode seed
            # means each episode sees a slightly different scene state.
            ev = self._controller.step(
                action="InitialRandomSpawn",
                randomSeed=int(randomize_seed) & 0x7FFFFFFF,
                forceVisible=False,
                numPlacementAttempts=5,
                placeStationary=True,
            )
            if not ev.metadata.get("lastActionSuccess", True):
                # Non-fatal: keep the deterministic layout if randomisation failed.
                pass
        self._last_event = self._controller.last_event
        return self._wrap("Done", None, success=True, error="")

    def reachable_positions(self) -> np.ndarray:
        ev = self._controller.step(action="GetReachablePositions")
        positions = ev.metadata.get("actionReturn") or []
        if not positions:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(
            [[p["x"], p["y"], p["z"]] for p in positions], dtype=np.float32
        )

    def teleport(
        self,
        position: dict | None = None,
        rotation: float | None = None,
        horizon: float | None = None,
    ) -> StepResult:
        kwargs: dict[str, Any] = {"action": "Teleport"}
        if position is not None:
            kwargs["position"] = position
        if rotation is not None:
            kwargs["rotation"] = {"x": 0, "y": rotation, "z": 0}
        if horizon is not None:
            kwargs["horizon"] = horizon
        ev = self._controller.step(**kwargs)
        self._last_event = ev
        return self._wrap("Teleport", None, success=ev.metadata["lastActionSuccess"],
                          error=ev.metadata.get("errorMessage", ""))

    def step(self, spec: ActionSpec) -> StepResult:
        ev = self._controller.step(**spec.to_thor_kwargs())
        self._last_event = ev
        return self._wrap(
            spec.name,
            spec.object_id or spec.receptacle_id,
            success=ev.metadata["lastActionSuccess"],
            error=ev.metadata.get("errorMessage", ""),
        )

    def stop(self) -> None:
        try:
            self._controller.stop()
        except Exception:  # noqa: BLE001 -- THOR sometimes raises on stop
            pass

    # ------------------------------------------------------------------ utils
    def _wrap(self, action_name: str, action_object_id: str | None,
              success: bool, error: str) -> StepResult:
        ev = self._last_event
        rgb = np.asarray(ev.frame, dtype=np.uint8)

        seg: np.ndarray | None = None
        if self._cfg.render_instance_segmentation:
            raw_seg = getattr(ev, "instance_segmentation_frame", None)
            if raw_seg is not None:
                # THOR returns RGB-coded ids; pack to a single 24-bit id per pixel.
                arr = np.asarray(raw_seg, dtype=np.uint32)
                packed = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]
                # Most scenes have <65k instances; compress to uint16 modulo overflow.
                seg = (packed & 0xFFFF).astype(np.uint16)

        depth: np.ndarray | None = None
        if self._cfg.render_depth:
            d = getattr(ev, "depth_frame", None)
            if d is not None:
                # AI2-THOR returns float32 metres. "Sky" / out-of-range pixels
                # come back as very large values (and sometimes +inf), which
                # overflow float16 (max ~65504). Clip to a sensible indoor
                # max of 20 m before casting -- anything farther is useless
                # geometry for our scenes anyway.
                d32 = np.asarray(d, dtype=np.float32)
                d32 = np.nan_to_num(d32, nan=20.0, posinf=20.0, neginf=0.0)
                np.clip(d32, 0.0, 20.0, out=d32)
                depth = d32.astype(np.float16)

        pose = _extract_pose(ev.metadata)
        tracked = _filter_objects(ev.metadata, self._tracked_types)
        return StepResult(
            rgb=rgb,
            instance_seg=seg,
            depth=depth,
            pose=pose,
            action_name=action_name,
            action_id=ACTION_TO_ID[action_name],
            action_object_id=action_object_id,
            action_success=success,
            error_message=error or "",
            tracked_object_states=tracked,
        )

    # ---------------------------------------------------------------- helpers
    def find_objects_by_type(self, object_type: str) -> list[dict]:
        return [
            o for o in self._last_event.metadata.get("objects", [])
            if o.get("objectType") == object_type
        ]

    def find_object_by_id(self, object_id: str) -> dict | None:
        for o in self._last_event.metadata.get("objects", []):
            if o.get("objectId") == object_id:
                return o
        return None
