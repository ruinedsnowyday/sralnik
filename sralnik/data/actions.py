"""Discrete action space mapping to AI2-THOR controller calls."""

from __future__ import annotations

from dataclasses import dataclass

# Movement actions (no arguments).
MOVEMENT_ACTIONS: tuple[str, ...] = (
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
)

# Interaction actions take an objectId argument.
INTERACTION_ACTIONS: tuple[str, ...] = (
    "OpenObject",
    "CloseObject",
    "PickupObject",
    "PutObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
)

# Special bookkeeping actions written into action logs but not sent to THOR.
BOOKKEEPING_ACTIONS: tuple[str, ...] = (
    "Done",        # explicit no-op, e.g. holding pose at query
    "Teleport",    # logged when we hard-reset the agent pose
)

ACTION_NAMES: tuple[str, ...] = (
    MOVEMENT_ACTIONS + INTERACTION_ACTIONS + BOOKKEEPING_ACTIONS
)
ACTION_TO_ID: dict[str, int] = {name: i for i, name in enumerate(ACTION_NAMES)}
ID_TO_ACTION: dict[int, str] = {i: name for i, name in enumerate(ACTION_NAMES)}


@dataclass(frozen=True)
class ActionSpec:
    """Concrete action issued by a policy/probe at a single step."""

    name: str
    object_id: str | None = None
    receptacle_id: str | None = None  # for PutObject

    def to_thor_kwargs(self) -> dict:
        if self.name in MOVEMENT_ACTIONS:
            return {"action": self.name}
        if self.name == "PutObject":
            assert self.receptacle_id is not None, "PutObject needs receptacle"
            return {
                "action": "PutObject",
                "objectId": self.receptacle_id,
                "forceAction": True,
            }
        if self.name in INTERACTION_ACTIONS:
            assert self.object_id is not None, f"{self.name} needs objectId"
            return {
                "action": self.name,
                "objectId": self.object_id,
                "forceAction": True,
            }
        raise ValueError(f"Action {self.name!r} is bookkeeping, do not send to THOR.")
