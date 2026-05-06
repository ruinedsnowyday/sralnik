"""Per-scene metadata: probe targets and tracked-object whitelist hints.

Object IDs in AI2-THOR look like ``Apple|+00.96|+00.94|-00.42``. We match by the
*type prefix* (everything before the first ``|``) so the same config works
regardless of position suffixes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTargets:
    """Object types relevant for each probe in a given scene."""

    receptacle_target: str | None  # e.g. "Fridge"
    receptacle_item: str | None    # e.g. "Apple"
    fallback_receptacle: str | None  # e.g. "Cabinet" if Fridge missing
    toggleable: str | None         # e.g. "FloorLamp" / "DeskLamp"
    pickupable: str | None         # for displacement probe
    displacement_dest: str | None  # surface to drop the object on
    rearrangement_items: tuple[str, ...] = ()
    rearrangement_surface: str | None = None


@dataclass(frozen=True)
class SceneConfig:
    name: str
    tracked_types: tuple[str, ...]
    probes: ProbeTargets


SCENE_CONFIGS: dict[str, SceneConfig] = {
    "FloorPlan1": SceneConfig(
        name="FloorPlan1",
        tracked_types=(
            "Fridge",
            "Microwave",
            "Toaster",
            "CoffeeMachine",
            "StoveBurner",
            "Cabinet",
            "Drawer",
            "CounterTop",
            "Apple",
            "Bread",
            "Bowl",
            "Mug",
            "Plate",
            "Pot",
            "Pan",
            "Tomato",
            "Lettuce",
            "Knife",
            "Fork",
            "Spoon",
        ),
        probes=ProbeTargets(
            receptacle_target="Fridge",
            receptacle_item="Apple",
            fallback_receptacle="Microwave",
            toggleable="Microwave",
            pickupable="Mug",
            displacement_dest="CounterTop",
            rearrangement_items=("Apple", "Bread", "Mug"),
            rearrangement_surface="CounterTop",
        ),
    ),
    "FloorPlan403": SceneConfig(
        name="FloorPlan403",
        tracked_types=(
            "Toilet",
            "ToiletPaper",
            "ToiletPaperHanger",
            "SinkBasin",
            "Sink",
            "Faucet",
            "Mirror",
            "Cabinet",
            "Drawer",
            "CounterTop",
            "TowelHolder",
            "Towel",
            "HandTowelHolder",
            "HandTowel",
            "ShowerHead",
            "ShowerCurtain",
            "ShowerDoor",
            "ShowerGlass",
            "SoapBar",
            "SoapBottle",
            "Candle",
            "SprayBottle",
            "TissueBox",
            "ScrubBrush",
            "Plunger",
            "Cloth",
            "DishSponge",
            "LightSwitch",
        ),
        probes=ProbeTargets(
            # Bathroom cabinet (under the sink / mirror cabinet) is openable +
            # receptacle in iTHOR. Drop a SoapBar inside as the probe item.
            receptacle_target="Cabinet",
            receptacle_item="SoapBar",
            fallback_receptacle="Drawer",
            toggleable="Faucet",
            pickupable="SoapBar",
            displacement_dest="CounterTop",
            rearrangement_items=("SoapBar", "Candle", "SprayBottle"),
            rearrangement_surface="CounterTop",
        ),
    ),
}


def get_scene_config(scene: str) -> SceneConfig:
    if scene not in SCENE_CONFIGS:
        raise KeyError(
            f"No scene config registered for {scene!r}. "
            f"Known scenes: {sorted(SCENE_CONFIGS)}"
        )
    return SCENE_CONFIGS[scene]
