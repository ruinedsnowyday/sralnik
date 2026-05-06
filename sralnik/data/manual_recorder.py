"""Keyboard-driven manual trajectory recorder.

Spins up an AI2-THOR controller, shows the current frame in a pygame window,
and lets the user drive the agent with the keyboard. Each step is captured as
the same ``StepResult`` shape used by scripted probes, so we can save with the
same ``EpisodeWriter`` and append to the same parquet manifest.

Controls:

* W / S        — MoveAhead / MoveBack
* A / D        — RotateLeft / RotateRight
* Q / E        — MoveLeft / MoveRight (strafe)
* R / F        — LookUp / LookDown
* Space        — Done (hold pose, no THOR step)
* 1..9 / 0     — select visible interactable shown in side panel (0 = clear)
* O / C        — OpenObject / CloseObject (on selection)
* P            — PickupObject (on selection)
* V            — PutObject (selection = receptacle, held object goes in)
* T / Y        — ToggleObjectOn / ToggleObjectOff (on selection)
* TAB          — cycle phase A -> B -> C
* Z            — undo last logged step (does NOT roll back THOR state)
* Enter        — save current episode, start a fresh one (same scene)
* Backspace    — discard current episode, start a fresh one
* Esc          — save (if non-empty) and quit

The recorder writes to ``<output>/episodes/<scene>/manual/<episode_id>.h5`` and
appends one row per saved episode to ``<output>/manifest.parquet`` (creating it
if absent).
"""

from __future__ import annotations

import datetime as _dt
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .actions import ACTION_TO_ID, ActionSpec, MOVEMENT_ACTIONS
from .config import ControllerConfig
from .controller import StepResult, ThorEnv
from .scenes import get_scene_config
from .storage import EpisodeRecord, EpisodeWriter
from .storage.manifest import ManifestBuilder


_PHASES: tuple[str, ...] = ("A", "B", "C")
_KEY_TO_MOVEMENT: dict[str, str] = {
    "w": "MoveAhead",
    "s": "MoveBack",
    "a": "RotateLeft",
    "d": "RotateRight",
    "q": "MoveLeft",
    "e": "MoveRight",
    "r": "LookUp",
    "f": "LookDown",
}


def _is_interactable(obj: dict) -> bool:
    return bool(
        obj.get("openable")
        or obj.get("toggleable")
        or obj.get("pickupable")
        or obj.get("receptacle")
    )


def _gather_visible_interactables(metadata: dict, limit: int = 10) -> list[dict]:
    out = []
    for o in metadata.get("objects", []):
        if not o.get("visible"):
            continue
        if not _is_interactable(o):
            continue
        out.append(o)
        if len(out) >= limit:
            break
    return out


def _episode_id(scene: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{scene}_manual_{stamp}"


def _load_manifest(root: Path) -> ManifestBuilder:
    """Create a fresh ManifestBuilder, prefilled with any existing rows."""

    mb = ManifestBuilder(root)
    existing = root / "manifest.parquet"
    if existing.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(existing)
            for _, row in df.iterrows():
                mb.add(EpisodeRecord(
                    episode_id=row["episode_id"],
                    scene=row["scene"],
                    episode_type=row["episode_type"],
                    probe_name=row.get("probe_name") if not _is_nan(row.get("probe_name")) else None,
                    gap_length=int(row["gap_length"]) if not _is_nan(row.get("gap_length")) else None,
                    seed=int(row["seed"]) if not _is_nan(row.get("seed")) else 0,
                    split=row["split"],
                    num_steps=int(row["num_steps"]),
                    success=bool(row["success"]),
                    failure_reason=row.get("failure_reason") if not _is_nan(row.get("failure_reason")) else None,
                    target_object_id=row.get("target_object_id") if not _is_nan(row.get("target_object_id")) else None,
                    target_receptacle_id=row.get("target_receptacle_id") if not _is_nan(row.get("target_receptacle_id")) else None,
                    relative_path=row["relative_path"],
                ))
        except Exception as exc:  # noqa: BLE001
            print(f"[recorder] warning: could not read existing manifest ({exc}), starting fresh.")
    return mb


def _is_nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and x != x)
    except Exception:  # noqa: BLE001
        return False


def run_recorder(
    output_dir: Path,
    scene: str,
    width: int = 128,
    height: int = 128,
    display_size: int = 768,
    split: str = "expert_eval",
    note: str | None = None,
) -> None:
    """Top-level recorder loop. Blocks until the user quits."""

    import pygame  # local import: heavy GUI dep

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(output_dir)

    scene_cfg = get_scene_config(scene)
    controller_cfg = ControllerConfig(
        width=width,
        height=height,
        render_instance_segmentation=True,
    )
    env = ThorEnv(controller_cfg, scene, scene_cfg.tracked_types)

    pygame.init()
    pygame.display.set_caption(f"sralnik recorder — {scene}")
    panel_w = 460
    win = pygame.display.set_mode((display_size + panel_w, display_size))
    font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 18, bold=True)

    try:
        _episode_loop(
            env=env,
            scene=scene,
            output_dir=output_dir,
            manifest=manifest,
            split=split,
            display_size=display_size,
            panel_w=panel_w,
            win=win,
            font=font,
            big_font=big_font,
            pygame=pygame,
            note=note,
        )
    finally:
        env.stop()
        pygame.quit()


def _episode_loop(
    *,
    env: ThorEnv,
    scene: str,
    output_dir: Path,
    manifest: ManifestBuilder,
    split: str,
    display_size: int,
    panel_w: int,
    win,
    font,
    big_font,
    pygame,
    note: str | None,
) -> None:
    running = True
    while running:
        ep_id = _episode_id(scene)
        steps: list[StepResult] = []
        phases: list[str] = []
        phase_idx = 0  # 0=A, 1=B, 2=C
        selected_obj: str | None = None
        target_object_id: str | None = None
        target_receptacle_id: str | None = None
        last = env.reset()
        steps.append(last)
        phases.append(_PHASES[phase_idx])
        message = f"new episode {ep_id} — phase A"

        while True:
            visible = _gather_visible_interactables(env.last_event.metadata)
            inventory = env.last_event.metadata.get("inventoryObjects", []) or []
            held_id = inventory[0]["objectId"] if inventory else None

            _draw(
                win=win,
                pygame=pygame,
                font=font,
                big_font=big_font,
                last=last,
                display_size=display_size,
                panel_w=panel_w,
                scene=scene,
                episode_id=ep_id,
                step_count=len(steps),
                phase=_PHASES[phase_idx],
                visible=visible,
                selected_obj=selected_obj,
                held_id=held_id,
                message=message,
                note=note,
            )
            pygame.display.flip()

            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                _maybe_save(steps, phases, ep_id, scene, output_dir, manifest,
                            split, target_object_id, target_receptacle_id, note)
                running = False
                return
            if event.type != pygame.KEYDOWN:
                continue

            key = event.key
            mods = event.mod
            if key == pygame.K_ESCAPE:
                _maybe_save(steps, phases, ep_id, scene, output_dir, manifest,
                            split, target_object_id, target_receptacle_id, note)
                manifest.write()
                return
            if key == pygame.K_RETURN:
                _maybe_save(steps, phases, ep_id, scene, output_dir, manifest,
                            split, target_object_id, target_receptacle_id, note)
                manifest.write()
                break  # break inner -> start new episode
            if key == pygame.K_BACKSPACE:
                message = f"discarded {ep_id}"
                break

            if key == pygame.K_TAB:
                phase_idx = (phase_idx + 1) % len(_PHASES)
                message = f"phase -> {_PHASES[phase_idx]}"
                continue

            if key == pygame.K_SPACE:
                # Hold pose: duplicate last frame, no THOR step.
                from dataclasses import replace
                held = replace(
                    last,
                    action_name="Done",
                    action_id=ACTION_TO_ID["Done"],
                    action_object_id=None,
                    action_success=True,
                    error_message="",
                )
                steps.append(held)
                phases.append(_PHASES[phase_idx])
                last = held
                message = "hold pose"
                continue

            if key == pygame.K_z:
                if len(steps) > 1:
                    steps.pop()
                    phases.pop()
                    last = steps[-1]
                    message = f"undo (back to step {len(steps)})"
                continue

            # Object selection by digit row.
            digit_keys = {
                pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
                pygame.K_5: 4, pygame.K_6: 5, pygame.K_7: 6, pygame.K_8: 7,
                pygame.K_9: 8, pygame.K_0: 9,
            }
            if key in digit_keys:
                idx = digit_keys[key]
                if idx < len(visible):
                    selected_obj = visible[idx]["objectId"]
                    message = f"selected {selected_obj}"
                else:
                    selected_obj = None
                    message = "selection cleared"
                continue

            # Movement keys.
            unicode = event.unicode.lower() if event.unicode else ""
            if unicode in _KEY_TO_MOVEMENT:
                action_name = _KEY_TO_MOVEMENT[unicode]
                spec = ActionSpec(action_name)
                last = env.step(spec)
                steps.append(last)
                phases.append(_PHASES[phase_idx])
                message = f"{action_name} success={last.action_success}"
                continue

            # Interactions (need a selection).
            interaction_map = {
                pygame.K_o: "OpenObject",
                pygame.K_c: "CloseObject",
                pygame.K_p: "PickupObject",
                pygame.K_t: "ToggleObjectOn",
                pygame.K_y: "ToggleObjectOff",
            }
            if key in interaction_map:
                if selected_obj is None:
                    message = "no object selected (press 1-9 first)"
                    continue
                spec = ActionSpec(interaction_map[key], object_id=selected_obj)
                last = env.step(spec)
                steps.append(last)
                phases.append(_PHASES[phase_idx])
                message = f"{spec.name} on {selected_obj}: success={last.action_success}"
                if not last.action_success:
                    message += f" ({last.error_message[:60]})"
                # Auto-track: first picked-up object in Phase A is target.
                if (
                    spec.name == "PickupObject"
                    and last.action_success
                    and target_object_id is None
                ):
                    target_object_id = selected_obj
                continue

            if key == pygame.K_v:
                if selected_obj is None:
                    message = "no receptacle selected"
                    continue
                if held_id is None:
                    message = "no object held"
                    continue
                spec = ActionSpec(
                    "PutObject", receptacle_id=selected_obj, object_id=held_id
                )
                last = env.step(spec)
                steps.append(last)
                phases.append(_PHASES[phase_idx])
                message = f"PutObject {held_id} -> {selected_obj}: success={last.action_success}"
                if last.action_success and target_receptacle_id is None:
                    target_receptacle_id = selected_obj
                continue

        if not running:
            return


def _maybe_save(
    steps: list[StepResult],
    phases: list[str],
    ep_id: str,
    scene: str,
    output_dir: Path,
    manifest: ManifestBuilder,
    split: str,
    target_object_id: str | None,
    target_receptacle_id: str | None,
    note: str | None,
) -> None:
    if len(steps) < 2:
        print(f"[recorder] skipping save for {ep_id}: only {len(steps)} step(s)")
        return
    out_path = output_dir / "episodes" / scene / "manual" / f"{ep_id}.h5"
    EpisodeWriter(out_path).write(
        steps=steps,
        phases=phases,
        attrs={
            "episode_id": ep_id,
            "scene": scene,
            "episode_type": "manual",
            "probe_name": None,
            "gap_length": -1,
            "seed": 0,
            "split": split,
            "target_object_id": target_object_id,
            "target_receptacle_id": target_receptacle_id,
            "note": note,
            "wallclock_seconds": 0.0,
        },
    )
    rel = out_path.relative_to(output_dir).as_posix()
    manifest.add(EpisodeRecord(
        episode_id=ep_id,
        scene=scene,
        episode_type="manual",
        probe_name=None,
        gap_length=None,
        seed=0,
        split=split,
        num_steps=len(steps),
        success=True,
        failure_reason=None,
        target_object_id=target_object_id,
        target_receptacle_id=target_receptacle_id,
        relative_path=rel,
    ))
    print(f"[recorder] saved {ep_id} ({len(steps)} steps) -> {out_path}")


def _draw(
    *,
    win,
    pygame,
    font,
    big_font,
    last: StepResult,
    display_size: int,
    panel_w: int,
    scene: str,
    episode_id: str,
    step_count: int,
    phase: str,
    visible: list[dict],
    selected_obj: str | None,
    held_id: str | None,
    message: str,
    note: str | None,
) -> None:
    win.fill((20, 20, 24))

    # Frame on the left, scaled up.
    frame = last.rgb
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    surf = pygame.transform.scale(surf, (display_size, display_size))
    win.blit(surf, (0, 0))

    # Right panel.
    px = display_size + 12
    py = 12
    line_h = 18

    def draw_line(s: str, *, big: bool = False, color=(220, 220, 220)) -> None:
        nonlocal py
        f = big_font if big else font
        win.blit(f.render(s, True, color), (px, py))
        py += line_h + (4 if big else 0)

    draw_line(f"Scene: {scene}", big=True)
    draw_line(f"Episode: {episode_id}")
    draw_line(f"Step: {step_count}    Phase: {phase}")
    draw_line(f"Held: {held_id or '-'}")
    draw_line(f"Selected: {selected_obj or '-'}")
    if note:
        draw_line(f"Note: {note}", color=(180, 180, 255))
    py += 8
    draw_line("Visible interactables:", big=True)
    if not visible:
        draw_line("  (none)")
    for i, obj in enumerate(visible[:10]):
        marker = "*" if obj["objectId"] == selected_obj else " "
        flags = []
        if obj.get("openable"):
            flags.append("open" if obj.get("isOpen") else "OPEN")
        if obj.get("toggleable"):
            flags.append("on" if obj.get("isToggled") else "OFF")
        if obj.get("pickupable"):
            flags.append("pick" if obj.get("isPickedUp") else "PICK")
        if obj.get("receptacle"):
            flags.append("recep")
        flag_s = " ".join(flags)
        s = f" [{(i + 1) % 10}]{marker} {obj['objectType']:<14} {flag_s}"
        draw_line(s)

    py = display_size - 230
    draw_line("Controls:", big=True)
    for s in (
        "W/S move ahead/back   A/D rotate L/R",
        "Q/E strafe L/R        R/F look up/down",
        "Space hold pose       1-9 select  0 clear",
        "O open  C close  P pickup  V put",
        "T toggle-on  Y toggle-off",
        "TAB cycle phase  Z undo last step",
        "Enter save+next  Backspace discard+next",
        "Esc save (if any) and quit",
    ):
        draw_line(s, color=(160, 200, 160))

    py = display_size - 28
    draw_line(message, color=(255, 220, 120))
