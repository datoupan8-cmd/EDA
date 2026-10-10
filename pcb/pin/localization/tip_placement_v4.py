"""GT-free normal extension policy for existing boundary-wire terminals.

P1-2.3 deliberately keeps candidate generation unchanged.  This module only
places the tip of an already generated ``boundary_wire_support`` candidate.
"""
from __future__ import annotations

from typing import Any


SIDES = ("left", "right", "top", "bottom")
NORMALS = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "top": (0.0, -1.0),
    "bottom": (0.0, 1.0),
}


def image_scale(image_shape: tuple[int, ...]) -> float:
    """Return the unchanged V3 global image scale."""
    height, width = image_shape[:2]
    return max(0.55, min(2.2, max(height, width) / 1200.0))


def scale_aware_extension(image_shape: tuple[int, ...]) -> float:
    """Use one conservative scale-aware length without side/case fitting.

    The V3 lower bound is preserved.  P1-2.1 showed that ``13 * scale`` was
    systematically too long on both Design and Check data, so P1-2.3 tests a
    single smaller coefficient instead of side-specific constants.
    """
    return max(6.0, 6.0 * image_scale(image_shape))


def place_boundary_tip(
    terminal: dict[str, Any],
    image_shape: tuple[int, ...],
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Return a new tip while preserving base, side, and tangential position.

    Missing/unknown geometry is a safe no-op fallback.  No image target or
    semantic field is consulted.
    """
    original = tuple(float(value) for value in terminal["tip"])
    base_value = terminal.get("base")
    side = str(terminal.get("side") or "")
    if base_value is None or side not in NORMALS:
        return original, {
            "applied": False,
            "reason": "missing_base_or_side",
            "old_tip": original,
            "new_tip": original,
        }
    base = tuple(float(value) for value in base_value)
    nx, ny = NORMALS[side]
    extension = scale_aware_extension(image_shape)
    height, width = image_shape[:2]
    new_tip = (
        min(float(width - 1), max(0.0, base[0] + nx * extension)),
        min(float(height - 1), max(0.0, base[1] + ny * extension)),
    )
    return new_tip, {
        "applied": True,
        "strategy": "scale_aware",
        "scale": image_scale(image_shape),
        "extension": extension,
        "side": side,
        "old_tip": original,
        "new_tip": new_tip,
    }

