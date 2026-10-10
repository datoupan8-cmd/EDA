"""Image-only continuity evidence for P2's outward terminal proposals."""
from __future__ import annotations

import numpy as np

from .directional_v5 import _normal_support


def continuity_features(ink: np.ndarray, terminal: dict, reach: int) -> dict[str, float]:
    """Measure one narrow, outward path starting at the component boundary.

    This does not classify a candidate or change its geometry. It exposes
    measurable image evidence for the separate P3 decision.
    """
    side = terminal["side"]
    base_x, base_y = terminal["base"]
    boundary = int(round(base_x if side in ("left", "right") else base_y))
    tangent = int(round(base_y if side in ("left", "right") else base_x))
    signal = _normal_support(ink, side, boundary, tangent, reach)
    positions = np.flatnonzero(signal)
    if not len(positions):
        return {"first_offset": float("inf"), "span": 0.0, "support": 0.0,
                "occupancy": 0.0, "gap_count": 0.0}
    # _normal_support starts at outward offset 2.
    first = int(positions[0])
    last = first
    missing = 0
    for index in range(first + 1, len(signal)):
        if signal[index]:
            last = index
            missing = 0
        else:
            missing += 1
            if missing >= 2:
                break
    path = signal[first:last + 1]
    return {
        "first_offset": float(first + 2),
        "span": float(last - first + 1),
        "support": float(path.sum()),
        "occupancy": float(path.mean()),
        "gap_count": float(len(path) - path.sum()),
    }


def verify_outward_continuity(ink: np.ndarray, terminal: dict, image_shape: tuple[int, ...]) -> tuple[bool, dict]:
    """Conservative continuity gate, using only original-image pixels."""
    height, width = image_shape[:2]
    scale = max(.55, min(2.2, max(height, width) / 1200.0))
    reach = max(8, int(round(20 * scale)))
    features = continuity_features(ink, terminal, reach)
    minimum_span = max(5, int(round(8 * scale)))
    accepted = (features["first_offset"] <= 4.0
                and features["span"] >= minimum_span
                and features["occupancy"] >= .75)
    return accepted, {**features, "minimum_span": minimum_span, "accepted": accepted}
