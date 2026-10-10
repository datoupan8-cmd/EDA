"""P3.1 experiment: retain a short boundary stroke if a nearby line resumes.

The original P3 gate is always applied first. This alternative only rescues
previously rejected proposals; it never changes terminal coordinates.
"""
from __future__ import annotations

import numpy as np

from .continuity_v6 import verify_outward_continuity
from .directional_v5 import _normal_support


def verify_outward_continuity_v7(ink: np.ndarray, terminal: dict,
                                 image_shape: tuple[int, ...]) -> tuple[bool, dict]:
    accepted, evidence = verify_outward_continuity(ink, terminal, image_shape)
    if accepted:
        return True, {**evidence, "decision": "p3_continuous"}

    side = terminal["side"]
    x, y = terminal["base"]
    boundary = int(round(x if side in ("left", "right") else y))
    tangent = int(round(y if side in ("left", "right") else x))
    height, width = image_shape[:2]
    scale = max(.55, min(2.2, max(height, width) / 1200.0))
    signal = _normal_support(ink, side, boundary, tangent, max(30, int(round(20 * scale))))

    # Distinguish an arrowhead or small gap before a real wire from a distant
    # text stroke. A near-boundary run must resume by offset 24, and the
    # resumed stroke must be long enough to be a line, not an isolated glyph.
    first_run = int(evidence["span"])
    second_start = None
    second_span = 0
    if evidence["first_offset"] <= 4 and 3 <= first_run < evidence["minimum_span"]:
        end = int(evidence["first_offset"] - 2 + first_run)
        for start in range(end + 1, min(len(signal), max(12, int(round(11 * scale))))):
            if not signal[start]:
                continue
            run = 0
            for value in signal[start:]:
                if not value:
                    break
                run += 1
            if run >= max(8, int(round(5 * scale))):
                second_start = start + 2
                second_span = run
                break
    rescued = second_start is not None
    return rescued, {**evidence, "accepted": rescued,
                     "decision": "resumed_wire" if rescued else "rejected",
                     "resumed_start": second_start, "resumed_span": second_span}
