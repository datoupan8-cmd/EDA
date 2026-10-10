"""Isolated frame suppression: delete supported box outlines, keep axial exits.

This experimental postprocessor is not registered in the production Pipeline.
It never adds pixels, changes Pins, or infers electrical connectivity. Frame
locations come from the image and existing body_bbox; no GT or case/source IDs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from pcb.core.interfaces import WireStageOutput


@dataclass(frozen=True)
class FramePolicy:
    search_band_stroke: float = 2.0
    cut_half_width_stroke: float = 0.75
    min_side_coverage: float = 0.80
    min_side_length: int = 16
    min_side_length_stroke: float = 8.0
    min_verified_sides: int = 3
    normal_kernel_min: int = 5
    normal_kernel_stroke: float = 3.0


POLICY = FramePolicy()


def detect_frame_sides(ink, box, stroke, policy=POLICY):
    """Find long straight ink near a box boundary; reject unverified sides."""
    h, w = ink.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    band = max(1, int(math.ceil(stroke * policy.search_band_stroke)))
    trim = max(1, int(math.ceil(stroke)))
    minimum = max(policy.min_side_length, int(math.ceil(stroke * policy.min_side_length_stroke)))
    sides = []
    for side, nominal, vertical in (("left", x1, True), ("right", x2, True),
                                     ("top", y1, False), ("bottom", y2, False)):
        begin, end = (max(0, y1 + trim), min(h, y2 - trim + 1)) if vertical else (max(0, x1 + trim), min(w, x2 - trim + 1))
        if end - begin < minimum:
            continue
        candidates = []
        for coordinate in range(max(0, nominal - band), min(w if vertical else h, nominal + band + 1)):
            values = ink[begin:end, coordinate] if vertical else ink[coordinate, begin:end]
            candidates.append((float(values.mean()), -abs(coordinate - nominal), coordinate))
        if not candidates:
            continue
        coverage, _, coordinate = max(candidates)
        if coverage >= policy.min_side_coverage:
            sides.append({"side": side, "coordinate": coordinate, "nominal": nominal,
                          "coverage": coverage, "search_band": band})
    return sides if len(sides) >= policy.min_verified_sides else []


def suppress_box_frames(image, scene, baseline: WireStageOutput,
                        policy: FramePolicy = POLICY) -> WireStageOutput:
    """Clear tangential frame bands while retaining outward axial mask support.

    A retained exit must contain a perpendicular morphological line AND ink in
    the outward part beyond the cut band. Thus a frame alone is not an exit.
    Decisions are local to image-verified box sides. The mask remains a subset
    of baseline; palette, text suppression and Topology algorithms stay fixed.
    """
    h, w = baseline.mask.shape
    stroke = max(1.0, float(scene.diagnostics.get("wire_stroke_width", 1.0)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ink = (gray < 205) | ((hsv[:, :, 1] >= 45) & (hsv[:, :, 2] < 248))
    radius = max(1, int(math.ceil(policy.cut_half_width_stroke * stroke)))
    kernel = max(policy.normal_kernel_min, int(math.ceil(policy.normal_kernel_stroke * stroke)), 2 * radius + 2)
    axes = {True: cv2.morphologyEx(baseline.mask, cv2.MORPH_OPEN, np.ones((1, kernel), np.uint8)),
            False: cv2.morphologyEx(baseline.mask, cv2.MORPH_OPEN, np.ones((kernel, 1), np.uint8))}
    cut = np.zeros((h, w), bool)
    decisions = []
    for component in scene.components:
        if component.type != "box":
            continue
        box = component.body_bbox or component.bbox
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        frame = detect_frame_sides(ink, box, stroke, policy)
        if not frame:
            decisions.append({"component": component.key, "action": "skip_unverified_frame"})
            continue
        for f in frame:
            side, p = f["side"], f["coordinate"]
            vertical = side in ("left", "right")
            normal = axes[vertical]
            if vertical:
                xa, xb = max(0, p - radius), min(w, p + radius + 1)
                ya, yb = max(0, y1), min(h, y2 + 1)
                oa, ob = (max(0, xa - kernel), xa) if side == "left" else (xb, min(w, xb + kernel))
                outward = normal[ya:yb, oa:ob].any(axis=1)
                protected = (normal[ya:yb, xa:xb] > 0) & outward[:, None]
            else:
                xa, xb = max(0, x1), min(w, x2 + 1)
                ya, yb = max(0, p - radius), min(h, p + radius + 1)
                oa, ob = (max(0, ya - kernel), ya) if side == "top" else (yb, min(h, yb + kernel))
                outward = normal[oa:ob, xa:xb].any(axis=0)
                protected = (normal[ya:yb, xa:xb] > 0) & outward[None, :]
            if xa >= xb or ya >= yb:
                continue
            occupied = baseline.mask[ya:yb, xa:xb] > 0
            remove = occupied & ~protected
            cut[ya:yb, xa:xb] |= remove
            decisions.append({"component": component.key, "action": "verified_frame_band",
                              **f, "cut_bbox": [xa, ya, xb, yb], "cut_radius": radius,
                              "removed_pixel_proposals": int(remove.sum()),
                              "protected_axial_exit_pixels": int((occupied & protected).sum())})
    mask = baseline.mask.copy()
    mask[cut] = 0
    detail = {"policy": asdict(policy), "normal_kernel": kernel, "stroke": stroke,
              "removed_pixels": int(cut.sum()), "added_pixels": 0, "pin_geometry_modified": False,
              "palette_modified": False, "decisions": decisions}
    scene.diagnostics["wire_frame_guard"] = detail
    suppressed = baseline.suppressed.copy() if baseline.suppressed is not None else np.zeros((h, w), np.uint8)
    suppressed[cut] = 255
    return WireStageOutput(mask, baseline.color_debug, suppressed, baseline.corridor,
                           baseline.diagnostics, baseline.debug)
