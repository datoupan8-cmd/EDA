"""P2 image-only, direction-aware proposals at component boundaries.

This module only proposes missing boundary-wire terminals. Existing V3
terminals, including their tip placement, are left untouched by the stage.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ...pin_detection import _groups


SIDES = ("left", "right", "top", "bottom")


def build_raw_ink(image: np.ndarray) -> np.ndarray:
    """Use V3's exact image-to-ink definition before its OCR suppression."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (gray < 220) | ((hsv[:, :, 1] > 45) & (hsv[:, :, 2] < 250))


def _normal_support(
    ink: np.ndarray,
    side: str,
    boundary: int,
    tangent: int,
    reach: int,
) -> np.ndarray:
    """Sample an outward normal in a narrow tangent band, not an OCR box."""
    height, width = ink.shape
    offsets = np.arange(2, reach + 1, dtype=int)
    if side == "left":
        normal = boundary - offsets
        tangents = np.arange(tangent - 1, tangent + 2)
        valid = (normal >= 0) & (normal < width)
        tangents = tangents[(tangents >= 0) & (tangents < height)]
        return ink[np.ix_(tangents, normal[valid])].any(axis=0) if len(tangents) else np.zeros(0, bool)
    if side == "right":
        normal = boundary + offsets
        tangents = np.arange(tangent - 1, tangent + 2)
        valid = (normal >= 0) & (normal < width)
        tangents = tangents[(tangents >= 0) & (tangents < height)]
        return ink[np.ix_(tangents, normal[valid])].any(axis=0) if len(tangents) else np.zeros(0, bool)
    if side == "top":
        normal = boundary - offsets
        tangents = np.arange(tangent - 1, tangent + 2)
        valid = (normal >= 0) & (normal < height)
        tangents = tangents[(tangents >= 0) & (tangents < width)]
        return ink[np.ix_(normal[valid], tangents)].any(axis=1) if len(tangents) else np.zeros(0, bool)
    normal = boundary + offsets
    tangents = np.arange(tangent - 1, tangent + 2)
    valid = (normal >= 0) & (normal < height)
    tangents = tangents[(tangents >= 0) & (tangents < width)]
    return ink[np.ix_(normal[valid], tangents)].any(axis=1) if len(tangents) else np.zeros(0, bool)


def _tip(side: str, base: tuple[float, float], extension: float, width: int, height: int) -> tuple[float, float]:
    x, y = base
    if side == "left":
        return max(0.0, x - extension), y
    if side == "right":
        return min(float(width - 1), x + extension), y
    if side == "top":
        return x, max(0.0, y - extension)
    return x, min(float(height - 1), y + extension)


def propose_boundary_terminals(
    ink: np.ndarray,
    component,
    existing: list[dict],
) -> tuple[list[dict], dict]:
    """Add image-supported outward candidates while preserving every V3 row.

    The first two outward pixels are excluded to avoid the component outline.
    A candidate needs ink near the boundary and farther along the normal.
    P3 will separately validate full outward continuity.
    """
    height, width = ink.shape
    x1, y1, x2, y2 = (int(round(value)) for value in (component.body_bbox or component.bbox))
    scale = max(.55, min(2.2, max(height, width) / 1200.0))
    reach = max(8, int(round(20 * scale)))
    margin = max(3, int(round(4 * scale)))
    minimum = max(3, int(round(4 * scale)))
    extension = max(6.0, 13.0 * scale)  # Keep V3 placement for isolated P2 A/B.
    bounds = {
        "left": (x1, range(max(0, y1 + margin), min(height, y2 - margin + 1))),
        "right": (x2, range(max(0, y1 + margin), min(height, y2 - margin + 1))),
        "top": (y1, range(max(0, x1 + margin), min(width, x2 - margin + 1))),
        "bottom": (y2, range(max(0, x1 + margin), min(width, x2 - margin + 1))),
    }
    proposals: list[dict] = []
    diagnostics: dict = {"supported_scan_positions": {}, "grouped": {}, "added": {}, "overlap_rejected": {}}
    for side in SIDES:
        boundary, positions = bounds[side]
        supported: list[int] = []
        for tangent in positions:
            signal = _normal_support(ink, side, boundary, tangent, reach)
            if len(signal) < minimum + 2:
                continue
            near_end = min(len(signal), max(4, int(round(6 * scale))))
            if signal[:near_end].sum() >= 2 and signal.sum() >= minimum and signal[near_end - 1 :].any():
                supported.append(tangent)
        groups = _groups(supported, gap=1)
        diagnostics["supported_scan_positions"][side] = len(supported)
        diagnostics["grouped"][side] = len(groups)
        added = 0
        overlap_rejected = 0
        for group in groups:
            # A thick band is more likely text or a diagram frame than one pin.
            if group[-1] - group[0] > max(4, int(round(6 * scale))):
                continue
            tangent = float(np.median(group))
            base = ((float(boundary), tangent) if side in ("left", "right")
                    else (tangent, float(boundary)))
            if any(row.get("side") == side and math.dist(row["base"], base) < 7 * scale for row in existing):
                overlap_rejected += 1
                continue
            proposal = {
                "tip": _tip(side, base, extension, width, height),
                "base": base,
                "side": side,
                "method": "directional_boundary_support",
                "wire_support_score": len(group),
            }
            proposals.append(proposal)
            added += 1
        diagnostics["added"][side] = added
        diagnostics["overlap_rejected"][side] = overlap_rejected
    return proposals, diagnostics
