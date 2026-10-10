"""E1: preserve image-supported box stubs without changing any Pin geometry.

This module is an isolated Wire strategy, never registered as a production
default. It adds real axial ink only when a continuous run reaches existing
wire outside the original terminal corridors. It does not bridge blank pixels,
infer connectivity, use GT, or change the original palette selection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import cv2
import numpy as np

from pcb.core.interfaces import WireStageOutput


NORMALS = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}


@dataclass(frozen=True)
class StubPolicy:
    """Single image-only policy; frozen before Design/Check comparison."""

    body_reach_fraction: float = 0.5
    pitch_reach_fraction: float = 1.5
    tangent_stroke_factor: float = 1.0
    tangent_pitch_fraction: float = 0.2
    line_kernel_stroke_factor: float = 3.0
    minimum_line_kernel: int = 5
    start_search_stroke_factor: float = 2.0


POLICY = StubPolicy()


def component_barrier(scene, shape: tuple[int, ...]) -> np.ndarray:
    """Exactly reproduce V3's interior barrier, including its 2px margin."""
    barrier = np.zeros(shape[:2], np.uint8)
    for component in scene.components:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), component.body_bbox or component.bbox)
        if x2 - x1 <= 2 or y2 - y1 <= 2:
            continue
        cv2.rectangle(barrier, (max(0, x1 + 2), max(0, y1 + 2)),
                      (min(shape[1] - 1, x2 - 2), min(shape[0] - 1, y2 - 2)), 255, -1)
    return barrier


def _local_limits(component, pin, stroke: float, policy: StubPolicy) -> tuple[int, int, float | None]:
    box = component.body_bbox or component.bbox
    horizontal = pin.side in {"left", "right"}
    normal_span = box[2] - box[0] if horizontal else box[3] - box[1]
    tangent_axis = 1 if horizontal else 0
    other_distances = [abs(q.base[tangent_axis] - pin.base[tangent_axis])
                       for q in component.pins if q is not pin and q.base is not None
                       and q.side == pin.side and abs(q.base[tangent_axis] - pin.base[tangent_axis]) > .5]
    pitch = min(other_distances) if other_distances else None
    reach = normal_span * policy.body_reach_fraction
    half_band = max(1, int(math.ceil(stroke * policy.tangent_stroke_factor)))
    if pitch is not None:
        reach = min(reach, pitch * policy.pitch_reach_fraction)
        half_band = min(half_band, max(1, int(math.floor(pitch * policy.tangent_pitch_fraction))))
    return max(0, int(math.floor(reach))), half_band, pitch


def restore_supported_stubs(image, scene, baseline: WireStageOutput,
                            policy: StubPolicy = POLICY) -> WireStageOutput:
    """Keep G Pins fixed and add supported pixels to the existing V3 Wire mask.

    A projection is only a candidate search. The accepted pixels must also
    belong to an actual 8-connected axial-ink component connecting the base
    neighbourhood and an existing external-wire anchor. No same-CC Pin union
    is performed here; unchanged Topology consumes the resulting mask.
    """
    h, w = image.shape[:2]
    stroke = max(1.0, float(scene.diagnostics.get("wire_stroke_width", 1.0)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    raw = (((gray < 205) | ((hsv[:, :, 1] >= 45) & (hsv[:, :, 2] < 248))).astype(np.uint8) * 255)
    kernel_length = max(policy.minimum_line_kernel, int(math.ceil(stroke * policy.line_kernel_stroke_factor)))
    axes = {
        True: cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((1, kernel_length), np.uint8)),
        False: cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((kernel_length, 1), np.uint8)),
    }
    barrier = component_barrier(scene, image.shape)
    for mask in axes.values():
        mask[barrier > 0] = 0
    old_corridor = (baseline.corridor if baseline.corridor is not None else np.zeros((h, w), np.uint8))
    backbone = (baseline.mask > 0) & (old_corridor == 0) & (barrier == 0)
    additions = np.zeros((h, w), np.uint8)
    decisions: list[dict[str, Any]] = []
    for component in scene.components:
        if component.type != "box":
            continue
        for pin in component.pins:
            if pin.base is None or pin.side not in NORMALS:
                continue
            nx, ny = NORMALS[pin.side]
            reach, band, pitch = _local_limits(component, pin, stroke, policy)
            record = {"component": component.key, "pin": pin.number, "side": pin.side,
                      "base": pin.base, "tip_unchanged": pin.tip, "reach": reach,
                      "half_band": band, "local_pitch": pitch, "restored_pixels": 0}
            if reach < kernel_length:
                decisions.append({**record, "reason": "insufficient_local_reach"})
                continue
            distances = np.arange(0, reach + 1)
            offsets = np.arange(-band, band + 1)
            xx = np.rint(pin.base[0] + distances[:, None] * nx - offsets[None, :] * ny).astype(int)
            yy = np.rint(pin.base[1] + distances[:, None] * ny + offsets[None, :] * nx).astype(int)
            valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            sx, sy = np.clip(xx, 0, w - 1), np.clip(yy, 0, h - 1)
            support = (axes[bool(nx)][sy, sx] > 0) & valid
            occupied = support.any(axis=1)
            start_limit = min(reach, max(1, int(math.ceil(stroke * policy.start_search_stroke_factor))))
            starts = np.flatnonzero(occupied[:start_limit + 1])
            if not len(starts):
                decisions.append({**record, "reason": "no_base_axial_support"})
                continue
            start = int(starts[0])
            missing = np.flatnonzero(~occupied[start:])
            end = start + int(missing[0]) - 1 if len(missing) else reach
            anchors = np.flatnonzero(((backbone[sy, sx] & support).any(axis=1))
                                     & (distances >= start) & (distances <= end))
            if not len(anchors):
                decisions.append({**record, "reason": "no_continuous_external_anchor", "continuous_end": end})
                continue
            anchor = int(anchors[0])
            # Prove actual 8-connected support, rather than merely occupied columns.
            strip = support[start:anchor + 1].astype(np.uint8)
            _, labels = cv2.connectedComponents(strip, connectivity=8)
            start_labels = set(labels[0]) - {0}
            anchor_mask = backbone[sy[anchor], sx[anchor]] & support[anchor]
            anchor_labels = set(labels[-1][anchor_mask]) - {0}
            accepted_labels = start_labels & anchor_labels
            if not accepted_labels:
                decisions.append({**record, "reason": "disconnected_ink_profile"})
                continue
            keep = np.isin(labels, sorted(accepted_labels)) & (labels > 0)
            # Half-pixel bases can round adjacent samples to the same pixel.
            # Deduplicate for truthful diagnostic counts; mask writes are identical.
            positions = np.unique(np.column_stack((sy[start:anchor + 1][keep], sx[start:anchor + 1][keep])), axis=0)
            py, px = positions[:, 0], positions[:, 1]
            new = (baseline.mask[py, px] == 0) & (additions[py, px] == 0)
            additions[py, px] = 255
            record.update({"reason": "supported_external_anchor", "anchor_distance": anchor,
                           "anchor": (float(pin.base[0] + anchor * nx), float(pin.base[1] + anchor * ny)),
                           "restored_pixels": int(new.sum())})
            decisions.append(record)
    additions[barrier > 0] = 0
    mask = cv2.bitwise_or(baseline.mask, additions)
    if np.any(mask[barrier > 0]):
        raise AssertionError("Wire crossed the unchanged component barrier")
    scene.diagnostics["wire_stub_preservation_e1"] = {
        "policy": asdict(policy), "added_pixels": int(((mask > 0) & (baseline.mask == 0)).sum()),
        "pin_geometry_modified": False, "palette_modified": False, "decisions": decisions,
    }
    corridor = cv2.bitwise_or(old_corridor, additions)
    return WireStageOutput(mask, baseline.color_debug, baseline.suppressed, corridor,
                           baseline.diagnostics, baseline.debug)
