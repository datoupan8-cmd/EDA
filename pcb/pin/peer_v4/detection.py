"""Conservative pixel-supported corrections to V3 boundary terminal tips.

A visible short stub endpoint is useful evidence, not proof of package pin
length. Continuous nets and ambiguous pixels retain the exact V3 coordinate.
Simple symbols keep their audited V3 layout and ordering unchanged.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ...pin_detection import terminal_candidates_v3

_NORMAL = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}


def pixel_evidence(image, texts=()):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ink = (gray < 220) | ((hsv[:, :, 1] > 45) & (hsv[:, :, 2] < 250))
    blocked = np.zeros(ink.shape, np.uint8)
    h, w = ink.shape
    for token in texts:
        a, b, c, d = [int(round(v)) for v in token.bbox]
        if c < 0 or d < 0 or a >= w or b >= h:
            continue
        cv2.rectangle(blocked, (max(0, a), max(0, b)), (min(w - 1, c), min(h - 1, d)), 1, -1)
    return ink, blocked


def refine_terminal(terminal, component, evidence):
    """Return a copy with an audited correction or an explicit fallback reason."""
    result = dict(terminal)
    result["v3_tip"] = tuple(terminal["tip"])

    def fallback(reason):
        result["localization_evidence"] = reason
        result["tip_adjusted"] = False
        return result

    if terminal.get("method") != "boundary_wire_support":
        return fallback("audited_layout_preserved")
    if terminal.get("base") is None or terminal.get("side") not in _NORMAL:
        return fallback("missing_frame")
    ink, blocked = evidence
    h, w = ink.shape
    scale = max(.55, min(2.2, max(h, w) / 1200))
    nx, ny = _NORMAL[terminal["side"]]
    tx, ty = -ny, nx
    bx, by = terminal["base"]
    x1, y1, x2, y2 = component.body_bbox or component.bbox
    depth = (x2 - x1) if nx else (y2 - y1)
    old_length = math.dist(terminal["tip"], terminal["base"])
    if depth <= 0 or not terminal.get("wire_support_score"):
        return fallback("insufficient_boundary_support")
    limit = int(max(12, min(50 * scale, .45 * depth, 3 * old_length)))
    radius = max(1, int(round(scale)))
    reach = max(5, int(round(6 * scale)))
    white_margin = max(3, int(round(3 * scale)))
    start = max(2, int(round(2 * scale)))

    def sample(distance):
        x, y = int(round(bx + nx * distance)), int(round(by + ny * distance))
        offsets = np.arange(-reach, reach + 1)
        xs, ys = x + tx * offsets, y + ty * offsets
        if xs.min() < 0 or ys.min() < 0 or xs.max() >= w or ys.max() >= h:
            return None
        row = ink[ys, xs]
        core = np.abs(offsets) <= radius
        if blocked[ys[core], xs[core]].any():
            return None
        return row, offsets, (x, y)

    # Require actual 8-connected pixels from the body edge, including the
    # small span skipped to avoid measuring the thick body outline as a stub.
    connected = None
    for distance in range(start):
        sampled = sample(distance)
        if sampled is None:
            return fallback("text_or_image_boundary")
        row, offsets, _ = sampled
        hits = offsets[row & (np.abs(offsets) <= radius)]
        if connected is not None:
            hits = np.array([offset for offset in hits if any(abs(offset - old) <= 1 for old in connected)])
        if not len(hits):
            return fallback("detached_from_body")
        connected = hits
    traced = []
    widths = []
    for distance in range(start, limit + 1):
        sampled = sample(distance)
        if sampled is None:
            return fallback("text_or_image_boundary")
        row, offsets, point = sampled
        core_hits = offsets[row & (np.abs(offsets) <= radius)]
        if not len(core_hits):
            # A one-pixel break followed by ink is uncertainty, not a stub end.
            following = [sample(d) for d in range(distance, distance + white_margin)]
            if distance + white_margin > limit or any(s is None for s in following):
                return fallback("unverified_end_margin")
            if any(s[0][np.abs(s[1]) <= radius].any() for s in following):
                return fallback("pixel_gap")
            if len(traced) < 2:
                return fallback("short_or_detached_support")
            endpoint = traced[-1]
            result.update(tip=endpoint, method="short_stub_endpoint_v4",
                          localization_evidence="continuous_short_stub_then_white_margin", tip_adjusted=True)
            return result
        core_hits = np.array([offset for offset in core_hits if any(abs(offset - old) <= 1 for old in connected)])
        if not len(core_hits):
            return fallback("disconnected_corridor_ink")
        connected = core_hits
        closest = int(core_hits[np.argmin(np.abs(core_hits))])
        center = reach + closest
        low = high = center
        while low > 0 and row[low - 1]:
            low -= 1
        while high + 1 < len(row) and row[high + 1]:
            high += 1
        width = high - low + 1
        expected = max(1.0, float(np.median(widths))) if widths else 1.0
        if width >= max(5, 3 * expected):
            negative, positive = low < reach - radius - 1, high > reach + radius + 1
            # Only a single L arm is eligible; dots, T and crossings are not.
            if negative == positive or len(traced) < 2:
                return fallback("branch_or_wide_ink")
            ahead = [sample(d) for d in range(distance + radius + 2, distance + radius + 2 + white_margin)]
            if distance + radius + 2 + white_margin > limit or any(s is None for s in ahead):
                return fallback("unverified_corner_margin")
            if any(s[0][np.abs(s[1]) <= radius].any() for s in ahead):
                return fallback("junction_or_continuing_net")
            result.update(tip=(float(point[0] + tx * closest), float(point[1] + ty * closest)), method="short_stub_corner_v4",
                          localization_evidence="single_nearby_L_corner", tip_adjusted=True)
            return result
        if width > max(4, int(round(3 * scale))):
            return fallback("wide_ink")
        # The corridor stays centred on the ORIGINAL terminal axis; no drift.
        traced.append((float(point[0] + tx * closest), float(point[1] + ty * closest)))
        widths.append(width)
    return fallback("continuous_wire_no_visible_pin_end")


def terminal_candidates_v4(image, component, texts=(), keypoint_provider=None, *, evidence=None):
    originals = terminal_candidates_v3(image, component, texts, keypoint_provider)
    if evidence is None:
        evidence = pixel_evidence(image, texts)
    return [refine_terminal(terminal, component, evidence) for terminal in originals]
