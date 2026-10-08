"""Point-first terminal localization for V3."""
from __future__ import annotations
import math
import cv2
import numpy as np

TWO_TERMINAL = {"r", "c", "l", "d", "led", "crystal", "crystal_2pin", "fuse"}


def _base(point, side, box):
    x, y = point; x1, y1, x2, y2 = box
    return {"left": (x1, y), "right": (x2, y), "top": (x, y1), "bottom": (x, y2)}[side]


def simple_terminals(component):
    """Audited simple-symbol geometry: tips are half a body size outside."""
    x1, y1, x2, y2 = component.body_bbox or component.bbox
    width, height = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if height >= width:
        points = [((cx, y2 + .5 * height), "bottom"), ((cx, y1 - .5 * height), "top")]
    else:
        points = [((x1 - .5 * width, cy), "left"), ((x2 + .5 * width, cy), "right")]
    return [{"tip": p, "side": side, "base": _base(p, side, (x1, y1, x2, y2)), "method": "audited_two_terminal_layout", "wire_support_score": None} for p, side in points]


def crystal4_terminals(component):
    x1, y1, x2, y2 = component.body_bbox or component.bbox
    width, height = x2 - x1, y2 - y1
    return [
        {"tip": (x1 + .75 * width, y2 + .25 * height), "side": "bottom", "base": (x1 + .75 * width, y2), "method": "crystal4_layout", "wire_support_score": None},
        {"tip": (x1 + .75 * width, y1 - .25 * height), "side": "top", "base": (x1 + .75 * width, y1), "method": "crystal4_layout", "wire_support_score": None},
        {"tip": (x1 + .25 * width, y1 - .25 * height), "side": "top", "base": (x1 + .25 * width, y1), "method": "crystal4_layout", "wire_support_score": None},
        {"tip": (x1 + .25 * width, y2 + .25 * height), "side": "bottom", "base": (x1 + .25 * width, y2), "method": "crystal4_layout", "wire_support_score": None},
    ]


def _groups(values, gap=3):
    values = sorted(values)
    groups = []
    for value in values:
        if not groups or value - groups[-1][-1] > gap:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def boundary_wire_terminals(image, component, texts=()):
    """Find wire stubs leaving each side of a component boundary."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ink = (gray < 220) | ((hsv[:, :, 1] > 45) & (hsv[:, :, 2] < 250))
    # Text outside the component must not masquerade as a terminal stub.
    text_mask = np.zeros((h, w), np.uint8)
    for token in texts:
        a, b, c, d = map(lambda x: int(round(x)), token.bbox)
        cv2.rectangle(text_mask, (max(0, a), max(0, b)), (min(w - 1, c), min(h - 1, d)), 1, -1)
    ink[text_mask > 0] = False
    x1, y1, x2, y2 = map(lambda x: int(round(x)), component.body_bbox or component.bbox)
    scale = max(.55, min(2.2, max(h, w) / 1200))
    reach = max(8, int(round(20 * scale)))
    minimum = max(3, int(round(4 * scale)))
    margin = max(3, int(round(4 * scale)))
    rows = {side: [] for side in ("left", "right", "top", "bottom")}
    for y in range(max(0, y1 + margin), min(h, y2 - margin + 1)):
        left = ink[y, max(0, x1 - reach):min(w, x1 + 2)]
        right = ink[y, max(0, x2 - 1):min(w, x2 + reach + 1)]
        if left.sum() >= minimum and left[-min(2, len(left)):].any(): rows["left"].append(y)
        if right.sum() >= minimum and right[:min(2, len(right))].any(): rows["right"].append(y)
    for x in range(max(0, x1 + margin), min(w, x2 - margin + 1)):
        top = ink[max(0, y1 - reach):min(h, y1 + 2), x]
        bottom = ink[max(0, y2 - 1):min(h, y2 + reach + 1), x]
        if top.sum() >= minimum and top[-min(2, len(top)):].any(): rows["top"].append(x)
        if bottom.sum() >= minimum and bottom[:min(2, len(bottom))].any(): rows["bottom"].append(x)
    output = []
    pin_length = max(6.0, 13.0 * scale)
    for side, coordinates in rows.items():
        for group in _groups(coordinates, max(2, int(round(3 * scale)))):
            coordinate = float(np.median(group))
            if side == "left": base, tip = (float(x1), coordinate), (max(0.0, x1 - pin_length), coordinate)
            elif side == "right": base, tip = (float(x2), coordinate), (min(float(w - 1), x2 + pin_length), coordinate)
            elif side == "top": base, tip = (coordinate, float(y1)), (coordinate, max(0.0, y1 - pin_length))
            else: base, tip = (coordinate, float(y2)), (coordinate, min(float(h - 1), y2 + pin_length))
            output.append({"tip": tip, "base": base, "side": side, "method": "boundary_wire_support", "wire_support_score": len(group)})
    # Collapse duplicate edge detections near corners.
    unique = []
    for item in output:
        old = next((candidate for candidate in unique if item["side"] == candidate["side"] and math.dist(item["tip"], candidate["tip"]) < 7 * scale), None)
        if old is not None:
            if item["wire_support_score"] > old["wire_support_score"]: old.update(item)
        else:
            unique.append(item)
    return unique


class KeypointProvider:
    """Optional interface for HAWP or another junction provider."""
    name = "disabled"

    def candidates(self, image):
        return []


def terminal_candidates_v3(image, component, texts=(), keypoint_provider=None):
    if component.type in TWO_TERMINAL:
        return simple_terminals(component)
    if component.type == "crystal_4pin":
        return crystal4_terminals(component)
    if component.type == "gnd":
        x1, y1, x2, _ = component.body_bbox or component.bbox
        cx = (x1 + x2) / 2
        return [{"tip": (cx, max(0.0, y1 - (y2 := max(4.0, (component.bbox[3] - component.bbox[1]) * .5)))), "base": (cx, y1), "side": "top", "method": "ground_geometry", "wire_support_score": None}]
    return boundary_wire_terminals(image, component, texts)
