"""V3 wire mask: component/text suppression with terminal corridors."""
from __future__ import annotations
import cv2
import numpy as np
from .v2 import extract_wire_v2


def _terminal_corridors(scene, shape, stroke):
    h, w = shape[:2]
    corridor = np.zeros((h, w), np.uint8)
    width = max(2, int(round(max(1.0, stroke) * 1.4)))
    for component in scene.components:
        for pin in component.pins:
            if pin.base is None:
                continue
            a = tuple(int(round(x)) for x in pin.base)
            b = tuple(int(round(x)) for x in pin.tip)
            cv2.line(corridor, a, b, 255, width * 2 + 1)
            # Keep a short outward continuation where the pin meets a bus.
            vector = np.asarray(pin.tip, float) - np.asarray(pin.base, float)
            norm = float(np.linalg.norm(vector))
            if norm > .5:
                end = np.asarray(pin.tip, float) + vector / norm * max(3, 2 * stroke)
                cv2.line(corridor, b, tuple(int(round(x)) for x in end), 255, width * 2 + 1)
    return corridor


def build_suppressed_wire_mask(image, scene):
    mask, color_debug, suppression = extract_wire_v2(image, scene)
    stroke = float(scene.diagnostics.get("wire_stroke_width", 1.0))
    corridor = _terminal_corridors(scene, image.shape, stroke)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    raw = ((gray < 205) | ((hsv[:, :, 1] >= 45) & (hsv[:, :, 2] < 248))).astype(np.uint8) * 255
    restore = cv2.bitwise_and(raw, corridor)
    mask = cv2.bitwise_or(mask, restore)
    # A component body remains an electrical barrier. Corridors begin at its
    # boundary and therefore do not punch a path through the interior.
    barrier = np.zeros(mask.shape, np.uint8)
    decisions = []
    for component in scene.components:
        x1, y1, x2, y2 = map(lambda x: int(round(x)), component.body_bbox or component.bbox)
        if x2 - x1 <= 2 or y2 - y1 <= 2:
            continue
        cv2.rectangle(barrier, (max(0, x1 + 2), max(0, y1 + 2)), (min(mask.shape[1] - 1, x2 - 2), min(mask.shape[0] - 1, y2 - 2)), 255, -1)
        decisions.append({"component": component.key, "bbox": (x1, y1, x2, y2), "action": "interior_removed"})
    removed = int(((mask > 0) & (barrier > 0)).sum())
    mask[barrier > 0] = 0
    suppression_with_barrier = cv2.bitwise_or(suppression, barrier)
    scene.diagnostics.update({
        "wire_extractor": "adaptive_v3_suppressed", "terminal_corridor_pixels": int((corridor > 0).sum()),
        "terminal_corridor_restored_pixels": int((restore > 0).sum()), "component_barrier_removed_pixels": removed,
        "component_barrier_decisions": decisions,
    })
    return mask, color_debug, suppression_with_barrier, corridor


extract_wire_v3 = build_suppressed_wire_mask
