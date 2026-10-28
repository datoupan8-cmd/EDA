"""Image-evidence-only power field recovery, isolated from the active profile.

These functions never receive target annotations or open files. Coordinates in
predictions are bottom-left; OCR coordinates are top-left.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass

from .geometry import bbox_iou
from .assignment import maximum_weight_assignment


GROUND_LABEL = re.compile(r"(?:[A-Z0-9]+[_-])?(?:[ADPSCIH]?GND|VSS|EARTH)[A-Z0-9_.+-]*", re.I)
SUPPLY_LABEL = re.compile(
    r"(?:[+-]?\d+(?:\.\d+)?V[A-Z0-9_.+-]*|"
    r"(?:VCC|VDD|VSS|VIN|VOUT|VBAT|VBATT|VBUS|AVDD|AVCC|DVDD|DVCC|IOVDD|PVDD|PVIN|VSYS|VREG|VREF)[A-Z0-9_.+-]*|"
    r"[A-Z][A-Z0-9]*[_+-](?:VCC|VDD|VBUS|VBAT|[+-]?\d+(?:\.\d+)?V)[A-Z0-9_.+-]*)", re.I)


@dataclass(frozen=True)
class PowerRecoveryOptions:
    minimum_ocr_confidence: float = .8
    maximum_gap_text_heights: float = 2.
    minimum_gap_limit_px: float = 12.
    physical_text_iou: float = .5
    minimum_assignment_score: float = .05


def top_box(component, height):
    b = component["bbox"]
    return b[0], height-b[3], b[2], height-b[1]


def gap(a, b):
    return math.hypot(max(a[0]-b[2], b[0]-a[2], 0), max(a[1]-b[3], b[1]-a[3], 0))


def label_kind(text):
    """Domain syntax, not a dictionary derived from evaluation targets."""
    text = text.strip()
    if GROUND_LABEL.fullmatch(text):
        return "gnd"
    if SUPPLY_LABEL.fullmatch(text):
        return "v"
    return None


def recover_power_values(prediction, tokens, image_height, options=PowerRecoveryOptions()):
    result = copy.deepcopy(prediction)
    bodies = [(key, c, top_box(c, image_height)) for key, c in result["components"].items()]
    powers = [(key, c, b) for key, c, b in bodies if c["type"] in {"gnd", "v"}]
    eligible = []
    for index, token in enumerate(tokens):
        if token.confidence < options.minimum_ocr_confidence or label_kind(token.text) is None:
            continue
        b = token.bbox_xyxy
        center = (b[0]+b[2])/2, (b[1]+b[3])/2
        # Chip pin names such as VCC/GND are not power-symbol labels.
        if any(c["type"] not in {"gnd", "v"} and body[0] <= center[0] <= body[2]
               and body[1] <= center[1] <= body[3] for _, c, body in bodies):
            continue
        eligible.append((index, token))
    # Union overlapping alternative OCR readings into one physical text owner.
    parents = list(range(len(eligible)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i, (_, a) in enumerate(eligible):
        for j in range(i):
            if bbox_iou(a.bbox_xyxy, eligible[j][1].bbox_xyxy) >= options.physical_text_iou:
                parents[find(i)] = find(j)
    grouped = {}
    for i, item in enumerate(eligible):
        grouped.setdefault(find(i), []).append(item)
    groups = list(grouped.values())
    weights, choices = [], {}
    for i, group in enumerate(groups):
        row = []
        for j, (_, c, body) in enumerate(powers):
            best = (0., None)
            for index, token in group:
                if label_kind(token.text) != c["type"]:
                    continue
                if c["value"] is not None and token.text.strip() != c["value"]:
                    continue
                b = token.bbox_xyxy
                th = max(3., b[3]-b[1])
                distance = gap(b, body)
                if distance > max(options.minimum_gap_limit_px, options.maximum_gap_text_heights*th):
                    continue
                tc = (b[0]+b[2])/2, (b[1]+b[3])/2
                bc = (body[0]+body[2])/2, (body[1]+body[3])/2
                score = token.confidence * math.exp(-distance/(1.5*th)-math.dist(tc, bc)/(3*th))
                if score > best[0]:
                    best = score, (index, token)
            row.append(best[0] if best[0] >= options.minimum_assignment_score else 0.)
            choices[i, j] = best[1]
        weights.append(row)
    events = []
    for i, j in maximum_weight_assignment(weights):
        if weights[i][j] <= 0:
            continue
        key, c, _ = powers[j]
        if c["value"] is not None:
            continue
        index, token = choices[i, j]
        c["value"] = token.text.strip()
        events.append({"key": key, "type": c["type"], "before": None, "after": c["value"],
                       "ocr_index": index, "ocr_bbox_top_left": list(token.bbox_xyxy),
                       "confidence": token.confidence, "assignment_score": weights[i][j],
                       "physical_group": i})
    return result, events


def restore_voltage_symbols(prediction, candidates, image_height, *, top_origin=True):
    """Restore frozen pre-filter v candidates, preserving their existing fields.

    No new detector threshold or GT-dependent selection is introduced. Existing
    components/pins/nets are preserved; each added owner gets an empty pin group.
    """
    result = copy.deepcopy(prediction)
    events = []
    for index, c in enumerate(candidates):
        if c["type"] != "v":
            continue
        b = c["bbox"]
        box = [round(v, 3) for v in ([b[0], image_height-b[3], b[2], image_height-b[1]] if top_origin else b)]
        if any(old["type"] == "v" and old["bbox"] == box for old in result["components"].values()):
            continue
        suffix = len(events)+1
        key = f"VREC_{suffix:04d}"
        while key in result["components"]:
            suffix += 1
            key = f"VREC_{suffix:04d}"
        result["components"][key] = {"Name": c.get("Name", c.get("name")) or key,
                                     "type": "v", "value": c.get("value"), "bbox": box}
        result["pins"][key] = {}
        events.append({"key": key, "candidate_index": index, "bbox": box,
                       "value": c.get("value"), "confidence": c.get("confidence")})
    return result, events
