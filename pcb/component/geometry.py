"""OCR-independent symbol proposals and global V3 text association."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json, math
import cv2
import numpy as np
from ..assignment import maximum_weight_assignment
from ..schema import Component, Text
from ..text_detection import classify_tokens, value_compatible
from ..annotation import unique_keys

PRIORS = json.loads((Path(__file__).resolve().parents[1] / "v3_priors.json").read_text(encoding="utf-8"))


def _linear_sum_assignment(cost):
    """Dependency-free equivalent for the finite cost matrices used here."""
    if cost.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    ceiling = float(np.max(cost)) + 1.0
    pairs = maximum_weight_assignment((ceiling - cost).tolist())
    return np.asarray([row for row, _ in pairs], dtype=int), np.asarray([column for _, column in pairs], dtype=int)


def source_from_name(name: str) -> str:
    for source in ("KiCad", "Altium Designer", "Datasheet", "jlc", "other"):
        if source.lower() in str(name).lower():
            return source
    return "other"


def _ink_mask(image, texts):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (((gray < 220) | ((hsv[:, :, 1] > 45) & (hsv[:, :, 2] < 250))).astype(np.uint8) * 255)
    text_mask = np.zeros(mask.shape, np.uint8)
    for token in texts:
        x1, y1, x2, y2 = map(lambda x: int(round(x)), token.bbox)
        cv2.rectangle(text_mask, (max(0, x1 - 1), max(0, y1 - 1)), (min(mask.shape[1] - 1, x2 + 1), min(mask.shape[0] - 1, y2 + 1)), 255, -1)
    symbol = mask.copy()
    symbol[text_mask > 0] = 0
    symbol[:2] = symbol[-2:] = 0
    symbol[:, :2] = symbol[:, -2:] = 0
    return mask, symbol, text_mask


def _merge_axis_lines(lines, axis):
    values = []
    for x1, y1, x2, y2 in lines:
        if axis == "h" and abs(y2 - y1) <= 3:
            values.append([float((y1 + y2) / 2), float(min(x1, x2)), float(max(x1, x2))])
        elif axis == "v" and abs(x2 - x1) <= 3:
            values.append([float((x1 + x2) / 2), float(min(y1, y2)), float(max(y1, y2))])
    values.sort()
    merged = []
    for pos, lo, hi in values:
        # Canny/Hough returns both edges of one antialiased stroke. Merge those
        # edges before looking for two distinct capacitor plates.
        hit = next((row for row in merged if abs(row[0] - pos) <= 4 and not (hi < row[1] - 14 or lo > row[2] + 14)), None)
        if hit:
            hit[0] = (hit[0] * hit[3] + pos) / (hit[3] + 1)
            hit[1], hit[2], hit[3] = min(hit[1], lo), max(hit[2], hi), hit[3] + 1
        else:
            merged.append([pos, lo, hi, 1])
    return merged


def _hough_lines(symbol, scale, long=False):
    edges = cv2.Canny(symbol, 40, 130)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(10, int((18 if long else 12) * scale)), minLineLength=max(6, int((28 if long else 7) * scale)), maxLineGap=max(2, int((10 if long else 3) * scale)))
    return [tuple(map(int, np.asarray(line).reshape(-1)[:4])) for line in raw] if raw is not None else []


def detect_large_box_components(image, symbol, scale):
    h, w = symbol.shape
    lines = _hough_lines(symbol, scale, long=True)
    hs = [x for x in _merge_axis_lines(lines, "h") if x[2] - x[1] >= 35 * scale]
    vs = [x for x in _merge_axis_lines(lines, "v") if x[2] - x[1] >= 35 * scale]
    candidates = []
    for top in hs:
        for bottom in hs:
            if bottom[0] - top[0] < 35 * scale:
                continue
            y1, y2 = top[0], bottom[0]
            lo, hi = max(top[1], bottom[1]), min(top[2], bottom[2])
            if hi - lo < 35 * scale:
                continue
            lefts = [v for v in vs if abs(v[0] - lo) <= 6 * scale and v[1] <= y1 + 8 * scale and v[2] >= y2 - 8 * scale]
            rights = [v for v in vs if abs(v[0] - hi) <= 6 * scale and v[1] <= y1 + 8 * scale and v[2] >= y2 - 8 * scale]
            if not lefts or not rights:
                continue
            x1 = min(lefts, key=lambda v: abs(v[0] - lo))[0]
            x2 = min(rights, key=lambda v: abs(v[0] - hi))[0]
            if x2 - x1 > .85 * w or y2 - y1 > .9 * h:
                continue
            candidates.append((x1, y1, x2, y2))
    # NMS by center/edge similarity.
    kept = []
    for box in sorted(candidates, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if any(max(abs(a - c) for a, c in zip(box, old)) <= 8 * scale for old in kept):
            continue
        kept.append(tuple(float(x) for x in box))
    return kept


def detect_capacitor_centers(symbol, scale):
    lines = _hough_lines(symbol, scale)
    output = []
    for axis, rows in (("h", _merge_axis_lines(lines, "h")), ("v", _merge_axis_lines(lines, "v"))):
        rows = [r for r in rows if 7 * scale <= r[2] - r[1] <= 90 * scale]
        for i, first in enumerate(rows):
            for second in rows[i + 1:]:
                sep = second[0] - first[0]
                if not 4 * scale <= sep <= 24 * scale:
                    continue
                overlap = min(first[2], second[2]) - max(first[1], second[1])
                shorter = min(first[2] - first[1], second[2] - second[1])
                if overlap < max(6 * scale, .55 * shorter):
                    continue
                along = (max(first[1], second[1]) + min(first[2], second[2])) / 2
                across = (first[0] + second[0]) / 2
                center = (along, across) if axis == "h" else (across, along)
                orientation = "vertical" if axis == "h" else "horizontal"
                if not any(math.dist(center, x[0]) < 7 * scale for x in output):
                    output.append((center, orientation, shorter, sep))
    return output


def detect_ground_components(symbol, scale):
    contours, _ = cv2.findContours(symbol, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, .06 * perimeter, True).reshape(-1, 2)
        if len(approx) != 3:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        area = abs(cv2.contourArea(contour))
        if not (10 * scale <= w <= 42 * scale and 9 * scale <= h <= 42 * scale and area >= 18 * scale * scale):
            continue
        bottom = approx[np.argmax(approx[:, 1])]
        top = approx[np.argsort(approx[:, 1])[:2]]
        if abs(int(top[0, 1]) - int(top[1, 1])) > max(4, 4 * scale) or abs(float(bottom[0]) - (x + w / 2)) > .3 * w:
            continue
        boxes.append((float(x), float(y), float(x + w), float(y + h)))
    kept = []
    for box in boxes:
        if not any(math.dist(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) < 8 * scale for b in kept):
            kept.append(box)
    return kept


def _clusters(source, typ):
    values = PRIORS["anchor_clusters"].get(f"{source}|{typ}") or PRIORS["anchor_clusters"].get(f"ALL|{typ}") or []
    if not values:
        return []
    maximum = max(x["count"] for x in values)
    return [x for x in values if x["count"] >= max(2, maximum * .025) and max(map(abs, x["center"][:2])) < 15]


def _prior_boxes(role, source, shape):
    h, w = shape[:2]
    th = max(4.0, role.token.bbox[3] - role.token.bbox[1])
    tx, ty = role.token.center
    output = []
    for row in _clusters(source, role.component_type):
        dx, dy, bw, bh = row["center"]
        cx, cy = tx + dx * th, ty + dy * th
        bw, bh = max(4, bw * th), max(4, bh * th)
        box = (max(0.0, cx - bw / 2), max(0.0, cy - bh / 2), min(float(w - 1), cx + bw / 2), min(float(h - 1), cy + bh / 2))
        if box[2] > box[0] and box[3] > box[1]:
            output.append((box, row["count"]))
    return output


def _box_visual_cost(box, symbol, prior_count):
    x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
    patch = symbol[max(0, y1):min(symbol.shape[0], y2 + 1), max(0, x1):min(symbol.shape[1], x2 + 1)]
    if patch.size == 0:
        return 100.0
    density = float((patch > 0).mean())
    density_penalty = 4.0 if density < .015 else abs(density - .16) * 3
    return density_penalty - .35 * math.log1p(prior_count)


def detect_components_v3(image, texts: list[Text], image_name=""):
    h, w = image.shape[:2]
    scale = max(.55, min(2.2, max(h, w) / 1200))
    raw_ink, symbol, text_mask = _ink_mask(image, texts)
    # Long box boundaries may pass directly behind pin labels.  Use raw ink for
    # the line proposal and reserve text-masked ink for small symbols.
    boxes = detect_large_box_components(image, raw_ink, scale)
    roles = classify_tokens(texts, boxes)
    caps = detect_capacitor_centers(symbol, scale)
    grounds = detect_ground_components(symbol, scale)
    src = source_from_name(image_name)
    components = []
    associations = []
    used_boxes = set()
    used_caps = set()
    anchors = [r for r in roles if r.role == "DESIGNATOR"]
    # Match box identifiers globally to geometry first.
    box_roles = [r for r in anchors if r.component_type == "box"]
    if box_roles and boxes:
        cost = np.full((len(box_roles), len(boxes)), 1e5, float)
        for i, role in enumerate(box_roles):
            for j, box in enumerate(boxes):
                tx, ty = role.token.center
                distance = math.hypot(max(box[0] - tx, 0, tx - box[2]), max(box[1] - ty, 0, ty - box[3]))
                cost[i, j] = distance / max(5, role.token.bbox[3] - role.token.bbox[1])
        rr, cc = _linear_sum_assignment(cost)
        for i, j in zip(rr, cc):
            if cost[i, j] > 6:
                continue
            role, box = box_roles[i], boxes[j]
            c = Component(role.designator, "box", box, observable_designator=role.designator, body_bbox=box, confidence=role.confidence / (1 + cost[i, j] / 5))
            components.append(c); used_boxes.add(j)
            associations.append({"token": role.token.text, "role": role.role, "component": c.key, "method": "large_rectangle", "score": float(cost[i, j]), "bbox": box})
    existing = {c.key for c in components}
    for role in anchors:
        if role.designator in existing:
            continue
        typ = role.component_type
        chosen = None; method = "aggregate_anchor_prior"; score = 99.0
        if typ == "c" and caps:
            expected = _prior_boxes(role, src, image.shape)
            for ci, (center, orientation, _, _) in enumerate(caps):
                if ci in used_caps:
                    continue
                d = min((math.dist(center, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) for b, _ in expected), default=math.dist(center, role.token.center))
                if d < score and d <= max(45 * scale, 5 * (role.token.bbox[3] - role.token.bbox[1])):
                    dims = PRIORS["dimensions"].get(f"{src}|c") or PRIORS["dimensions"].get("ALL|c")
                    bw, bh = dims["width"]["median"], dims["height"]["median"]
                    if orientation == "horizontal" and bh > bw: bw, bh = bh, bw
                    if orientation == "vertical" and bw > bh: bw, bh = bh, bw
                    chosen = (center[0] - bw / 2, center[1] - bh / 2, center[0] + bw / 2, center[1] + bh / 2)
                    score = d; method = "parallel_plate_geometry"; chosen_cap = ci
        if chosen is None:
            priors = _prior_boxes(role, src, image.shape)
            if priors:
                ranked = sorted((_box_visual_cost(box, symbol, count), box) for box, count in priors)
                score, chosen = ranked[0]
        if chosen is None:
            # Last-resort body is compact and centered next to the OCR token,
            # still independent of any case coordinates.
            th = max(6.0, role.token.bbox[3] - role.token.bbox[1])
            cx, cy = role.token.center[0], role.token.center[1] + 1.5 * th
            chosen = (cx - th, cy - th, cx + th, cy + th)
            method = "fallback_anchor_box"
        if method == "parallel_plate_geometry":
            used_caps.add(chosen_cap)
        c = Component(role.designator, typ, tuple(map(float, chosen)), observable_designator=role.designator, body_bbox=tuple(map(float, chosen)), confidence=max(.05, role.confidence / (1 + max(0, score) / 8)))
        components.append(c); existing.add(c.key)
        associations.append({"token": role.token.text, "role": role.role, "component": c.key, "method": method, "score": float(score), "bbox": chosen})
    # Ground serials are explicitly scoring-insensitive; geometry proposals can
    # therefore represent symbols whose internal official IDs are unobservable.
    for index, box in enumerate(grounds, 1):
        if any(math.dist(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), ((c.bbox[0] + c.bbox[2]) / 2, (c.bbox[1] + c.bbox[3]) / 2)) < 12 * scale for c in components):
            continue
        components.append(Component(f"GND_{index}", "gnd", box, body_bbox=box, confidence=.65, net_label="GND"))
    unique_keys(components)
    # Values/models are assigned one-to-one after components exist.
    value_roles = [r for r in roles if r.role == "VALUE"]
    eligible = [c for c in components if c.type in ("r", "c", "l")]
    if eligible and value_roles:
        cost = np.full((len(eligible), len(value_roles)), 1e5, float)
        for i, comp in enumerate(eligible):
            for j, role in enumerate(value_roles):
                if not value_compatible(comp.type, role.token.text):
                    continue
                cx, cy = (comp.bbox[0] + comp.bbox[2]) / 2, (comp.bbox[1] + comp.bbox[3]) / 2
                cost[i, j] = math.dist((cx, cy), role.token.center) / max(5, role.token.bbox[3] - role.token.bbox[1]) + (1 - role.confidence)
        rr, cc = _linear_sum_assignment(cost)
        for i, j in zip(rr, cc):
            if cost[i, j] <= 8:
                eligible[i].value = value_roles[j].token.text.strip()
                associations.append({"token": value_roles[j].token.text, "role": "VALUE", "component": eligible[i].key, "method": "global_value_assignment", "score": float(cost[i, j])})
    diagnostics = {
        "source_style": src, "text_roles": [{"text": r.token.text, "bbox": r.token.bbox, "role": r.role, "reason": r.reason, "confidence": r.confidence} for r in roles],
        "large_box_candidates": boxes, "capacitor_geometry_candidates": [{"center": x[0], "orientation": x[1]} for x in caps],
        "ground_candidates": grounds, "text_component_associations": associations,
        "component_candidates": len(anchors) + len(boxes) + len(grounds), "component_final": len(components),
    }
    return components, roles, diagnostics, {"symbol_mask": symbol, "text_mask": text_mask}
