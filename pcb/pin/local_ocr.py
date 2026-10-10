"""Targeted image-only pin OCR with rotation-aware full-image coordinates.

This experiment never creates terminals or reads annotations. It freezes V7's
accepted pins and only attempts to recover abstained box/block terminals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..io import write_json
from ..ocr_backends import _bbox_iou, deduplicate_texts
from ..pin_semantics import PIN_NUMBER, SIGNAL
from ..pin_semantics_v5 import _name_score, _number_score
from ..pin_semantics_v7 import assign_pin_semantics_v7, directional_candidate_is_exportable
from ..schema import Text
from ..text_detection import classify_tokens


@dataclass(frozen=True)
class LocalPinOCRConfig:
    rotations: tuple[int, ...] = (0, 90, 180, 270)
    magnification: float = 3.0
    padding: int = 12
    min_confidence: float = .8
    min_joint_score: float = 6.0
    maximum_normal_extent: float = 120.0
    minimum_tangent_half_width: float = 8.0
    maximum_tangent_half_width: float = 24.0
    conflict_iou: float = .5
    conflict_confidence_margin: float = .12
    detector_max_side: int = 512


def rotate_points(points, width: int, height: int, angle: int, *, inverse=False):
    """Transform pixel-center coordinates through np.rot90 (counterclockwise)."""
    p = np.asarray(points, dtype=float)
    x, y = p[..., 0], p[..., 1]
    if angle == 0:
        return p.copy()
    if angle not in (90, 180, 270):
        raise ValueError("Only quarter-turn OCR rotations are supported")
    if inverse:
        if angle == 90:
            return np.stack((width - 1 - y, x), axis=-1)
        if angle == 180:
            return np.stack((width - 1 - x, height - 1 - y), axis=-1)
        return np.stack((y, height - 1 - x), axis=-1)
    if angle == 90:
        return np.stack((y, width - 1 - x), axis=-1)
    if angle == 180:
        return np.stack((width - 1 - x, height - 1 - y), axis=-1)
    return np.stack((height - 1 - y, x), axis=-1)


def terminal_rois(component, terminals, index: int, image_shape, config):
    """Separate outside number and inside name windows; use same-side pitch."""
    terminal = terminals[index]
    side = terminal.get("side")
    if side not in ("left", "right", "top", "bottom"):
        return []
    x1, y1, x2, y2 = component.body_bbox or component.bbox
    bx, by = terminal["base"]
    vertical_side = side in ("left", "right")
    tangent = by if vertical_side else bx
    distances = [abs((t["base"][1] if vertical_side else t["base"][0]) - tangent)
                 for j, t in enumerate(terminals) if j != index and t.get("side") == side]
    pitch = min((d for d in distances if d > 0), default=40.0)
    half = min(config.maximum_tangent_half_width,
               max(config.minimum_tangent_half_width, .45 * pitch))
    span = max(1.0, (x2 - x1) if vertical_side else (y2 - y1))
    outside = max(30.0, min(config.maximum_normal_extent, .25 * span))
    inside = min(config.maximum_normal_extent, .5 * span)
    if side == "left":
        boxes = [(x1-outside, by-half, x1+2, by+half),
                 (x1+2, by-half, x1+inside, by+half)]
    elif side == "right":
        boxes = [(x2-2, by-half, x2+outside, by+half),
                 (x2-inside, by-half, x2-2, by+half)]
    elif side == "top":
        boxes = [(bx-half, y1-outside, bx+half, y1+2),
                 (bx-half, y1+2, bx+half, y1+inside)]
    else:
        boxes = [(bx-half, y2-2, bx+half, y2+outside),
                 (bx-half, y2-inside, bx+half, y2-2)]
    height, width = image_shape[:2]
    result = []
    for kind, box in zip(("number", "name"), boxes):
        a, b, c, d = box
        clipped = (max(0, math.floor(a)), max(0, math.floor(b)),
                   min(width, math.ceil(c)), min(height, math.ceil(d)))
        if clipped[2] - clipped[0] >= 3 and clipped[3] - clipped[1] >= 3:
            result.append((kind, clipped))
    return result


class RotatedLocalPinOCR:
    """Detect and recognize each local crop; preserve polygons across rotations."""

    def __init__(self, cache_dir=None, *, config=None, engine=None):
        self.config = config or LocalPinOCRConfig()
        self.cache = Path(cache_dir) if cache_dir else None
        self.engine = engine
        self.calls = self.cache_hits = 0

    def recognize_roi(self, image, box, kind):
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        if not crop.size:
            return []
        try:
            package = version("rapidocr-onnxruntime")
        except PackageNotFoundError:
            package = "unknown"
        signature = json.dumps({"version": "pin-local-v2-max-det", "package": package,
                                "shape": list(crop.shape), "config": asdict(self.config)},
                               sort_keys=True).encode()
        key = hashlib.sha256(crop.tobytes() + signature).hexdigest()
        path = self.cache / (key + ".json") if self.cache else None
        if path and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.cache_hits += 1
        else:
            if self.engine is None:
                from rapidocr_onnxruntime import RapidOCR
                # Full-image defaults upscale the crop's short side to 2048.
                # Local crops are already explicitly enlarged; cap the long
                # side instead, without changing the existing full-image OCR.
                self.engine = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=2,
                    det_limit_side_len=self.config.detector_max_side, det_limit_type="max")
            scaled = cv2.resize(crop, None, fx=self.config.magnification,
                                fy=self.config.magnification, interpolation=cv2.INTER_CUBIC)
            sy, sx = scaled.shape[0] / crop.shape[0], scaled.shape[1] / crop.shape[1]
            pad = self.config.padding
            padded = cv2.copyMakeBorder(scaled, pad, pad, pad, pad,
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
            height, width = padded.shape[:2]
            raw = []
            for angle in self.config.rotations:
                rotated = np.ascontiguousarray(np.rot90(padded, angle // 90))
                output, _ = self.engine(rotated, use_cls=False)
                self.calls += 1
                for polygon, reading, score in output or []:
                    if float(score) < self.config.min_confidence:
                        continue
                    points = rotate_points(polygon, width, height, angle, inverse=True)
                    points[:, 0] = (points[:, 0] - pad) / sx
                    points[:, 1] = (points[:, 1] - pad) / sy
                    a, b = np.min(points, axis=0)
                    c, d = np.max(points, axis=0)
                    a, b = max(0.0, float(a)), max(0.0, float(b))
                    c, d = min(float(crop.shape[1]), float(c)), min(float(crop.shape[0]), float(d))
                    if c > a and d > b:
                        raw.append({"text": str(reading), "bbox": [a, b, c, d],
                                    "score": float(score), "rotation": angle})
            if path:
                write_json(path, raw)
        return [Text(row["text"], (row["bbox"][0]+x1, row["bbox"][1]+y1,
                                   row["bbox"][2]+x1, row["bbox"][3]+y1), row["score"],
                     source_id=f"pin_local:{kind}:{row['rotation']}") for row in raw]


def _usable_tokens(tokens, config):
    """Abstain when high-confidence rotations disagree on the same text box."""
    dedup = deduplicate_texts(tokens)
    ambiguous = set()
    for i, a in enumerate(dedup):
        for j, b in enumerate(dedup[:i]):
            if (_bbox_iou(a.bbox, b.bbox) >= config.conflict_iou
                    and abs(a.score - b.score) < config.conflict_confidence_margin):
                ambiguous.update((i, j))
    return [t for i, t in enumerate(dedup) if i not in ambiguous], len(ambiguous)


def refine_pin_semantics_local(image, component, terminals, roles, reader):
    """Try joint local OCR recovery without replacing any accepted baseline pin."""
    before, before_events = assign_pin_semantics_v7(component, terminals, roles)
    statistics = {"roi_count": 0, "local_token_count": 0, "ambiguous_token_count": 0,
                  "attempted_terminals": 0, "recovered_pins": 0, "candidates": []}
    if (component.type not in ("box", "block")
            or str(component.key).startswith("UNRESOLVED_")):
        return before, before_events, statistics
    target_indices = [i for i, pin in enumerate(before) if not pin.exportable]
    if not target_indices:
        return before, before_events, statistics
    local_tokens = []
    for index in target_indices:
        statistics["attempted_terminals"] += 1
        terminal = terminals[index]
        for kind, box in terminal_rois(component, terminals, index, image.shape, reader.config):
            tokens = reader.recognize_roi(image, box, kind)
            statistics["roi_count"] += 1
            local_tokens.extend(tokens)
            statistics["candidates"].append({"terminal_index": index, "kind": kind,
                "roi": box, "tokens": [asdict(token) for token in tokens]})
    usable, ambiguous = _usable_tokens(local_tokens, reader.config)
    statistics["ambiguous_token_count"] = ambiguous
    statistics["local_token_count"] = len(usable)
    local_roles = classify_tokens(usable, [component.body_bbox or component.bbox])
    # Keep the same geometric number/name scores. A local reading is useful
    # only in the relevant role and in an existing terminal's actual row.
    accepted_roles = []
    for role in local_roles:
        appropriate = ((PIN_NUMBER.fullmatch(role.normalized)
                        and max((_number_score(role, terminals[i], component)
                                 for i in target_indices), default=0.0) > 0)
                       or (SIGNAL.fullmatch(role.normalized)
                           and max((_name_score(role, terminals[i], component)
                                    for i in target_indices), default=0.0) > 0))
        if appropriate:
            accepted_roles.append(role)
    if not accepted_roles:
        return before, before_events, statistics
    after, after_events = assign_pin_semantics_v7(component, terminals, [*roles, *accepted_roles])
    used_numbers = {pin.number for pin in before if pin.exportable}
    for index in target_indices:
        pin, event = after[index], after_events[index]
        local_evidence = any(
            (role.token.text == event.get("number_token") and _number_score(role, terminals[index], component) > 0)
            or (role.token.text == event.get("name_token") and _name_score(role, terminals[index], component) > 0)
            for role in accepted_roles)
        accepted = bool(pin.exportable and local_evidence
                        and directional_candidate_is_exportable(event)
                        and float(event.get("joint_assignment_score") or 0) >= reader.config.min_joint_score
                        and pin.number not in used_numbers)
        before_events[index]["local_ocr_attempted"] = True
        before_events[index]["local_ocr_recovered"] = accepted
        if not accepted:
            before_events[index]["local_ocr_rejection"] = (
                "duplicate_number" if pin.number in used_numbers else "insufficient_joint_local_evidence")
            continue
        pin.number_source = "targeted_rotated_local_ocr_ordered"
        event["local_ocr_attempted"] = True
        event["local_ocr_recovered"] = True
        event["semantic_method"] = "targeted_rotated_local_ocr_ordered"
        before[index], before_events[index] = pin, event
        used_numbers.add(pin.number)
        statistics["recovered_pins"] += 1
    # Geometry comes exclusively from the original localization candidates.
    if len(before) != len(terminals):
        raise AssertionError("Local OCR changed terminal cardinality")
    for pin, terminal in zip(before, terminals):
        if pin.tip != terminal["tip"] or pin.base != terminal["base"] or pin.side != terminal["side"]:
            raise AssertionError("Local OCR changed terminal geometry")
    return before, before_events, statistics
