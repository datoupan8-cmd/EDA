"""Isolated matched-box experiment inspired by SINA's skeleton annulus.

Only box terminal candidate generation changes. Every variant keeps V3's
normal tip extension; an annulus centroid is a boundary coordinate, not a tip.
No target, case identifier, source style, model or OCR service is consumed here.
SINA reference copyright (c) 2025 MEDAL Research Group, MIT; see
docs/SINA_LICENSE.txt and reports/pin_sina_source_audit.md for adaptations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np
from skimage.morphology import skeletonize

from ...core.interfaces import PinLocalizationOutput
from ...pin_detection import boundary_wire_terminals
from .v3 import PinLocalizationStageV3


@dataclass(frozen=True)
class BoxCandidateConfig:
    # A retained stroke must start at the body and continue outward, with
    # independently visible ink outside the OCR mask. No synthesized pixels.
    minimum_run_factor: float = 2.0
    annulus_pixels: float = 3.0  # SINA PIN_DIST, scaled to this image's V3 scale.


CONFIG = BoxCandidateConfig()
SIDES = ("left", "right", "top", "bottom")


def raw_ink(image: np.ndarray) -> np.ndarray:
    """Exact V3 intensity/color rule; no source-name or color-only gate."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (gray < 220) | ((hsv[:, :, 1] > 45) & (hsv[:, :, 2] < 250))


def text_mask(shape: tuple[int, int], texts) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    for token in texts:
        a, b, c, d = (int(round(v)) for v in token.bbox)
        cv2.rectangle(mask, (max(0, a), max(0, b)), (min(width - 1, c), min(height - 1, d)), 1, -1)
    return mask.astype(bool)


def dimensions(shape, component):
    height, width = shape
    box = tuple(int(round(v)) for v in (component.body_bbox or component.bbox))
    scale = max(.55, min(2.2, max(height, width) / 1200))
    return box, scale, max(8, int(round(20 * scale))), max(3, int(round(4 * scale)))


def preserve_strokes(ink, masked, component, config=CONFIG):
    """Restore only raw pixels on continuous outward strokes with mask-free anchors.

    A one-dimensional run touches the body, is at least twice V3's minimum
    support, and has V3's minimum number of unmasked outward ink pixels.
    Grouping/dedup/tip are not changed by this function.
    """
    height, width = ink.shape
    (x1, y1, x2, y2), scale, reach, margin = dimensions(ink.shape, component)
    minimum = max(3, int(round(4 * scale)))
    required = max(minimum + 1, int(math.ceil(config.minimum_run_factor * minimum)))
    restored = np.zeros_like(ink)
    events = []
    bounds = {
        "left": (x1, range(max(0, y1 + margin), min(height, y2 - margin + 1))),
        "right": (x2, range(max(0, y1 + margin), min(height, y2 - margin + 1))),
        "top": (y1, range(max(0, x1 + margin), min(width, x2 - margin + 1))),
        "bottom": (y2, range(max(0, x1 + margin), min(width, x2 - margin + 1))),
    }
    for side, (boundary, tangents) in bounds.items():
        for tangent in tangents:
            offsets = np.arange(1, reach + 1)
            normal = boundary - offsets if side in {"left", "top"} else boundary + offsets
            limit = width if side in {"left", "right"} else height
            normal = normal[(normal >= 0) & (normal < limit)]
            if not len(normal):
                continue
            yy, xx = (np.full(len(normal), tangent), normal) if side in {"left", "right"} else (normal, np.full(len(normal), tangent))
            profile = ink[yy, xx]
            missing = np.flatnonzero(~profile)
            run = int(missing[0]) if len(missing) else len(profile)
            if run < required or int((~masked[yy[:run], xx[:run]]).sum()) < minimum:
                continue
            covered = masked[yy[:run], xx[:run]]
            if not covered.any():
                continue
            restored[yy[:run][covered], xx[:run][covered]] = True
            events.append({"side": side, "tangent": tangent, "outward_run": run, "visible_anchor_count": int((~covered).sum()), "restored_pixels": int(covered.sum())})
    kept = (ink & ~masked) | restored
    assert not np.any(kept & ~ink), "Restoration must not fabricate ink"
    return kept, restored, events


def tip_from_base(base, side, extension, shape):
    height, width = shape
    x, y = base
    if side == "left":
        return max(0., x - extension), y
    if side == "right":
        return min(float(width - 1), x + extension), y
    if side == "top":
        return x, max(0., y - extension)
    return x, min(float(height - 1), y + extension)


def skeleton_terminals(kept, component, components, config=CONFIG):
    """Cluster skeleton components in a dilated bbox's outer ring, like SINA.

    Adaptations: V3 ink, frozen bodies, ROI computation, per-owner/per-side
    output and V3 tip placement. No morphological gap filling or global NMS.
    """
    height, width = kept.shape
    (x1, y1, x2, y2), scale, reach, margin = dimensions(kept.shape, component)
    radius = max(1, int(round(config.annulus_pixels * scale)))
    pad = reach + radius + 2
    a, b, c, d = max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad + 1), min(height, y2 + pad + 1)
    if a >= c or b >= d:
        return [], {"annulus_pixels": radius, "regions": [], "rejected": []}, np.zeros((0, 0), bool)
    sk = skeletonize(kept[b:d, a:c]).astype(np.uint8)
    # SINA also clears body interiors after skeletonization. Preserve current
    # body boxes, and never reconnect two terminals through a component.
    for other in components:
        ox1, oy1, ox2, oy2 = (int(round(v)) for v in (other.body_bbox or other.bbox))
        u, v, s, t = max(a, ox1), max(b, oy1), min(c - 1, ox2), min(d - 1, oy2)
        if u <= s and v <= t:
            sk[v-b:t-b+1, u-a:s-a+1] = 0
    body = np.zeros_like(sk)
    cv2.rectangle(body, (x1-a, y1-b), (x2-a, y2-b), 1, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    ring = cv2.dilate(body, kernel) & (1-body)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(sk & ring, connectivity=8)
    rows, rejected = [], []
    for label in range(1, count):
        py, px = np.where(labels == label)
        gx, gy = px+a, py+b
        sides = {"left": bool(np.any(gx < x1)), "right": bool(np.any(gx > x2)), "top": bool(np.any(gy < y1)), "bottom": bool(np.any(gy > y2))}
        active = [side for side in SIDES if sides[side]]
        if len(active) != 1:
            rejected.append({"reason": "ambiguous_corner_side", "sides": active, "area": int(stats[label, cv2.CC_STAT_AREA])})
            continue
        side = active[0]
        cx, cy = centroids[label]+np.asarray([a, b])
        tangent = float(cy if side in {"left", "right"} else cx)
        low, high = (y1+margin, y2-margin) if side in {"left", "right"} else (x1+margin, x2-margin)
        if not low <= tangent <= high:
            rejected.append({"reason": "v3_corner_margin", "side": side, "tangent": tangent})
            continue
        base = {"left": (float(x1), tangent), "right": (float(x2), tangent), "top": (tangent, float(y1)), "bottom": (tangent, float(y2))}[side]
        base = (min(width-1., max(0., base[0])), min(height-1., max(0., base[1])))
        rows.append({"tip": tip_from_base(base, side, max(6., 13.*scale), kept.shape), "base": base, "side": side, "method": "sina_skeleton_annulus", "wire_support_score": int(stats[label, cv2.CC_STAT_AREA]), "skeleton_centroid": (float(cx), float(cy))})
    # Mirror V3 side iteration and close same-side duplicate suppression.
    output = []
    for side in SIDES:
        for item in sorted((r for r in rows if r['side'] == side), key=lambda r: r['base'][1] if side in {'left','right'} else r['base'][0]):
            old = next((r for r in output if r['side'] == side and math.dist(r['tip'], item['tip']) < 7*scale), None)
            if old is None:
                output.append(item)
            elif item['wire_support_score'] > old['wire_support_score']:
                old.update(item)
    return output, {"annulus_pixels": radius, "regions": rows, "rejected": rejected, "crop": (a,b,c,d)}, sk


class BoxCandidateExperimentStage:
    """Config-selectable box-only strategy; production V3 is unchanged."""
    def __init__(self, *, preserve: bool, skeleton: bool):
        self.preserve = preserve
        self.skeleton = skeleton

    def run(self, image, component, context) -> PinLocalizationOutput:
        baseline = PinLocalizationStageV3().run(image, component, context)
        ink = raw_ink(image)
        masked = text_mask(ink.shape, component.texts)
        rows, diagnostics = [], []
        for item, original in baseline.terminals:
            if item.type != "box":
                rows.append((item, original))
                continue
            kept, restored, events = preserve_strokes(ink, masked, item) if self.preserve else (ink & ~masked, np.zeros_like(ink), [])
            if self.skeleton:
                terminals, detail, _ = skeleton_terminals(kept, item, component.components)
            else:
                cleaned = image.copy()
                cleaned[masked & ~restored] = 255
                terminals = boundary_wire_terminals(cleaned, item, ())
                detail = {}
            for terminal in terminals:
                for field in ("tip", "base"):
                    x, y = terminal[field]
                    terminal[field] = (min(context.width-1., max(0., x)), min(context.height-1., max(0., y)))
            rows.append((item, terminals))
            diagnostics.append({"component": item.key, "original_count": len(original), "candidate_count": len(terminals), "preserved_pixels": int(restored.sum()), "preservation_events": events, **detail})
        return PinLocalizationOutput(rows, {"box_candidate_experiment": {"preserve": self.preserve, "skeleton": self.skeleton, "rows": diagnostics}})


class BoxScanPreserveStage(BoxCandidateExperimentStage):
    def __init__(self):
        super().__init__(preserve=True, skeleton=False)


class BoxSkeletonMaskedStage(BoxCandidateExperimentStage):
    def __init__(self):
        super().__init__(preserve=False, skeleton=True)


class BoxSkeletonPreserveStage(BoxCandidateExperimentStage):
    def __init__(self):
        super().__init__(preserve=True, skeleton=True)
