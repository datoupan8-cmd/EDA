"""Exact V3 + F wire sequence used by the validated L39 combination."""
import copy

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .versions import WireStageV3
from .frame_guard import suppress_box_frames


class WireStageV3FrameGuard:
    def run(self, image, scene, context):
        return suppress_box_frames(image, scene, WireStageV3().run(image, scene, context))


class WireStageV4PaletteExpanded:
    """Experimental V3 pipeline retaining all palettes above the 72% gate."""
    def run(self, image, scene, context):
        from ..core.interfaces import WireStageOutput
        from .v3 import build_suppressed_wire_mask
        mask, color_debug, suppressed, corridor = build_suppressed_wire_mask(
            image, scene, max_palette_candidates=None, palette_score_ratio=.72)
        return WireStageOutput(mask, color_debug, suppressed, corridor)


class WireStageV4PaletteExpandedFrameGuard:
    def run(self, image, scene, context):
        return suppress_box_frames(image, scene, WireStageV4PaletteExpanded().run(image, scene, context))


class WireStageV5PaletteRecall:
    """Experimental V3 pipeline with uncapped palettes and a 71% score gate."""
    def run(self, image, scene, context):
        from ..core.interfaces import WireStageOutput
        from .v3 import build_suppressed_wire_mask
        mask, color_debug, suppressed, corridor = build_suppressed_wire_mask(
            image, scene, max_palette_candidates=None, palette_score_ratio=.71)
        return WireStageOutput(mask, color_debug, suppressed, corridor)


class WireStageV5PaletteRecallFrameGuard:
    def run(self, image, scene, context):
        return suppress_box_frames(image, scene, WireStageV5PaletteRecall().run(image, scene, context))


def filter_bent_candidate_bridges(baseline_mask, candidate_mask, stroke_width):
    """Reject bent new-only components that join separate baseline components.

    Straight extensions remain eligible because local mask evidence cannot
    distinguish a real missing straight wire from a false straight bridge.
    Existing baseline pixels are never removed. This opt-in heuristic is not
    an electrical-connectivity decision.
    """
    baseline = np.asarray(baseline_mask) > 0
    candidate = np.asarray(candidate_mask) > 0
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate masks must have identical shapes")
    filtered = np.asarray(candidate_mask).copy()
    added = (candidate & ~baseline).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(added, 8)
    baseline_count, baseline_labels, _, _ = cv2.connectedComponentsWithStats(
        baseline.astype(np.uint8), 8
    )
    rejected = np.zeros(baseline.shape, dtype=bool)
    decisions = []
    deviation_limit = max(2.0, 2.0 * max(1.0, float(stroke_width)))

    for label in range(1, component_count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        expanded = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        touched = np.unique(baseline_labels[expanded & baseline])
        touched = touched[touched > 0]
        if touched.size < 2:
            continue

        points_yx = np.argwhere(skeletonize(component))
        if points_yx.shape[0] < 3:
            continue
        points = points_yx[:, ::-1].astype(np.float64)
        centered = points - points.mean(axis=0)
        try:
            _, _, axes = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        normal = np.array([-axes[0, 1], axes[0, 0]])
        max_deviation = float(np.max(np.abs(centered @ normal)))
        if max_deviation <= deviation_limit:
            continue

        filtered[component] = 0
        rejected[component] = True
        x, y, width, height, _ = map(int, stats[label])
        decisions.append({
            "action": "reject_bent_bridge",
            "bbox_xyxy": [x, y, x + width, y + height],
            "candidate_pixels": area,
            "baseline_components_joined": int(touched.size),
            "max_axis_deviation_px": round(max_deviation, 3),
            "deviation_limit_px": round(deviation_limit, 3),
        })

    return filtered, rejected, {
        "baseline_component_count": max(0, baseline_count - 1),
        "candidate_added_component_count": max(0, component_count - 1),
        "rejected_component_count": len(decisions),
        "rejected_pixel_count": int(rejected.sum()),
        "deviation_limit_px": round(deviation_limit, 3),
        "decisions": decisions,
        "limitations": [
            "straight bridges are retained because mask geometry alone is ambiguous",
            "bent legitimate wires joining baseline fragments may be rejected",
        ],
    }


class WireStageV6PaletteBridgeGuard:
    """Experimental v5 recall candidate with a conservative bent-bridge guard."""

    def run(self, image, scene, context):
        from ..core.interfaces import WireStageOutput

        baseline_scene = copy.deepcopy(scene)
        baseline = WireStageV3FrameGuard().run(image, baseline_scene, context)
        candidate = WireStageV5PaletteRecallFrameGuard().run(image, scene, context)
        stroke = float(scene.diagnostics.get("wire_stroke_width", 1.0))
        mask, rejected, detail = filter_bent_candidate_bridges(
            baseline.mask, candidate.mask, stroke
        )
        scene.diagnostics["wire_candidate_bridge_guard"] = detail
        suppressed = (candidate.suppressed.copy() if candidate.suppressed is not None
                      else np.zeros(mask.shape, dtype=np.uint8))
        suppressed[rejected] = 255
        return WireStageOutput(
            mask, candidate.color_debug, suppressed, candidate.corridor,
            candidate.diagnostics, candidate.debug,
        )


def retain_corridor_anchored_components(candidate_masks, corridor_mask):
    """Keep image-derived, line-supported components that meet a pin corridor.

    A one-pixel dilation tolerates raster alignment at a terminal. This does
    not invent pixels or connect pins: retained pixels all come from a
    candidate mask produced by the ordinary Wire color and line-support
    pipeline.
    """
    corridor = np.asarray(corridor_mask) > 0
    if corridor.ndim != 2:
        raise ValueError("corridor_mask must be two-dimensional")
    near_corridor = cv2.dilate(corridor.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    result = np.zeros(corridor.shape, dtype=np.uint8)
    retained_components = 0
    for mask in candidate_masks:
        candidate = np.asarray(mask)
        if candidate.shape != corridor.shape:
            raise ValueError("candidate masks and corridor must have identical shapes")
        count, labels, _, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), 8)
        for label in range(1, count):
            component = labels == label
            if np.any(component & near_corridor):
                result[component] = 255
                retained_components += 1
    return result, retained_components


def filter_candidate_bridges(baseline_mask, candidate_mask):
    """Reject supplemental components that join multiple baseline regions.

    The complete baseline is preserved. Unlike the v6 bent-only guard, this
    conservative rule rejects straight as well as bent bridges. It is a mask
    safety heuristic, not an electrical-connectivity classifier.
    """
    baseline = np.asarray(baseline_mask) > 0
    candidate = np.asarray(candidate_mask) > 0
    if baseline.ndim != 2 or candidate.shape != baseline.shape:
        raise ValueError("baseline and candidate masks must be matching 2D arrays")
    filtered = candidate.astype(np.uint8) * 255
    filtered[baseline] = 255
    added = (candidate & ~baseline).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(added, 8)
    baseline_count, baseline_labels, _, _ = cv2.connectedComponentsWithStats(
        baseline.astype(np.uint8), 8
    )
    rejected = np.zeros(baseline.shape, dtype=bool)
    decisions = []
    for label in range(1, count):
        component = labels == label
        expanded = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        touched = np.unique(baseline_labels[expanded & baseline])
        touched = touched[touched > 0]
        if touched.size < 2:
            continue
        filtered[component] = 0
        rejected[component] = True
        x, y, width, height, area = map(int, stats[label])
        decisions.append({
            "action": "reject_component_joining_baseline_regions",
            "bbox_xyxy": [x, y, x + width, y + height],
            "candidate_pixels": area,
            "baseline_components_joined": int(touched.size),
        })
    return filtered, rejected, {
        "baseline_component_count": max(0, baseline_count - 1),
        "candidate_added_component_count": max(0, count - 1),
        "rejected_component_count": len(decisions),
        "rejected_pixel_count": int(rejected.sum()),
        "decisions": decisions,
        "limitations": [
            "components that bridge distinct baseline regions are rejected even when the wire may be genuine",
            "candidate components joining only one baseline region may still alter topology at crossings",
            "this image-mask rule does not determine electrical net identity",
        ],
    }


class WireStageV7PaletteCorridorRecall:
    """Non-default recall experiment anchored to pin corridors and base mask.

    It keeps supported palette components that meet predicted terminal
    corridors, then rejects supplemental components that bridge separate
    baseline mask regions. The default v3 output is always retained.
    """

    def run(self, image, scene, context):
        from ..core.interfaces import WireStageOutput
        from .v3 import build_suppressed_wire_mask, _terminal_corridors

        baseline_scene = copy.deepcopy(scene)
        baseline = WireStageV3FrameGuard().run(image, baseline_scene, context)
        candidate_scene = copy.deepcopy(scene)
        stroke = float(baseline_scene.diagnostics.get("wire_stroke_width", 1.0))
        terminal_corridor = _terminal_corridors(candidate_scene, image.shape, stroke)
        extra = np.zeros(terminal_corridor.shape, dtype=np.uint8)
        retained_components = 0

        def collect_candidate(_name, mask):
            nonlocal retained_components
            retained, count = retain_corridor_anchored_components(
                (mask,), terminal_corridor
            )
            np.bitwise_or(extra, retained, out=extra)
            retained_components += count

        _, color_debug, suppressed, corridor = build_suppressed_wire_mask(
            image, candidate_scene, max_palette_candidates=None,
            palette_score_ratio=0.0, _candidate_callback=collect_candidate,
        )

        # Match V3's component-body barrier, but only apply it to supplemental
        # pixels so the validated baseline is never damaged.
        barrier = np.zeros(extra.shape, dtype=np.uint8)
        for component in candidate_scene.components:
            x1, y1, x2, y2 = map(lambda value: int(round(value)),
                                 component.body_bbox or component.bbox)
            if x2 - x1 <= 2 or y2 - y1 <= 2:
                continue
            cv2.rectangle(
                barrier, (max(0, x1 + 2), max(0, y1 + 2)),
                (min(extra.shape[1] - 1, x2 - 2), min(extra.shape[0] - 1, y2 - 2)),
                255, -1,
            )
        extra[barrier > 0] = 0
        extra = cv2.morphologyEx(extra, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        extra[:2, :] = 0
        extra[-2:, :] = 0
        extra[:, :2] = 0
        extra[:, -2:] = 0
        supplemental = WireStageOutput(extra, color_debug, suppressed, corridor)
        supplemental = suppress_box_frames(image, candidate_scene, supplemental)
        combined = cv2.bitwise_or(baseline.mask, supplemental.mask)
        mask, rejected, detail = filter_candidate_bridges(baseline.mask, combined)

        scene.diagnostics.update(baseline_scene.diagnostics)
        scene.diagnostics["wire_candidate_corridor_recall"] = {
            **detail,
            "retained_supported_components": int(retained_components),
            "added_mask_pixels": int(((mask > 0) & (baseline.mask == 0)).sum()),
            "rejected_by_bridge_guard_pixels": int(rejected.sum()),
            "palette_selection": "all line-supported palette candidates; component must meet a terminal corridor",
        }
        marked_suppressed = (baseline.suppressed.copy() if baseline.suppressed is not None
                             else np.zeros(mask.shape, dtype=np.uint8))
        marked_suppressed[rejected] = 255
        return WireStageOutput(
            mask, baseline.color_debug, marked_suppressed,
            baseline.corridor if baseline.corridor is not None else corridor,
            baseline.diagnostics, baseline.debug,
        )
