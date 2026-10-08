"""Component V4: YOLO proposals + V3 text roles + optional geometry recovery."""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .component_detector_yolo import YoloComponentDetector
from .component_proposal_fusion import ComponentProposal, fuse_proposals
from .component_text_assignment import (
    assign_designators, assign_names, assign_values, build_components,
    recover_near_box_designators,
)
from .schema import Text
from .text_detection import classify_tokens


STAGES = ("B", "C", "D", "E", "F", "BEST")


def _geometry_proposals(image: np.ndarray, texts: Sequence[Text]) -> tuple[list[ComponentProposal], dict[str, object], dict[str, np.ndarray]]:
    # Reuse only the generic V3 geometry detectors. Source-name priors and the
    # 0057 binary MOSFET template are deliberately excluded.
    from .component_detection import (
        _ink_mask,
        detect_capacitor_centers,
        detect_ground_components,
        detect_large_box_components,
    )

    height, width = image.shape[:2]
    scale = max(.55, min(2.2, max(height, width) / 1200))
    raw_ink, symbol, text_mask = _ink_mask(image, list(texts))
    boxes = detect_large_box_components(image, raw_ink, scale)
    grounds = detect_ground_components(symbol, scale)
    capacitors = detect_capacitor_centers(symbol, scale)
    proposals: list[ComponentProposal] = [
        ComponentProposal(tuple(box), "box", .76, "large_box_geometry", {}) for box in boxes
    ]
    proposals.extend(ComponentProposal(tuple(box), "gnd", .74, "ground_geometry", {}) for box in grounds)
    for center, orientation, plate_length, separation in capacitors:
        # This fallback intentionally frames the visible plate body and a short
        # terminal corridor. It is lower confidence than YOLO and enters only F.
        if orientation == "vertical":
            half_w, half_h = max(4.0, plate_length * .65), max(4.0, separation * 1.25)
        else:
            half_w, half_h = max(4.0, separation * 1.25), max(4.0, plate_length * .65)
        box = (
            max(0.0, center[0] - half_w), max(0.0, center[1] - half_h),
            min(float(width - 1), center[0] + half_w), min(float(height - 1), center[1] + half_h),
        )
        proposals.append(ComponentProposal(box, "c", .58, "capacitor_geometry", {"orientation": orientation}))
    diagnostics = {
        "large_box_count": len(boxes), "ground_count": len(grounds),
        "capacitor_count": len(capacitors), "proposal_count": len(proposals),
    }
    return proposals, diagnostics, {"symbol_mask": symbol, "text_mask": text_mask}


def detect_components_v4(
    image: np.ndarray,
    texts: list[Text],
    image_name: str = "",
    detector: YoloComponentDetector | None = None,
    stage: str = "F",
    text_rules: str = "v4_1",
):
    """Return V3-compatible Components, text roles, diagnostics, and debug data."""
    del image_name  # V4 never branches on source/file name.
    stage = stage.upper()
    if stage not in STAGES:
        raise ValueError(f"Unknown Component V4 stage {stage!r}; choose one of {STAGES}")
    if detector is None:
        raise ValueError("Component V4 requires a YoloComponentDetector")

    yolo, yolo_diagnostics = detector.detect(image)
    geometry: list[ComponentProposal] = []
    geometry_diagnostics: dict[str, object] = {"enabled": False}
    debug: dict[str, object] = {}
    if stage == "F":
        geometry, geometry_diagnostics, debug = _geometry_proposals(image, texts)
        geometry_diagnostics["enabled"] = True

    fused, fusion_rejected = fuse_proposals([*yolo, *geometry])
    large_boxes = [proposal.bbox for proposal in fused if proposal.type in {"box", "block"}]
    expanded_rules = text_rules == "v4_1"
    if text_rules not in {"v4", "v4_1"}:
        raise ValueError("text_rules must be 'v4' or 'v4_1'")
    roles = classify_tokens(texts, large_boxes, expanded_rules=expanded_rules)
    roles, near_box_recovery = recover_near_box_designators(fused, roles)
    designators, designator_diagnostics = assign_designators(fused, roles, global_assignment=stage != "B")
    components, provenance = build_components(fused, designators)
    method_by_index = {row["proposal_index"]: row["method"] for row in designator_diagnostics["accepted"]}
    for index, row in enumerate(provenance):
        if index in method_by_index:
            row["designator_source"] = "ocr_" + str(method_by_index[index])

    name_diagnostics = {"enabled": False, "accepted": [], "candidate_count": 0}
    value_diagnostics = {"enabled": False, "accepted": [], "candidate_count": 0}
    if stage in {"D", "E", "F"}:
        name_diagnostics = assign_names(components, roles, provenance)
        name_diagnostics["enabled"] = True
    if stage in {"E", "F", "BEST"}:
        value_diagnostics = assign_values(components, roles, provenance,
                                          expanded_rules=expanded_rules, designators=designators)
        value_diagnostics["enabled"] = True

    diagnostics = {
        "frontend": "component_v4_1_yolo_text_assignment",
        "component_stage": stage,
        "text_rules": text_rules,
        "source_name_used": False,
        "automatic_designator_generation": False,
        "fixed_case_template_used": False,
        "yolo": yolo_diagnostics,
        "geometry": geometry_diagnostics,
        "proposal_counts": {"yolo": len(yolo), "geometry": len(geometry), "fused": len(fused)},
        "fusion_rejected": fusion_rejected,
        "text_roles": [
            {"index": role.index, "text": role.token.text, "bbox": role.token.bbox, "role": role.role,
             "reason": role.reason, "confidence": role.confidence, "normalized": role.normalized}
            for role in roles
        ],
        "near_box_designator_recovery": near_box_recovery,
        "designator_assignment": designator_diagnostics,
        "name_assignment": name_diagnostics,
        "value_assignment": value_diagnostics,
        "component_provenance": provenance,
        "component_final": len(components),
    }
    debug.update({"yolo_proposals": yolo, "geometry_proposals": geometry, "fused_proposals": fused})
    return components, roles, diagnostics, debug
