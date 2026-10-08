"""Component V4 wired into the unchanged V3 pin interface."""
from __future__ import annotations

from .component_detection_v4 import detect_components_v4
from .component_text_assignment import BOX_FAMILY
from .official_types import to_official_type
from .pin_detection import terminal_candidates_v3
from .pin_semantics import assign_pin_semantics
from .schema import Scene


VALUE_TYPES = frozenset({
    "r", "c", "l", "v", "battary", "motor", "fuse", "circle_header",
    "crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin",
})


def component_refinement_regions(image, components, diagnostics):
    """Select and expand only components whose text fields need another look."""
    height, width = image.shape[:2]
    provenance = {row["component"]: row for row in diagnostics.get("component_provenance", [])}
    designator_scores = {
        row.get("proposal_index"): float(row.get("score", 0.0))
        for row in diagnostics.get("designator_assignment", {}).get("accepted", [])
    }
    regions = []
    for component in components:
        if component.type in {"gnd", "v", "net_input", "net_output", "net_bidirection", "net_short"}:
            continue
        row = provenance.get(component.key, {})
        reasons = []
        if component.observable_designator is None or component.key.startswith("UNRESOLVED_"):
            reasons.append("missing_designator")
        proposal_index = row.get("proposal_index")
        if proposal_index in designator_scores and designator_scores[proposal_index] < .12:
            reasons.append("low_designator_association_score")
        if component.type in BOX_FAMILY and row.get("name_source") != "ocr_inside_or_near_box":
            reasons.append("missing_model_name")
        if component.type in VALUE_TYPES and component.value is None:
            reasons.append("missing_value")
        if not reasons:
            continue
        x1, y1, x2, y2 = component.bbox
        box_width, box_height = max(4.0, x2 - x1), max(4.0, y2 - y1)
        # Small passives need more outside context; large boxes mainly need
        # their interior plus a narrow band for the visible U/IC designator.
        if component.type in BOX_FAMILY:
            pad_x, pad_y = max(24.0, .22 * box_width), max(24.0, .22 * box_height)
        else:
            pad_x, pad_y = max(36.0, 2.0 * box_width), max(30.0, 2.0 * box_height)
        regions.append({
            "component": component.key,
            "reasons": reasons,
            "bbox": (
                max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
                min(float(width), x2 + pad_x), min(float(height), y2 + pad_y),
            ),
        })
    return regions


def detect_scene_v4(image, ocr, detector, image_name="", component_stage="F", keypoint_provider=None, text_rules="v4_1"):
    height, width = image.shape[:2]
    texts = ocr.recognize(image)
    components, roles, diagnostics, debug = detect_components_v4(image, texts, image_name, detector, component_stage, text_rules)
    if hasattr(ocr, "recognize_component_regions"):
        regions = component_refinement_regions(image, components, diagnostics)
        initial_summary = {
            "component_count": len(components),
            "unresolved_count": sum(component.observable_designator is None for component in components),
            "missing_value_count": sum(component.type in VALUE_TYPES and component.value is None for component in components),
        }
        if regions:
            texts = ocr.recognize_component_regions(image, regions)
            components, roles, diagnostics, debug = detect_components_v4(
                image, texts, image_name, detector, component_stage, text_rules
            )
        diagnostics["selective_local_refinement"] = {
            "enabled": True,
            "selected_component_count": len(regions),
            "regions": regions,
            "initial": initial_summary,
            "final": {
                "component_count": len(components),
                "unresolved_count": sum(component.observable_designator is None for component in components),
                "missing_value_count": sum(component.type in VALUE_TYPES and component.value is None for component in components),
            },
        }
    pin_events = []
    for component in components:
        component.type = to_official_type(component.type)
        terminals = terminal_candidates_v3(image, component, texts, keypoint_provider)
        for terminal in terminals:
            terminal["tip"] = (min(width - 1.0, max(0.0, terminal["tip"][0])), min(height - 1.0, max(0.0, terminal["tip"][1])))
            terminal["base"] = (min(width - 1.0, max(0.0, terminal["base"][0])), min(height - 1.0, max(0.0, terminal["base"][1])))
        pins, events = assign_pin_semantics(component, terminals, roles)
        component.pins = pins
        pin_events.extend({"component": component.key, **event} for event in events)
    diagnostics.update({
        "mode": "image-only", "pipeline": "v4_component", "pin_v3_frozen": True,
        "ocr_backend": getattr(ocr, "name", "rapidocr"),
        "ocr_diagnostics": dict(getattr(ocr, "last_diagnostics", {}) or {}),
        "terminal_candidates": sum(len(c.pins) for c in components),
        "exportable_pins": sum(p.exportable for c in components for p in c.pins),
        "pin_events": pin_events, "keypoint_provider": getattr(keypoint_provider, "name", "disabled"),
    })
    return Scene(width, height, components, texts, diagnostics=diagnostics), debug
