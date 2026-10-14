"""OCR-supported small-symbol recovery through a second detection scale."""
from __future__ import annotations

import copy
import cv2

from .latest.text_assignment import (BOX_FAMILY, assign_values, build_components,
                                     designator_weight, NO_DESIGNATOR_TYPES)
from .proposal_fusion import ComponentProposal, bbox_iou
from .text_enrichment import _claimed_value_roles, enrich_fields
from ..assignment import maximum_weight_assignment
from ..core.interfaces import ComponentStageOutput


def duplicate_body(left, right):
    """Catch both similar boxes and a tiny duplicate inside a larger body."""
    if bbox_iou(left, right) >= .25:
        return True
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    smaller = min((left[2] - left[0]) * (left[3] - left[1]),
                  (right[2] - right[0]) * (right[3] - right[1]))
    return width * height / max(1e-9, smaller) >= .5


def recover_slice_components(image, original, detector):
    """Add only detections supported by an unclaimed printed reference label.

    Existing components and their geometry are immutable. No label numbering
    is generated; unsupported second-scale proposals are not exported.
    """
    components = copy.deepcopy(original.components)
    diagnostics = copy.deepcopy(original.diagnostics)
    used = {component.key for component in components}
    unused = [role for role in original.roles if role.role == "DESIGNATOR"
              and role.designator not in used and role.confidence >= .85 and not role.tail
              and role.component_type not in NO_DESIGNATOR_TYPES
              and role.component_type not in BOX_FAMILY]
    trace = {"version": "slice_recovery_v1", "scale": 2, "unclaimed_designator_count": len(unused),
             "new_components": [], "GT_used_in_inference": False, "existing_geometry_changed": False}
    if not unused:
        diagnostics["component_slice_recovery"] = trace
        return ComponentStageOutput(components, original.texts, original.roles, diagnostics, original.debug)
    doubled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    raw, detector_trace = detector.detect(doubled)
    proposals = []
    for proposal in raw:
        if proposal.confidence < .25 or proposal.type in NO_DESIGNATOR_TYPES or proposal.type in BOX_FAMILY:
            continue
        box = tuple(value / 2 for value in proposal.bbox)
        if any(duplicate_body(box, component.bbox) for component in components):
            continue
        candidate = ComponentProposal(box, proposal.type, proposal.confidence, "slice_2x_yolo", {})
        if any(designator_weight(candidate, role) >= .055 for role in unused):
            proposals.append(candidate)
    weights = [[designator_weight(proposal, role) for proposal in proposals] for role in unused]
    accepted, assigned = [], {}
    for token_index, proposal_index in maximum_weight_assignment(weights):
        role, proposal = unused[token_index], proposals[proposal_index]
        if weights[token_index][proposal_index] < .055 or role.designator in used:
            continue
        if any(duplicate_body(proposal.bbox, previous.bbox) for previous in accepted):
            continue
        used.add(role.designator)
        assigned[len(accepted)] = role
        accepted.append(proposal)
    added, provenance = build_components(accepted, assigned)
    claimed = _claimed_value_roles(components, original.roles, diagnostics)
    claimed.update(row["index"] for row in diagnostics.get("component_contextual_values", {}).get(
        "rejected_ambiguous_numeric_readings", []))
    free_roles = [role for role in original.roles if role.index not in claimed]
    value_trace = assign_values(added, free_roles, provenance, expanded_rules=True, designators=assigned)
    added, added_diagnostics = enrich_fields(added, free_roles, {"value_assignment": value_trace})
    trace.update({"second_scale_yolo": detector_trace, "supported_candidate_count": len(proposals),
                  "new_components": [{"key": c.key, "type": c.type, "bbox": c.bbox,
                                      "confidence": c.confidence} for c in added],
                  "value_assignment": added_diagnostics.get("value_assignment"),
                  "text_enrichment": added_diagnostics.get("component_text_enrichment")})
    offset = len(components)
    components.extend(added)
    diagnostics["component_provenance"].extend({**row, "proposal_index": offset + row["proposal_index"]}
                                                for row in provenance)
    diagnostics["component_slice_recovery"] = trace
    return ComponentStageOutput(components, original.texts, original.roles, diagnostics, original.debug)
