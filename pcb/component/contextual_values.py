"""Resolve ambiguous OCR value readings using the same printed text region."""
from __future__ import annotations

import copy
from dataclasses import replace
import re

from .latest.text_assignment import assign_values, BOX_FAMILY, designator_weight
from .proposal_fusion import ComponentProposal, bbox_iou
from .text_enrichment import enrich_fields
from ..core.interfaces import ComponentStageOutput
from ..vision_v4 import VALUE_TYPES

BARE_NUMBER = re.compile(r"^\d{1,4}$")


def contextual_value_roles(roles, components, *, conflicts=True, pin_boxes=False):
    """Retain literal readings; reject numeric alternatives with stronger context.

    A higher-confidence designator/signal reading in the same physical region
    beats a bare number. IoU .3 is the existing OCR duplicate-region criterion.
    Optional pin-box filtering is kept separate for a development ablation.
    """
    textual = [role for role in roles if role.role in {"DESIGNATOR", "MODEL_TEXT", "PIN_NAME"}
               and role.confidence >= .85 and any(char.isalpha() for char in role.token.text)]
    boxes = [component.bbox for component in components if component.type in BOX_FAMILY]
    output, rejected = [], []
    for role in roles:
        if role.role != "VALUE" or not BARE_NUMBER.fullmatch(role.token.text.strip()):
            output.append(role)
            continue
        witnesses = [other for other in textual if other.confidence > role.confidence
                     and bbox_iou(other.token.bbox, role.token.bbox) >= .3] if conflicts else []
        inside = []
        if pin_boxes:
            x, y = role.token.center
            inside = [box for box in boxes if box[0] <= x <= box[2] and box[1] <= y <= box[3]]
        if witnesses or inside:
            reason = "stronger_text_in_same_region" if witnesses else "bare_number_inside_chip"
            output.append(replace(role, role="PIN_NUMBER", reason=reason))
            rejected.append({"index": role.index, "text": role.token.text, "bbox": role.token.bbox,
                             "reason": reason, "witnesses": [other.token.text for other in witnesses]})
        else:
            output.append(role)
    return output, rejected


def accepted_designators(original):
    """Recover the existing accepted physical designator tokens without reassigning keys."""
    by_index = {}
    for event in original.diagnostics.get("designator_assignment", {}).get("accepted", []):
        index = event["proposal_index"]
        if index >= len(original.components):
            raise AssertionError("Designator provenance is inconsistent with component order")
        component = original.components[index]
        proposal = ComponentProposal(component.bbox, component.type, component.confidence, "frozen", {})
        candidates = [role for role in original.roles if role.role == "DESIGNATOR"
                      and role.token.text == event["token"] and role.designator == component.key]
        if candidates:
            by_index[index] = max(candidates, key=lambda role: designator_weight(proposal, role))
    return by_index


def refine_contextual_values(original, *, conflicts=True, pin_boxes=False):
    """Re-run only value assignment on frozen detections, then apply V1 enrichment."""
    roles, rejected = contextual_value_roles(original.roles, original.components,
                                             conflicts=conflicts, pin_boxes=pin_boxes)
    if not rejected:
        components, diagnostics = enrich_fields(original.components, original.roles, original.diagnostics)
    else:
        components = copy.deepcopy(original.components)
        diagnostics = copy.deepcopy(original.diagnostics)
        provenance = diagnostics["component_provenance"]
        for index, component in enumerate(components):
            if component.type in VALUE_TYPES:
                component.value = None
                provenance[index]["value_source"] = "unknown"
        diagnostics["value_assignment"] = {
            **assign_values(components, roles, provenance, expanded_rules=True,
                            designators=accepted_designators(original)), "enabled": True}
        components, diagnostics = enrich_fields(components, roles, diagnostics)
    diagnostics["component_contextual_values"] = {
        "version": "contextual_values_v2", "conflicts": conflicts, "pin_boxes": pin_boxes,
        "rejected_ambiguous_numeric_readings": rejected, "GT_used_in_inference": False,
        "identity_and_geometry_changed": False}
    # The Component-owned value decision does not change Pin's token roles.
    return ComponentStageOutput(components, original.texts, original.roles, diagnostics, original.debug)


class ComponentStageContextualValues:
    def __init__(self):
        from .latest.stage import ComponentStageV4
        self.baseline = ComponentStageV4()

    def run(self, image, text, context):
        return refine_contextual_values(self.baseline.run(image, text, context),
                                        conflicts=True, pin_boxes=True)
