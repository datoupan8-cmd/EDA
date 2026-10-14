"""Use reference identifiers, rather than duplicate OCR rows, as assignment nodes."""
from __future__ import annotations
import copy

from .latest.text_assignment import build_components, assign_values, designator_weight
from .text_enrichment import enrich_fields
from .contextual_values import refine_contextual_values
from .proposal_fusion import ComponentProposal
from ..assignment import maximum_weight_assignment
from ..core.interfaces import ComponentStageOutput


def assign_unique_designators(proposals, roles):
    grouped = {}
    for role in roles:
        if role.role == "DESIGNATOR" and role.designator:
            grouped.setdefault(role.designator, []).append(role)
    keys = list(grouped)
    weights, chosen = [], []
    for key in keys:
        scores, witnesses = [], []
        for proposal in proposals:
            role = max(grouped[key], key=lambda item: designator_weight(proposal, item))
            scores.append(designator_weight(proposal, role))
            witnesses.append(role)
        weights.append(scores)
        chosen.append(witnesses)
    assigned, accepted = {}, []
    for row, column in maximum_weight_assignment(weights):
        if weights[row][column] < .055:
            continue
        assigned[column] = chosen[row][column]
        accepted.append({"proposal_index": column, "token": chosen[row][column].token.text,
                         "score": weights[row][column], "method": "unique_designator_global_assignment"})
    return assigned, {"accepted": accepted, "rejected": [], "candidate_count": sum(map(len, grouped.values())),
                      "unique_identifier_count": len(grouped),
                      "duplicate_ocr_candidates": sum(len(rows) - 1 for rows in grouped.values())}


def refine_unique_designators(original, *, numeric_context=False):
    proposals = original.debug.get("fused_proposals")
    if proposals is None:
        # Used only by pure unit tests; actual PNG runs retain detector proposals.
        proposals = [ComponentProposal(c.bbox, c.type, c.confidence, c.source_id.split(":")[0], {})
                     for c in original.components]
    assigned, trace = assign_unique_designators(proposals, original.roles)
    components, provenance = build_components(proposals, assigned)
    diagnostics = copy.deepcopy(original.diagnostics)
    diagnostics["component_provenance"] = provenance
    diagnostics["designator_assignment"] = trace
    diagnostics["value_assignment"] = {
        **assign_values(components, original.roles, provenance, expanded_rules=True, designators=assigned),
        "enabled": True}
    diagnostics["component_unique_designators"] = {
        "version": "unique_designators_v1", "GT_used_in_inference": False,
        "geometry_changed": False, "generated_identifiers": False}
    result = ComponentStageOutput(components, original.texts, original.roles, diagnostics, original.debug)
    if numeric_context:
        return refine_contextual_values(result, conflicts=True, pin_boxes=True)
    result.components, result.diagnostics = enrich_fields(components, original.roles, diagnostics)
    return result
