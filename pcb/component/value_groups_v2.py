"""Literal unit-aware Component value matching using observable text layout.

The panel-edge negative evidence is derived from component boxes and OCR only;
no pin locator, net stage, case filename, target or per-case rule is consulted.
"""
from __future__ import annotations

import copy
import math
import re

from .contextual_values import accepted_designators, contextual_value_roles
from .latest.text_assignment import BOX_FAMILY
from .local_joint import grouped_roles, line_group_strength, printed_value_compatible
from .proposal_fusion import bbox_iou
from .text_enrichment import _value_weight
from ..assignment import maximum_weight_assignment
from ..core.interfaces import ComponentStageOutput


BARE_NUMBER = re.compile(r"^\d{1,4}$")
PANEL_TYPES = BOX_FAMILY | {"pin", "circle_header"}


def explicit_unit_rank(kind, text):
    text = re.sub(r"\s+", "", text)
    if kind == "c":
        return 2 if text.lower().endswith("f") else 1 if re.search(r"[pnuµμm]$", text) else 0
    if kind == "l":
        return 2 if text.lower().endswith("h") else 0
    if kind == "r":
        return 2 if re.search(r"(?:Ω|ohm|R)$", text, re.I) else 1 if re.search(r"[kKmMgG]$", text) else 0
    return 0


def select_literal(component, group):
    candidates = [r for r in group if printed_value_compatible(component.type, r.token.text)]
    if not candidates:
        return None
    # The alternatives describe one physical region. A well-read printed unit
    # is evidence; never synthesize units or change the literal output string.
    complete = [r for r in candidates if r.confidence >= .75 and explicit_unit_rank(component.type, r.token.text) > 0]
    if complete:
        rank = max(explicit_unit_rank(component.type, r.token.text) for r in complete)
        candidates = [r for r in complete if explicit_unit_rank(component.type, r.token.text) == rank]
    return max(candidates, key=lambda r: _value_weight(component, r))


def panel_edge_number(role, components, numeric_roles):
    if not BARE_NUMBER.fullmatch(role.token.text.strip()):
        return False
    token = role.token
    x, y = token.center
    font = max(4., token.bbox[3] - token.bbox[1])
    for component in components:
        if component.type not in PANEL_TYPES:
            continue
        x1, y1, x2, y2 = component.bbox
        inside = x1 <= x <= x2 and y1 <= y <= y2
        beside = y1 - .5 * font <= y <= y2 + .5 * font and min(abs(x - x1), abs(x - x2)) <= 1.5 * font
        if inside:
            return True
        if beside:
            aligned = [other for other in numeric_roles if other.index != role.index
                       and abs(other.token.center[0] - x) <= .65 * font
                       and .7 * font <= abs(other.token.center[1] - y) <= 6 * font]
            if aligned:
                return True
    return False


def filter_value_evidence(roles, components, designators):
    candidates = [role for role in roles if role.role == "VALUE"]
    numeric = [r for r in candidates if BARE_NUMBER.fullmatch(r.token.text.strip())]
    owned = list(designators.values())
    kept, rejected = [], []
    for role in candidates:
        reason = None
        if any(bbox_iou(role.token.bbox, anchor.token.bbox) >= .3
               and (BARE_NUMBER.fullmatch(role.token.text.strip()) or anchor.reason == "designator_value_pair")
               for anchor in owned):
            reason = "physical_text_already_owned_by_designator"
        elif panel_edge_number(role, components, numeric):
            reason = "numeric_column_at_panel_edge"
        if reason:
            rejected.append({"token_index": role.index, "text": role.token.text,
                             "bbox": role.token.bbox, "reason": reason})
        else:
            kept.append(role)
    return kept, rejected


def annotation_strength(anchor, value, *, ordered=False):
    strength = line_group_strength(anchor, value)
    # Tight printed rows can have overlapping OCR rectangles even though their
    # glyphs occupy separate baselines. Distinguish those from alternate reads
    # of one text line by baseline separation and left/center alignment.
    overlap = bbox_iou(anchor.bbox, value.bbox)
    if not strength and .15 < overlap <= .35:
        ah, vh = max(3., anchor.bbox[3] - anchor.bbox[1]), max(3., value.bbox[3] - value.bbox[1])
        font = (ah + vh) / 2
        column_offset = min(abs(anchor.bbox[0] - value.bbox[0]), abs(anchor.center[0] - value.center[0])) / font
        if min(ah, vh) / max(ah, vh) >= .55 and .5 * font <= abs(anchor.center[1] - value.center[1]) <= 1.6 * font and column_offset <= .65:
            strength = math.exp(-column_offset)
    if not ordered or not strength:
        return strength
    font = max(4., (anchor.bbox[3] - anchor.bbox[1] + value.bbox[3] - value.bbox[1]) / 2)
    dx, dy = value.center[0] - anchor.center[0], value.center[1] - anchor.center[1]
    # Soft reading-order prior. Reverse layouts remain candidates and still
    # participate in the geometric fallback; no direction is forbidden.
    if dy < -.5 * font or (abs(dy) <= .3 * font and dx < 0):
        strength *= .45
    return strength


def refine_guarded_values(original, *, ordered=False):
    components = copy.deepcopy(original.components)
    diagnostics = copy.deepcopy(original.diagnostics)
    roles, numeric_conflicts = contextual_value_roles(original.roles, components, conflicts=True, pin_boxes=True)
    anchors = accepted_designators(original)
    candidates, rejected = filter_value_evidence(roles, components, anchors)
    groups = grouped_roles(candidates)
    provenance = diagnostics["component_provenance"]
    bound, accepted = set(), []
    for index, anchor in anchors.items():
        component = components[index]
        if (component.type in {"r", "c", "l"} and anchor.reason == "designator_value_pair"
                and component.key == anchor.designator and printed_value_compatible(component.type, anchor.tail)):
            component.value = anchor.tail.strip()
            provenance[index]["value_source"] = "ocr_designator_tail"
            bound.add(index)
            accepted.append({"component": component.key, "token": component.value, "token_index": anchor.index,
                             "bbox": anchor.token.bbox, "method": "designator_tail", "score": anchor.confidence})
    indices = [i for i, c in enumerate(components) if c.type in {"r", "c", "l"} and i not in bound]
    weights, choices, supports = [], [], []
    for group in groups:
        selections = [select_literal(components[index], group) for index in indices]
        anchor_scores = []
        for index, role in zip(indices, selections):
            anchor = anchors.get(index)
            strength = (annotation_strength(anchor.token, role.token, ordered=ordered) * role.confidence
                        if role and anchor and anchor.confidence >= .8 else 0.)
            anchor_scores.append(strength)
        ranked = sorted(anchor_scores, reverse=True)
        unambiguous = bool(ranked and ranked[0] >= .35 and
                           (len(ranked) == 1 or ranked[0] >= 1.4 * ranked[1]))
        row, tags = [], []
        for column, (index, role) in enumerate(zip(indices, selections)):
            weight = _value_weight(components[index], role) if role else 0.
            use_group = bool(role and unambiguous and anchor_scores[column] == ranked[0] and weight > 0)
            row.append(weight + anchor_scores[column] if use_group else weight)
            tags.append(use_group)
        weights.append(row)
        choices.append(selections)
        supports.append(tags)
    for index in indices:
        components[index].value = None
        provenance[index]["value_source"] = "unknown"
    for row, column in maximum_weight_assignment(weights):
        role = choices[row][column]
        if weights[row][column] < .055 or role is None:
            continue
        index = indices[column]
        components[index].value = role.token.text.strip()
        method = "ocr_ordered_group" if supports[row][column] and ordered else "ocr_guarded_group" if supports[row][column] else "ocr_global_assignment"
        provenance[index]["value_source"] = method
        accepted.append({"component": components[index].key, "token": components[index].value,
                         "token_index": role.index, "bbox": role.token.bbox, "physical_group": row,
                         "score": weights[row][column], "method": method})
    for before, after in zip(original.components, components):
        assert (before.key, before.name, before.type, before.bbox) == (after.key, after.name, after.type, after.bbox)
        if before.type not in {"r", "c", "l"}:
            assert before.value == after.value
    diagnostics["frontend"] = "component_ordered_value_groups_v2" if ordered else "component_guarded_value_groups_v2"
    diagnostics["value_assignment"] = {"accepted": accepted, "enabled": True,
                                        "candidate_count": len(groups), "bound_count": len(bound)}
    diagnostics["component_text_refinement"] = {
        "extra_OCR": False, "identity_name_geometry_types_changed": False, "GT_used_in_inference": False,
        "ordered": ordered, "rejected_numeric_alternatives": numeric_conflicts,
        "rejected_value_evidence": rejected, "literal_units_preserved": True}
    # No second shorthand fill: it could reuse a physical OCR region already
    # claimed by the global assignment. Shorthand is supported in the matrix.
    return ComponentStageOutput(components, original.texts, original.roles, diagnostics, original.debug)


class ComponentStageGuardedValues:
    ordered = False

    def __init__(self):
        from .contextual_values import ComponentStageContextualValues
        self.baseline = ComponentStageContextualValues()

    def refine(self, image, original, context):
        del image
        if context.config.component_stage != "BEST" or context.config.text_rules != "v4_1":
            raise ValueError("guarded values require BEST / v4_1")
        return refine_guarded_values(original, ordered=self.ordered)

    def run(self, image, text, context):
        return self.refine(image, self.baseline.run(image, text, context), context)


class ComponentStageOrderedValues(ComponentStageGuardedValues):
    ordered = True
