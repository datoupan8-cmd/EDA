"""Component V4 text roles and constrained one-to-one association."""
from __future__ import annotations

import math
import re
from collections.abc import Sequence

from ...assignment import maximum_weight_assignment
from ...component_proposal_fusion import ComponentProposal
from ...schema import Component
from .text_roles import DESIGNATOR_RE, TokenRole, compatible_types_for_designator, prefix_type, value_compatible


BOX_FAMILY = frozenset({"box", "block", "amp", "amp_3pin", "amp_5pin", "opt", "other"})
CRYSTAL_FAMILY = frozenset({"crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin"})
MOSFET_FAMILY = frozenset({"mosfet", "mosfet_npn", "mosfet_pnp"})
BJT_FAMILY = frozenset({"bjt", "bjt_npn", "bjt_pnp"})
PIN_FAMILY = frozenset({"pin", "circle_header"})
DIODE_FAMILY = frozenset({"d", "led", "esd"})
TRANSISTOR_FAMILY = frozenset({"mosfet", "mosfet_npn", "mosfet_pnp", "bjt", "bjt_npn", "bjt_pnp"})
NO_DESIGNATOR_TYPES = frozenset({"gnd", "v", "net_input", "net_output", "net_bidirection", "net_short"})
SIGNAL_RE = re.compile(r"^(?:VDD|VSS|VCC|GND|VIN|VOUT|VBUS|SCL|SDA|CLK|DATA|GPIO|TX|RX|CS|AGND|DGND)[A-Z0-9_./+\-]*$", re.I)
MODEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.+\-/]{2,40}$")


def bbox_gap(left, right) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def _same_family(left: str, right: str) -> bool:
    return left == right or any(left in family and right in family for family in (BOX_FAMILY, CRYSTAL_FAMILY, MOSFET_FAMILY, BJT_FAMILY, PIN_FAMILY, DIODE_FAMILY, TRANSISTOR_FAMILY))


def type_designator_compatible(component_type: str, role: TokenRole) -> bool:
    if role.role != "DESIGNATOR" or not role.designator:
        return False
    allowed = compatible_types_for_designator(role.designator)
    return component_type in allowed or any(_same_family(component_type, candidate) for candidate in allowed)


def _inside(point, box, padding=0.0):
    return box[0] - padding <= point[0] <= box[2] + padding and box[1] - padding <= point[1] <= box[3] + padding


def designator_weight(proposal: ComponentProposal, role: TokenRole) -> float:
    if not type_designator_compatible(proposal.type, role):
        return 0.0
    token = role.token
    tw = max(3.0, token.bbox[2] - token.bbox[0])
    th = max(3.0, token.bbox[3] - token.bbox[1])
    pw = max(3.0, proposal.bbox[2] - proposal.bbox[0])
    ph = max(3.0, proposal.bbox[3] - proposal.bbox[1])
    gap = bbox_gap(token.bbox, proposal.bbox)
    if gap > max(100.0, 6.0 * th, 2.2 * max(pw, ph)):
        return 0.0
    pc = ((proposal.bbox[0] + proposal.bbox[2]) / 2, (proposal.bbox[1] + proposal.bbox[3]) / 2)
    tc = token.center
    center_distance = math.dist(pc, tc) / max(8.0, math.sqrt(pw * ph), 2 * th)
    token_inside = _inside(tc, proposal.bbox, padding=1.0)
    if proposal.type in BOX_FAMILY:
        layout = 1.30 if token_inside else 1.0
    else:
        # Ordinary designators are usually outside the symbol. Text inside a
        # small symbol is more likely a pin/name token.
        layout = .45 if token_inside else 1.10
        horizontal_offset = abs(tc[0] - pc[0]) / max(pw, tw)
        vertical_offset = abs(tc[1] - pc[1]) / max(ph, th)
        if min(horizontal_offset, vertical_offset) < 1.2:
            layout *= 1.10
    return float(proposal.confidence) * max(.05, float(role.confidence)) * layout * math.exp(-.85 * center_distance - gap / max(10.0, 3 * th))


def recover_near_box_designators(
    proposals: Sequence[ComponentProposal], roles: Sequence[TokenRole],
) -> tuple[list[TokenRole], dict[str, object]]:
    """Rescue passive designators suppressed by a nearby chip box only when a symbol supports them."""
    recovered = []
    accepted = []
    for role in roles:
        if role.reason != "near_large_box_boundary":
            recovered.append(role)
            continue
        match = DESIGNATOR_RE.fullmatch(role.normalized)
        if match is None:
            recovered.append(role)
            continue
        key, tail = match.group(1).upper(), match.group(2).strip()
        component_type = prefix_type(key)
        if component_type not in {"r", "c", "l"}:
            recovered.append(role)
            continue
        candidate = TokenRole(role.index, role.token, "DESIGNATOR", role.normalized,
                              key, component_type, tail, role.confidence,
                              "near_box_with_component_candidate")
        text_height = max(3.0, role.token.bbox[3] - role.token.bbox[1])
        nearby = [
            (designator_weight(proposal, candidate), index)
            for index, proposal in enumerate(proposals)
            if proposal.type == component_type
            and bbox_gap(role.token.bbox, proposal.bbox) <= 2.0 * text_height
        ]
        if nearby and max(nearby)[0] >= .055:
            score, proposal_index = max(nearby)
            recovered.append(candidate)
            accepted.append({"token": role.token.text, "proposal_index": proposal_index, "score": score})
        else:
            recovered.append(role)
    return recovered, {"accepted": accepted, "count": len(accepted)}


def assign_designators(
    proposals: Sequence[ComponentProposal],
    roles: Sequence[TokenRole],
    *,
    global_assignment: bool,
) -> tuple[dict[int, TokenRole], dict[str, object]]:
    candidates = [role for role in roles if role.role == "DESIGNATOR"]
    assignments: dict[int, TokenRole] = {}
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    if global_assignment and proposals and candidates:
        weights = [[designator_weight(proposal, role) for proposal in proposals] for role in candidates]
        for token_index, proposal_index in maximum_weight_assignment(weights):
            weight = weights[token_index][proposal_index]
            if weight >= .055:
                assignments[proposal_index] = candidates[token_index]
                accepted.append({"proposal_index": proposal_index, "token": candidates[token_index].token.text, "score": weight, "method": "global_assignment"})
        for token_index, role in enumerate(candidates):
            if all(row["token"] != role.token.text for row in accepted):
                ranked = sorted(((weights[token_index][j], j) for j in range(len(proposals))), reverse=True)
                rejected.append({"token": role.token.text, "reason": "below_threshold_or_conflict", "best": ranked[0] if ranked else None})
    else:
        used_tokens: set[int] = set()
        # Greedy nearest/type-compatible association is the stage-B control.
        for proposal_index, proposal in sorted(enumerate(proposals), key=lambda row: row[1].confidence, reverse=True):
            ranked = [(designator_weight(proposal, role), i, role) for i, role in enumerate(candidates) if i not in used_tokens]
            ranked.sort(key=lambda row: row[0], reverse=True)
            if ranked and ranked[0][0] >= .055:
                score, token_index, role = ranked[0]
                assignments[proposal_index] = role
                used_tokens.add(token_index)
                accepted.append({"proposal_index": proposal_index, "token": role.token.text, "score": score, "method": "greedy_nearest"})
    return assignments, {"accepted": accepted, "rejected": rejected, "candidate_count": len(candidates)}


def build_components(
    proposals: Sequence[ComponentProposal],
    designators: dict[int, TokenRole],
) -> tuple[list[Component], list[dict[str, object]]]:
    components: list[Component] = []
    provenance: list[dict[str, object]] = []
    used: set[str] = set()
    unresolved_index = 0
    gnd_index = 0
    for index, proposal in enumerate(proposals):
        role = designators.get(index)
        if role is not None and role.designator and role.designator not in used:
            key = role.designator
            observable = key
            name = key
            designator_source = "ocr_global_assignment" if role else "unknown"
        elif proposal.type == "gnd":
            gnd_index += 1
            key, observable, name = f"GND_{gnd_index}", None, "GND"
            designator_source = "gnd_scoring_insensitive_geometry"
        else:
            unresolved_index += 1
            key, observable, name = f"UNRESOLVED_{unresolved_index:04d}", None, None
            while key in used:
                unresolved_index += 1
                key = f"UNRESOLVED_{unresolved_index:04d}"
            designator_source = "unknown"
        used.add(key)
        component = Component(
            key=key, type=proposal.type, bbox=proposal.bbox, name=name,
            body_bbox=proposal.bbox, confidence=proposal.confidence,
            source_id=f"{proposal.source}:{index}", observable_designator=observable,
        )
        components.append(component)
        provenance.append({
            "component": key, "proposal_index": index, "proposal_source": proposal.source,
            "proposal_confidence": proposal.confidence, "designator_source": designator_source,
            "name_source": "designator_default" if name else "unknown", "value_source": "unknown",
        })
    return components, provenance


def _looks_like_model(role: TokenRole) -> bool:
    text = re.sub(r"\s+", "", role.token.text.strip())
    if role.role in {"DESIGNATOR", "VALUE", "PIN_NUMBER"} or not MODEL_RE.fullmatch(text):
        return False
    if SIGNAL_RE.fullmatch(text):
        return False
    return any(ch.isalpha() for ch in text) and (any(ch.isdigit() for ch in text) or "-" in text)


def assign_names(components: Sequence[Component], roles: Sequence[TokenRole], provenance: list[dict[str, object]]):
    indices = [i for i, component in enumerate(components) if component.type in BOX_FAMILY]
    candidates = [role for role in roles if _looks_like_model(role)]
    if not indices or not candidates:
        return {"accepted": [], "candidate_count": len(candidates)}
    weights = []
    for role in candidates:
        row = []
        for component_index in indices:
            component = components[component_index]
            box, token = component.bbox, role.token
            width, height = max(4.0, box[2] - box[0]), max(4.0, box[3] - box[1])
            gap = bbox_gap(token.bbox, box)
            if gap > max(30.0, .25 * max(width, height)):
                row.append(0.0); continue
            pc = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            distance = abs(token.center[0] - pc[0]) / width + abs(token.center[1] - pc[1]) / height
            inside_bonus = 1.35 if _inside(token.center, box, 2) else 1.0
            row.append(float(role.confidence) * inside_bonus * math.exp(-1.6 * distance - gap / max(width, height)))
        weights.append(row)
    accepted = []
    for token_index, local_component_index in maximum_weight_assignment(weights):
        score = weights[token_index][local_component_index]
        if score < .08:
            continue
        component_index = indices[local_component_index]
        value = candidates[token_index].token.text.strip()
        components[component_index].name = value
        provenance[component_index]["name_source"] = "ocr_inside_or_near_box"
        accepted.append({"component": components[component_index].key, "token": value, "score": score})
    return {"accepted": accepted, "candidate_count": len(candidates)}


def assign_values(components: Sequence[Component], roles: Sequence[TokenRole], provenance: list[dict[str, object]], expanded_rules: bool = True,
                  designators: dict[int, TokenRole] | None = None):
    supported = {"r", "c", "l", "v", "battary", "motor", "fuse", "circle_header", "crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin"} if expanded_rules else {"r", "c", "l"}
    accepted = []
    bound_indices = set()
    for index, role in (designators or {}).items():
        if (index >= len(components) or role.reason != "designator_value_pair"
                or components[index].type not in supported
                or components[index].key != role.designator
                or not value_compatible(components[index].type, role.tail, expanded_rules)):
            continue
        value = role.tail.strip()
        components[index].value = value
        provenance[index]["value_source"] = "ocr_designator_tail"
        bound_indices.add(index)
        accepted.append({"component": components[index].key, "token": value,
                         "score": role.confidence, "method": "designator_tail"})
    indices = [i for i, component in enumerate(components) if component.type in supported and i not in bound_indices]
    candidates = [role for role in roles if role.role == "VALUE"]
    if not indices or not candidates:
        return {"accepted": accepted, "candidate_count": len(candidates) + len(bound_indices),
                "bound_count": len(bound_indices)}
    weights = []
    for role in candidates:
        row = []
        for component_index in indices:
            component = components[component_index]
            if not value_compatible(component.type, role.token.text, expanded_rules):
                row.append(0.0); continue
            width = max(5.0, component.bbox[2] - component.bbox[0])
            height = max(5.0, component.bbox[3] - component.bbox[1])
            gap = bbox_gap(role.token.bbox, component.bbox)
            if gap > max(80.0, 5 * (role.token.bbox[3] - role.token.bbox[1]), 2.5 * max(width, height)):
                row.append(0.0); continue
            pc = ((component.bbox[0] + component.bbox[2]) / 2, (component.bbox[1] + component.bbox[3]) / 2)
            normalized = math.dist(pc, role.token.center) / max(8.0, math.sqrt(width * height))
            row.append(float(role.confidence) * math.exp(-.9 * normalized))
        weights.append(row)
    for token_index, local_component_index in maximum_weight_assignment(weights):
        score = weights[token_index][local_component_index]
        if score < .055:
            continue
        component_index = indices[local_component_index]
        value = candidates[token_index].token.text.strip()
        components[component_index].value = value
        provenance[component_index]["value_source"] = "ocr_global_assignment"
        accepted.append({"component": components[component_index].key, "token": value, "score": score})
    return {"accepted": accepted, "candidate_count": len(candidates) + len(bound_indices),
            "bound_count": len(bound_indices)}
