"""E6 single factor: local OCR role reinterpretation, with frozen geometry.

No image/GT/file access. Existing name/number scores and ordered association
are reused. The scorer adapter is process-local, serial, and always restored.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcb.pin_semantics import SIGNAL
from pcb.pin_semantics_v5 import _name_score
from pcb.schema import Component, Scene
from pcb.text_detection import TokenRole
from pin_skeleton_followup import number_bbox_score


def _inside(role: TokenRole, box: tuple) -> bool:
    x, y = role.token.center
    a, b, c, d = box
    # Reuse the original semantic boundary tolerance; no fitted parameter.
    return a + 2 < x < c - 2 and b + 2 < y < d - 2


def _outside(role: TokenRole, terminal: dict, box: tuple) -> bool:
    x, y = role.token.center
    a, b, c, d = box
    return {"left": x <= a, "right": x >= c,
            "top": y <= b, "bottom": y >= d}.get(terminal["side"], False)


def local_roles(component: Component, terminals: list[dict], roles: list[TokenRole]
                ) -> tuple[list[TokenRole], list[dict[str, Any]]]:
    """Promote ambiguous inside-box tokens only with independent number evidence."""
    output = list(roles)
    if component.type != "box":
        return output, []
    box = component.body_bbox or component.bbox
    own_text = {str(x).strip().upper() for x in
                (component.key, component.name, component.value, component.model) if x}
    evidence: dict[int, list[tuple[float, TokenRole]]] = {}
    for index, terminal in enumerate(terminals):
        supports = []
        for number in roles:
            if _outside(number, terminal, box):
                score = number_bbox_score(number, terminal, component)
                if score > 0:
                    supports.append((score, number))
        evidence[index] = supports
    decisions = []
    for position, role in enumerate(roles):
        if role.role != "PIN_NUMBER" or not SIGNAL.fullmatch(role.normalized):
            continue
        reason = "not_strictly_inside_body"
        proposed = replace(role, role="PIN_NAME")
        options = []
        if _inside(role, box):
            reason = "no_existing_terminal_name_support"
            if role.normalized.upper() in own_text:
                reason = "component_identity_text"
            else:
                for index, terminal in enumerate(terminals):
                    score = _name_score(proposed, terminal, component)
                    if score <= 0:
                        continue
                    reason = "no_independent_outside_number"
                    for number_score, number in evidence[index]:
                        if number.index != role.index:
                            options.append((score + number_score, index, score, number_score, number))
        detail = {"component": component.key, "role_index": role.index,
                  "text": role.token.text, "bbox": list(role.token.bbox),
                  "confidence": role.confidence, "original_role": role.role,
                  "new_role": role.role, "changed": bool(options), "reason": reason}
        if options:
            _, index, name_score, number_score, number = max(options, key=lambda r: (r[0], -r[1]))
            output[position] = proposed
            detail.update(new_role="PIN_NAME", reason="inside_name_with_outside_number_evidence",
                          candidate_index=index, side=terminals[index]["side"],
                          name_score=name_score, number_score=number_score,
                          outside_number_index=number.index, outside_number=number.token.text,
                          outside_number_bbox=list(number.token.bbox))
        decisions.append(detail)
    return output, decisions


def assert_invariants(before: Scene, after: Scene, raw: list[dict]) -> None:
    """Semantics may change; Component data and all terminal geometry may not."""
    if len(before.components) != len(after.components) or len(raw) != len(before.components):
        raise AssertionError("Owner count changed")
    for old, new, block in zip(before.components, after.components, raw):
        a, b = asdict(old), asdict(new)
        ap, bp = a.pop("pins"), b.pop("pins")
        if a != b or old.key != block["component"] or len(ap) != len(bp):
            raise AssertionError("Component/candidate count changed")
        if len(bp) != len(block["terminals"]):
            raise AssertionError("Pin/raw terminal count mismatch")
        if old.type != "box" and ap != bp:
            raise AssertionError("Nonbox Pin metadata changed")
        for p, q, terminal in zip(ap, bp, block["terminals"]):
            for field in ("tip", "base", "side"):
                if p[field] != q[field] or tuple(q[field]) != tuple(terminal[field]):
                    raise AssertionError("Terminal geometry changed")


def refine_scene(scene: Scene, raw: list[dict], roles: list[TokenRole]
                 ) -> tuple[Scene, list[dict[str, Any]]]:
    """Replace box semantics only; leave untouched owners bit-equivalent."""
    from pcb import pin_semantics_v6 as ordered
    from pin_skeleton_followup import event_lookup

    before_roles = copy.deepcopy(roles)
    result = copy.deepcopy(scene)
    previous = event_lookup(raw, scene.diagnostics)
    events, decisions = [], []
    score = ordered._number_score
    try:
        ordered._number_score = number_bbox_score
        for component, block in zip(result.components, raw):
            adapted, details = local_roles(component, block["terminals"], roles)
            decisions.extend(details)
            if not any(d["changed"] for d in details):
                events.extend(copy.deepcopy(previous[component.key]))
                continue
            pins, new_events = ordered.assign_pin_semantics_v6(component, block["terminals"], adapted)
            if len(pins) != len(component.pins):
                raise AssertionError("Association changed candidate count")
            for old_pin, pin in zip(component.pins, pins):
                # Keep the exact frozen geometry, including tuple/list representation.
                pin.tip, pin.base, pin.side = old_pin.tip, old_pin.base, old_pin.side
            component.pins = pins
            for old, event in zip(previous[component.key], new_events):
                # Retain geometry/provenance diagnostics from the frozen provider.
                events.append({**old, **event, "component": component.key,
                               "contextual_role_e6": True})
    finally:
        ordered._number_score = score
    if roles != before_roles:
        raise AssertionError("Global OCR roles were mutated")
    result.diagnostics["pin_events"] = events
    result.diagnostics["pin_contextual_roles_e6"] = decisions
    result.diagnostics["exportable_pins"] = sum(p.exportable for c in result.components for p in c.pins)
    assert_invariants(scene, result, raw)
    event_lookup(raw, result.diagnostics)
    return result, decisions
