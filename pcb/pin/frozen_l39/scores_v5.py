"""P6-S2 experimental side-aware, one-to-one pin text assignment."""
from __future__ import annotations

from ...assignment import maximum_weight_assignment
from .detection_v3 import TWO_TERMINAL
from .semantics_v3 import PIN_NUMBER, SIGNAL, assign_pin_semantics
from ...schema import Pin


CONTEXT_NUMERIC_TYPES = frozenset({"box", "block"})


def _center(role):
    return role.token.center


def _tangent_distance(role, terminal):
    tx, ty = _center(role)
    bx, by = terminal["base"]
    return abs(ty - by) if terminal["side"] in ("left", "right") else abs(tx - bx)


def _tangent_limit(role, terminal, box):
    x1, y1, x2, y2 = box
    a, b, c, d = role.token.bbox
    text_cross = d - b if terminal["side"] in ("left", "right") else c - a
    tangent_span = y2 - y1 if terminal["side"] in ("left", "right") else x2 - x1
    return max(7.0, .55 * text_cross, .025 * tangent_span)


def _number_normal_distance(role, terminal, box):
    tx, ty = _center(role)
    x1, y1, x2, y2 = box
    side = terminal["side"]
    if side == "left" and tx <= x1 + 2:
        return max(0.0, x1 - tx)
    if side == "right" and tx >= x2 - 2:
        return max(0.0, tx - x2)
    if side == "top" and ty <= y1 + 2:
        return max(0.0, y1 - ty)
    if side == "bottom" and ty >= y2 - 2:
        return max(0.0, ty - y2)
    return None


def _number_score(role, terminal, component):
    if not PIN_NUMBER.fullmatch(role.normalized):
        return 0.0
    if role.role == "PIN_NUMBER":
        role_bonus = .25
    elif (component.type in CONTEXT_NUMERIC_TYPES and role.role == "VALUE"
          and role.normalized.isdigit()):
        role_bonus = 0.0
    else:
        return 0.0
    box = component.body_bbox or component.bbox
    normal = _number_normal_distance(role, terminal, box)
    if normal is None:
        return 0.0
    x1, y1, x2, y2 = box
    normal_span = x2 - x1 if terminal["side"] in ("left", "right") else y2 - y1
    normal_limit = max(30.0, min(120.0, .25 * normal_span))
    tangent = _tangent_distance(role, terminal)
    tangent_limit = _tangent_limit(role, terminal, box)
    if normal > normal_limit or tangent > tangent_limit:
        return 0.0
    return (1.0 + 2.0 * (1.0 - tangent / tangent_limit)
            + .5 * (1.0 - normal / normal_limit)
            + float(role.confidence) + role_bonus)


def _name_score(role, terminal, component):
    if role.role not in ("PIN_NAME", "MODEL_TEXT") or not SIGNAL.fullmatch(role.normalized):
        return 0.0
    box = component.body_bbox or component.bbox
    x1, y1, x2, y2 = box
    tx, ty = _center(role)
    if not (x1 - 2 <= tx <= x2 + 2 and y1 - 2 <= ty <= y2 + 2):
        return 0.0
    # A same-row name on the opposite half of a large symbol belongs to the
    # opposing terminal.  V3 ignored this normal-direction relationship.
    middle_x, middle_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = terminal["side"]
    if ((side == "left" and tx > middle_x + 2)
            or (side == "right" and tx < middle_x - 2)
            or (side == "top" and ty > middle_y + 2)
            or (side == "bottom" and ty < middle_y - 2)):
        return 0.0
    tangent = _tangent_distance(role, terminal)
    tangent_limit = _tangent_limit(role, terminal, box)
    if tangent > tangent_limit:
        return 0.0
    return 1.0 + 2.0 * (1.0 - tangent / tangent_limit) + float(role.confidence)


def _assign(terminals, roles, score_function, component):
    if not terminals or not roles:
        return {}, {}
    weights = [[score_function(role, terminal, component) for role in roles]
               for terminal in terminals]
    proposed = [(row, column, weights[row][column])
                for row, column in maximum_weight_assignment(weights)
                if weights[row][column] > 0]
    return ({row: roles[column] for row, column, _ in proposed},
            {row: score for row, _, score in proposed})


def assign_pin_semantics_v5(component, terminals, roles):
    """Preserve V3 priors; globally assign complex-component number/name text."""
    if component.type in TWO_TERMINAL or component.type in {"crystal_4pin", "gnd"}:
        return assign_pin_semantics(component, terminals, roles)

    number_roles = [role for role in roles if (
        role.role == "PIN_NUMBER" or
        (component.type in CONTEXT_NUMERIC_TYPES and role.role == "VALUE" and role.normalized.isdigit())
    ) and PIN_NUMBER.fullmatch(role.normalized)]
    name_roles = [role for role in roles if role.role in ("PIN_NAME", "MODEL_TEXT")
                  and SIGNAL.fullmatch(role.normalized)]
    numbers, number_scores = _assign(terminals, number_roles, _number_score, component)
    names, name_scores = _assign(terminals, name_roles, _name_score, component)

    # Distinct OCR boxes can contain the same number.  Keep only the strongest
    # assignment for an exported key and leave weaker duplicates unresolved.
    best_number = {}
    for row, role in numbers.items():
        current = best_number.get(role.normalized)
        if current is None or number_scores[row] > number_scores[current]:
            best_number[role.normalized] = row

    pins, events = [], []
    for index, terminal in enumerate(terminals, 1):
        number_role = numbers.get(index - 1)
        duplicate = bool(number_role and best_number.get(number_role.normalized) != index - 1)
        number = number_role.normalized if number_role and not duplicate else f"UNK{index}"
        name_role = names.get(index - 1)
        exportable = number_role is not None and not duplicate
        name = name_role.token.text.strip() if name_role else ""
        reclassified = bool(number_role and number_role.role == "VALUE")
        source = ("contextual_numeric_global" if reclassified else
                  "local_ocr_global" if exportable else "unrecognized")
        pin = Pin(number, name, terminal["tip"], base=terminal["base"], side=terminal["side"],
                  inferred_number=not exportable,
                  number_confidence=number_role.confidence if number_role and not duplicate else 0.0,
                  name_confidence=name_role.confidence if name_role else 0.0,
                  exportable=exportable, observable_number=number if exportable else None,
                  number_source=source)
        pins.append(pin)
        events.append({
            **terminal, "number": number, "pinname": name,
            "number_confidence": pin.number_confidence,
            "name_confidence": pin.name_confidence,
            "exportable": exportable,
            "semantic_method": "side_aware_global_assignment",
            "number_token": number_role.token.text if number_role else None,
            "number_token_original_role": number_role.role if number_role else None,
            "number_role_reclassified": reclassified,
            "number_duplicate": duplicate,
            "number_assignment_score": number_scores.get(index - 1),
            "name_token": name_role.token.text if name_role else None,
            "name_assignment_score": name_scores.get(index - 1),
        })
    return pins, events
