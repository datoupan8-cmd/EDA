"""P6.1 experimental side-separated, order-preserving pin text alignment.

The implementation deliberately reuses the P6-S2 geometric scores.  Its only
new hypothesis is structural: terminals and OCR rows on each component side
must keep their visual order, while either sequence may contain gaps.
"""
from __future__ import annotations

from dataclasses import dataclass

from .pin_detection import TWO_TERMINAL
from .pin_semantics import PIN_NUMBER, SIGNAL, assign_pin_semantics
from .pin_semantics_v5 import (
    CONTEXT_NUMERIC_TYPES,
    _name_score,
    _number_score,
    assign_pin_semantics_v5,
)
from .schema import Pin


SIDES = ("left", "right", "top", "bottom")


def _abstain_unresolved_component(component, pins, events):
    """Keep terminal diagnostics but never export pins under an unknown key."""
    if not str(component.key).startswith("UNRESOLVED_"):
        return pins, events
    for pin in pins:
        if pin.exportable:
            pin.exportable = False
            pin.inferred_number = True
            pin.observable_number = None
            pin.number_source = "unresolved_component_abstention"
    for event in events:
        if event.get("exportable"):
            event["exportable"] = False
            event["semantic_abstention"] = "unresolved_component_key"
    return pins, events


def _tangent_point(point, side):
    return float(point[1] if side in ("left", "right") else point[0])


def _role_tangent(role, side):
    return _tangent_point(role.token.center, side)


def _terminal_tangent(terminal):
    return _tangent_point(terminal["base"], terminal["side"])


def _monotone_assignment(items, roles, score_function):
    """Maximum-score non-crossing one-to-one alignment with free gaps."""
    rows, columns = len(items), len(roles)
    if not rows or not columns:
        return {}, {}
    scores = [[float(score_function(role, item)) for role in roles] for item in items]
    dp = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    trace = [[None] * (columns + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        trace[i][0] = "skip_item"
    for j in range(1, columns + 1):
        trace[0][j] = "skip_role"
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            candidates = [
                (dp[i - 1][j], 0, "skip_item"),
                (dp[i][j - 1], 1, "skip_role"),
            ]
            score = scores[i - 1][j - 1]
            if score > 0:
                candidates.append((dp[i - 1][j - 1] + score, 2, "match"))
            value, _, action = max(candidates, key=lambda row: (row[0], row[1]))
            dp[i][j], trace[i][j] = value, action
    assigned, assigned_scores = {}, {}
    i, j = rows, columns
    while i or j:
        action = trace[i][j]
        if action == "match":
            assigned[i - 1] = roles[j - 1]
            assigned_scores[i - 1] = scores[i - 1][j - 1]
            i -= 1
            j -= 1
        elif action == "skip_item":
            i -= 1
        else:
            j -= 1
    return assigned, assigned_scores


def _best_side(role, side_terminals, component, score_function):
    options = []
    for side_index, side in enumerate(SIDES):
        terminals = side_terminals.get(side, [])
        score = max((score_function(role, terminal, component) for _, terminal in terminals), default=0.0)
        if score > 0:
            options.append((score, -side_index, side))
    return max(options, default=(0.0, 0, None))[2]


def _row_pair_limit(number_role, name_role, side, box):
    x1, y1, x2, y2 = box
    tangent_span = y2 - y1 if side in ("left", "right") else x2 - x1
    na, nb, nc, nd = number_role.token.bbox
    aa, ab, ac, ad = name_role.token.bbox
    number_cross = nd - nb if side in ("left", "right") else nc - na
    name_cross = ad - ab if side in ("left", "right") else ac - aa
    return max(8.0, .75 * max(number_cross, name_cross), .025 * tangent_span)


def _pair_number_names(number_roles, name_roles, side, box):
    """Create ordered OCR rows, retaining unmatched numbers and names as gaps."""
    numbers = sorted(number_roles, key=lambda role: _role_tangent(role, side))
    names = sorted(name_roles, key=lambda role: _role_tangent(role, side))

    def pair_score(name_role, number_role):
        distance = abs(_role_tangent(name_role, side) - _role_tangent(number_role, side))
        limit = _row_pair_limit(number_role, name_role, side, box)
        if distance > limit:
            return 0.0
        return 1.0 + 2.0 * (1.0 - distance / limit)

    # _monotone_assignment indexes its first sequence.  Names are the items;
    # numbers are the roles so each OCR token is consumed at most once.
    name_to_number, _ = _monotone_assignment(names, numbers, pair_score)
    paired_numbers = set()
    rows = []
    for name_index, name_role in enumerate(names):
        number_role = name_to_number.get(name_index)
        if number_role is not None:
            paired_numbers.add(id(number_role))
        tangent_values = [_role_tangent(name_role, side)]
        if number_role is not None:
            tangent_values.append(_role_tangent(number_role, side))
        rows.append({"number": number_role, "name": name_role,
                     "tangent": sum(tangent_values) / len(tangent_values)})
    for number_role in numbers:
        if id(number_role) not in paired_numbers:
            rows.append({"number": number_role, "name": None,
                         "tangent": _role_tangent(number_role, side)})
    return sorted(rows, key=lambda row: row["tangent"])


def _assign_side(terminals, rows, component):
    ordered_terminals = sorted(terminals, key=lambda row: _terminal_tangent(row[1]))

    def joint_score(row, indexed_terminal):
        _, terminal = indexed_terminal
        number_score = _number_score(row["number"], terminal, component) if row["number"] else 0.0
        name_score = _name_score(row["name"], terminal, component) if row["name"] else 0.0
        if number_score <= 0 and name_score <= 0:
            return 0.0
        return number_score + name_score + (0.5 if number_score > 0 and name_score > 0 else 0.0)

    matched, scores = _monotone_assignment(ordered_terminals, rows, joint_score)
    assignments = {}
    for ordered_index, row in matched.items():
        original_index, terminal = ordered_terminals[ordered_index]
        assignments[original_index] = (row, scores[ordered_index],
                                       ordered_index, _terminal_tangent(terminal))
    return assignments


def assign_pin_semantics_v6(component, terminals, roles):
    """Assign box-like pin text per side with order-preserving gaps."""
    if component.type in TWO_TERMINAL or component.type in {"crystal_4pin", "gnd"}:
        return _abstain_unresolved_component(
            component, *assign_pin_semantics(component, terminals, roles))
    # P6.1 only changes the box/block hypothesis.  Other complex types retain
    # the S2 behavior so the A/B isolates sequence alignment.
    if component.type not in CONTEXT_NUMERIC_TYPES:
        return _abstain_unresolved_component(
            component, *assign_pin_semantics_v5(component, terminals, roles))

    box = component.body_bbox or component.bbox
    side_terminals = {side: [] for side in SIDES}
    for index, terminal in enumerate(terminals):
        if terminal.get("side") in side_terminals:
            side_terminals[terminal["side"]].append((index, terminal))

    number_roles = [role for role in roles if (
        role.role == "PIN_NUMBER" or (role.role == "VALUE" and role.normalized.isdigit())
    ) and PIN_NUMBER.fullmatch(role.normalized)]
    name_roles = [role for role in roles if role.role in ("PIN_NAME", "MODEL_TEXT")
                  and SIGNAL.fullmatch(role.normalized)]

    numbers_by_side = {side: [] for side in SIDES}
    names_by_side = {side: [] for side in SIDES}
    for role in number_roles:
        side = _best_side(role, side_terminals, component, _number_score)
        if side:
            numbers_by_side[side].append(role)
    for role in name_roles:
        side = _best_side(role, side_terminals, component, _name_score)
        if side:
            names_by_side[side].append(role)

    assignments = {}
    side_diagnostics = {}
    for side in SIDES:
        rows = _pair_number_names(numbers_by_side[side], names_by_side[side], side, box)
        assignments.update(_assign_side(side_terminals[side], rows, component))
        side_diagnostics[side] = {
            "terminal_count": len(side_terminals[side]),
            "number_token_count": len(numbers_by_side[side]),
            "name_token_count": len(names_by_side[side]),
            "text_row_count": len(rows),
        }

    # Duplicate normalized numbers remain invalid target keys.  Keep only the
    # assignment with the strongest joint score.
    best_number = {}
    for index, (row, score, _, _) in assignments.items():
        number_role = row.get("number")
        if number_role is None:
            continue
        current = best_number.get(number_role.normalized)
        if current is None or score > assignments[current][1]:
            best_number[number_role.normalized] = index

    pins, events = [], []
    for index, terminal in enumerate(terminals):
        assignment = assignments.get(index)
        row = assignment[0] if assignment else {"number": None, "name": None}
        joint_score = assignment[1] if assignment else None
        sequence_rank = assignment[2] if assignment else None
        number_role, name_role = row.get("number"), row.get("name")
        duplicate = bool(number_role and best_number.get(number_role.normalized) != index)
        exportable = number_role is not None and not duplicate
        number = number_role.normalized if exportable else f"UNK{index + 1}"
        name = name_role.token.text.strip() if name_role else ""
        reclassified = bool(number_role and number_role.role == "VALUE")
        source = ("contextual_numeric_ordered" if reclassified else
                  "local_ocr_ordered" if exportable else "unrecognized")
        pin = Pin(number, name, terminal["tip"], base=terminal["base"], side=terminal["side"],
                  inferred_number=not exportable,
                  number_confidence=number_role.confidence if exportable else 0.0,
                  name_confidence=name_role.confidence if name_role else 0.0,
                  exportable=exportable, observable_number=number if exportable else None,
                  number_source=source)
        pins.append(pin)
        events.append({
            **terminal, "number": number, "pinname": name,
            "number_confidence": pin.number_confidence,
            "name_confidence": pin.name_confidence,
            "exportable": exportable,
            "semantic_method": "side_ordered_joint_alignment",
            "number_token": number_role.token.text if number_role else None,
            "number_token_original_role": number_role.role if number_role else None,
            "number_role_reclassified": reclassified,
            "number_duplicate": duplicate,
            "name_token": name_role.token.text if name_role else None,
            "joint_assignment_score": joint_score,
            "sequence_side": terminal.get("side"),
            "sequence_rank": sequence_rank,
            "row_has_number_and_name": bool(number_role and name_role),
            "side_alignment": side_diagnostics.get(terminal.get("side"), {}),
        })
    return _abstain_unresolved_component(component, pins, events)

