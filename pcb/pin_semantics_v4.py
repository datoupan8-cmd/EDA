"""Conservative local pin labels with a component-wide one-to-one assignment.

Unobserved pin identities remain stable internal geometry IDs, never fabricated
package pin numbers. Pure numeric OCR VALUE tokens may be numbers only when
they pass the same local terminal/edge constraints as explicit PIN_NUMBER roles.
"""
from __future__ import annotations

import math
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from .pin_detection import TWO_TERMINAL
from .pin_semantics import PIN_NUMBER, SIGNAL, assign_pin_semantics
from .schema import Pin

_DIRECTIONS = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}
_SIDE_ORDER = {side: index for index, side in enumerate(_DIRECTIONS)}
_LETTER_NUMBER = re.compile(r"^[A-Z]{1,2}\d{1,3}$")


def _frame(component, terminal):
    side = terminal["side"]
    nx, ny = _DIRECTIONS[side]
    tip = terminal["tip"]
    x1, y1, x2, y2 = component.body_bbox or component.bbox
    base = terminal.get("base")
    if base is None:
        base = {"left": (x1, tip[1]), "right": (x2, tip[1]),
                "top": (tip[0], y1), "bottom": (tip[0], y2)}[side]
    depth = x2 - x1 if nx else y2 - y1
    coordinate = base[1] if nx else base[0]
    lead = max(2.0, math.dist(tip, base))
    return base, nx, ny, max(1.0, depth), coordinate, lead


def _pitches(component, terminals):
    output = []
    frames = [_frame(component, terminal) for terminal in terminals]
    for index, terminal in enumerate(terminals):
        coordinate = frames[index][4]
        distances = [abs(coordinate - frame[4]) for j, frame in enumerate(frames)
                     if j != index and terminals[j]["side"] == terminal["side"]
                     and abs(coordinate - frame[4]) > 1.0]
        output.append(min(distances) if distances else None)
    return frames, output


def _local_cost(component, terminal, frame, pitch, role, *, name=False):
    """Bound both along-edge displacement and outward/inward reach.

    Using the local pitch prevents a large package height from permitting a
    number several rows away; a bounded edge strip also permits printed numbers
    just inside a symbol without accepting unrelated central model text.
    """
    base, nx, ny, depth, _, lead = frame
    tx, ty = role.token.center
    a, b, c, d = role.token.bbox
    font = max(3.0, min(abs(c - a), abs(d - b)))
    normal = (tx - base[0]) * nx + (ty - base[1]) * ny
    tangent = abs((ty - base[1]) if nx else (tx - base[0]))
    tolerance = max(5.0, .72 * font)
    if pitch is not None:
        tolerance = min(tolerance, max(4.0, .58 * pitch))
    if tangent > tolerance:
        return None
    inside_limit = min(.29 * depth, max(12.0, 1.8 * font))
    outside_limit = min(max(24.0, 2.2 * lead, 2.4 * font), max(24.0, .35 * depth))
    if name:
        # Names may be long: the edge-facing end, rather than centre, must be
        # close to the component side. They must still be on the same row.
        near_normal = {"left": base[0] - a, "right": c - base[0],
                       "top": base[1] - b, "bottom": d - base[1]}[terminal["side"]]
        if normal > 3.0 or near_normal < -min(.4 * depth, max(18.0, 2.5 * font)):
            return None
        normal_cost = min(1.0, abs(near_normal) / max(inside_limit, 1.0))
    else:
        if not -inside_limit <= normal <= outside_limit:
            return None
        normal_cost = abs(normal) / (inside_limit if normal < 0 else outside_limit)
        # A central number belongs to no side, even on tiny symbols.
        x1, y1, x2, y2 = component.body_bbox or component.bbox
        if x1 <= tx <= x2 and y1 <= ty <= y2:
            distances = {"left": tx - x1, "right": x2 - tx,
                         "top": ty - y1, "bottom": y2 - ty}
            if distances[terminal["side"]] > min(distances.values()) + font * .35:
                return None
    confidence = max(0.0, min(1.0, float(role.confidence)))
    return 1.1 * tangent / tolerance + .45 * normal_cost + .2 * (1.0 - confidence)


def _unknown_ids(component, terminals):
    ranks = {}
    for side in _DIRECTIONS:
        indices = [i for i, terminal in enumerate(terminals) if terminal["side"] == side]
        indices.sort(key=lambda i: (_frame(component, terminals[i])[4],
                                    tuple(terminals[i]["tip"])))
        for rank, index in enumerate(indices, 1):
            ranks[index] = f"UNK_{side}_{rank}"
    return ranks


def _candidate_costs(component, terminal, frame, pitch, role):
    """Apply identical semantic/geometric gates during ownership and matching."""
    normalized = re.sub(r"\s+", "", role.normalized).upper()
    if component.type == "gnd":
        return normalized, None, None
    x1, y1, x2, y2 = component.body_bbox or component.bbox
    tx, ty = role.token.center
    inside_letter_number = (role.role == "PIN_NUMBER" and bool(_LETTER_NUMBER.fullmatch(normalized))
                            and x1 <= tx <= x2 and y1 <= ty <= y2)
    # A1/P12 inside a schematic package normally names a signal, while an
    # outside A1 may be a physical BGA ball. Pure digits remain eligible inside.
    is_number = not inside_letter_number and (
        (role.role == "PIN_NUMBER" and PIN_NUMBER.fullmatch(normalized)) or
        (role.role == "VALUE" and re.fullmatch(r"\d{1,4}", normalized)))
    number_cost = _local_cost(component, terminal, frame, pitch, role) if is_number else None
    if number_cost is not None:
        number_cost += .08 if role.role == "VALUE" else 0.0
    is_name = (role.role in {"PIN_NAME", "MODEL_TEXT"} or inside_letter_number) and SIGNAL.fullmatch(normalized)
    name_cost = _local_cost(component, terminal, frame, pitch, role, name=True) if is_name else None
    return normalized, number_cost, name_cost


def assign_role_owners(localized, roles):
    """Assign each physical OCR token to at most one component before matching.

    This does not globally deduplicate number strings: different tokens reading
    ``1`` may belong to different chips. Exact duplicate text/box observations
    share an owner. Tie-breaking uses stable component identity and geometry,
    not the iteration order, except for indistinguishable duplicate components.
    """
    grouped_roles = {}
    for role in roles:
        token_key = (tuple(role.token.bbox), str(role.token.text).strip().casefold())
        grouped_roles.setdefault(token_key, []).append(role)
    candidates = {}
    representative_roles = {}
    for component_index, (component, terminals) in enumerate(localized):
        if component.type in TWO_TERMINAL or component.type in {"crystal_4pin", "gnd"} or not terminals:
            continue
        frames, pitches = _pitches(component, terminals)
        stable_component = (component.key.casefold(), component.key,
                            tuple(component.body_bbox or component.bbox), component.type, component_index)
        for token_key, token_roles in grouped_roles.items():
            best = None
            best_role = None
            for role in token_roles:
                for terminal, frame, pitch in zip(terminals, frames, pitches):
                    _, number_cost, name_cost = _candidate_costs(component, terminal, frame, pitch, role)
                    for kind, cost in (("number", number_cost), ("name", name_cost)):
                        if cost is None or cost >= 2.0:
                            continue
                        distance = math.dist(role.token.center, frame[0])
                        option = (float(cost), distance, stable_component, kind, str(role.index))
                        if best is None or option < best:
                            best = option
                            best_role = role
            if best is not None:
                candidates.setdefault(token_key, []).append((best, component_index))
                representative_roles[(token_key, component_index)] = best_role
    owned = [[] for _ in localized]
    events = []
    for token_key, choices in candidates.items():
        best, component_index = min(choices)
        # Repeated observations are still one physical token, including within
        # the owner's per-pin name assignment.
        owned[component_index].append(representative_roles[(token_key, component_index)])
        events.append({"text": grouped_roles[token_key][0].token.text,
                       "bbox": list(token_key[0]),
                       "token_indices": [role.index for role in grouped_roles[token_key]],
                       "owner_component": localized[component_index][0].key,
                       "owner_component_index": component_index,
                       "candidate_components": [localized[index][0].key for _, index in sorted(choices)],
                       "candidate_component_count": len(choices),
                       "assignment_cost": best[0], "candidate_kind": best[3]})
    return owned, events


def assign_pin_semantics_v4(component, terminals, roles):
    """Return pins and audit events without changing inputs or guessing IDs."""
    if component.type in TWO_TERMINAL or component.type == "crystal_4pin":
        return assign_pin_semantics(component, terminals, roles)
    if not terminals:
        return [], []
    frames, pitches = _pitches(component, terminals)
    unknown = _unknown_ids(component, terminals)
    candidates = {}
    names = {}
    numbers = set()
    is_ground = component.type == "gnd"
    for index, terminal in enumerate(terminals):
        for role in roles:
            normalized, number_cost, name_cost = _candidate_costs(component, terminal, frames[index], pitches[index], role)
            if number_cost is not None:
                key = (index, normalized)
                previous = candidates.get(key)
                tie = (number_cost, str(role.token.text), tuple(role.token.bbox))
                if previous is None or tie < previous[0]:
                    candidates[key] = (tie, role)
                numbers.add(normalized)
            if name_cost is not None:
                key = str(role.index)
                names[(index, key)] = (name_cost, role)

    # Each observed number is a single global column, so duplicate OCR readings
    # compete for one pin. Dummy columns retain unknowns instead of forcing IDs.
    number_list = sorted(numbers)
    count = len(terminals)
    costs = np.full((count, len(number_list) + count), 10000.0)
    costs[:, len(number_list):] = 2.0
    number_to_column = {number: i for i, number in enumerate(number_list)}
    for (index, number), (tie, role) in candidates.items():
        costs[index, number_to_column[number]] = tie[0]
    assignments = {}
    if number_list:
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            if column < len(number_list) and costs[row, column] < 2.0:
                number = number_list[column]
                assignments[int(row)] = (number, candidates[(int(row), number)][1])

    # Name text is also physical evidence: never share one OCR token across pins.
    name_assignments = {}
    name_keys = sorted({key for _, key in names})
    if name_keys:
        name_columns = {key: index for index, key in enumerate(name_keys)}
        name_costs = np.full((count, len(name_keys) + count), 10000.0)
        name_costs[:, len(name_keys):] = 2.0
        for (index, key), (cost, role) in names.items():
            name_costs[index, name_columns[key]] = cost
        rows, columns = linear_sum_assignment(name_costs)
        for row, column in zip(rows, columns):
            if column < len(name_keys) and name_costs[row, column] < 2.0:
                name_assignments[int(row)] = names[(int(row), name_keys[column])][1]

    pins, events = [], []
    for index, terminal in enumerate(terminals):
        number, number_role = assignments.get(index, (unknown[index], None))
        name_role = name_assignments.get(index)
        exportable = number_role is not None
        name = name_role.token.text.strip() if name_role else ""
        source = "local_ocr_v4" if exportable else ("unobservable_internal_id" if is_ground else "unrecognized")
        pin = Pin(number, name, terminal["tip"], base=frames[index][0], side=terminal["side"],
                  inferred_number=not exportable, exportable=exportable,
                  observable_number=number if exportable else None,
                  number_confidence=float(number_role.confidence) if number_role else 0.0,
                  name_confidence=float(name_role.confidence) if name_role else 0.0,
                  number_source=source)
        pins.append(pin)
        events.append({**terminal, "number": number, "pinname": name, "exportable": exportable,
                       "inferred_number": pin.inferred_number, "observable_number": pin.observable_number,
                       "number_confidence": pin.number_confidence, "name_confidence": pin.name_confidence,
                       "semantic_method": "local_global_assignment_v4" if not is_ground else "internal_id_not_guessed",
                       "number_token": number_role.token.text if number_role else None,
                       "name_token": name_role.token.text if name_role else None,
                       "signal_label_reclassified": bool(name_role and name_role.role == "PIN_NUMBER"),
                       "number_role": number_role.role if number_role else None,
                       "promoted_numeric_value": bool(number_role and number_role.role == "VALUE"),
                       "number_candidate_count": sum(i == index for i, _ in candidates),
                       "assignment_cost": float(costs[index, number_to_column[number]]) if exportable else None})
    return pins, events
