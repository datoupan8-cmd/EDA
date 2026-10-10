"""Localized pin identity and name assignment."""
from __future__ import annotations
import math, re
from ...schema import Pin
from .detection_v3 import TWO_TERMINAL

PIN_NUMBER = re.compile(r"^(?:\d{1,4}|[A-Z]{1,2}\d{1,3}|EP|PAD)$", re.I)
SIGNAL = re.compile(r"^[A-Za-z][A-Za-z0-9_./+\-]{1,40}$")


def _aligned(token, terminal, component):
    x, y = terminal["tip"]; side = terminal["side"]
    x1, y1, x2, y2 = component.body_bbox or component.bbox
    tx, ty = token.center
    if side in ("left", "right"):
        return abs(ty - y) <= max(6, .035 * (y2 - y1)), abs(tx - x)
    return abs(tx - x) <= max(6, .035 * (x2 - x1)), abs(ty - y)


def assign_pin_semantics(component, terminals, roles):
    if component.type in TWO_TERMINAL:
        return [Pin(str(i), str(i), t["tip"], base=t["base"], side=t["side"], inferred_number=False, number_confidence=.98, name_confidence=.98, observable_number=str(i), number_source="audited_type_prior") for i, t in enumerate(terminals, 1)], [
            {**t, "number": str(i), "pinname": str(i), "number_confidence": .98, "name_confidence": .98, "semantic_method": "audited_two_terminal_prior"} for i, t in enumerate(terminals, 1)
        ]
    if component.type == "crystal_4pin":
        names = ("OSC1", "OSC2", "GND", "GND")
        return [Pin(str(i), names[i - 1], t["tip"], base=t["base"], side=t["side"], number_confidence=.85, name_confidence=.7, observable_number=str(i), number_source="crystal4_type_prior") for i, t in enumerate(terminals, 1)], [
            {**t, "number": str(i), "pinname": names[i - 1], "number_confidence": .85, "name_confidence": .7, "semantic_method": "crystal4_type_prior"} for i, t in enumerate(terminals, 1)
        ]
    if component.type == "gnd":
        # The official public targets use unobservable internal pin IDs here.
        t = terminals[0] if terminals else None
        if not t: return [], []
        pin = Pin("UNK1", "", t["tip"], base=t["base"], side=t["side"], inferred_number=True, exportable=False, number_source="unobservable_internal_id")
        return [pin], [{**t, "number": "UNK1", "pinname": "", "exportable": False, "semantic_method": "internal_id_not_guessed"}]
    x1, y1, x2, y2 = component.body_bbox or component.bbox
    pins, events, used_numbers = [], [], set()
    for index, terminal in enumerate(terminals, 1):
        aligned = []
        for role in roles:
            ok, distance = _aligned(role.token, terminal, component)
            if not ok: continue
            tx, ty = role.token.center
            inside = x1 - 2 <= tx <= x2 + 2 and y1 - 2 <= ty <= y2 + 2
            aligned.append((distance, inside, role))
        number_candidates = [(d, role) for d, inside, role in aligned if not inside and role.role == "PIN_NUMBER" and PIN_NUMBER.fullmatch(role.normalized)]
        name_candidates = [(d, role) for d, inside, role in aligned if inside and role.role in ("PIN_NAME", "MODEL_TEXT") and SIGNAL.fullmatch(role.normalized)]
        number_role = min(number_candidates, default=(None, None), key=lambda x: x[0])[1]
        name_role = min(name_candidates, default=(None, None), key=lambda x: x[0])[1]
        number = number_role.normalized if number_role else f"UNK{index}"
        duplicate = number in used_numbers
        exportable = number_role is not None and not duplicate
        used_numbers.add(number)
        name = name_role.token.text.strip() if name_role else ""
        pin = Pin(number, name, terminal["tip"], base=terminal["base"], side=terminal["side"], inferred_number=not exportable,
                  number_confidence=number_role.confidence if number_role else 0.0, name_confidence=name_role.confidence if name_role else 0.0,
                  exportable=exportable, observable_number=number if exportable else None, number_source="local_ocr" if exportable else "unrecognized")
        pins.append(pin)
        events.append({**terminal, "number": number, "pinname": name, "number_confidence": pin.number_confidence, "name_confidence": pin.name_confidence,
                       "exportable": exportable, "semantic_method": "terminal_local_text", "number_token": number_role.token.text if number_role else None,
                       "name_token": name_role.token.text if name_role else None})
    return pins, events
