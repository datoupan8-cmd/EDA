"""Experimental P6-S1 pin semantics with contextual numeric reclassification.

The global OCR role classifier is deliberately unchanged.  This module only
allows an aligned, outside-body, bare numeric VALUE token to act as a pin
number for box-like multi-pin components.  All V3 geometry and greedy
assignment behavior is retained so the A/B isolates this one hypothesis.
"""
from __future__ import annotations

from .pin_detection import TWO_TERMINAL
from .pin_semantics import PIN_NUMBER, SIGNAL, _aligned, assign_pin_semantics
from .schema import Pin


CONTEXT_NUMERIC_TYPES = frozenset({"box", "block"})


def assign_pin_semantics_v4(component, terminals, roles):
    """Run V3 except for one context-limited VALUE→pin-number reinterpretation."""
    if component.type in TWO_TERMINAL or component.type in {"crystal_4pin", "gnd"}:
        return assign_pin_semantics(component, terminals, roles)

    x1, y1, x2, y2 = component.body_bbox or component.bbox
    pins, events, used_numbers = [], [], set()
    for index, terminal in enumerate(terminals, 1):
        aligned = []
        for role in roles:
            ok, distance = _aligned(role.token, terminal, component)
            if not ok:
                continue
            tx, ty = role.token.center
            inside = x1 - 2 <= tx <= x2 + 2 and y1 - 2 <= ty <= y2 + 2
            aligned.append((distance, inside, role))

        def number_candidate(row):
            _, inside, role = row
            if inside or not PIN_NUMBER.fullmatch(role.normalized):
                return False
            if role.role == "PIN_NUMBER":
                return True
            return (component.type in CONTEXT_NUMERIC_TYPES
                    and role.role == "VALUE"
                    and role.normalized.isdigit())

        number_candidates = [(distance, role) for distance, inside, role in aligned
                             if number_candidate((distance, inside, role))]
        name_candidates = [(distance, role) for distance, inside, role in aligned
                           if inside and role.role in ("PIN_NAME", "MODEL_TEXT")
                           and SIGNAL.fullmatch(role.normalized)]
        number_role = min(number_candidates, default=(None, None), key=lambda row: row[0])[1]
        name_role = min(name_candidates, default=(None, None), key=lambda row: row[0])[1]
        number = number_role.normalized if number_role else f"UNK{index}"
        duplicate = number in used_numbers
        exportable = number_role is not None and not duplicate
        used_numbers.add(number)
        name = name_role.token.text.strip() if name_role else ""
        reclassified = bool(number_role and number_role.role == "VALUE")
        source = ("contextual_numeric_value" if reclassified else
                  "local_ocr" if exportable else "unrecognized")
        pin = Pin(number, name, terminal["tip"], base=terminal["base"], side=terminal["side"],
                  inferred_number=not exportable,
                  number_confidence=number_role.confidence if number_role else 0.0,
                  name_confidence=name_role.confidence if name_role else 0.0,
                  exportable=exportable, observable_number=number if exportable else None,
                  number_source=source)
        pins.append(pin)
        events.append({
            **terminal,
            "number": number,
            "pinname": name,
            "number_confidence": pin.number_confidence,
            "name_confidence": pin.name_confidence,
            "exportable": exportable,
            "semantic_method": "terminal_local_text_v4_numeric_context",
            "number_token": number_role.token.text if number_role else None,
            "number_token_original_role": number_role.role if number_role else None,
            "number_role_reclassified": reclassified,
            "number_duplicate": duplicate,
            "name_token": name_role.token.text if name_role else None,
        })
    return pins, events
