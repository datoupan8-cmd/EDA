"""P6.5: fill audited D/LED names without inferring symbol polarity.

This policy does not claim that geometric terminal order identifies cathode.
It only makes the existing reference number and name consistent: 1 -> K,
2 -> A. Terminal ownership, number, geometry and export decisions are frozen.
"""
from __future__ import annotations

from typing import Any

from ...schema import Pin

DIODE_TYPES = frozenset({"d", "led"})
NUMBER_TO_NAME = {"1": "K", "2": "A"}


def mapped_diode_name(component_type: str, number: str, name: str | None) -> str | None:
    """Map only the existing numeric placeholder name; preserve recognized names."""
    if component_type in DIODE_TYPES and number in NUMBER_TO_NAME and name == number:
        return NUMBER_TO_NAME[number]
    return name


def apply_diode_pinname_mapping(
    component_type: str, pins: list[Pin], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Change only Pin.name and its event mirror, never number/geometry/confidence."""
    if len(pins) != len(events):
        raise ValueError("Pin/event cardinality mismatch in diode name mapping")
    changes = []
    for index, (pin, event) in enumerate(zip(pins, events)):
        new_name = mapped_diode_name(component_type, pin.number, pin.name)
        if new_name == pin.name:
            continue
        old_name = pin.name
        pin.name = new_name
        event["pinname"] = new_name
        event["pinname_source"] = "audited_diode_number_name_mapping"
        event["diode_pinname_previous"] = old_name
        changes.append({
            "terminal_index": index, "number": pin.number,
            "old_pinname": old_name, "pinname": new_name,
            "exportable": pin.exportable,
        })
    return changes

