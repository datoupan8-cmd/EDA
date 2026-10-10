"""P6.3 confidence gate for terminals added by directional localization.

The gate is deliberately applied after V6's ordered assignment.  It leaves
every V3 terminal unchanged and only abstains from exporting a P2/P3-added
terminal when the OCR row lacks mutually supporting number and name evidence.
"""
from __future__ import annotations

from .pin_semantics_v6 import assign_pin_semantics_v6


MIN_JOINT_ASSIGNMENT_SCORE = 6.0
MIN_NUMBER_CONFIDENCE = 0.8
MIN_NAME_CONFIDENCE = 0.8
GATED_METHODS = {"directional_boundary_support"}


def directional_candidate_is_exportable(event: dict) -> bool:
    """Return whether a new directional terminal has strong joint OCR evidence."""
    return bool(
        event.get("row_has_number_and_name")
        and float(event.get("joint_assignment_score") or 0.0) >= MIN_JOINT_ASSIGNMENT_SCORE
        and float(event.get("number_confidence") or 0.0) >= MIN_NUMBER_CONFIDENCE
        and float(event.get("name_confidence") or 0.0) >= MIN_NAME_CONFIDENCE
    )


def apply_directional_candidate_gate(pins, events):
    """Abstain on weak P2/P3 additions without changing legacy terminals."""
    if len(pins) != len(events):
        raise ValueError("Pin/event cardinality mismatch in directional confidence gate")
    for pin, event in zip(pins, events):
        method = event.get("method")
        if method not in GATED_METHODS:
            event["directional_gate_applied"] = False
            continue
        passed = directional_candidate_is_exportable(event)
        event["directional_gate_applied"] = True
        event["directional_gate_passed"] = passed
        if pin.exportable and not passed:
            pin.exportable = False
            pin.inferred_number = True
            pin.observable_number = None
            pin.number_source = "directional_candidate_confidence_gate"
            event["exportable"] = False
            event["semantic_abstention"] = "directional_candidate_low_confidence"
    return pins, events


def assign_pin_semantics_v7(component, terminals, roles):
    """Run V6 semantics, then gate only directional localization additions."""
    pins, events = assign_pin_semantics_v6(component, terminals, roles)
    return apply_directional_candidate_gate(pins, events)
