"""L37 public-library number constraints with image-only model evidence.

No paths, GT, OCR calls, terminal generation, or library-internal ID mapping.
All existing component geometry and OCR-selected names are preserved.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import dataclass
import math
import re
import unicodedata

from pcb.schema import Component, Scene, Text


@dataclass(frozen=True)
class LibraryPolicy:
    model_confidence: float = .80
    name_confidence: float = .80
    model_min_chars: int = 5
    model_distance_heights: float = 3.0


POLICY = LibraryPolicy()


def model_key(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text))).upper()


def name_key(text: str) -> str:
    # This affects lookup only, never the exported name.
    return model_key(text).replace("_", "")


def library_candidates(text: str, library: dict) -> tuple[str, ...]:
    normalized = model_key(text)
    exact = sorted(k for k in library if model_key(k) == normalized)
    if exact:
        return tuple(exact)
    # Only documented supplier-part suffixes; do not guess chip families.
    return tuple(sorted(k for k in library
                        if model_key(re.sub(r"_C\d+(?:_\d+)?$", "", k)) == normalized))


def name_numbers(keys: tuple[str, ...], library: dict, name: str) -> tuple[str, ...]:
    normalized = name_key(name)
    if not normalized:
        return ()
    values = set()
    for key in keys:
        for pin in library[key]["pins"]:
            if name_key(pin.get("name", "")) == normalized:
                number = str(pin.get("number", "")).strip().upper()
                if number and re.fullmatch(r"(?:\d{1,4}|[A-Z]{1,2}\d{1,3}|EP|PAD)", number):
                    values.add(number)
    return tuple(sorted(values))


def model_matches(components: list[Component], texts: list[Text], library: dict,
                  policy: LibraryPolicy = POLICY) -> tuple[dict, list]:
    """Exact model OCR matching followed by unique nearest-body ownership."""
    boxes = [c for c in components if c.type == "box" and not c.key.startswith("UNRESOLVED_")]
    evidence = defaultdict(list)
    events = []
    for index, token in enumerate(texts):
        normalized = model_key(token.text)
        if (len(normalized) < policy.model_min_chars or token.score < policy.model_confidence
                or not re.search(r"[A-Z]", normalized) or not re.search(r"\d", normalized)):
            continue
        keys = library_candidates(token.text, library)
        if not keys:
            continue
        cx, cy = token.center
        height = max(1., token.bbox[3] - token.bbox[1])
        owners = []
        for c in boxes:
            x1, y1, x2, y2 = c.body_bbox or c.bbox
            dx = max(x1 - cx, 0., cx - x2)
            dy = max(y1 - cy, 0., cy - y2)
            distance = math.hypot(dx, dy)
            if distance <= policy.model_distance_heights * height:
                owners.append((distance, c.key))
        owners.sort()
        record = {"token_index": index, "text": token.text, "bbox": token.bbox,
                  "confidence": token.score, "library_keys": keys, "owner_candidates": owners}
        if not owners:
            events.append({**record, "status": "no_resolved_box_owner"}); continue
        if len(owners) > 1 and abs(owners[0][0] - owners[1][0]) < 1e-9:
            events.append({**record, "status": "ambiguous_owner"}); continue
        record.update(component=owners[0][1], status="owned")
        evidence[owners[0][1]].append(record)
        events.append(record)
    accepted = {}
    for component, records in evidence.items():
        choices = {tuple(r["library_keys"]) for r in records}
        if len(choices) != 1:
            events.append({"component": component, "status": "multiple_model_evidence"})
            continue
        strongest = max(records, key=lambda r: r["confidence"])
        accepted[component] = {"library_keys": next(iter(choices)), "evidence": records,
                               "model_confidence": strongest["confidence"]}
    return accepted, events


def stage(scene: Scene, library: dict, policy: LibraryPolicy = POLICY):
    """Repair existing Pin numbers only when the selected name uniquely maps.

    Simultaneous proposals allow number-release chains, e.g. wrong 8->7 then
    missing->8. Unresolved collision components are left unchanged.
    """
    result = copy.deepcopy(scene)
    models, model_events = model_matches(result.components, result.texts, library, policy)
    changes = []
    reviews = []
    for c in result.components:
        if c.type != "box" or c.key not in models:
            continue
        model = models[c.key]
        proposals = {}
        entries = {}
        for index, pin in enumerate(c.pins):
            options = name_numbers(tuple(model["library_keys"]), library, pin.name or "")
            row = {"component": c.key, "terminal_index": index, "tip": pin.tip,
                   "side": pin.side, "name": pin.name, "name_confidence": pin.name_confidence,
                   "old_number": pin.number, "old_exportable": pin.exportable,
                   "library_number_candidates": options,
                   "library_keys": model["library_keys"]}
            entries[index] = row
            if pin.name_confidence < policy.name_confidence:
                row["status"] = "low_or_missing_name_confidence"
            elif len(options) != 1:
                row["status"] = "ambiguous_name_number" if options else "no_library_name_support"
            elif pin.exportable and pin.number == options[0]:
                row["status"] = "already_consistent"
            else:
                row["status"] = "proposed"
                proposals[index] = options[0]
        duplicate_numbers = {number for number, count in Counter(proposals.values()).items() if count > 1}
        for index in list(proposals):
            if proposals[index] in duplicate_numbers:
                entries[index]["status"] = "duplicate_proposed_number"
                del proposals[index]
        # Fixed-point removal prevents a rejected proposal's original number
        # from being taken by another proposal. No scorer/GT-based tie-break.
        while True:
            reserved = {p.number for i, p in enumerate(c.pins)
                        if i not in proposals and p.exportable}
            blocked = [i for i, number in proposals.items() if number in reserved]
            if not blocked:
                break
            for index in blocked:
                entries[index]["status"] = "number_reserved_by_unchanged_pin"
                del proposals[index]
        for index, number in proposals.items():
            pin = c.pins[index]
            entries[index]["status"] = "accepted"
            entries[index]["new_number"] = number
            entries[index]["new_exportable"] = True
            changes.append(entries[index])
            pin.number = number
            pin.exportable = True
            pin.observable_number = None  # Inferred from a library, not read.
            pin.inferred_number = True
            pin.number_source = "L37_public_library_from_image_model_and_OCR_name"
            pin.number_confidence = min(model["model_confidence"], pin.name_confidence)
        reviews.extend(entries.values())
    # Retain all existing events and their OCR provenance.
    event_groups = defaultdict(list)
    for event in result.diagnostics.get("pin_events", []):
        event_groups[event["component"]].append(event)
    for change in changes:
        c = next(c for c in result.components if c.key == change["component"])
        group = event_groups[c.key]
        if group:
            if len(group) != len(c.pins):
                raise AssertionError("Incomplete original Pin events")
            event = group[change["terminal_index"]]
            pin = c.pins[change["terminal_index"]]
            event.update(number=pin.number, exportable=True, number_confidence=pin.number_confidence,
                         semantic_method="L37_library_number_after_L36",
                         library_number_evidence=copy.deepcopy(change))
    trace = {"models": models, "model_events": model_events, "pin_reviews": reviews,
             "accepted_changes": changes, "public_library_pin_id_used": False,
             "GT_read_in_inference": False}
    result.diagnostics["L37_public_library"] = trace
    return result, trace
