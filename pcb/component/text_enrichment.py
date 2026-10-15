"""Recover printed model names and capacitor shorthand without changing detections."""
from __future__ import annotations

import copy
from functools import lru_cache
import json
import math
from pathlib import Path
import re

from ..assignment import maximum_weight_assignment
from ..core.interfaces import ComponentStageOutput
from .latest.text_assignment import BOX_FAMILY, MODEL_RE, bbox_gap

CAPACITOR_SHORTHAND = re.compile(r"^[<>]?[-+]?(?:\d+(?:[.,]\d+)?|\.\d+)\s*[pnuµμm]$", re.I)


@lru_cache(maxsize=1)
def _known_models():
    library = json.loads((Path(__file__).resolve().parents[2] / "assets/cpntLibrary.json").read_text(encoding="utf-8"))
    # Single/two-terminal symbols and supply labels are not chip model evidence.
    return frozenset(re.sub(r"\s+", "", name).upper() for name, entry in library.items()
                     if len(entry.get("pins", [])) >= 3)


def _assign_model_names(components, roles, claimed, known_models):
    indices = [i for i, c in enumerate(components) if c.type in BOX_FAMILY]
    candidates = []
    for role in roles:
        text = re.sub(r"\s+", "", role.token.text.strip())
        known = text.upper() in known_models
        # The generic classifier deliberately labels e.g. ball identifiers as
        # PIN_NUMBER. Only an exact public model can override that ambiguity.
        readable_model = (role.role == "MODEL_TEXT" and MODEL_RE.fullmatch(text)
                          and sum(char.isdigit() for char in text) >= 3)
        if role.index not in claimed and (known or readable_model):
            candidates.append(role)
    weights = []
    for role in candidates:
        row = []
        for index in indices:
            component = components[index]
            box, token = component.bbox, role.token
            width, height = max(4.0, box[2] - box[0]), max(4.0, box[3] - box[1])
            gap = bbox_gap(token.bbox, box)
            if gap > max(30.0, .25 * max(width, height)):
                row.append(0.0)
                continue
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            distance = abs(token.center[0] - center[0]) / width + abs(token.center[1] - center[1]) / height
            inside = box[0] - 2 <= token.center[0] <= box[2] + 2 and box[1] - 2 <= token.center[1] <= box[3] + 2
            row.append(float(role.confidence) * (1.35 if inside else 1.0) * math.exp(-1.6 * distance - gap / max(width, height)))
        weights.append(row)
    accepted = []
    for token_index, column in maximum_weight_assignment(weights):
        score = weights[token_index][column]
        if score < .08:
            continue
        owner = components[indices[column]]
        role = candidates[token_index]
        owner.name = role.token.text.strip()
        accepted.append({"component": owner.key, "token": owner.name, "score": score,
                         "public_model": re.sub(r"\s+", "", owner.name).upper() in known_models})
    return {"accepted": accepted, "candidate_count": len(candidates)}


def _value_weight(component, role):
    width = max(5.0, component.bbox[2] - component.bbox[0])
    height = max(5.0, component.bbox[3] - component.bbox[1])
    gap = bbox_gap(role.token.bbox, component.bbox)
    if gap > max(80.0, 5 * (role.token.bbox[3] - role.token.bbox[1]), 2.5 * max(width, height)):
        return 0.0
    center = ((component.bbox[0] + component.bbox[2]) / 2,
              (component.bbox[1] + component.bbox[3]) / 2)
    normalized = math.dist(center, role.token.center) / max(8.0, math.sqrt(width * height))
    return float(role.confidence) * math.exp(-.9 * normalized)


def _claimed_value_roles(components, roles, diagnostics):
    """Respect physical tokens already assigned by the frozen value stage."""
    owners = {c.key: c for c in components}
    claimed = set()
    for event in diagnostics.get("value_assignment", {}).get("accepted", []):
        owner = owners.get(event.get("component"))
        if owner is None or event.get("method") == "designator_tail":
            continue
        for role in roles:
            if role.role != "VALUE" or role.token.text.strip() != event.get("token"):
                continue
            if math.isclose(_value_weight(owner, role), float(event["score"]), rel_tol=1e-7, abs_tol=1e-9):
                claimed.add(role.index)
    return claimed


def enrich_fields(components, roles, diagnostics, *, names=True, shorthand=True, known_models=None):
    """Keep identity/geometry and existing trusted fields; use visible OCR only."""
    result = copy.deepcopy(components)
    detail = copy.deepcopy(diagnostics)
    changes = []
    if names:
        proposals = copy.deepcopy(result)
        model_trace = _assign_model_names(proposals, roles, _claimed_value_roles(result, roles, detail),
                                         _known_models() if known_models is None else known_models)
        for original, proposal in zip(result, proposals):
            if original.name in (None, original.key) and proposal.name not in (None, original.key):
                changes.append({"component": original.key, "field": "name", "before": original.name,
                                "after": proposal.name, "source": "image_model_text_evidence"})
                original.name = proposal.name
        detail["component_model_text_assignment"] = model_trace
    if shorthand:
        indices = [i for i, c in enumerate(result) if c.type == "c" and c.value is None]
        claimed = _claimed_value_roles(result, roles, detail)
        tokens = [role for role in roles if role.role == "VALUE" and role.index not in claimed
                  and CAPACITOR_SHORTHAND.fullmatch(role.token.text.strip())]
        weights = [[_value_weight(result[index], role) for index in indices] for role in tokens]
        for token_index, column in maximum_weight_assignment(weights):
            if weights[token_index][column] < .055:
                continue
            component = result[indices[column]]
            value = tokens[token_index].token.text.strip()
            component.value = value
            changes.append({"component": component.key, "field": "value", "before": None,
                            "after": value, "source": "printed_capacitor_shorthand"})
    detail["component_text_enrichment"] = {"version": "text_enriched_v1", "changes": changes,
        "identity_and_geometry_changed": False, "GT_used_in_inference": False}
    return result, detail


class ComponentStageTextEnriched:
    def __init__(self, baseline=None):
        if baseline is None:
            from .latest.stage import ComponentStageV4
            baseline = ComponentStageV4()
        self.baseline = baseline

    def run(self, image, text, context):
        original = self.baseline.run(image, text, context)
        components, diagnostics = enrich_fields(original.components, original.roles, original.diagnostics)
        return ComponentStageOutput(components, original.texts, original.roles, diagnostics, original.debug)
