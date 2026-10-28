"""Literal OCR candidate union: preserve differing readings until assignment."""
from __future__ import annotations
import copy

from .geometry import bbox_iou
from .tokens import OcrToken


def merge_literal_candidates(*sources: list[OcrToken]) -> list[OcrToken]:
    """Deduplicate identical text at the same place; retain case/space variants.

    This is image-only candidate generation. There is no target vocabulary,
    designator/model format gate, or alteration of the recognized text.
    """
    kept: list[OcrToken] = []
    for source in sources:
        for token in source:
            if not token.text.strip():
                continue
            duplicate = next((i for i, old in enumerate(kept)
                              if old.text == token.text and bbox_iou(old.bbox_xyxy, token.bbox_xyxy) >= .5), None)
            if duplicate is None:
                kept.append(token)
            elif token.confidence > kept[duplicate].confidence:
                kept[duplicate] = token
    return kept


def recover_missing_fields(baseline: dict, alternative: dict) -> tuple[dict, list[dict]]:
    """Recover unresolved identities; preserve every resolved component.

    This adapter is for the components-only stage, before pins/nets exist.
    Matching uses identical detector geometry/type, never annotations. Resolved
    identities, existing names/values, detection count and order are fixed.
    """
    for scene in (baseline, alternative):
        if scene.get("nets") or any(scene.get("pins", {}).values()):
            raise ValueError("Candidate recovery must run before pins/net inference")
    result = copy.deepcopy(baseline)
    locations: dict[tuple, list] = {}
    for key, component in alternative["components"].items():
        locations.setdefault((component["type"], tuple(component["bbox"])), []).append((key, component))
    used = set(baseline["components"])
    events = []
    components = {}
    for old_key, old in baseline["components"].items():
        key, component = old_key, copy.deepcopy(old)
        matches = locations.get((old["type"], tuple(old["bbox"])), [])
        changes = {}
        if old_key.startswith("UNRESOLVED_") and len(matches) == 1 and old["type"] not in {"v", "gnd"} and not old["type"].startswith("net_"):
            new_key, new = matches[0]
            if old_key.startswith("UNRESOLVED_") and not new_key.startswith("UNRESOLVED_") and new_key not in used:
                key = new_key
                used.add(key)
                changes["key"] = key
                if old["Name"] in (None, old_key):
                    component["Name"] = new["Name"] or key
                    changes["Name"] = component["Name"]
            # A null value on an already resolved part may be intentional.
            # Auxiliary fields are filled only with a recovered identity.
            if key != old_key and component["Name"] in (None, old_key, key) and new["Name"] not in (None, new_key):
                component["Name"] = new["Name"]
                changes["Name"] = component["Name"]
            if key != old_key and old["value"] is None and new["value"] is not None:
                component["value"] = new["value"]
                changes["value"] = component["value"]
        components[key] = component
        if changes:
            events.append({"old_key": old_key, "changes": changes, "bbox": old["bbox"]})
    result["components"] = components
    result["pins"] = {key: {} for key in components}
    return result, events


def rescue_observed_proposals(prediction: dict, before_filter: list[dict], height: int) -> tuple[dict, list[dict]]:
    """Restore filtered proposals with an image-observed, unused identity.

    No target text or confidence threshold is used. Existing predictions are
    preserved; unresolved proposals and automatic symbols are not synthesized.
    Inputs are the frozen frontend's own components before its P01 filter.
    """
    if prediction.get("nets") or any(prediction.get("pins",{}).values()):
        raise ValueError("Proposal rescue must precede pins/net inference")
    result=copy.deepcopy(prediction);events=[]
    for component in before_filter:
        key=component["key"]
        if component["type"] in {"v","gnd","VCC","GND"} or component["type"].startswith("net_"):
            continue
        if not component.get("observable_designator") or key.startswith("UNRESOLVED_") or key in result["components"]:
            continue
        box=component["bbox"]
        exported=[round(v,3) for v in (box[0],height-box[3],box[2],height-box[1])]
        if any(old["type"]==component["type"] and old["bbox"]==exported for old in result["components"].values()):
            continue
        result["components"][key]={"Name":component.get("name") or component["observable_designator"],
            "type":component["type"],"value":component.get("value"),"bbox":exported}
        result["pins"][key]={}
        events.append({"key":key,"bbox":exported,"confidence":component.get("confidence")})
    return result,events


def append_new_component_predictions(baseline: dict, alternative: dict) -> tuple[dict,list[dict]]:
    """Protect existing fields and append only unused identities at new bodies.

    Image-only proposal union before pin/net inference. Competing readings of
    an existing body cannot replace the baseline, even if they use a new key.
    """
    for scene in (baseline,alternative):
        if scene.get("nets") or any(scene.get("pins",{}).values()):
            raise ValueError("Guarded proposal union must precede pins/net inference")
    result=copy.deepcopy(baseline);events=[]
    for key,c in alternative["components"].items():
        if (key in result["components"] or key.startswith("UNRESOLVED_")
                or c["type"] in {"v","gnd","VCC","GND"} or c["type"].startswith("net_")):
            continue
        if any(bbox_iou(c["bbox"],old["bbox"])>=.5 for old in result["components"].values()):continue
        result["components"][key]=copy.deepcopy(c);result["pins"][key]={}
        events.append({"key":key,"bbox":c["bbox"],"type":c["type"]})
    return result,events
