"""Component-only selective PP-OCRv5 rereading and printed label grouping.

No case names, annotations, inferred model names or generated designators enter
this module. The detector proposals are frozen across the two OCR passes.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics

import cv2

from .alternative_ocr import PPOcrV5Backend, ROOT
from .contextual_values import accepted_designators, contextual_value_roles, refine_contextual_values
from .latest.text_assignment import (
    BOX_FAMILY, NO_DESIGNATOR_TYPES, assign_values, bbox_gap, build_components,
    designator_weight, recover_near_box_designators,
)
from .latest.text_roles import classify_tokens, value_compatible
from .proposal_fusion import bbox_iou
from .text_enrichment import CAPACITOR_SHORTHAND, _value_weight, enrich_fields
from ..assignment import maximum_weight_assignment
from ..core.interfaces import ComponentStageOutput
from ..schema import Text
from ..text.ocr_backends import deduplicate_texts


@dataclass(frozen=True)
class LocalOCRConfig:
    # Bounded work per page; these are algorithm settings, never case overrides.
    max_regions: int = 24
    max_recrops: int = 12
    max_crop_side: int = 1280
    min_confidence: float = .70
    version: str = "component_local_v1"


def select_regions(image_shape, original, config=LocalOCRConfig()):
    height, width = image_shape[:2]
    heights = [t.bbox[3] - t.bbox[1] for t in original.texts
               if 4 <= t.bbox[3] - t.bbox[1] <= 64]
    font = float(statistics.median(heights)) if heights else 12.
    confidence = {event["proposal_index"]: event["score"] for event in
                  original.diagnostics.get("designator_assignment", {}).get("accepted", [])}
    candidates = []
    for index, component in enumerate(original.components):
        if component.type in NO_DESIGNATOR_TYPES:
            continue
        reasons = []
        if component.observable_designator is None:
            reasons.append("missing_designator")
        if component.type in {"r", "c", "l"} and component.value is None:
            reasons.append("missing_value")
        if component.type in BOX_FAMILY and component.name in (None, component.key):
            reasons.append("missing_model")
        if not reasons:
            continue
        x1, y1, x2, y2 = component.bbox
        margin = min(160., max(4. * font, .65 * min(x2 - x1, y2 - y1)))
        region = (max(0, math.floor(x1 - margin)), max(0, math.floor(y1 - margin)),
                  min(width, math.ceil(x2 + margin)), min(height, math.ceil(y2 + margin)))
        priority = (3 * ("missing_designator" in reasons)
                    + ("missing_value" in reasons) + .25 * ("missing_model" in reasons))
        candidates.append((-priority, confidence.get(index, 0), index, region, reasons))
    selected = []
    for _, _, index, region, reasons in sorted(candidates):
        if any(bbox_iou(region, row["bbox"]) >= .65 for row in selected):
            continue
        selected.append({"component_index": index, "bbox": region, "reasons": reasons,
                         "font_height": font})
        if len(selected) >= config.max_regions:
            break
    return selected


def merge_local_readings(existing, additions):
    """Keep literal hypotheses; do not let a weak reread replace trusted text."""
    result = list(existing)
    rejected = []
    for token in sorted(additions, key=lambda t: t.score, reverse=True):
        conflicts = [old for old in result if bbox_iou(token.bbox, old.bbox) >= .55]
        if conflicts and all(token.text.strip() != old.text.strip() for old in conflicts):
            best = max(conflicts, key=lambda t: t.score)
            if token.score < max(.85, best.score + .08):
                rejected.append({"text": token.text, "reason": "stronger_existing_reading"})
                continue
            result = [old for old in result if old not in conflicts]
        result.append(token)
    return deduplicate_texts(result, .3), rejected


class SelectivePPOCR:
    def __init__(self, cache, config=LocalOCRConfig(), backend=None):
        self.cache = Path(cache)
        self.config = config
        self.backend = backend or PPOcrV5Backend(self.cache / "backend")

    def recognize(self, image, original):
        regions = select_regions(image.shape, original, self.config)
        fingerprint = json.dumps({"config": asdict(self.config), "regions": regions,
                                  "models": self.backend.signature.decode(),
                                  "texts": [asdict(t) for t in original.texts]}, sort_keys=True).encode()
        key = hashlib.sha256(image.tobytes() + fingerprint).hexdigest()[:24]
        path = self.cache / f"{key}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [Text(**row) for row in payload["texts"]], {**payload["diagnostics"], "cache_hit": True}
        additions = []
        recrops = []
        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            patch = image[y1:y2, x1:x2]
            scale = min(3., max(1., 28. / region["font_height"]),
                        self.config.max_crop_side / max(patch.shape[:2]))
            patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for token in self.backend._read(patch, left=x1, top=y1, scale=scale):
                if token.score >= self.config.min_confidence:
                    token.source_id = "ppocrv5_component_local"
                    additions.append(token)
        # Recognition-only rereads use a padded text crop, not a whole symbol.
        for token in sorted(original.texts, key=lambda t: t.score):
            if len(recrops) >= self.config.max_recrops:
                break
            if token.score >= .90 or not any(bbox_iou(token.bbox, row["bbox"]) > 0 for row in regions):
                continue
            th = max(4., token.bbox[3] - token.bbox[1])
            x1 = max(0, math.floor(token.bbox[0] - .20 * th))
            y1 = max(0, math.floor(token.bbox[1] - .12 * th))
            x2 = min(image.shape[1], math.ceil(token.bbox[2] + .20 * th))
            y2 = min(image.shape[0], math.ceil(token.bbox[3] + .12 * th))
            patch = image[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            patch = cv2.resize(patch, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            output = self.backend._ensure_engine()(patch, use_det=False, use_cls=False, use_rec=True)
            if output.txts is not None:
                for value, score in zip(output.txts, output.scores):
                    if str(value).strip() and float(score) >= self.config.min_confidence:
                        additions.append(Text(str(value), token.bbox, float(score), "ppocrv5_component_recrop"))
            recrops.append({"bbox": token.bbox, "before": token.text})
        texts, rejected = merge_local_readings(original.texts, additions)
        diagnostics = {"version": self.config.version, "cache_hit": False, "regions": regions,
                       "recrops": recrops, "added_readings": len(additions), "before": len(original.texts),
                       "after": len(texts), "rejected": rejected, "GT_used_in_inference": False}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"texts": [asdict(t) for t in texts], "diagnostics": diagnostics},
                                   ensure_ascii=False), encoding="utf-8")
        return texts, diagnostics


def line_group_strength(anchor, other):
    """Pair co-aligned lines in the same small printed annotation block."""
    a, b = anchor.bbox, other.bbox
    ah, bh = max(3., a[3] - a[1]), max(3., b[3] - b[1])
    if min(ah, bh) / max(ah, bh) < .55 or bbox_iou(a, b) > .15:
        return 0.
    font = (ah + bh) / 2
    ax, ay = anchor.center
    bx, by = other.center
    vertical_gap = max(a[1] - b[3], b[1] - a[3], 0.)
    horizontal_gap = max(a[0] - b[2], b[0] - a[2], 0.)
    column_offset = min(abs(a[0] - b[0]), abs(ax - bx)) / font
    if column_offset <= .65 and vertical_gap <= 1.5 * font and abs(ay - by) >= .6 * font:
        return math.exp(-column_offset - vertical_gap / (1.5 * font))
    row_offset = abs(ay - by) / font
    if row_offset <= .30 and horizontal_gap <= 1.5 * font and abs(ax - bx) >= font:
        return .85 * math.exp(-row_offset - horizontal_gap / (1.5 * font))
    return 0.


def printed_value_compatible(kind, text):
    return value_compatible(kind, text, True) or (kind == "c" and bool(CAPACITOR_SHORTHAND.fullmatch(text.strip())))


def grouped_roles(roles):
    """OCR alternatives at one physical location must occupy only one slot."""
    groups = []
    for role in sorted(roles, key=lambda r: r.confidence, reverse=True):
        hit = next((group for group in groups if any(bbox_iou(role.token.bbox, r.token.bbox) >= .55 for r in group)), None)
        if hit is None:
            groups.append([role])
        else:
            hit.append(role)
    return groups


def assign_joint_designators(proposals, roles):
    candidates = [role for role in roles if role.role == "DESIGNATOR"]
    groups = grouped_roles(candidates)
    supports = [r for r in roles if r.role in {"VALUE", "MODEL_TEXT"} and r.confidence >= .8]
    weights, choices = [], []
    for group in groups:
        scores, selections = [], []
        for proposal in proposals:
            best_score, best_role = 0., group[0]
            for role in group:
                weight = designator_weight(proposal, role)
                if weight <= 0:
                    continue
                support = 0.
                for other in supports:
                    suitable = (printed_value_compatible(proposal.type, other.token.text) if other.role == "VALUE"
                                else proposal.type in BOX_FAMILY)
                    if not suitable:
                        continue
                    strength = line_group_strength(role.token, other.token)
                    # Group evidence must agree with the proposed component's location.
                    near = math.exp(-bbox_gap(other.token.bbox, proposal.bbox) /
                                    max(12., 4 * (other.token.bbox[3] - other.token.bbox[1])))
                    support = max(support, strength * near * other.confidence)
                score = weight * (1 + .6 * support)
                if score > best_score:
                    best_score, best_role = score, role
            scores.append(best_score)
            selections.append(best_role)
        weights.append(scores)
        choices.append(selections)
    assignments, accepted = {}, []
    for row, column in maximum_weight_assignment(weights):
        if weights[row][column] >= .055:
            role = choices[row][column]
            assignments[column] = role
            accepted.append({"proposal_index": column, "token": role.token.text,
                             "score": weights[row][column], "method": "printed_group_assignment"})
    return assignments, {"accepted": accepted, "candidate_count": len(candidates),
                         "physical_regions": len(groups)}


def assign_joint_values(components, roles, provenance, designators):
    # Preserve explicit "R1 10k" observations as hard bindings.
    diagnostics = assign_values(components, roles, provenance, expanded_rules=True, designators=designators)
    bound = {row["component"] for row in diagnostics["accepted"] if row.get("method") == "designator_tail"}
    indices = [i for i, c in enumerate(components) if c.type in {"r", "c", "l"} and c.key not in bound]
    groups = grouped_roles([r for r in roles if r.role == "VALUE"])
    weights, choices, grouped = [], [], []
    for group in groups:
        row, selected, tags = [], [], []
        anchor_scores = []
        for index in indices:
            anchor = designators.get(index)
            anchor_scores.append(max((line_group_strength(anchor.token, r.token) * r.confidence for r in group), default=0.)
                                 if anchor and anchor.confidence >= .8 else 0.)
        ranked = sorted(anchor_scores, reverse=True)
        unambiguous = bool(ranked and ranked[0] >= .35 and
                           (len(ranked) == 1 or ranked[0] >= 1.4 * ranked[1]))
        for column, index in enumerate(indices):
            component = components[index]
            options = [( _value_weight(component, r), r) for r in group
                       if printed_value_compatible(component.type, r.token.text)]
            weight, role = max(options, key=lambda pair: pair[0]) if options else (0., group[0])
            use_group = unambiguous and anchor_scores[column] == ranked[0] and weight > 0
            if use_group:
                weight += anchor_scores[column]
            row.append(weight)
            selected.append(role)
            tags.append(use_group)
        weights.append(row)
        choices.append(selected)
        grouped.append(tags)
    accepted = [event for event in diagnostics["accepted"] if event["component"] not in {components[i].key for i in indices}]
    for index in indices:
        components[index].value = None
        provenance[index]["value_source"] = "unknown"
    for row, column in maximum_weight_assignment(weights):
        if weights[row][column] < .055:
            continue
        index, role = indices[column], choices[row][column]
        components[index].value = role.token.text.strip()
        provenance[index]["value_source"] = "ocr_printed_group" if grouped[row][column] else "ocr_global_assignment"
        accepted.append({"component": components[index].key, "token": components[index].value,
                         "score": weights[row][column], "method": provenance[index]["value_source"]})
    return {"enabled": True, "accepted": accepted, "candidate_count": len(groups), "bound_count": len(bound)}


def rebuild_fields(original, texts, *, joint=False):
    """Reclassify printed annotations while keeping the exact fused boxes/types."""
    from .latest.text_assignment import assign_designators
    proposals = original.debug["fused_proposals"]
    roles = classify_tokens(texts, [p.bbox for p in proposals if p.type in {"box", "block"}], expanded_rules=True)
    roles, recovered = recover_near_box_designators(proposals, roles)
    if joint:
        designators, assignment = assign_joint_designators(proposals, roles)
    else:
        designators, assignment = assign_designators(proposals, roles, global_assignment=True)
    components, provenance = build_components(proposals, designators)
    diagnostics = copy.deepcopy(original.diagnostics)
    diagnostics.update({"designator_assignment": assignment, "component_provenance": provenance,
                        "near_box_designator_recovery": recovered,
                        "text_roles": [{"index": r.index, "text": r.token.text, "bbox": r.token.bbox,
                                        "role": r.role, "confidence": r.confidence, "reason": r.reason} for r in roles]})
    diagnostics["value_assignment"] = {**assign_values(components, roles, provenance, True, designators), "enabled": True}
    result = refine_contextual_values(ComponentStageOutput(components, texts, roles, diagnostics, original.debug),
                                      conflicts=True, pin_boxes=True)
    if joint:
        value_roles, rejected = contextual_value_roles(roles, result.components, conflicts=True, pin_boxes=True)
        components = copy.deepcopy(result.components)
        diagnostics = copy.deepcopy(result.diagnostics)
        diagnostics["value_assignment"] = assign_joint_values(components, value_roles, diagnostics["component_provenance"], designators)
        components, diagnostics = enrich_fields(components, value_roles, diagnostics)
        diagnostics["printed_label_groups"] = {"enabled": True, "rejected_numeric_alternatives": rejected}
        result = ComponentStageOutput(components, texts, roles, diagnostics, original.debug)
    assert [(c.type, c.bbox) for c in result.components] == [(c.type, c.bbox) for c in original.components]
    return result


class ComponentStageLocalOCR:
    def __init__(self, *, joint=False):
        from .contextual_values import ComponentStageContextualValues
        self.baseline = ComponentStageContextualValues()
        self.joint = joint
        self.reader = None

    def refine(self, image, original, context):
        if context.config.component_stage != "BEST" or context.config.text_rules != "v4_1":
            raise ValueError("local OCR variants require BEST / v4_1")
        if self.reader is None:
            resources = context.resources.get("component", {})
            cache = Path(resources.get("cache_dir", ROOT / "runs/component_alternative_cache"))
            self.reader = SelectivePPOCR(cache / "selective_local_v1")
        texts, local_diagnostics = self.reader.recognize(image, original)
        result = rebuild_fields(original, texts, joint=self.joint)
        result.diagnostics["component_local_ocr"] = local_diagnostics
        result.diagnostics["frontend"] = "component_local_joint_v1" if self.joint else "component_local_ocr_v1"
        # Component-specific rereads are private to this stage. Other consumers
        # keep the original text-stage contract and token indexing.
        result.texts, result.roles = original.texts, original.roles
        return result

    def run(self, image, text, context):
        return self.refine(image, self.baseline.run(image, text, context), context)


class ComponentStageLocalJoint(ComponentStageLocalOCR):
    def __init__(self):
        super().__init__(joint=True)


class ComponentStagePrintedGroups:
    """Selected low-cost variant: printed group association without extra OCR."""
    def __init__(self):
        from .contextual_values import ComponentStageContextualValues
        self.baseline = ComponentStageContextualValues()

    def refine(self, image, original, context):
        del image
        if context.config.component_stage != "BEST" or context.config.text_rules != "v4_1":
            raise ValueError("printed groups require BEST / v4_1")
        result = rebuild_fields(original, original.texts, joint=True)
        result.diagnostics["frontend"] = "component_printed_groups_v1"
        result.diagnostics["component_text_refinement"] = {
            "extra_OCR": False, "symbol_geometry_and_types_changed": False,
            "GT_used_in_inference": False}
        result.texts, result.roles = original.texts, original.roles
        return result

    def run(self, image, text, context):
        return self.refine(image, self.baseline.run(image, text, context), context)


class ComponentStageValueGroups(ComponentStagePrintedGroups):
    """Conservative ablation: preserve identities/names; group R/C/L values."""
    def refine(self, image, original, context):
        del image
        if context.config.component_stage != "BEST" or context.config.text_rules != "v4_1":
            raise ValueError("printed value groups require BEST / v4_1")
        components = copy.deepcopy(original.components)
        diagnostics = copy.deepcopy(original.diagnostics)
        roles, rejected = contextual_value_roles(original.roles, components, conflicts=True, pin_boxes=True)
        diagnostics["value_assignment"] = assign_joint_values(
            components, roles, diagnostics["component_provenance"], accepted_designators(original))
        components, diagnostics = enrich_fields(components, roles, diagnostics, names=False)
        for before, after in zip(original.components, components):
            # Only the target value fields are owned by this experiment.
            if before.type not in {"r", "c", "l"}:
                after.value = before.value
            assert (after.key, after.name, after.type, after.bbox) == (before.key, before.name, before.type, before.bbox)
        diagnostics["frontend"] = "component_printed_value_groups_v1"
        diagnostics["component_text_refinement"] = {
            "extra_OCR": False, "identity_name_geometry_types_changed": False,
            "rejected_numeric_alternatives": rejected, "GT_used_in_inference": False}
        return ComponentStageOutput(components, original.texts, original.roles, diagnostics, original.debug)
