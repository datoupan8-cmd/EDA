"""Unified proposals and spatial duplicate suppression for Component V4."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .official_types import to_official_type


@dataclass
class ComponentProposal:
    bbox: tuple[float, float, float, float]
    type: str
    confidence: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def bbox_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_iou(left, right):
    overlap = (
        max(left[0], right[0]), max(left[1], right[1]),
        min(left[2], right[2]), min(left[3], right[3]),
    )
    intersection = bbox_area(overlap)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def center_distance_ratio(left, right):
    lx, ly = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
    rx, ry = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
    scale = max(4.0, min(left[2] - left[0], left[3] - left[1], right[2] - right[0], right[3] - right[1]))
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** .5 / scale


def fuse_proposals(
    proposals: Iterable[ComponentProposal],
    *,
    same_type_iou: float = .42,
    conflict_iou: float = .72,
) -> tuple[list[ComponentProposal], list[dict[str, Any]]]:
    """Merge YOLO and geometry proposals by position, never by generated key."""
    normalized = [
        ComponentProposal(tuple(map(float, p.bbox)), to_official_type(p.type), float(p.confidence), p.source, dict(p.metadata))
        for p in proposals
        if p.bbox[2] > p.bbox[0] and p.bbox[3] > p.bbox[1]
    ]
    kept: list[ComponentProposal] = []
    rejected: list[dict[str, Any]] = []
    source_bonus = {"yolo": .04, "large_box_geometry": .03, "ground_geometry": .03, "capacitor_geometry": 0.0}
    for candidate in sorted(normalized, key=lambda p: p.confidence + source_bonus.get(p.source, 0), reverse=True):
        duplicate = None
        for old in kept:
            iou = bbox_iou(candidate.bbox, old.bbox)
            same_type = candidate.type == old.type
            if (same_type and (iou >= same_type_iou or (iou >= .20 and center_distance_ratio(candidate.bbox, old.bbox) <= .35))) or iou >= conflict_iou:
                duplicate = (old, iou)
                break
        if duplicate:
            rejected.append({"proposal": candidate.as_dict(), "reason": "spatial_duplicate", "kept": duplicate[0].as_dict(), "iou": duplicate[1]})
        else:
            kept.append(candidate)
    return kept, rejected

