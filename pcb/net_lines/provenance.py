"""Optional, run-local provenance for NetLine output segments.

This sidecar deliberately does not alter the shared GeometryResult contract.
"""
from dataclasses import dataclass
from typing import Literal

from ..network_shared.contracts import Point, Segment

SourceType = Literal["path_edges", "reference_segment", "contact_point"]
Resolution = Literal["complete", "unknown"]


@dataclass(frozen=True)
class SegmentSource:
    """One known contributor to an output segment.

    ``source_edge_indices`` is path-local and uses the half-open edge range
    represented by its explicit indices. ``None`` means the upstream contract
    does not provide that mapping; it is never inferred from coordinates.
    """

    source_type: SourceType
    path_id: int | None = None
    source_edge_indices: tuple[int, ...] | None = None
    reference_segment_index: int | None = None
    contact_point_index: int | None = None
    resolution: Resolution = "complete"
    reason: str | None = None


@dataclass(frozen=True)
class SegmentProvenance:
    """Run-local stable reference to one output segment within its group."""

    group_id: int
    segment_index: int
    endpoints: Segment
    sources: tuple[SegmentSource, ...]


@dataclass(frozen=True)
class GroupProvenance:
    group_id: int
    segments: tuple[SegmentProvenance, ...]


@dataclass(frozen=True)
class GeometryProvenance:
    """Optional sidecar aligned exactly with GeometryResult group/segment order."""

    groups: tuple[GroupProvenance, ...]


class _GroupGeometryBuilder:
    """Keep geometry order/dedup semantics and optional source refs aligned."""

    def __init__(self, group_id: int, track_provenance: bool):
        self.group_id = group_id
        self.track_provenance = track_provenance
        self.segments: list[Segment] = []
        self._sources: list[list[SegmentSource]] = []

    def append(self, segment: Segment, source: SegmentSource | None = None) -> None:
        self.segments.append(segment)
        if self.track_provenance:
            self._sources.append([] if source is None else [source])

    def add_contact_once(
        self, segment: Segment, source: SegmentSource | None = None
    ) -> None:
        """Mirror the existing ``if edge not in segments`` contact behavior."""
        try:
            index = self.segments.index(segment)
        except ValueError:
            self.append(segment, source)
        else:
            if self.track_provenance and source is not None:
                self._sources[index].append(source)

    def finish(self, include_provenance: bool):
        from ..network_shared.contracts import GroupLines

        geometry = GroupLines(self.group_id, tuple(self.segments))
        if not include_provenance or not self.track_provenance:
            return geometry, None
        traces = tuple(
            SegmentProvenance(
                self.group_id,
                index,
                segment,
                tuple(self._sources[index]),
            )
            for index, segment in enumerate(self.segments)
        )
        return geometry, GroupProvenance(self.group_id, traces)


def _validated_reference_edge_range(graph, path_id, reference_index, segment):
    """Return a trusted raw path edge range, or an explicit unknown reason."""
    mappings = getattr(graph, "reference_segment_edge_ranges", None)
    if mappings is None:
        return None, "WireGraph has no reference-segment source mapping"
    if not isinstance(mappings, tuple) or len(mappings) != len(graph.paths):
        return None, "WireGraph reference source mapping is not path-aligned"
    if type(path_id) is not int or not 0 <= path_id < len(graph.paths):
        return None, "reference path ID is outside WireGraph.paths"
    if type(reference_index) is not int:
        return None, "reference segment index is invalid"
    if not isinstance(graph.reference_segments, tuple):
        return None, "WireGraph reference segments are not tuple-aligned"
    if path_id >= len(graph.reference_segments):
        return None, "reference path ID is outside WireGraph.reference_segments"
    if not isinstance(mappings[path_id], tuple):
        return None, "reference path source mapping is malformed"
    if len(mappings[path_id]) != len(graph.reference_segments[path_id]):
        return None, "reference source mapping is not segment-aligned"
    if not 0 <= reference_index < len(mappings[path_id]):
        return None, "reference segment index is outside its source mapping"

    interval = mappings[path_id][reference_index]
    if (
        not isinstance(interval, tuple)
        or len(interval) != 2
        or any(type(index) is not int for index in interval)
    ):
        return None, "reference segment source interval is unknown or malformed"
    start, end = interval
    path = graph.paths[path_id]
    if not isinstance(path, tuple):
        return None, "reference source path is malformed"
    if not 0 <= start < end < len(path):
        return None, "reference segment source interval is outside its path"

    try:
        expected = (
            (float(path[start][1]), float(path[start][0])),
            (float(path[end][1]), float(path[end][0])),
        )
    except (IndexError, TypeError, ValueError, OverflowError):
        return None, "reference source path vertices are malformed"
    if segment != expected:
        return None, "reference segment endpoints do not match its source interval"
    return interval, None
