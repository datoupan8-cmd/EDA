"""Unchanged legacy segment materialisation, separately replaceable."""
from ..network_shared.contracts import GeometryResult
from .provenance import (
    GeometryProvenance,
    SegmentSource,
    _GroupGeometryBuilder,
    _validated_reference_edge_range,
)


class LegacyLines:
    def run(self, graph, plan):
        geometry, _ = self._run(graph, plan, include_provenance=False)
        return geometry

    def run_with_provenance(self, graph, plan):
        """Return legacy geometry plus honest reference/contact provenance.

        A reference segment's raw path edges are recorded only when the
        WireGraph's aligned half-open source range passes validation against
        that path's coordinates and the segment endpoints. Missing or
        inconsistent mappings remain explicitly unknown; they are never
        inferred from coordinate overlap.
        """
        return self._run(graph, plan, include_provenance=True)

    def _run(self, graph, plan, include_provenance):
        groups = []
        group_traces = []
        for group in plan.groups:
            builder = _GroupGeometryBuilder(group.group_id, include_provenance)
            for path_id in group.path_ids:
                for reference_index, segment in enumerate(
                    graph.reference_segments[path_id]
                ):
                    source = None
                    if include_provenance:
                        edge_range, unknown_reason = _validated_reference_edge_range(
                            graph, path_id, reference_index, segment
                        )
                        source = SegmentSource(
                            source_type="reference_segment",
                            path_id=path_id,
                            reference_segment_index=reference_index,
                            source_edge_indices=(
                                tuple(range(*edge_range))
                                if edge_range is not None
                                else None
                            ),
                            resolution="complete" if edge_range is not None else "unknown",
                            reason=unknown_reason,
                        )
                    builder.append(segment, source)
            for contact_index, point in enumerate(group.contact_points):
                edge = (point, point)
                source = (
                    SegmentSource(
                        source_type="contact_point",
                        contact_point_index=contact_index,
                    )
                    if include_provenance
                    else None
                )
                builder.add_contact_once(edge, source)
            geometry_group, trace_group = builder.finish(include_provenance)
            groups.append(geometry_group)
            if include_provenance:
                group_traces.append(trace_group)

        geometry = GeometryResult(tuple(groups))
        if not include_provenance:
            return geometry, None
        return geometry, GeometryProvenance(tuple(group_traces))
