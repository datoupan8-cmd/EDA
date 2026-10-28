"""Path-derived candidate line geometry, separate from the legacy reference output."""
from ..network_shared.contracts import GeometryResult
from .provenance import GeometryProvenance, SegmentSource, _GroupGeometryBuilder


class PathLines:
    def run(self, graph, plan):
        geometry, _ = self._run(graph, plan, include_provenance=False)
        return geometry

    def run_with_provenance(self, graph, plan):
        """Return geometry and exact path-local source-edge references."""
        return self._run(graph, plan, include_provenance=True)

    def _run(self, graph, plan, include_provenance):
        groups = []
        group_traces = []
        for group in plan.groups:
            builder = _GroupGeometryBuilder(group.group_id, include_provenance)
            for path_id in group.path_ids:
                path = graph.paths[path_id]
                points = tuple((float(column), float(row)) for row, column in path)
                for edge_index, (start, end) in enumerate(zip(points, points[1:])):
                    source = (
                        SegmentSource(
                            source_type="path_edges",
                            path_id=path_id,
                            source_edge_indices=(edge_index,),
                        )
                        if include_provenance
                        else None
                    )
                    builder.append((start, end), source)
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
