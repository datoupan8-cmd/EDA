"""Path geometry candidate that merges only exact, forward-collinear runs."""
from ..network_shared.contracts import GeometryResult
from .provenance import GeometryProvenance, SegmentSource, _GroupGeometryBuilder


def _merge_path_with_edges(points):
    """Return merged segments paired with every path-local source edge index."""
    if len(points) < 2:
        return []

    segments = []
    start_index = 0
    previous_index = 1
    current_index = 1
    previous_vector = (
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
    )

    for current_index in range(2, len(points)):
        previous = points[previous_index]
        current = points[current_index]
        vector = (current[0] - previous[0], current[1] - previous[1])
        cross = previous_vector[0] * vector[1] - previous_vector[1] * vector[0]
        dot = previous_vector[0] * vector[0] + previous_vector[1] * vector[1]
        if cross == 0 and dot > 0:
            previous_index = current_index
            previous_vector = vector
            continue
        segments.append(
            (points[start_index], points[previous_index], tuple(range(start_index, previous_index)))
        )
        start_index = previous_index
        previous_index = current_index
        previous_vector = vector

    segments.append(
        (points[start_index], points[previous_index], tuple(range(start_index, previous_index)))
    )
    return segments


def _merge_path(points):
    """Return geometry only, preserving the original helper's behavior."""
    if len(points) < 2:
        return []

    segments = []
    start = previous = points[0]
    current = points[1]
    previous_vector = (current[0] - previous[0], current[1] - previous[1])
    previous = current

    for current in points[2:]:
        vector = (current[0] - previous[0], current[1] - previous[1])
        cross = previous_vector[0] * vector[1] - previous_vector[1] * vector[0]
        dot = previous_vector[0] * vector[0] + previous_vector[1] * vector[1]
        if cross == 0 and dot > 0:
            previous = current
            previous_vector = vector
            continue
        segments.append((start, previous))
        start = previous
        previous = current
        previous_vector = vector

    segments.append((start, previous))
    return segments


class CollinearPathLines:
    """Independent candidate; never merges across paths or groups."""

    def run(self, graph, plan):
        geometry, _ = self._run(graph, plan, include_provenance=False)
        return geometry

    def run_with_provenance(self, graph, plan):
        """Return merged geometry and its exact, ordered source-edge indices."""
        return self._run(graph, plan, include_provenance=True)

    def _run(self, graph, plan, include_provenance):
        groups = []
        group_traces = []
        for group in plan.groups:
            builder = _GroupGeometryBuilder(group.group_id, include_provenance)
            for path_id in group.path_ids:
                path = graph.paths[path_id]
                points = tuple((float(column), float(row)) for row, column in path)
                if include_provenance:
                    path_segments = (
                        (start, end, source_edge_indices)
                        for start, end, source_edge_indices in _merge_path_with_edges(points)
                    )
                else:
                    path_segments = (
                        (start, end, None) for start, end in _merge_path(points)
                    )
                for start, end, source_edge_indices in path_segments:
                    source = None
                    if include_provenance:
                        source = SegmentSource(
                            source_type="path_edges",
                            path_id=path_id,
                            source_edge_indices=source_edge_indices,
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
