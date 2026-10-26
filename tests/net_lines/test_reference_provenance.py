import unittest
from dataclasses import replace

import numpy as np

from pcb.net_lines.legacy_v2 import LegacyLines
from pcb.network_shared.contracts import NetGroup, NetworkPlan
from pcb.network_shared.graph import build_graph
from pcb.wire_stage.v1 import (
    _unique_ordered_vertex_indices,
    simplify,
    simplify_with_indices,
)


class ReferenceSegmentProvenanceTests(unittest.TestCase):
    def test_simplifier_preserves_indices_and_half_open_source_edge_ranges(self):
        path = ((4, 1), (4, 3), (4, 6), (7, 6), (10, 6))

        indexed = simplify_with_indices(path)

        self.assertEqual(list(indexed.segments), simplify(path))
        self.assertEqual(indexed.source_vertex_indices, (0, 2, 4))
        self.assertEqual(indexed.segment_edge_ranges, ((0, 2), (2, 4)))
        for (start, end), (first, stop) in zip(
            indexed.segments, indexed.segment_edge_ranges
        ):
            self.assertEqual(start, (float(path[first][1]), float(path[first][0])))
            self.assertEqual(end, (float(path[stop][1]), float(path[stop][0])))

        reverse = simplify_with_indices(((4, 6), (4, 3), (4, 1)))
        self.assertEqual(reverse.source_vertex_indices, (0, 2))
        self.assertEqual(reverse.segment_edge_ranges, ((0, 2),))
        self.assertEqual(reverse.segments[0], ((6.0, 4.0), (1.0, 4.0)))

        empty = simplify_with_indices(())
        self.assertEqual((empty.vertices, empty.source_vertex_indices,
                          empty.segments, empty.segment_edge_ranges), ((), (), (), ()))
        single = simplify_with_indices(((5, 9),))
        self.assertEqual(single.source_vertex_indices, (0,))
        self.assertEqual(single.segments, ())
        self.assertEqual(single.segment_edge_ranges, ())

        repeated = simplify_with_indices(((2, 2), (2, 2)))
        self.assertIsNone(repeated.source_vertex_indices)
        self.assertEqual(repeated.segments, ())
        self.assertEqual(repeated.segment_edge_ranges, ())

        # If repeated coordinates admit more than one ordered source occurrence,
        # the helper refuses to guess an index.
        self.assertIsNone(
            _unique_ordered_vertex_indices(
                ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)),
                ((0.0, 0.0), (0.0, 1.0)),
            )
        )

    def test_build_graph_mapping_aligns_with_references_and_legacy_sidecar(self):
        paths = (
            ((4, 1), (4, 3), (4, 6), (7, 6), (10, 6)),
            ((1, 2), (4, 2), (7, 2)),
        )
        mask = np.zeros((16, 16), dtype=np.uint8)
        skeleton = np.zeros_like(mask)
        graph = build_graph(mask, traced=(paths, skeleton, {}))

        self.assertEqual(len(graph.reference_segment_edge_ranges), len(paths))
        for path_id, path in enumerate(paths):
            indexed = simplify_with_indices(path)
            self.assertEqual(graph.reference_segments[path_id], indexed.segments)
            self.assertEqual(
                graph.reference_segment_edge_ranges[path_id],
                indexed.segment_edge_ranges,
            )
            self.assertEqual(
                len(graph.reference_segment_edge_ranges[path_id]),
                len(graph.reference_segments[path_id]),
            )

        plan = NetworkPlan((NetGroup(31, (), (1, 0)),))
        geometry = LegacyLines().run(graph, plan)
        traced_geometry, provenance = LegacyLines().run_with_provenance(graph, plan)
        self.assertEqual(geometry, traced_geometry)
        self.assertEqual(len(geometry.groups[0].segments), len(provenance.groups[0].segments))
        for item in provenance.groups[0].segments:
            source = item.sources[0]
            expected_range = graph.reference_segment_edge_ranges[source.path_id][
                source.reference_segment_index
            ]
            self.assertEqual(source.source_type, "reference_segment")
            self.assertEqual(source.resolution, "complete")
            self.assertEqual(source.source_edge_indices, tuple(range(*expected_range)))
            self.assertEqual(item.endpoints, geometry.groups[0].segments[item.segment_index])

    def test_legacy_trace_marks_invalid_source_range_unknown(self):
        path = ((4, 1), (4, 3), (4, 6), (7, 6), (10, 6))
        mask = np.zeros((16, 16), dtype=np.uint8)
        graph = build_graph(mask, traced=((path,), np.zeros_like(mask), {}))
        path_ranges = list(graph.reference_segment_edge_ranges[0])
        path_ranges[0] = (-1, 2)
        invalid_graph = replace(
            graph, reference_segment_edge_ranges=(tuple(path_ranges),)
        )

        _, provenance = LegacyLines().run_with_provenance(
            invalid_graph, NetworkPlan((NetGroup(32, (), (0,)),))
        )

        source = provenance.groups[0].segments[0].sources[0]
        self.assertEqual(source.resolution, "unknown")
        self.assertIsNone(source.source_edge_indices)
        self.assertIn("outside its path", source.reason)


if __name__ == "__main__":
    unittest.main()
