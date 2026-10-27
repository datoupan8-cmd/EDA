import unittest
from types import MappingProxyType

from pcb.net_lines.legacy_v2 import LegacyLines
from pcb.net_lines.path_collinear_v1 import CollinearPathLines
from pcb.net_lines.path_v1 import PathLines
from pcb.network_shared.contracts import (
    GeometryResult,
    NetGroup,
    NetworkPlan,
    WireGraph,
)


BACKENDS = (LegacyLines, PathLines, CollinearPathLines)


class NetLineProvenanceContract(unittest.TestCase):
    def make_graph(self, paths, reference_segments=None):
        paths = tuple(tuple(path) for path in paths)
        if reference_segments is None:
            reference_segments = tuple(() for _ in paths)
        return WireGraph(
            width=32,
            height=32,
            paths=paths,
            adjacency=MappingProxyType({}),
            reference_segments=tuple(reference_segments),
            mask=memoryview(b"mask").toreadonly(),
            thickness=memoryview(b"thickness").toreadonly(),
            skeleton=memoryview(b"skeleton").toreadonly(),
        )

    def assert_trace_aligned(self, geometry, trace):
        self.assertEqual(
            tuple(group.group_id for group in trace.groups),
            tuple(group.group_id for group in geometry.groups),
        )
        for output_group, trace_group in zip(geometry.groups, trace.groups):
            self.assertEqual(len(output_group.segments), len(trace_group.segments))
            for index, (segment, item) in enumerate(
                zip(output_group.segments, trace_group.segments)
            ):
                self.assertEqual(item.group_id, output_group.group_id)
                self.assertEqual(item.segment_index, index)
                self.assertEqual(item.endpoints, segment)
                self.assertTrue(item.sources)

    def test_run_contract_and_trace_opt_in_preserve_geometry_for_all_backends(self):
        graph = self.make_graph(
            (((1, 1), (1, 3), (1, 7), (4, 7)), ()),
            reference_segments=(
                (((1.0, 1.0), (7.0, 1.0)),),
                (),
            ),
        )
        plan = NetworkPlan(
            (
                NetGroup(4, ("U1.1",), (0,), ((12.0, 13.0),)),
                NetGroup(8, (), (1,), ((12.0, 13.0),)),
            )
        )
        for backend_type in BACKENDS:
            with self.subTest(backend=backend_type.__name__):
                backend = backend_type()
                geometry = backend.run(graph, plan)
                traced_geometry, trace = backend.run_with_provenance(graph, plan)
                self.assertIsInstance(geometry, GeometryResult)
                self.assertEqual(geometry, traced_geometry)
                self.assertIsInstance(trace.groups, tuple)
                self.assert_trace_aligned(geometry, trace)

    def test_path_v1_maps_each_output_to_exact_path_local_edge(self):
        graph = self.make_graph(((), ((2, 3), (5, 3), (5, 8), (8, 8))))
        plan = NetworkPlan((NetGroup(17, (), (1,)),))

        geometry, trace = PathLines().run_with_provenance(graph, plan)

        self.assertEqual(
            geometry.groups[0].segments,
            (((3.0, 2.0), (3.0, 5.0)),
             ((3.0, 5.0), (8.0, 5.0)),
             ((8.0, 5.0), (8.0, 8.0))),
        )
        self.assertEqual(
            tuple((s.path_id, s.source_edge_indices) for item in trace.groups[0].segments
                  for s in item.sources),
            ((1, (0,)), (1, (1,)), (1, (2,))),
        )
        self.assert_trace_aligned(geometry, trace)

    def test_collinear_trace_covers_each_source_edge_once_and_keeps_corners(self):
        paths = (
            ((2, 1), (2, 3), (2, 6), (5, 6), (8, 6)),
            ((10, 2), (10, 4)),
        )
        graph = self.make_graph(paths)
        plan = NetworkPlan((NetGroup(3, (), (0, 1)),))

        geometry, trace = CollinearPathLines().run_with_provenance(graph, plan)

        self.assertEqual(
            geometry.groups[0].segments,
            (((1.0, 2.0), (6.0, 2.0)),
             ((6.0, 2.0), (6.0, 8.0)),
             ((2.0, 10.0), (4.0, 10.0))),
        )
        sources = tuple(item.sources[0] for item in trace.groups[0].segments)
        self.assertEqual(
            tuple((source.path_id, source.source_edge_indices) for source in sources),
            ((0, (0, 1)), (0, (2, 3)), (1, (0,))),
        )
        mapped = [
            (source.path_id, edge_index)
            for item in trace.groups[0].segments
            for source in item.sources
            for edge_index in source.source_edge_indices or ()
        ]
        self.assertEqual(len(mapped), len(set(mapped)))
        self.assertEqual(set(mapped), {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)})
        self.assertIn((6.0, 2.0), {point for seg in geometry.groups[0].segments for point in seg})
        self.assert_trace_aligned(geometry, trace)

    def test_group_and_path_boundaries_are_retained_in_trace(self):
        graph = self.make_graph(
            (((2, 1), (2, 4)), ((2, 4), (2, 9)))
        )
        plan = NetworkPlan(
            (NetGroup(10, (), (0,)), NetGroup(20, (), (1,)))
        )

        geometry, trace = CollinearPathLines().run_with_provenance(graph, plan)

        self.assertEqual(tuple(g.group_id for g in geometry.groups), (10, 20))
        self.assertEqual(tuple(g.group_id for g in trace.groups), (10, 20))
        self.assertEqual(
            tuple(s.sources[0].path_id for g in trace.groups for s in g.segments),
            (0, 1),
        )
        self.assertEqual(
            tuple(s.sources[0].source_edge_indices for g in trace.groups for s in g.segments),
            ((0,), (0,)),
        )

    def test_duplicate_contacts_are_all_referenced_without_duplicate_geometry(self):
        point = (7.5, 9.5)
        graph = self.make_graph(((),))
        plan = NetworkPlan((NetGroup(7, (), (0,), (point, point, point)),))

        for backend_type in BACKENDS:
            with self.subTest(backend=backend_type.__name__):
                geometry, trace = backend_type().run_with_provenance(graph, plan)

                self.assertEqual(geometry.groups[0].segments, ((point, point),))
                sources = trace.groups[0].segments[0].sources
                self.assertEqual(tuple(source.source_type for source in sources),
                                 ("contact_point",) * 3)
                self.assertEqual(tuple(source.contact_point_index for source in sources),
                                 (0, 1, 2))
                self.assertTrue(all(source.path_id is None for source in sources))

    def test_contact_can_share_a_zero_length_path_output_without_fake_pin_link(self):
        point = (3.0, 2.0)
        graph = self.make_graph((((2, 3), (2, 3)),))
        plan = NetworkPlan((NetGroup(7, (), (0,), (point, point)),))

        geometry, trace = PathLines().run_with_provenance(graph, plan)

        self.assertEqual(geometry.groups[0].segments, ((point, point),))
        sources = trace.groups[0].segments[0].sources
        self.assertEqual(tuple(source.source_type for source in sources),
                         ("path_edges", "contact_point", "contact_point"))
        self.assertEqual(sources[0].source_edge_indices, (0,))
        self.assertEqual(tuple(s.contact_point_index for s in sources[1:]), (0, 1))
        self.assertFalse(any(hasattr(source, "pin_id") for source in sources))

    def test_legacy_reference_segments_are_explicitly_unknown_at_raw_edge_level(self):
        reference = (((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0), (7.0, 8.0)))
        graph = self.make_graph((((2, 1), (4, 3)),), (reference,))
        plan = NetworkPlan((NetGroup(9, (), (0,)),))

        geometry, trace = LegacyLines().run_with_provenance(graph, plan)

        self.assertEqual(geometry.groups[0].segments, reference)
        sources = tuple(item.sources[0] for item in trace.groups[0].segments)
        self.assertEqual(tuple(source.reference_segment_index for source in sources), (0, 1))
        for source in sources:
            self.assertEqual(source.source_type, "reference_segment")
            self.assertEqual(source.path_id, 0)
            self.assertIsNone(source.source_edge_indices)
            self.assertEqual(source.resolution, "unknown")
            self.assertIn("source mapping", source.reason)

    def test_empty_and_single_point_paths_keep_empty_groups_in_sidecar(self):
        graph = self.make_graph(((), ((5, 9),)))
        plan = NetworkPlan((NetGroup(5, (), (0, 1)), NetGroup(6, (), ())))

        for backend_type in (PathLines, CollinearPathLines):
            with self.subTest(backend=backend_type.__name__):
                geometry, trace = backend_type().run_with_provenance(graph, plan)

                self.assertEqual(tuple(group.segments for group in geometry.groups), ((), ()))
                self.assertEqual(tuple(group.segments for group in trace.groups), ((), ()))
                self.assertEqual(tuple(group.group_id for group in trace.groups), (5, 6))

    def test_collinear_trace_keeps_reversal_and_degenerate_edges_separate(self):
        paths = (
            ((3, 1), (3, 5), (3, 2)),
            ((2, 1), (2, 3), (2, 3), (2, 5)),
        )
        graph = self.make_graph(paths)
        plan = NetworkPlan((NetGroup(12, (), (0, 1)),))

        geometry, trace = CollinearPathLines().run_with_provenance(graph, plan)

        self.assertEqual(
            geometry.groups[0].segments,
            (((1.0, 3.0), (5.0, 3.0)),
             ((5.0, 3.0), (2.0, 3.0)),
             ((1.0, 2.0), (3.0, 2.0)),
             ((3.0, 2.0), (3.0, 2.0)),
             ((3.0, 2.0), (5.0, 2.0))),
        )
        self.assertEqual(
            tuple((source.path_id, source.source_edge_indices)
                  for item in trace.groups[0].segments for source in item.sources),
            ((0, (0,)), (0, (1,)), (1, (0,)), (1, (1,)), (1, (2,))),
        )

    def test_trace_is_deterministic_and_does_not_mutate_graph_or_plan(self):
        graph = self.make_graph(
            (((1, 1), (1, 2), (1, 4), (3, 4)), ()),
            reference_segments=((((1.0, 1.0), (4.0, 1.0)),), ()),
        )
        plan = NetworkPlan((NetGroup(2, (), (0, 1), ((9.0, 10.0),)),))
        snapshot = (
            graph.paths,
            graph.reference_segments,
            bytes(graph.mask),
            bytes(graph.thickness),
            bytes(graph.skeleton),
            plan,
        )
        backend = CollinearPathLines()

        first = backend.run_with_provenance(graph, plan)
        second = backend.run_with_provenance(graph, plan)

        self.assertEqual(first, second)
        self.assertEqual(
            (graph.paths, graph.reference_segments, bytes(graph.mask),
             bytes(graph.thickness), bytes(graph.skeleton), plan),
            snapshot,
        )


if __name__ == "__main__":
    unittest.main()
