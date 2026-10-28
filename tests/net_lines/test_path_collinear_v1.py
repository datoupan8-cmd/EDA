import unittest
from types import MappingProxyType

from pcb.net_lines.path_collinear_v1 import CollinearPathLines
from pcb.network_shared.contracts import (
    GeometryResult,
    GroupLines,
    NetGroup,
    NetworkPlan,
    WireGraph,
)


class CollinearPathLinesContract(unittest.TestCase):
    def make_graph(self, paths):
        return WireGraph(
            width=32,
            height=32,
            paths=tuple(tuple(path) for path in paths),
            adjacency=MappingProxyType({}),
            reference_segments=tuple(() for _ in paths),
            mask=memoryview(b"mask").toreadonly(),
            thickness=memoryview(b"thickness").toreadonly(),
            skeleton=memoryview(b"skeleton").toreadonly(),
        )

    def run_paths(self, paths, groups):
        return CollinearPathLines().run(self.make_graph(paths), NetworkPlan(tuple(groups)))

    def test_horizontal_collinear_segments_merge(self):
        result = self.run_paths((((4, 2), (4, 5), (4, 9)),), (NetGroup(7, (), (0,)),))
        self.assertEqual(result.groups[0].segments, (((2.0, 4.0), (9.0, 4.0)),))

    def test_vertical_collinear_segments_merge(self):
        result = self.run_paths((((2, 6), (5, 6), (11, 6)),), (NetGroup(7, (), (0,)),))
        self.assertEqual(result.groups[0].segments, (((6.0, 2.0), (6.0, 11.0)),))

    def test_right_angle_corner_is_preserved(self):
        result = self.run_paths((((1, 2), (3, 2), (3, 5)),), (NetGroup(7, (), (0,)),))
        self.assertEqual(
            result.groups[0].segments,
            (((2.0, 1.0), (2.0, 3.0)), ((2.0, 3.0), (5.0, 3.0))),
        )

    def test_multiple_collinear_edges_merge_as_one_run(self):
        path = ((8, 1), (8, 2), (8, 4), (8, 7), (8, 12))
        result = self.run_paths((path,), (NetGroup(7, (), (0,)),))
        self.assertEqual(result.groups[0].segments, (((1.0, 8.0), (12.0, 8.0)),))

    def test_slightly_offset_edge_is_not_merged(self):
        result = self.run_paths(
            (((2, 1), (2, 3), (3, 5)),), (NetGroup(7, (), (0,)),)
        )
        self.assertEqual(
            result.groups[0].segments,
            (((1.0, 2.0), (3.0, 2.0)), ((3.0, 2.0), (5.0, 3.0))),
        )

    def test_touching_distinct_paths_in_one_group_are_not_merged(self):
        paths = (((2, 1), (2, 4)), ((2, 4), (2, 9)))
        result = self.run_paths(paths, (NetGroup(7, (), (0, 1)),))
        self.assertEqual(
            result.groups[0].segments,
            (((1.0, 2.0), (4.0, 2.0)), ((4.0, 2.0), (9.0, 2.0))),
        )

    def test_group_boundary_is_never_merged(self):
        paths = (((2, 1), (2, 4)), ((2, 4), (2, 9)))
        result = self.run_paths(
            paths, (NetGroup(11, (), (0,)), NetGroup(22, (), (1,)))
        )
        self.assertEqual(tuple(group.group_id for group in result.groups), (11, 22))
        self.assertEqual(result.groups[0].segments, (((1.0, 2.0), (4.0, 2.0)),))
        self.assertEqual(result.groups[1].segments, (((4.0, 2.0), (9.0, 2.0)),))

    def test_duplicate_contact_points_emit_one_zero_length_segment(self):
        point = (4.5, 6.5)
        result = self.run_paths(((),), (NetGroup(7, (), (0,), (point, point)),))
        self.assertEqual(result.groups[0].segments, ((point, point),))

    def test_empty_and_single_point_paths_emit_no_path_segments(self):
        result = self.run_paths(
            ((), ((5, 9),)), (NetGroup(7, (), (0, 1)),)
        )
        self.assertEqual(result.groups[0].segments, ())

    def test_reversal_does_not_merge_and_preserves_covered_edges(self):
        result = self.run_paths(
            (((3, 1), (3, 5), (3, 2)),), (NetGroup(7, (), (0,)),)
        )
        self.assertEqual(
            result.groups[0].segments,
            (((1.0, 3.0), (5.0, 3.0)), ((5.0, 3.0), (2.0, 3.0))),
        )



    def test_oblique_collinear_edges_merge(self):
        path = ((1, 1), (3, 4), (5, 7), (9, 13))
        result = self.run_paths((path,), (NetGroup(7, (), (0,)),))
        self.assertEqual(result.groups[0].segments, (((1.0, 1.0), (13.0, 9.0)),))

    def test_continuous_raster_curve_keeps_each_direction_change(self):
        path = ((1, 1), (2, 2), (3, 3), (3, 4), (3, 5), (4, 6), (5, 7))
        result = self.run_paths((path,), (NetGroup(7, (), (0,)),))
        self.assertEqual(
            result.groups[0].segments,
            (((1.0, 1.0), (3.0, 3.0)),
             ((3.0, 3.0), (5.0, 3.0)),
             ((5.0, 3.0), (7.0, 5.0))),
        )

    def test_dense_corners_are_all_preserved(self):
        path = ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3))
        result = self.run_paths((path,), (NetGroup(7, (), (0,)),))
        self.assertEqual(
            result.groups[0].segments,
            (((1.0, 1.0), (2.0, 1.0)),
             ((2.0, 1.0), (2.0, 2.0)),
             ((2.0, 2.0), (3.0, 2.0)),
             ((3.0, 2.0), (3.0, 3.0))),
        )

    def test_one_pixel_noncollinearity_is_not_merged(self):
        path = ((4, 1), (4, 2), (5, 3))
        result = self.run_paths((path,), (NetGroup(7, (), (0,)),))
        self.assertEqual(
            result.groups[0].segments,
            (((1.0, 4.0), (2.0, 4.0)), ((2.0, 4.0), (3.0, 5.0))),
        )

    def test_repeated_path_coordinate_is_preserved_as_zero_length_edge(self):
        path = ((2, 1), (2, 3), (2, 3), (2, 5))
        result = self.run_paths((path,), (NetGroup(7, (), (0,)),))
        self.assertEqual(
            result.groups[0].segments,
            (((1.0, 2.0), (3.0, 2.0)),
             ((3.0, 2.0), (3.0, 2.0)),
             ((3.0, 2.0), (5.0, 2.0))),
        )

    def test_source_edges_remain_covered_and_turns_are_endpoints(self):
        path = ((1, 1), (2, 2), (3, 3), (3, 4), (3, 5), (4, 6), (5, 7))
        result = self.run_paths((path,), (NetGroup(7, (), (0,)),))
        xy = tuple((float(column), float(row)) for row, column in path)
        segments = result.groups[0].segments

        def on_segment(point, segment):
            (ax, ay), (bx, by) = segment
            px, py = point
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            return (cross == 0 and min(ax, bx) <= px <= max(ax, bx)
                    and min(ay, by) <= py <= max(ay, by))

        for start, end in zip(xy, xy[1:]):
            self.assertTrue(any(on_segment(start, segment) and on_segment(end, segment)
                                for segment in segments))
        endpoints = {point for segment in segments for point in segment}
        self.assertIn((3.0, 3.0), endpoints)
        self.assertIn((5.0, 3.0), endpoints)

    def test_requested_path_order_is_preserved_after_merging(self):
        paths = (((1, 1), (1, 2), (1, 4)), ((8, 3), (8, 5), (8, 9)))
        result = self.run_paths(paths, (NetGroup(7, (), (1, 0)),))
        self.assertEqual(
            result.groups[0].segments,
            (((3.0, 8.0), (9.0, 8.0)), ((1.0, 1.0), (4.0, 1.0))),
        )
    def test_input_graph_and_plan_remain_unchanged(self):
        graph = self.make_graph((((4, 1), (4, 3), (4, 8)),))
        plan = NetworkPlan((NetGroup(7, (), (0,), ((2.5, 6.5),)),))
        graph_before = (
            graph.paths, graph.reference_segments,
            graph.mask.tobytes(), graph.thickness.tobytes(), graph.skeleton.tobytes(),
            dict(graph.adjacency),
        )
        self.assertEqual(CollinearPathLines().run(graph, plan).groups[0].segments,
                         (((1.0, 4.0), (8.0, 4.0)), ((2.5, 6.5), (2.5, 6.5))))
        self.assertEqual(
            (graph.paths, graph.reference_segments,
             graph.mask.tobytes(), graph.thickness.tobytes(), graph.skeleton.tobytes(),
             dict(graph.adjacency)),
            graph_before,
        )

    def test_repeated_execution_is_deterministic(self):
        graph = self.make_graph((((1, 1), (2, 2), (3, 3), (3, 4)), ((8, 2), (8, 6))))
        plan = NetworkPlan((NetGroup(3, (), (1, 0), ((10.0, 12.0),)),))
        backend = CollinearPathLines()
        self.assertEqual(backend.run(graph, plan), backend.run(graph, plan))

if __name__ == "__main__":
    unittest.main()
