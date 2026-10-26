import unittest
from types import MappingProxyType

from pcb.net_lines.path_v1 import PathLines
from pcb.network_shared.contracts import (
    GeometryResult,
    GroupLines,
    NetGroup,
    NetworkPlan,
    WireGraph,
)


class PathLinesContract(unittest.TestCase):
    def make_graph(self, paths, reference_segments=None):
        paths = tuple(tuple(path) for path in paths)
        if reference_segments is None:
            reference_segments = tuple(() for _ in paths)
        return WireGraph(
            width=20,
            height=20,
            paths=paths,
            adjacency=MappingProxyType({}),
            reference_segments=tuple(reference_segments),
            mask=memoryview(b"mask").toreadonly(),
            thickness=memoryview(b"thickness").toreadonly(),
            skeleton=memoryview(b"skeleton").toreadonly(),
        )

    def test_horizontal_path_converts_row_column_to_xy(self):
        graph = self.make_graph((((4, 2), (4, 5), (4, 8)),))
        plan = NetworkPlan((NetGroup(7, (), (0,)),))

        result = PathLines().run(graph, plan)

        self.assertEqual(
            result,
            GeometryResult((GroupLines(7, (((2.0, 4.0), (5.0, 4.0)),
                                           ((5.0, 4.0), (8.0, 4.0)))),)),
        )

    def test_vertical_path_converts_row_column_to_xy(self):
        graph = self.make_graph((((2, 6), (5, 6), (8, 6)),))
        plan = NetworkPlan((NetGroup(7, (), (0,)),))

        result = PathLines().run(graph, plan)

        self.assertEqual(
            result.groups[0].segments,
            (((6.0, 2.0), (6.0, 5.0)), ((6.0, 5.0), (6.0, 8.0))),
        )

    def test_right_angle_corner_is_preserved(self):
        graph = self.make_graph((((1, 2), (3, 2), (3, 5)),))
        plan = NetworkPlan((NetGroup(7, (), (0,)),))

        result = PathLines().run(graph, plan)

        self.assertEqual(
            result.groups[0].segments,
            (((2.0, 1.0), (2.0, 3.0)), ((2.0, 3.0), (5.0, 3.0))),
        )

    def test_path_order_and_segment_order_follow_plan_path_ids(self):
        graph = self.make_graph(
            (((1, 1), (1, 2)), ((8, 3), (9, 3))),
            reference_segments=((((99.0, 99.0), (100.0, 100.0)),), ()),
        )
        plan = NetworkPlan((NetGroup(7, (), (1, 0)),))

        result = PathLines().run(graph, plan)

        self.assertEqual(
            result.groups[0].segments,
            (((3.0, 8.0), (3.0, 9.0)), ((1.0, 1.0), (2.0, 1.0))),
        )

    def test_group_ids_and_path_membership_do_not_cross(self):
        graph = self.make_graph((((1, 1), (1, 2)), ((8, 3), (9, 3))))
        plan = NetworkPlan((NetGroup(11, (), (0,)), NetGroup(22, (), (1,))))

        result = PathLines().run(graph, plan)

        self.assertEqual(tuple(group.group_id for group in result.groups), (11, 22))
        self.assertEqual(result.groups[0].segments, (((1.0, 1.0), (2.0, 1.0)),))
        self.assertEqual(result.groups[1].segments, (((3.0, 8.0), (3.0, 9.0)),))

    def test_contact_points_are_zero_length_and_duplicates_are_removed(self):
        graph = self.make_graph(((),))
        point = (4.5, 6.5)
        plan = NetworkPlan((NetGroup(7, (), (0,), (point, point)),))

        result = PathLines().run(graph, plan)

        self.assertEqual(result.groups[0].segments, ((point, point),))

    def test_empty_and_single_point_paths_emit_no_path_segment(self):
        graph = self.make_graph(((), ((5, 9),)))
        plan = NetworkPlan((NetGroup(7, (), (0, 1)),))

        result = PathLines().run(graph, plan)

        self.assertEqual(result.groups[0].segments, ())

    def test_group_without_paths_or_contacts_is_preserved_empty(self):
        graph = self.make_graph(())
        plan = NetworkPlan((NetGroup(7, (), ()),))

        result = PathLines().run(graph, plan)

        self.assertEqual(result, GeometryResult((GroupLines(7, ()),)))

    def test_input_graph_data_is_unchanged(self):
        refs = ((((90.0, 91.0), (92.0, 93.0)),),)
        graph = self.make_graph((((2, 3), (4, 5)),), refs)
        plan = NetworkPlan((NetGroup(7, (), (0,), ((10.0, 11.0),)),))
        original_plan = plan
        original = (
            graph.paths,
            graph.reference_segments,
            bytes(graph.mask),
            bytes(graph.thickness),
            bytes(graph.skeleton),
            graph.adjacency,
        )

        PathLines().run(graph, plan)

        self.assertEqual(plan, original_plan)
        self.assertEqual(graph.paths, original[0])
        self.assertEqual(graph.reference_segments, original[1])
        self.assertEqual(bytes(graph.mask), original[2])
        self.assertEqual(bytes(graph.thickness), original[3])
        self.assertEqual(bytes(graph.skeleton), original[4])
        self.assertIs(graph.adjacency, original[5])
        self.assertTrue(graph.mask.readonly)
        self.assertTrue(graph.thickness.readonly)
        self.assertTrue(graph.skeleton.readonly)


if __name__ == "__main__":
    unittest.main()
