import unittest
from pcb.net_lines.legacy_v2 import LegacyLines
from pcb.net_hypergraph.legacy_v2 import LegacyHypergraph
from pcb.network_shared.graph import build_graph
from pcb.network_shared.contracts import NetGroup,NetworkPlan,WireGraph,snapshot
from tests.topology.split_cases import cases


class LinesContract(unittest.TestCase):
    @staticmethod
    def graph(reference_segments,paths=()):
        return WireGraph(100,100,tuple(paths),{},tuple(reference_segments),None,None,None)

    def test_returns_geometry_only(self):
        _,s,mask=next(c for c in cases() if c[0]=='straight');graph=build_graph(mask)
        logical=LegacyHypergraph().run(snapshot(s),graph)
        geometry=LegacyLines().run(graph,logical.plan)
        self.assertEqual(geometry.groups[0].segments,(((10.,50.),(90.,50.)),))
        self.assertFalse(hasattr(geometry.groups[0],'pins'))
        self.assertEqual(logical.plan.groups[0].pins,('U1.1','U2.1'))

    def test_multiple_groups_preserve_plan_order_and_group_ids(self):
        plan=NetworkPlan((NetGroup(7,(),(0,)),NetGroup(19,(),(1,))))
        graph=self.graph(((((1.,2.),(3.,4.)),),(((5.,6.),(7.,8.)),)))

        geometry=LegacyLines().run(graph,plan)

        self.assertEqual(tuple(row.group_id for row in geometry.groups),
                         tuple(group.group_id for group in plan.groups))
        self.assertEqual(tuple(row.segments for row in geometry.groups),
                         ((((1.,2.),(3.,4.)),),(((5.,6.),(7.,8.)),)))

    def test_multiple_path_ids_merge_reference_segments_in_path_order(self):
        first=((1.,2.),(3.,4.));second=((10.,20.),(30.,40.));third=((50.,60.),(70.,80.))
        plan=NetworkPlan((NetGroup(3,(),(1,0)),))
        graph=self.graph(((first,),(second,third)))

        geometry=LegacyLines().run(graph,plan)

        self.assertEqual(geometry.groups[0].segments,(second,third,first))

    def test_contact_point_materializes_degenerate_segment(self):
        point=(12.,34.)
        plan=NetworkPlan((NetGroup(4,(),(),(point,)),))

        geometry=LegacyLines().run(self.graph(()),plan)

        self.assertEqual(geometry.groups[0].segments,((point,point),))

    def test_duplicate_contact_points_do_not_duplicate_segment(self):
        point=(12.,34.)
        plan=NetworkPlan((NetGroup(4,(),(),(point,point)),))

        geometry=LegacyLines().run(self.graph(()),plan)

        self.assertEqual(geometry.groups[0].segments,((point,point),))

    def test_empty_paths_keep_group_and_can_emit_contact_point(self):
        point=(21.,43.)
        plan=NetworkPlan((NetGroup(5,(),()),NetGroup(8,(),(),(point,))))

        geometry=LegacyLines().run(self.graph(()),plan)

        self.assertEqual(tuple(row.group_id for row in geometry.groups),(5,8))
        self.assertEqual(geometry.groups[0].segments,())
        self.assertEqual(geometry.groups[1].segments,((point,point),))

    def test_nonhorizontal_segment_keeps_xy_not_skeleton_row_column(self):
        segment=((12.,23.),(41.,67.))
        # The skeleton path is (row, column); the fixed reference segment is (x, y).
        graph=self.graph(((segment,),),paths=(((23,12),(67,41)),))
        plan=NetworkPlan((NetGroup(6,(),(0,)),))

        geometry=LegacyLines().run(graph,plan)

        self.assertEqual(geometry.groups[0].segments,(segment,))
        self.assertNotEqual(geometry.groups[0].segments,(((23.,12.),(67.,41.)),))


if __name__=='__main__':unittest.main()
