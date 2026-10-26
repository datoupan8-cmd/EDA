import unittest
from dataclasses import replace
from pcb.net_lines.legacy_v2 import LegacyLines
from pcb.net_hypergraph.legacy_v2 import LegacyHypergraph
from pcb.network_shared.graph import build_graph
from pcb.network_shared.contracts import snapshot
from tests.topology.split_cases import cases


class LinesContract(unittest.TestCase):
    def test_returns_geometry_only(self):
        _,s,mask=next(c for c in cases() if c[0]=='straight');graph=build_graph(mask)
        logical=LegacyHypergraph().run(snapshot(s),graph)
        geometry=LegacyLines().run(graph,logical.plan)
        self.assertEqual(geometry.groups[0].segments,(((10.,50.),(90.,50.)),))
        self.assertFalse(hasattr(geometry.groups[0],'pins'))
        self.assertEqual(logical.plan.groups[0].pins,('U1.1','U2.1'))

    def test_contact_point_becomes_zero_length_segment(self):
        _,s,mask=next(c for c in cases() if c[0]=='straight');graph=build_graph(mask)
        logical=LegacyHypergraph().run(snapshot(s),graph)
        point=(50.,40.)
        group=replace(logical.plan.groups[0],contact_points=(point,))
        plan=replace(logical.plan,groups=(group,))
        geometry=LegacyLines().run(graph,plan)
        self.assertIn((point,point),geometry.groups[0].segments)

    def test_duplicate_contact_point_is_emitted_once(self):
        _,s,mask=next(c for c in cases() if c[0]=='straight');graph=build_graph(mask)
        logical=LegacyHypergraph().run(snapshot(s),graph)
        point=(50.,40.)
        group=replace(logical.plan.groups[0],contact_points=(point,point))
        plan=replace(logical.plan,groups=(group,))
        geometry=LegacyLines().run(graph,plan)
        self.assertEqual(geometry.groups[0].segments.count((point,point)),1)


if __name__=='__main__':unittest.main()
