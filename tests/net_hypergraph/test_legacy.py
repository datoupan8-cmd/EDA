import unittest
from pcb.net_hypergraph.legacy_v2 import LegacyHypergraph
from pcb.network_shared.graph import build_graph
from pcb.network_shared.contracts import snapshot
from tests.topology.split_cases import cases


class HypergraphContract(unittest.TestCase):
    def test_returns_membership_with_no_geometry_or_upstream_mutation(self):
        _,s,mask=next(c for c in cases() if c[0]=='straight')
        result=LegacyHypergraph().run(snapshot(s),build_graph(mask))
        self.assertEqual(len(result.plan.groups),1)
        self.assertEqual(result.plan.groups[0].pins,('U1.1','U2.1'))
        self.assertFalse(hasattr(result.plan.groups[0],'segments'))
        self.assertEqual(s.nets,[])


if __name__=='__main__':unittest.main()
