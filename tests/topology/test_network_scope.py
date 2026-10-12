import unittest
from tools.check_network_change_scope import allowed,wire_fields_allowed


class ChangeScope(unittest.TestCase):
    def test_line_owner_cannot_change_other_algorithms(self):
        self.assertTrue(allowed('pcb/net_lines/v4.py','net_lines'))
        self.assertTrue(allowed('configs/net_lines/current.json','net_lines'))
        for path in ('pcb/pin/frozen_l39/stages.py','pcb/component/current.py','pcb/net_hypergraph/legacy_v2.py',
                     'pcb/network_shared/graph.py','pcb/submission.py','evaluate_v2.py'):
            self.assertFalse(allowed(path,'net_lines'))

    def test_hypergraph_and_full_network_owner_boundaries(self):
        self.assertTrue(allowed('pcb/net_hypergraph/v4.py','net_hypergraph'))
        self.assertFalse(allowed('pcb/net_lines/v4.py','net_hypergraph'))
        self.assertTrue(allowed('pcb/net_lines/v4.py','network'))
        self.assertTrue(allowed('pcb/wire_stage/v4.py','network'))
        self.assertFalse(allowed('pcb/core/pipeline.py','network'))

    def test_wire_cannot_switch_topology_or_labels(self):
        old={'wire':'v3_frame_guard','topology':'v3','variant':'baseline','enable_flying_labels':False}
        self.assertTrue(wire_fields_allowed(old,{**old,'wire':'v4'}))
        self.assertFalse(wire_fields_allowed(old,{**old,'topology':'v1'}))
        self.assertFalse(wire_fields_allowed(old,{**old,'enable_flying_labels':True}))

    def test_run_report_snapshot_cannot_modify_selected_backends(self):
        from types import SimpleNamespace
        from tools.run_team_pipeline import network_version_snapshot
        stage=SimpleNamespace(network_versions={'net_hypergraph':'legacy_v2','net_lines':'legacy_v2'})
        result=network_version_snapshot(SimpleNamespace(topology=stage))
        result['net_lines']='other'
        self.assertEqual(stage.network_versions['net_lines'],'legacy_v2')
        self.assertIsNone(network_version_snapshot(SimpleNamespace(topology=object())))


if __name__=='__main__':unittest.main()
