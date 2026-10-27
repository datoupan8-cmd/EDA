import copy
from dataclasses import asdict, replace
import unittest
import numpy as np
from pcb.schema import Component, Pin, Scene
from pcb.network_shared.contracts import snapshot
from pcb.network_shared.graph import build_graph
from pcb.net_hypergraph.legacy_v2 import LegacyHypergraph
from pcb.net_lines.legacy_v2 import LegacyLines
from pcb.network_shared.composite import assemble, validate_plan
from pcb.submission import export, validate_strict
from pcb.topology_stage.power_terminals import attach_power_terminals
from pcb.core.team_config import load_team_config


def fixture(power_pins=None):
    c = Component('P', 'v', (18, 10, 22, 15), value='3.3V', pins=power_pins or [])
    r = Component('R1', 'r', (18, 45, 22, 50), pins=[Pin('1', '1', (20., 40.), base=(20., 45.), side='top')])
    scene = Scene(60, 60, [c, r], diagnostics={'wire_stroke_width': 1.})
    mask = np.zeros((60, 60), np.uint8)
    mask[17:42, 20] = 255
    graph = build_graph(mask)
    logical = LegacyHypergraph().run(snapshot(scene), graph, mode='directional', merge_labels=False)
    return scene, graph, logical


class PowerTerminals(unittest.TestCase):
    def test_missing_power_pin_becomes_valid_reference_without_merging_old_groups(self):
        scene, graph, logical = fixture()
        old = copy.deepcopy(scene)
        plan, trace = attach_power_terminals(scene, graph, logical.plan, logical.diagnostics)
        self.assertEqual(len(trace['events']), 1)
        self.assertEqual(asdict(scene.components[1]), asdict(old.components[1]))
        self.assertEqual([(g.group_id, g.path_ids) for g in plan.groups],
                         [(g.group_id, g.path_ids) for g in logical.plan.groups])
        self.assertEqual([g.pins for g in logical.plan.groups],
                         [tuple(r for r in g.pins if r != 'P.1') for g in plan.groups])
        validate_plan(plan, snapshot(scene), graph)
        scene.nets = assemble(plan, LegacyLines().run(graph, plan))
        result = export(scene)
        self.assertIn('pin_1', result['pins']['P'])
        self.assertTrue(any('P.1' in n['hyperGraph'] for n in result['nets'].values()))
        validate_strict(result, (60, 60))

    def test_existing_hidden_pin_position_is_preserved(self):
        pin = Pin('UNK1', '', (20., 18.), base=(20., 15.), side='bottom', exportable=False)
        scene, graph, logical = fixture([pin])
        plan, trace = attach_power_terminals(scene, graph, logical.plan, logical.diagnostics)
        self.assertEqual(trace['events'][0]['method'], 'existing_directional_snap')
        self.assertEqual(scene.components[0].pins[0].tip, (20., 18.))
        self.assertTrue(scene.components[0].pins[0].exportable)
        self.assertFalse(any('P.UNK1' in g.pins for g in plan.groups))

    def test_absent_wire_does_not_create_or_export_terminal(self):
        scene, _, _ = fixture()
        graph = build_graph(np.zeros((60, 60), np.uint8))
        logical = LegacyHypergraph().run(snapshot(scene), graph, mode='directional', merge_labels=False)
        _, trace = attach_power_terminals(scene, graph, logical.plan, logical.diagnostics)
        self.assertFalse(trace['events'])
        self.assertEqual(scene.components[0].pins, [])

    def test_existing_exported_power_pin_is_protected(self):
        pin = Pin('9', 'V', (20., 18.), base=(20., 15.), side='bottom')
        scene, graph, logical = fixture([pin])
        before = asdict(scene.components[0])
        _, trace = attach_power_terminals(scene, graph, logical.plan, logical.diagnostics)
        self.assertFalse(trace['events'])
        self.assertEqual(asdict(scene.components[0]), before)

    def test_idempotence(self):
        scene, graph, logical = fixture()
        first, _ = attach_power_terminals(scene, graph, logical.plan, logical.diagnostics)
        second, trace = attach_power_terminals(scene, graph, first, logical.diagnostics)
        self.assertEqual(first, second)
        self.assertFalse(trace['events'])

    def test_unknown_value_does_not_create_default_vcc_rail(self):
        scene, graph, logical = fixture()
        scene.components[0].value = None
        _, trace = attach_power_terminals(scene, graph, logical.plan, logical.diagnostics)
        self.assertFalse(trace['events'])
        self.assertEqual(trace['rejected'][0]['reason'], 'missing_supply_identity')
        self.assertIsNone(scene.components[0].value)

    def test_ground_without_value_has_type_identity(self):
        scene, graph, logical = fixture()
        scene.components[0].type = 'gnd'
        scene.components[0].value = None
        plan, trace = attach_power_terminals(scene, graph, logical.plan, logical.diagnostics)
        self.assertEqual(len(trace['events']), 1)
        validate_plan(plan, snapshot(scene), graph)

    def test_wire_only_group_can_receive_a_power_terminal(self):
        scene, graph, _ = fixture()
        scene.components = scene.components[:1]
        logical = LegacyHypergraph().run(snapshot(scene), graph, mode='directional', merge_labels=False)
        self.assertFalse(any(g.pins for g in logical.plan.groups))
        plan, trace = attach_power_terminals(scene, graph, logical.plan, logical.diagnostics)
        self.assertEqual(len(trace['events']), 1)
        scene.nets = assemble(plan, LegacyLines().run(graph, plan))
        self.assertEqual(len(export(scene)['nets']), 1)

    def test_equally_close_different_groups_are_rejected(self):
        scene = Scene(50, 50, [Component('P', 'v', (18, 20, 22, 24), value='VCC')],
                      diagnostics={'wire_stroke_width': 1.})
        mask = np.zeros((50, 50), np.uint8)
        mask[3:18, 20] = 255
        mask[27:43, 20] = 255
        graph = build_graph(mask)
        logical = LegacyHypergraph().run(snapshot(scene), graph, mode='directional', merge_labels=False)
        _, trace = attach_power_terminals(scene, graph, logical.plan, logical.diagnostics)
        self.assertFalse(trace['events'])
        self.assertEqual(trace['rejected'][0]['reason'], 'ambiguous_wire_groups')

    def test_selected_profile_changes_only_topology(self):
        from tools.run_team_pipeline import runtime_registry
        from pcb.topology_stage.power_v1 import PowerTerminalTopology
        baseline = load_team_config('configs/team.json')
        candidate = replace(baseline, topology='v3_power_terminals_v1')
        a, b = baseline.as_dict(), candidate.as_dict()
        self.assertEqual({k for k in a if a[k] != b[k]}, {'topology'})
        self.assertIsInstance(runtime_registry(candidate).create('topology', candidate.topology), PowerTerminalTopology)


if __name__ == '__main__':
    unittest.main()
