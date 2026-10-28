import unittest
import numpy as np

from pcb.schema import Scene, Component, Pin
from pcb.network_shared.graph import build_graph
from pcb.topology_stage.third_stage_audit import (
    identity_ceiling, identity_stages, observed_topology, destination_status, gt_edge_index,
)


class ThirdStageAudit(unittest.TestCase):
    def test_identity_loss_waterfall_is_exclusive_and_conserves_missing_nodes(self):
        scene = Scene(40, 40, [Component('U1', 'box', (1, 1, 5, 5), pins=[
            Pin('1', '', (6, 2)), Pin('2', '', (6, 3), exportable=False)]),
            Component('R1', 'r', (20, 1, 24, 5), pins=[Pin('1', '', (25, 3))])])
        cs = {'U1': {'type': 'box'}, 'R1': {'type': 'r'}}
        pred = {'components': cs, 'pins': {'U1': {'pin_1': {}}, 'R1': {'pin_1': {}}},
                'nets': {'n': {'hyperGraph': '(U1.1)', 'edges': {}}}}
        gt = {'components': {**cs, 'U2': {'type': 'box'}}}
        gg = [{('ic', 'U1', '1'), ('ic', 'U1', '2'), ('ic', 'U1', '3'),
               ('ic', 'U2', '1'), ('component', 'R1')}]
        result = identity_stages(scene, pred, gt, gg)
        self.assertEqual((result['GT_unique'], result['correct_in_net'], result['missing_in_net']), (5, 1, 4))
        self.assertEqual([sum(x.values()) for x in result['exclusive_losses'].values()], [1, 1, 1, 1])
        self.assertEqual(sum(result['exclusive_pair_losses'].values()), 10)
        ceilings = list(result['ceilings'].values())
        self.assertEqual(ceilings, sorted(ceilings, reverse=True))

    def test_optimistic_ceiling_allows_dropping_weak_networks_and_ignores_extra_ids(self):
        a = ('component', 'A')
        rest = {('component', str(i)) for i in range(10)}
        available = {a, ('component', '0')}
        self.assertAlmostEqual(identity_ceiling([{a}, rest], available), 2/3)
        self.assertEqual(identity_ceiling([{a}, rest], available),
                         identity_ceiling([{a}, rest], available | {('component', 'extra')}))
        self.assertEqual(identity_ceiling([{a}, rest], set()), 0.)

    def test_copy_only_instrumentation_preserves_topology_and_captures_all_boundaries(self):
        scene = Scene(40, 40, [Component('U1', 'box', (1, 15, 4, 25),
                    pins=[Pin('1', '', (5, 20), base=(4, 20))]),
                    Component('R1', 'r', (32, 15, 36, 25),
                    pins=[Pin('1', '', (30, 20), base=(32, 20))])])
        mask = np.zeros((40, 40), np.uint8)
        mask[20, 5:31] = 255
        result, captured = observed_topology(scene, build_graph(mask))
        self.assertEqual(len(captured['snapped']), 2)
        self.assertTrue({'junction', 'bridge', 'snap', 'contact', 'duplicate'} <= captured.keys())
        self.assertEqual(len(result.diagnostics['snaps']), 2)

    def test_geometry_diagnostic_keeps_ambiguity_and_missing_evidence_separate(self):
        a, b = ('component', 'A'), ('component', 'B')
        gn = [{'vertices': {a}, 'edges': [[{'x': 0, 'y': 20}, {'x': 40, 'y': 20}]]},
              {'vertices': {b}, 'edges': [[{'x': 20, 'y': 0}, {'x': 20, 'y': 40}]]}]
        index = gt_edge_index(gn)
        self.assertEqual(destination_status(a, (5, 20), gn, index), 'supported_correct_net')
        self.assertEqual(destination_status(a, (20, 5), gn, index), 'wrong_net_geometry')
        self.assertEqual(destination_status(a, (20, 20), gn, index), 'ambiguous_GT_nets')
        self.assertEqual(destination_status(a, (100, 100), gn, index), 'no_GT_line_nearby')
        self.assertEqual(destination_status(('component', 'missing'), (5, 20), gn, index), 'wrong_identity')


if __name__ == '__main__':
    unittest.main()
