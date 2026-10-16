"""Synthetic contracts for the already accepted ground export policy."""
import copy
from dataclasses import asdict
import math
import unittest

from pcb.pin.ground_export_T import project
from pcb.schema import Component, Net, Pin, Scene
from pcb.submission import export


def scene(*, distance=0, number_source='unobservable_internal_id'):
    ground = Component('G1', 'gnd', (5, 5, 15, 15), pins=[
        Pin('internal', None, (10, 10 + distance), exportable=False, number_source=number_source)])
    chip = Component('U1', 'box', (18, 5, 30, 20), pins=[Pin('1', 'A', (20, 10))])
    return Scene(100, 100, [ground, chip], nets=[Net(['G1.internal', 'U1.1'], [((10, 10), (20, 10))])])


class GroundExportTContracts(unittest.TestCase):
    def test_recovers_single_ground_reference_without_mutating_input(self):
        raw = scene()
        original = copy.deepcopy(asdict(raw))
        before = export(raw)
        projected, prediction, changes, rejected = project(raw)
        self.assertEqual(asdict(raw), original)
        self.assertEqual(len(changes), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(projected.nets[0].pins, ['G1.1', 'U1.1'])
        self.assertEqual(prediction['components'], before['components'])
        self.assertEqual(prediction['pins']['U1'], before['pins']['U1'])
        self.assertEqual(projected.nets[0].segments, raw.nets[0].segments)

    def test_rejects_terminal_without_local_wire_support(self):
        raw = scene(distance=6)
        _, prediction, changes, rejected = project(raw)
        self.assertEqual(prediction, export(raw))
        self.assertEqual(changes, [])
        self.assertEqual(rejected[0]['reason'], 'no_local_wire_support')

    def test_rejects_nonunique_existing_network(self):
        raw = scene()
        raw.nets.append(Net(['G1.internal'], [((10, 10), (10, 20))]))
        _, prediction, changes, rejected = project(raw)
        self.assertEqual(prediction, export(raw))
        self.assertEqual(changes, [])
        self.assertEqual(rejected[0]['reason'], 'nonunique_or_absent_existing_net')

    def test_rejects_unqualified_number_source(self):
        raw = scene(number_source='printed_number')
        _, prediction, changes, rejected = project(raw)
        self.assertEqual(prediction, export(raw))
        self.assertEqual(changes, [])
        self.assertEqual(rejected[0]['reason'], 'unqualified_number_source')

    def test_preserves_already_exported_ground(self):
        raw = scene()
        raw.components[0].pins[0].number = '1'
        raw.components[0].pins[0].exportable = True
        raw.nets[0].pins[0] = 'G1.1'
        _, prediction, changes, rejected = project(raw)
        self.assertEqual(prediction, export(raw))
        self.assertEqual(changes, [])
        self.assertEqual(rejected[0]['reason'], 'already_exported_protected')

    def test_ground_only_activation_is_explicitly_preserved_as_a_known_risk(self):
        raw = scene()
        raw.nets[0].pins = ['G1.internal']
        projected, prediction, changes, _ = project(raw)
        self.assertEqual(len(changes), 1)
        self.assertEqual(export(raw)['nets'], {})
        self.assertTrue(prediction['nets'])
        self.assertEqual(projected.nets[0].segments, raw.nets[0].segments)


if __name__ == '__main__':
    unittest.main()
