import copy
import contextlib
from dataclasses import asdict
import io
import json
from types import SimpleNamespace
import unittest
import numpy as np

from pcb.schema import Component, Scene
from pcb.core.interfaces import WireStageOutput
from pcb.network_shared.graph import build_graph
from pcb.network_shared.contracts import snapshot
from pcb.net_hypergraph.legacy_v2 import LegacyHypergraph
from pcb.topology_stage.power_terminals import attach_power_terminals
from pcb.topology_stage.power_v2 import attach_power_v2, power_crops, PowerEvidenceWire


def setup_case(ink=True, obstacle=False):
    scene = Scene(80, 80, [Component('P', 'v', (18, 10, 22, 20), value='3.3V')],
                  diagnostics={'wire_stroke_width': 1.})
    if obstacle:
        scene.components.append(Component('C1', 'c', (17, 22, 23, 27)))
    mask = np.zeros((80, 80), np.uint8)
    mask[28:60, 20] = 255
    image = np.full((80, 80, 3), 255, np.uint8)
    if ink:
        image[20:60, 20] = (0, 130, 0)
    graph = build_graph(mask)
    logical = LegacyHypergraph().run(snapshot(scene), graph, mode='directional', merge_labels=False)
    return scene, graph, logical, image


class PowerV2(unittest.TestCase):
    def test_real_ink_recovers_masked_stem_without_merging_or_new_paths(self):
        scene, graph, logical, image = setup_case()
        original = copy.deepcopy(scene)
        v1, trace1 = attach_power_terminals(copy.deepcopy(scene), graph, logical.plan, logical.diagnostics)
        self.assertFalse(trace1['events'])
        plan, trace = attach_power_v2(scene, graph, logical.plan, logical.diagnostics, power_crops(image, scene))
        self.assertEqual(trace['v2_extra_refs'], 1)
        self.assertEqual([(g.group_id, g.path_ids, g.contact_points) for g in plan.groups],
                         [(g.group_id, g.path_ids, g.contact_points) for g in v1.groups])
        self.assertEqual(scene.components[0].value, original.components[0].value)

    def test_empty_image_cannot_use_larger_radius(self):
        scene, graph, logical, image = setup_case(ink=False)
        _, trace = attach_power_v2(scene, graph, logical.plan, logical.diagnostics, power_crops(image, scene))
        self.assertEqual(trace['v2_extra_refs'], 0)

    def test_cannot_jump_across_capacitor(self):
        scene, graph, logical, image = setup_case(obstacle=True)
        ordinary = asdict(scene.components[1])
        _, trace = attach_power_v2(scene, graph, logical.plan, logical.diagnostics, power_crops(image, scene))
        self.assertEqual(trace['v2_extra_refs'], 0)
        self.assertEqual(asdict(scene.components[1]), ordinary)

    def test_unknown_value_stays_rejected(self):
        scene, graph, logical, image = setup_case()
        scene.components[0].value = None
        _, trace = attach_power_v2(scene, graph, logical.plan, logical.diagnostics, power_crops(image, scene))
        self.assertFalse(trace['events'])

    def test_explicit_wire_crop_wrapper_preserves_masks(self):
        scene, _, _, image = setup_case()
        base = WireStageOutput(np.zeros((80, 80), np.uint8))
        stage = SimpleNamespace(run=lambda *args: base)
        result = PowerEvidenceWire(stage).run(image, scene, None)
        self.assertIs(result.mask, base.mask)
        crop = result.debug['power_symbol_crops']['P']['image']
        self.assertFalse(crop.flags.writeable)
        self.assertNotIn('power_symbol_crops', base.debug)

    def test_cli_selects_adopted_v2_and_keeps_v1_and_no_power_modes(self):
        from tools.run_team_pipeline import main
        for options, expected in (([], 'v3'), (['--power-terminals'], 'v3_power_terminals_v2'),
                                  (['--power-terminals', '--power-version', 'v1'], 'v3_power_terminals_v1')):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main(options+['--list-stages']), 0)
            self.assertEqual(json.loads(stream.getvalue())['config']['topology'], expected)


if __name__ == '__main__':
    unittest.main()
