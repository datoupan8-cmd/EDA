import unittest

import cv2
import numpy as np

from pcb.schema import Component, Pin, Scene
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import WireStageOutput
from pcb.topology_stage.versions import TopologyStageV3
from pcb.wire_stage.registry import register_stages
from pcb.wire_stage.retained import filter_bent_candidate_bridges


class PaletteBridgeGuardTests(unittest.TestCase):
    def test_straight_recovery_between_fragments_is_preserved(self):
        baseline = np.zeros((40, 100), np.uint8)
        cv2.line(baseline, (5, 20), (35, 20), 255, 1)
        cv2.line(baseline, (65, 20), (95, 20), 255, 1)
        candidate = baseline.copy()
        cv2.line(candidate, (35, 20), (65, 20), 255, 1)

        result, rejected, detail = filter_bent_candidate_bridges(baseline, candidate, 1.0)

        self.assertTrue(np.array_equal(result, candidate))
        self.assertFalse(rejected.any())
        self.assertEqual(detail["rejected_component_count"], 0)

    def test_bent_candidate_bridge_between_separate_wires_is_rejected(self):
        baseline = np.zeros((80, 100), np.uint8)
        cv2.line(baseline, (5, 15), (30, 15), 255, 1)
        cv2.line(baseline, (65, 55), (95, 55), 255, 1)
        candidate = baseline.copy()
        cv2.line(candidate, (30, 15), (45, 15), 255, 1)
        cv2.line(candidate, (45, 15), (45, 55), 255, 1)
        cv2.line(candidate, (45, 55), (65, 55), 255, 1)

        result, rejected, detail = filter_bent_candidate_bridges(baseline, candidate, 1.0)

        self.assertEqual(detail["rejected_component_count"], 1)
        self.assertEqual(detail["baseline_component_count"], 2)
        self.assertTrue(rejected.any())
        self.assertFalse(np.any(result[rejected]))
        self.assertTrue(np.array_equal(result[baseline > 0], baseline[baseline > 0]))
        self.assertEqual(cv2.connectedComponents(result, 8)[0] - 1, 2)

    def test_unmarked_cross_pixels_are_left_for_topology_crossing_policy(self):
        baseline = np.zeros((100, 100), np.uint8)
        cv2.line(baseline, (10, 50), (90, 50), 255, 1)
        candidate = baseline.copy()
        cv2.line(candidate, (50, 10), (50, 90), 255, 1)

        result, rejected, _ = filter_bent_candidate_bridges(baseline, candidate, 1.0)

        self.assertFalse(rejected.any())
        self.assertTrue(np.array_equal(result, candidate))
        pins = [
            Component(f"P{i}", "pin", (x - 1, y - 1, x + 1, y + 1),
                      pins=[Pin("1", "", (x, y))])
            for i, (x, y) in enumerate(((10, 50), (90, 50), (50, 10), (50, 90)))
        ]
        scene = Scene(100, 100, pins)
        TopologyStageV3(network_versions={
            "net_hypergraph": "legacy_v2", "net_lines": "legacy_v2"
        }).run(scene, WireStageOutput(result),
               PipelineContext("cross.png", 100, 100, PipelineConfig(), None))
        self.assertEqual(
            {frozenset(net.pins) for net in scene.nets},
            {frozenset(("P0.1", "P1.1")), frozenset(("P2.1", "P3.1"))},
        )

    def test_true_t_junction_branch_is_preserved(self):
        baseline = np.zeros((80, 100), np.uint8)
        cv2.line(baseline, (10, 50), (90, 50), 255, 1)
        candidate = baseline.copy()
        cv2.line(candidate, (50, 10), (50, 50), 255, 1)

        result, rejected, detail = filter_bent_candidate_bridges(baseline, candidate, 1.0)

        self.assertFalse(rejected.any())
        self.assertEqual(detail["rejected_component_count"], 0)
        self.assertTrue(np.array_equal(result, candidate))
        points = ((10, 50), (90, 50), (50, 10))
        scene = Scene(100, 80, [
            Component(f"U{i}", "box", (x - 1, y - 1, x + 1, y + 1),
                      pins=[Pin("1", "", (x, y))])
            for i, (x, y) in enumerate(points, start=1)
        ])
        TopologyStageV3(network_versions={
            "net_hypergraph": "legacy_v2", "net_lines": "legacy_v2"
        }).run(scene, WireStageOutput(result),
               PipelineContext("t-junction.png", 100, 80, PipelineConfig(), None))
        self.assertEqual(len(scene.nets), 1)
        self.assertEqual(len(scene.nets[0].pins), 3)

    def test_pin_supported_baseline_continuation_is_never_removed(self):
        # The baseline segment represents a previously supported pin lead.
        baseline = np.zeros((60, 100), np.uint8)
        cv2.line(baseline, (10, 30), (60, 30), 255, 1)
        candidate = baseline.copy()
        cv2.line(candidate, (60, 30), (85, 30), 255, 1)

        result, _, _ = filter_bent_candidate_bridges(baseline, candidate, 1.0)

        self.assertTrue(np.array_equal(result[baseline > 0], baseline[baseline > 0]))
        self.assertTrue(result[30, 60:86].all())

    def test_candidate_version_is_registered_without_changing_default(self):
        class Registry:
            def __init__(self):
                self.entries = {}

            def register(self, stage, name, cls):
                self.entries[name] = cls

        registry = Registry()
        register_stages(registry)
        self.assertIn("v6_palette_bridge_guard", registry.entries)

    def test_registered_candidate_runs_and_keeps_an_isolated_palette_wire(self):
        class Registry:
            def __init__(self):
                self.entries = {}

            def register(self, stage, name, cls):
                self.entries[name] = cls

        registry = Registry()
        register_stages(registry)
        image = np.full((120, 120, 3), 255, np.uint8)
        cv2.line(image, (15, 20), (105, 20), (0, 0, 200), 1)
        cv2.line(image, (15, 55), (105, 55), (0, 200, 200), 1)
        cv2.line(image, (15, 90), (105, 90), (200, 0, 0), 1)
        output = registry.entries["v6_palette_bridge_guard"]().run(
            image, Scene(120, 120, []), None
        )

        self.assertTrue(output.mask[89:92, 49:52].any())

    def test_candidate_wire_at_a_pin_terminal_is_kept(self):
        from pcb.wire_stage.retained import WireStageV6PaletteBridgeGuard

        image = np.full((120, 120, 3), 255, np.uint8)
        cv2.line(image, (15, 20), (105, 20), (0, 0, 200), 1)
        cv2.line(image, (15, 55), (105, 55), (0, 200, 200), 1)
        cv2.line(image, (45, 90), (105, 90), (200, 0, 0), 1)
        component = Component(
            "R1", "resistor", (20, 82, 36, 98),
            pins=[Pin("1", "1", (45, 90), base=(35, 90), side="right")],
        )
        scene = Scene(120, 120, [component])

        output = WireStageV6PaletteBridgeGuard().run(image, scene, None)

        self.assertTrue(output.mask[89:93, 48:106].any(axis=0).all())
        self.assertEqual(
            scene.diagnostics["wire_candidate_bridge_guard"]["rejected_component_count"], 0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
