import copy
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import WireStageOutput
from pcb.core.registry import StageRegistry
from pcb.schema import Component, Pin, Scene
from pcb.topology_stage.versions import TopologyStageV3
from pcb.wire_stage.registry import register_stages
from pcb.wire_stage.retained import (
    WireStageV3FrameGuard,
    WireStageV7PaletteCorridorRecall,
    filter_candidate_bridges,
    retain_corridor_anchored_components,
)


class PaletteCorridorRecallTests(unittest.TestCase):
    def test_corridor_keeps_supported_component_and_rejects_unanchored_component(self):
        candidates = np.zeros((40, 80), np.uint8)
        cv2.line(candidates, (5, 20), (35, 20), 255, 1)
        cv2.line(candidates, (50, 8), (70, 8), 255, 1)
        corridor = np.zeros_like(candidates)
        corridor[20, 18] = 255

        result, count = retain_corridor_anchored_components([candidates], corridor)

        self.assertEqual(count, 1)
        self.assertTrue(result[20, 5:36].all())
        self.assertFalse(result[8, 50:71].any())

    def test_normal_continuous_extension_from_one_baseline_region_is_kept(self):
        baseline = np.zeros((40, 100), np.uint8)
        cv2.line(baseline, (5, 20), (35, 20), 255, 1)
        candidate = baseline.copy()
        cv2.line(candidate, (35, 20), (65, 20), 255, 1)

        result, rejected, detail = filter_candidate_bridges(baseline, candidate)

        self.assertFalse(rejected.any())
        self.assertEqual(detail["rejected_component_count"], 0)
        self.assertTrue(result[20, 5:66].all())

    def test_added_straight_bridge_between_independent_wires_is_rejected(self):
        baseline = np.zeros((40, 100), np.uint8)
        cv2.line(baseline, (5, 20), (35, 20), 255, 1)
        cv2.line(baseline, (65, 20), (95, 20), 255, 1)
        candidate = baseline.copy()
        cv2.line(candidate, (35, 20), (65, 20), 255, 1)

        result, rejected, detail = filter_candidate_bridges(baseline, candidate)

        self.assertTrue(rejected[20, 36:65].all())
        self.assertFalse(result[20, 36:65].any())
        self.assertTrue(np.array_equal(result[baseline > 0], baseline[baseline > 0]))
        self.assertEqual(detail["rejected_component_count"], 1)
        self.assertEqual(cv2.connectedComponents(result, 8)[0] - 1, 2)

    def test_unmarked_crossing_remains_separate_under_topology_policy(self):
        baseline = np.zeros((100, 100), np.uint8)
        cv2.line(baseline, (10, 50), (90, 50), 255, 1)
        candidate = baseline.copy()
        cv2.line(candidate, (50, 10), (50, 90), 255, 1)
        result, rejected, _ = filter_candidate_bridges(baseline, candidate)
        self.assertFalse(rejected.any())

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

    def test_true_t_branch_and_pin_corridor_evidence_are_preserved(self):
        baseline = np.zeros((80, 100), np.uint8)
        cv2.line(baseline, (10, 50), (90, 50), 255, 1)
        branch = np.zeros_like(baseline)
        cv2.line(branch, (50, 10), (50, 50), 255, 1)
        corridor = np.zeros_like(baseline)
        corridor[10:15, 48:53] = 255
        anchored, count = retain_corridor_anchored_components([branch], corridor)
        self.assertEqual(count, 1)
        candidate = cv2.bitwise_or(baseline, anchored)
        result, rejected, _ = filter_candidate_bridges(baseline, candidate)
        self.assertFalse(rejected.any())
        self.assertTrue(result[10:51, 50].all())

        scene = Scene(100, 80, [
            Component(f"U{i}", "box", (x - 1, y - 1, x + 1, y + 1),
                      pins=[Pin("1", "", (x, y))])
            for i, (x, y) in enumerate(((10, 50), (90, 50), (50, 10)), start=1)
        ])
        TopologyStageV3(network_versions={
            "net_hypergraph": "legacy_v2", "net_lines": "legacy_v2"
        }).run(scene, WireStageOutput(result),
               PipelineContext("t-branch.png", 100, 80, PipelineConfig(), None))
        self.assertEqual(len(scene.nets), 1)
        self.assertEqual(len(scene.nets[0].pins), 3)

    def test_candidate_is_registered_and_default_configuration_is_unchanged(self):
        registry = StageRegistry()
        register_stages(registry)
        self.assertIsInstance(registry.create("wire", "v7_palette_corridor_recall"),
                              WireStageV7PaletteCorridorRecall)
        config_path = Path(__file__).resolve().parents[2] / "configs" / "wire_topology" / "current.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["wire"], "v3_frame_guard")

    def test_candidate_stage_runs_without_mutating_default_pixels_or_pins(self):
        image = np.full((120, 120, 3), 255, np.uint8)
        cv2.line(image, (15, 55), (105, 55), (0, 0, 200), 1)
        component = Component(
            "R1", "resistor", (10, 45, 30, 65),
            pins=[Pin("1", "1", (40, 55), base=(30, 55), side="right")],
        )
        scene = Scene(120, 120, [component])
        context = PipelineContext("candidate.png", 120, 120, PipelineConfig(), None)
        baseline = WireStageV3FrameGuard().run(image, copy.deepcopy(scene), context)
        original_image = image.copy()

        output = WireStageV7PaletteCorridorRecall().run(image, scene, context)

        self.assertEqual(output.mask.shape, (120, 120))
        self.assertTrue(output.mask.any())
        self.assertTrue(np.all(output.mask[baseline.mask > 0] > 0))
        self.assertTrue(np.array_equal(image, original_image))
        self.assertEqual(len(scene.components), 1)
        self.assertEqual(scene.components[0].pins[0].tip, (40, 55))
        self.assertGreaterEqual(
            scene.diagnostics["wire_candidate_corridor_recall"]["added_mask_pixels"], 0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
