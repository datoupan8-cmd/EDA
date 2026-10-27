import copy
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.registry import StageRegistry
from pcb.schema import Component, Pin, Scene
from pcb.wire_stage.registry import register_stages
from pcb.wire_stage.retained import (
    WireStageV3FrameGuard,
    WireStageV8BaselineAnchoredRecall,
    combine_candidate_anchor_masks,
    filter_candidate_bridges,
    retain_corridor_anchored_components,
)


class BaselineAnchoredRecallTests(unittest.TestCase):
    def test_baseline_mask_anchors_candidate_without_terminal_corridor(self):
        baseline = np.zeros((40, 100), np.uint8)
        cv2.line(baseline, (10, 20), (25, 20), 255, 1)
        terminal = np.zeros_like(baseline)
        anchors = combine_candidate_anchor_masks(baseline, terminal)
        candidates = np.zeros_like(baseline)
        cv2.line(candidates, (25, 20), (45, 20), 255, 1)
        cv2.line(candidates, (60, 8), (90, 8), 255, 1)

        retained, count = retain_corridor_anchored_components((candidates,), anchors)

        self.assertEqual(count, 1)
        self.assertTrue(retained[20, 25:46].all())
        self.assertFalse(retained[8, 60:91].any())

    def test_bridge_guard_still_rejects_component_joining_two_baseline_regions(self):
        baseline = np.zeros((40, 100), np.uint8)
        cv2.line(baseline, (5, 20), (30, 20), 255, 1)
        cv2.line(baseline, (70, 20), (95, 20), 255, 1)
        candidate = baseline.copy()
        cv2.line(candidate, (30, 20), (70, 20), 255, 1)
        anchors = combine_candidate_anchor_masks(baseline, np.zeros_like(baseline))
        retained, _ = retain_corridor_anchored_components((candidate,), anchors)

        result, rejected, detail = filter_candidate_bridges(baseline, retained)

        self.assertEqual(detail["rejected_component_count"], 1)
        self.assertTrue(rejected[20, 31:70].all())
        self.assertFalse(result[20, 31:70].any())
        self.assertEqual(cv2.connectedComponents(result, 8)[0] - 1, 2)

    def test_candidate_is_registered_and_default_remains_v3(self):
        registry = StageRegistry()
        register_stages(registry)
        self.assertIsInstance(registry.create("wire", "v8_baseline_anchored_recall"),
                              WireStageV8BaselineAnchoredRecall)
        config_path = Path(__file__).resolve().parents[2] / "configs" / "wire_topology" / "current.json"
        self.assertEqual(json.loads(config_path.read_text(encoding="utf-8"))["wire"],
                         "v3_frame_guard")

    def test_v8_retains_every_v7_baseline_pixel(self):
        image = np.full((120, 120, 3), 255, np.uint8)
        cv2.line(image, (15, 55), (105, 55), (0, 0, 200), 1)
        scene = Scene(120, 120, [Component(
            "R1", "resistor", (10, 45, 30, 65),
            pins=[Pin("1", "1", (40, 55), base=(30, 55), side="right")],
        )])
        context = PipelineContext("baseline.png", 120, 120, PipelineConfig(), None)
        v7 = WireStageV3FrameGuard().run(image, copy.deepcopy(scene), context)

        v8 = WireStageV8BaselineAnchoredRecall().run(image, scene, context)

        self.assertTrue(np.all(v8.mask[v7.mask > 0] > 0))
        self.assertEqual(scene.diagnostics["wire_candidate_corridor_recall"]["guard_baseline"],
                         "v7_palette_corridor_recall")

    def test_anchor_masks_must_be_matching_two_dimensional_arrays(self):
        with self.assertRaisesRegex(ValueError, "matching 2D arrays"):
            combine_candidate_anchor_masks(np.zeros((8, 8), np.uint8),
                                           np.zeros((8, 9), np.uint8))
        with self.assertRaisesRegex(ValueError, "matching 2D arrays"):
            combine_candidate_anchor_masks(np.zeros((8, 8, 3), np.uint8),
                                           np.zeros((8, 8), np.uint8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
