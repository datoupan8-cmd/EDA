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
    WireStageV8BaselineAnchoredRecall,
    WireStageV9BranchSalvageRecall,
    WireStageV7PaletteCorridorRecall,
    filter_candidate_bridges,
    filter_candidate_bridges_with_branch_salvage,
)


def line(mask, start, end):
    cv2.line(mask, start, end, 255, 1)


def baseline_touch_counts(mask, baseline):
    base = baseline > 0
    _, labels, _, _ = cv2.connectedComponentsWithStats(base.astype(np.uint8), 8)
    count, added_labels, _, _ = cv2.connectedComponentsWithStats(
        ((mask > 0) & ~base).astype(np.uint8), 8
    )
    result = []
    for label in range(1, count):
        piece = added_labels == label
        near = cv2.dilate(piece.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        touched = np.unique(labels[near & base])
        result.append(int(np.count_nonzero(touched)))
    return result


class PaletteBranchSalvageTests(unittest.TestCase):
    def test_bent_wire_extension_from_one_baseline_is_preserved(self):
        baseline = np.zeros((60, 80), np.uint8)
        line(baseline, (5, 30), (20, 30))
        candidate = baseline.copy()
        line(candidate, (20, 30), (35, 30))
        line(candidate, (35, 30), (35, 45))

        output, _, detail = filter_candidate_bridges_with_branch_salvage(
            baseline, candidate, stroke_width=1
        )

        self.assertTrue(np.all(output[30, 20:36] > 0))
        self.assertTrue(np.all(output[30:46, 35] > 0))
        self.assertEqual(detail["incremental_guard_rejected_pieces"], 0)

    def test_bent_bridge_between_two_baselines_remains_rejected(self):
        baseline = np.zeros((60, 100), np.uint8)
        line(baseline, (3, 20), (20, 20))
        line(baseline, (45, 45), (70, 45))
        candidate = baseline.copy()
        line(candidate, (20, 20), (20, 45))
        line(candidate, (20, 45), (45, 45))

        output, _, detail = filter_candidate_bridges_with_branch_salvage(
            baseline, candidate, stroke_width=1
        )

        self.assertEqual(cv2.connectedComponents(output, 8)[0] - 1, 2)
        self.assertEqual(detail["branch_salvage_retained_pixels"], 0)

    def test_unmarked_crossing_arms_are_kept_separate(self):
        baseline = np.zeros((51, 51), np.uint8)
        line(baseline, (3, 25), (19, 25))
        line(baseline, (31, 25), (47, 25))
        line(baseline, (25, 3), (25, 19))
        line(baseline, (25, 31), (25, 47))
        candidate = baseline.copy()
        line(candidate, (19, 25), (31, 25))
        line(candidate, (25, 19), (25, 31))

        output, _, detail = filter_candidate_bridges_with_branch_salvage(
            baseline, candidate, stroke_width=1
        )

        self.assertEqual(cv2.connectedComponents(output, 8)[0] - 1, 4)
        self.assertTrue(all(touch_count <= 1 for touch_count in baseline_touch_counts(output, baseline)))
        self.assertGreater(detail["branch_salvage_retained_pixels"], 0)

    def test_t_junction_with_one_baseline_is_preserved(self):
        baseline = np.zeros((61, 81), np.uint8)
        line(baseline, (5, 30), (25, 30))
        candidate = baseline.copy()
        line(candidate, (25, 30), (55, 30))
        line(candidate, (40, 15), (40, 45))

        output, _, detail = filter_candidate_bridges_with_branch_salvage(
            baseline, candidate, stroke_width=1
        )

        self.assertTrue(np.all(output[30, 25:56] > 0))
        self.assertTrue(np.all(output[15:46, 40] > 0))
        self.assertEqual(detail["branch_salvage_retained_pixels"], 0)

    def test_multibranch_component_recovers_only_single_baseline_arms(self):
        baseline = np.zeros((61, 91), np.uint8)
        line(baseline, (3, 30), (10, 30))
        line(baseline, (40, 30), (48, 30))
        candidate = baseline.copy()
        line(candidate, (10, 30), (40, 30))
        line(candidate, (25, 12), (25, 30))
        v8_output, _, _ = filter_candidate_bridges(baseline, candidate)

        output, _, detail = filter_candidate_bridges_with_branch_salvage(
            baseline, candidate, stroke_width=1
        )

        self.assertTrue(np.all(output[v8_output > 0] > 0))
        self.assertEqual(cv2.connectedComponents(output, 8)[0] - 1, 2)
        self.assertTrue(np.any(output[30, 11:24]))
        self.assertTrue(np.any(output[30, 27:40]))
        self.assertFalse(output[30, 24:27].any())
        self.assertGreater(detail["branch_salvage_retained_pixels"], 0)
        self.assertTrue(all(touch_count <= 1 for touch_count in baseline_touch_counts(output, baseline)))

    def test_candidate_is_registered_without_changing_default(self):
        registry = StageRegistry()
        register_stages(registry)
        self.assertIsInstance(registry.create("wire", "v9_branch_salvage_recall"),
                              WireStageV9BranchSalvageRecall)
        config = Path(__file__).resolve().parents[2] / "configs" / "wire_topology" / "current.json"
        self.assertEqual(json.loads(config.read_text(encoding="utf-8"))["wire"],
                         "v3_frame_guard")

    def test_v9_stage_retains_v8_output_pixels(self):
        image = np.full((120, 120, 3), 255, np.uint8)
        cv2.line(image, (15, 55), (105, 55), (0, 0, 200), 1)
        scene = Scene(120, 120, [Component(
            "R1", "resistor", (10, 45, 30, 65),
            pins=[Pin("1", "1", (40, 55), base=(30, 55), side="right")],
        )])
        context = PipelineContext("v9.png", 120, 120, PipelineConfig(), None)
        expected = WireStageV8BaselineAnchoredRecall().run(
            image, copy.deepcopy(scene), context
        )

        actual = WireStageV9BranchSalvageRecall().run(image, scene, context)

        self.assertTrue(np.all(actual.mask[expected.mask > 0] > 0))
        detail = scene.diagnostics["wire_candidate_corridor_recall"]
        self.assertIn("branch_salvage_retained_pixels", detail)
        self.assertEqual(detail["guard_baseline"], "v7_palette_corridor_recall")


if __name__ == "__main__":
    unittest.main(verbosity=2)
