import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from pcb.schema import Scene
from pcb.wire_stage.v2 import _select_palette_candidates, extract_wire_v2
from pcb.wire_stage.registry import register_stages


def candidate(score, name, density=100):
    return (score, name, None, density, 0, 0, 0)


class PaletteCandidateSelectionTests(unittest.TestCase):
    def test_default_two_candidate_policy_is_unchanged(self):
        scored = [candidate(100, "best"), candidate(95, "second"),
                  candidate(80, "third"), candidate(70, "below_gate")]
        selected = _select_palette_candidates(scored, minimum=20)
        self.assertEqual([item[1] for item in selected], ["best", "second"])

    def test_expanded_policy_restores_supported_non_top_two_hue_lines(self):
        image = np.full((120, 120, 3), 255, np.uint8)
        cv2.line(image, (15, 20), (105, 20), (0, 0, 200), 1)       # red hue
        cv2.line(image, (15, 55), (105, 55), (0, 200, 200), 1)     # yellow hue
        cv2.line(image, (15, 90), (105, 90), (200, 0, 0), 1)       # blue hue
        baseline, _, _ = extract_wire_v2(image, Scene(120, 120, []))
        expanded, _, _ = extract_wire_v2(
            image, Scene(120, 120, []), max_palette_candidates=None,
            palette_score_ratio=.72)
        def line_present(mask, y):
            return bool(mask[y - 1:y + 2, 49:52].any())
        self.assertTrue(line_present(baseline, 20))
        self.assertTrue(line_present(baseline, 55))
        self.assertFalse(line_present(baseline, 90))
        self.assertTrue(line_present(expanded, 20))
        self.assertTrue(line_present(expanded, 55))
        self.assertTrue(line_present(expanded, 90))
        self.assertGreater(int(expanded.sum()), int(baseline.sum()))

    def test_expanded_policy_keeps_candidates_passing_existing_gate(self):
        scored = [candidate(100, "best"), candidate(95, "second"),
                  candidate(80, "third"), candidate(71, "fourth"),
                  candidate(70.9, "below_gate")]
        selected = _select_palette_candidates(
            scored, minimum=20, score_ratio=.72, max_candidates=None)
        self.assertEqual([item[1] for item in selected], ["best", "second", "third"])

    def test_recall_policy_uses_same_candidates_with_explicit_71_percent_gate(self):
        scored = [candidate(100, "best"), candidate(95, "second"),
                  candidate(80, "third"), candidate(71, "fourth"),
                  candidate(70.9, "below_gate")]
        selected = _select_palette_candidates(
            scored, minimum=20, score_ratio=.71, max_candidates=None)
        self.assertEqual([item[1] for item in selected],
                         ["best", "second", "third", "fourth"])

    def test_private_stage_trace_does_not_change_extractor_mask(self):
        image = np.full((120, 120, 3), 255, np.uint8)
        cv2.line(image, (15, 20), (105, 20), (0, 0, 200), 1)
        cv2.line(image, (15, 55), (105, 55), (0, 200, 200), 1)
        scene = Scene(120, 120, [])
        plain, _, _ = extract_wire_v2(image, scene)
        trace = {}
        traced, _, _ = extract_wire_v2(image, scene, _trace=trace)
        self.assertTrue(np.array_equal(plain, traced))
        self.assertTrue(np.array_equal(trace["after_close_and_border"], traced))
        self.assertGreaterEqual(
            int((trace["raw_palette_union"] > 0).sum()),
            int((trace["after_line_support_union"] > 0).sum()),
        )

    def test_candidate_callback_matches_trace_candidates_without_changing_mask(self):
        image = np.full((120, 120, 3), 255, np.uint8)
        cv2.line(image, (15, 20), (105, 20), (0, 0, 200), 1)
        cv2.line(image, (15, 55), (105, 55), (0, 200, 200), 1)
        trace = {}
        traced, _, _ = extract_wire_v2(
            image, Scene(120, 120, []), max_palette_candidates=None,
            palette_score_ratio=0, _trace=trace)
        callback_masks = {}
        collected, _, _ = extract_wire_v2(
            image, Scene(120, 120, []), max_palette_candidates=None,
            palette_score_ratio=0,
            _candidate_callback=lambda name, mask: callback_masks.setdefault(name, mask.copy()))
        self.assertTrue(np.array_equal(traced, collected))
        self.assertEqual(callback_masks.keys(), trace["line_supported_candidates"].keys())
        for name, mask in callback_masks.items():
            self.assertTrue(np.array_equal(mask, trace["line_supported_candidates"][name]))

    def test_new_versions_are_explicit_and_default_config_is_unchanged(self):
        class Registry:
            def __init__(self): self.entries = {}
            def register(self, stage, name, cls): self.entries[name] = cls
        registry = Registry()
        register_stages(registry)
        self.assertIn("v4_palette_expanded", registry.entries)
        self.assertIn("v5_palette_recall", registry.entries)
        config = Path(__file__).resolve().parents[2] / "configs/wire_topology/current.json"
        self.assertEqual(json.loads(config.read_text(encoding="utf-8"))["wire"],
                         "v3_frame_guard")
        image = np.full((120, 120, 3), 255, np.uint8)
        cv2.line(image, (15, 20), (105, 20), (0, 0, 200), 1)
        cv2.line(image, (15, 55), (105, 55), (0, 200, 200), 1)
        cv2.line(image, (15, 90), (105, 90), (200, 0, 0), 1)
        for version in ("v4_palette_expanded", "v5_palette_recall"):
            output = registry.entries[version]().run(image, Scene(120, 120, []), None)
            self.assertTrue(output.mask[89:92, 49:52].any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
