"""Dataset-free safety and invariance tests for pin-stage comparisons."""
import copy
from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from pcb.core.config import PipelineConfig
from pcb.core.interfaces import PinLocalizationOutput, PinSemanticsOutput
from pcb.schema import Component, Pin, Scene, Text
from tools.benchmark_pins_v4 import (
    frozen_component_fields, internal_copy, restore_roles, run_pin_stages, verify_reference,
)


class PinBenchmarkTests(unittest.TestCase):
    def scene(self):
        scene = Scene(100, 100, [Component("U1", "box", (20, 20, 80, 80), pins=[
            Pin("1", "VCC", (10, 30), exportable=True),
            Pin("1", "", (10, 40), exportable=False),
            Pin("UNK1", "", (10, 50), exportable=False),
        ])], [Text("1", (11, 27, 17, 33))])
        scene.diagnostics["text_roles"] = [{"index": 0, "text": "1", "bbox": [11, 27, 17, 33],
                                           "role": "VALUE", "normalized": "1", "confidence": .87,
                                           "reason": "number is ambiguous"}]
        return scene

    def test_internal_unknowns_are_unique_and_do_not_mutate_official_pins(self):
        original = self.scene()
        snapshot = asdict(original)
        internal, audit = internal_copy(original)
        self.assertEqual(asdict(original), snapshot)
        numbers = [pin.number for pin in internal.components[0].pins]
        self.assertEqual(len(set(numbers)), 3)
        self.assertEqual(numbers[0], "1")
        self.assertTrue(all(pin.exportable for pin in internal.components[0].pins))
        self.assertEqual([row["recognized_number"] for row in audit], [True, False, False])
        self.assertEqual(frozen_component_fields(original), frozen_component_fields(internal))

    def test_internal_ids_do_not_collide_with_existing_real_ids(self):
        scene = self.scene()
        scene.components[0].pins[0].number = "__TMP_2"
        internal, _ = internal_copy(scene)
        self.assertEqual([pin.number for pin in internal.components[0].pins][:2], ["__TMP_2", "__TMP_2_"])

    def test_roles_preserve_ocr_geometry_and_evidence(self):
        roles = restore_roles(self.scene())
        self.assertEqual(roles[0].token.text, "1")
        self.assertEqual(roles[0].token.bbox, (11, 27, 17, 33))
        self.assertEqual(roles[0].role, "VALUE")
        self.assertEqual(roles[0].confidence, .87)

    def test_pin_stage_cannot_modify_component_or_text(self):
        class Localization:
            def run(self, image, output, context):
                output.components[0].bbox = (0, 0, 1, 1)
                return PinLocalizationOutput([])

        class Semantics:
            def run(self, output, localization, context):
                return PinSemanticsOutput(output.components)

        class Registry:
            def create(self, kind, version):
                return Localization() if kind == "pin_localization" else Semantics()

        scene = self.scene()
        before = copy.deepcopy(scene)
        with self.assertRaisesRegex(AssertionError, "component or OCR"):
            run_pin_stages(np.zeros((100, 100, 3), np.uint8), Path("fixture.png"),
                           scene, PipelineConfig(), Registry())
        self.assertEqual(scene, before)

    def test_reference_guard_rejects_holdout_incomplete_and_duplicate_cases(self):
        ids = [f"{n:04d}" for n in range(1, 151)]
        base = {"paired_case_ids": ids, "success_count": 150, "holdout_used": False,
                "cases": [{"case_id": key} for key in ids]}
        bad = []
        row = copy.deepcopy(base); row["paired_case_ids"][-1] = "0151"; bad.append(row)
        row = copy.deepcopy(base); row["holdout_used"] = True; bad.append(row)
        row = copy.deepcopy(base); row["cases"][-1]["case_id"] = "0001"; bad.append(row)
        row = copy.deepcopy(base); row["success_count"] = 149; bad.append(row)
        with patch("tools.benchmark_pins_v4.sha256_file") as reader:
            for payload in bad:
                with self.assertRaises(ValueError):
                    verify_reference(payload)
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
