from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from pcb.component.alternative_ocr import PPOcrV5Backend, TextStagePPOcrV5
from pcb.component.local_joint import (
    LocalOCRConfig, SelectivePPOCR, assign_joint_designators, assign_joint_values,
    ComponentStagePrintedGroups, ComponentStageValueGroups, line_group_strength, merge_local_readings, select_regions,
)
from pcb.component.latest.text_roles import classify_tokens
from pcb.component.proposal_fusion import ComponentProposal
from pcb.core.interfaces import ComponentStageOutput, TextStageOutput
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.schema import Component, Text
from tools.evaluate_component_local_joint import select_cases


class ComponentLocalJointTests(unittest.TestCase):
    def test_value_group_stage_preserves_identity_model_and_input_objects(self):
        from pcb.component.contextual_values import ComponentStageContextualValues
        class Detector:
            def detect(self, image):
                return [ComponentProposal((40, 40, 55, 70), "r", .95, "test")], {}
        texts = [Text("R1", (20, 20, 35, 30), .98), Text("10k", (20, 33, 40, 43), .98)]
        image = np.zeros((100, 100, 3), np.uint8)
        context = PipelineContext("unseen", 100, 100, PipelineConfig(), None, Detector())
        original = ComponentStageContextualValues().run(image, TextStageOutput(texts), context)
        original.components[0].name = "printed_literal_name"
        original.components[0].value = None
        result = ComponentStageValueGroups().refine(image, original, context)
        self.assertEqual(result.components[0].key, original.components[0].key)
        self.assertEqual(result.components[0].name, "printed_literal_name")
        self.assertEqual(result.components[0].value, "10k")
        self.assertIsNone(original.components[0].value)
        self.assertIs(result.roles, original.roles)
        self.assertIs(result.texts, original.texts)

    def test_selected_stage_runs_detector_once_and_keeps_public_text_contract(self):
        class Detector:
            calls = 0
            def detect(self, image):
                self.calls += 1
                return [ComponentProposal((40, 40, 55, 70), "r", .95, "test")], {}
        detector = Detector()
        texts = [Text("R1", (20, 20, 35, 30), .98), Text("10k", (20, 33, 40, 43), .98)]
        context = PipelineContext("unseen_image", 100, 100,
                                  PipelineConfig(component="printed_groups_v1"), None, detector)
        result = ComponentStagePrintedGroups().run(np.zeros((100, 100, 3), np.uint8), TextStageOutput(texts), context)
        self.assertEqual(detector.calls, 1)
        self.assertIs(result.texts, texts)
        self.assertEqual(len(result.roles), len(texts))
        self.assertFalse(result.diagnostics["component_text_refinement"]["extra_OCR"])
        self.assertEqual(result.components[0].bbox, (40., 40., 55., 70.))
        self.assertEqual(result.components[0].key, "R1")
        self.assertEqual(result.components[0].value, "10k")

    def test_text_stage_keeps_context_resources_free_of_model_handles(self):
        backend = SimpleNamespace(recognize=lambda image: [], last_diagnostics={"backend": "fake"})
        stage = TextStagePPOcrV5()
        stage.backend = backend
        context = SimpleNamespace(resources={})
        output = stage.run(np.zeros((10, 10, 3), np.uint8), context)
        self.assertEqual(context.resources, {})
        self.assertEqual(output.texts, [])
        self.assertEqual(output.diagnostics, {"backend": "fake"})

    def test_padded_local_crop_maps_coordinates_and_cache_is_identical(self):
        class Backend:
            signature = b"test"
            calls = 0

            def _read(self, image, *, left, top, scale):
                self.calls += 1
                return [Text("R7", (left + 8 / scale, top + 4 / scale,
                                    left + 24 / scale, top + 16 / scale), .98)]
        component = Component("UNRESOLVED_0001", "r", (1, 1, 12, 18))
        original = ComponentStageOutput([component], [], [], {}, {})
        image = np.zeros((90, 120, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend()
            reader = SelectivePPOCR(directory, backend=backend)
            first, trace = reader.recognize(image, original)
            second, cached = reader.recognize(image, original)
            self.assertEqual(first[0].bbox, tuple(second[0].bbox))
            self.assertGreater(first[0].bbox[0], 0)
            self.assertFalse(trace["cache_hit"])
            self.assertTrue(cached["cache_hit"])
            self.assertEqual(backend.calls, 1)

    def test_full_ocr_restores_detector_after_recognition_only_call(self):
        calls = []
        reader = PPOcrV5Backend.__new__(PPOcrV5Backend)
        reader.engine = lambda image, **kwargs: (calls.append(kwargs) or SimpleNamespace(txts=None, boxes=None))
        self.assertEqual(reader._read(np.zeros((12, 12, 3), np.uint8)), [])
        self.assertTrue(calls[0]["use_det"])
        self.assertTrue(calls[0]["use_rec"])

    def test_local_work_is_bounded_and_power_labels_are_skipped(self):
        components = [Component(f"UNRESOLVED_{i}", "r", (i * 90, 0, i * 90 + 20, 20)) for i in range(40)]
        components.append(Component("GND_1", "gnd", (0, 30, 20, 50)))
        original = ComponentStageOutput(components, [], [], {}, {})
        regions = select_regions((100, 4000, 3), original, LocalOCRConfig(max_regions=5))
        self.assertEqual(len(regions), 5)
        self.assertTrue(all(row["component_index"] < 40 for row in regions))
        self.assertTrue(all(0 <= x <= 4000 and 0 <= y <= 100
                            for row in regions for x, y in [row["bbox"][:2], row["bbox"][2:]]))

    def test_weak_local_alternative_does_not_replace_trusted_designator(self):
        texts, rejected = merge_local_readings([Text("R10", (10, 10, 30, 20), .97)],
                                               [Text("R1O", (10, 10, 30, 20), .90)])
        self.assertEqual([t.text for t in texts], ["R10"])
        self.assertEqual(len(rejected), 1)

    def test_multiline_value_stays_with_its_designator_despite_neighbor(self):
        roles = classify_tokens([Text("R1", (10, 10, 24, 20), .98),
                                 Text("R2", (70, 10, 84, 20), .98),
                                 Text("10k", (10, 22, 28, 32), .98)])
        components = [Component("R1", "r", (30, 50, 40, 90)),
                      Component("R2", "r", (30, 26, 40, 36))]
        provenance = [{"value_source": "unknown"}, {"value_source": "unknown"}]
        diag = assign_joint_values(components, roles, provenance, {0: roles[0], 1: roles[1]})
        self.assertEqual(components[0].value, "10k")
        self.assertIsNone(components[1].value)
        self.assertEqual(len(diag["accepted"]), 1)

    def test_one_physical_text_cannot_supply_two_designators(self):
        proposals = [ComponentProposal((10, 35, 25, 50), "r", .9, "test"),
                     ComponentProposal((30, 35, 45, 50), "c", .9, "test")]
        roles = classify_tokens([Text("R1", (10, 10, 30, 20), .96),
                                 Text("C1", (10, 10, 30, 20), .95)])
        assigned, _ = assign_joint_designators(proposals, roles)
        self.assertEqual(len(assigned), 1)

    def test_one_physical_value_cannot_supply_two_components(self):
        roles = classify_tokens([Text("10k", (10, 10, 30, 20), .96),
                                 Text("100k", (10, 10, 30, 20), .95)])
        components = [Component("R1", "r", (0, 25, 12, 40)), Component("R2", "r", (30, 25, 42, 40))]
        assign_joint_values(components, roles, [{}, {}], {})
        self.assertEqual(sum(c.value is not None for c in components), 1)

    def test_grouping_rejects_unaligned_or_overlapping_text(self):
        anchor = Text("R1", (10, 10, 25, 20))
        self.assertEqual(line_group_strength(anchor, Text("10k", (90, 25, 110, 35))), 0)
        self.assertEqual(line_group_strength(anchor, Text("R7", (10, 10, 25, 20))), 0)

    def test_reserved_and_duplicate_case_ids_are_rejected_before_io(self):
        with self.assertRaises(PermissionError):
            select_cases(Path("200_train_cases"), "151")
        with self.assertRaises(PermissionError):
            select_cases(Path("200_train_cases"), "1,1")

    def test_missing_development_directory_stops_instead_of_expanding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "200_train_cases"
            root.mkdir()
            with self.assertRaises(ValueError):
                select_cases(root)


if __name__ == "__main__":
    unittest.main()
