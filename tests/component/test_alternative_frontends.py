from pathlib import Path
import tempfile
import unittest

import numpy as np

from pcb.component.alternative_rcnn import (CLASSES, RcnnConfig, RcnnDetector,
                                            assemble_rcnn_component_output)
from pcb.component.proposal_fusion import ComponentProposal
from pcb.component.experiments.train_faster_rcnn import TrainingGuard, TRAIN_IDS, CHECK_IDS
from pcb.schema import Text


class TrainingScopeTests(unittest.TestCase):
    def guard(self):
        guard = object.__new__(TrainingGuard)
        guard.dataset = (Path(tempfile.gettempdir()) / "synthetic_scope/200_train_cases").resolve()
        guard.allowed = None
        guard.target_opens = 0
        return guard

    def test_training_split_is_exactly_120_of_first_150_cases(self):
        self.assertEqual(len(TRAIN_IDS), 120)
        self.assertEqual(len(CHECK_IDS), 30)
        self.assertFalse(set(TRAIN_IDS) & set(CHECK_IDS))
        self.assertEqual(set(TRAIN_IDS) | set(CHECK_IDS), {f"{n:04d}" for n in range(1, 151)})

    def test_training_cannot_read_check_cases_or_reserved_cases(self):
        guard = self.guard()
        for cid in ("0005", "0150", "0151", "0200"):
            with self.subTest(cid=cid), self.assertRaises(PermissionError):
                guard.audit("open", (str(guard.dataset / cid / "example.png"), "r", 0))

    def test_annotations_require_exact_explicit_training_path(self):
        guard = self.guard()
        path = guard.dataset / "0001/example_target.json"
        with self.assertRaises(PermissionError):
            guard.audit("open", (str(path), "r", 0))
        guard.allowed = path.resolve()
        guard.audit("open", (str(path), "r", 0))
        self.assertEqual(guard.target_opens, 1)
        with self.assertRaises(PermissionError):
            guard.audit("open", (str(guard.dataset / "0002/example_target.json"), "r", 0))


class AlternativeFrontendTests(unittest.TestCase):
    def test_two_stage_detections_are_used_directly_and_do_not_invent_labels(self):
        proposals = [ComponentProposal((100, 100, 120, 116), "r", .99, "faster_rcnn", {})]
        texts = [Text("R3", (100, 87, 115, 97), .99), Text("10kΩ", (120, 82, 150, 94), .98)]
        result = assemble_rcnn_component_output(proposals, texts, {"detector": "fake_rcnn"})
        self.assertEqual(result.components[0].key, "R3")
        self.assertEqual(result.components[0].bbox, proposals[0].bbox)
        self.assertEqual(result.components[0].value, "10kΩ")
        self.assertFalse(result.diagnostics["yolo_used"])
        empty = assemble_rcnn_component_output(proposals, [], {})
        self.assertIsNone(empty.components[0].observable_designator)

    def test_tile_predictions_are_restored_to_original_coordinates(self):
        import torch

        class FakeModel:
            def __call__(self, images):
                return [{"boxes": torch.tensor([[10., 20., 30., 40.]]),
                         "scores": torch.tensor([.9]), "labels": torch.tensor([CLASSES.index("r")])}]

        detector = object.__new__(RcnnDetector)
        detector.weight_sha256 = "synthetic"
        detector.config = RcnnConfig()
        detector.cache = None
        detector.device = "cpu"
        detector.model = FakeModel()
        proposals, diagnostic = detector.detect(np.zeros((500, 1500, 3), np.uint8))
        self.assertEqual(diagnostic["tile_count"], 3)
        self.assertEqual([proposal.bbox for proposal in proposals],
                         [(10., 20., 30., 40.), (586., 20., 606., 40.), (742., 20., 762., 40.)])


if __name__ == "__main__":
    unittest.main()
