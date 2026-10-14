import copy
import unittest
import numpy as np

from pcb.component.latest.text_assignment import assign_values
from pcb.component.latest.text_roles import classify_tokens
from pcb.component.proposal_fusion import ComponentProposal
from pcb.component.slice_recovery import recover_slice_components
from pcb.core.interfaces import ComponentStageOutput
from pcb.schema import Component, Text


class FakeDetector:
    def __init__(self, proposals):
        self.proposals = proposals
        self.calls = []

    def detect(self, image):
        self.calls.append(image.shape)
        return self.proposals, {"fake": True}


class SliceRecoveryTests(unittest.TestCase):
    def fixture(self, reference=True):
        components = [Component("R1", "r", (10, 10, 30, 22), name="R1", value="10k",
                                observable_designator="R1")]
        texts = [Text("10k", (10, 0, 30, 8), .99)]
        if reference:
            texts += [Text("R2", (100, 84, 116, 94), .99), Text("22k", (120, 82, 150, 94), .99)]
        roles = classify_tokens(texts)
        provenance = [{"component": "R1", "proposal_index": 0}]
        trace = assign_values(components, roles, provenance)
        return ComponentStageOutput(components, texts, roles,
                                    {"component_provenance": provenance, "value_assignment": trace})

    def test_new_symbol_requires_visible_unclaimed_reference_and_restores_coordinates(self):
        original = self.fixture()
        frozen = copy.deepcopy(original.components)
        detector = FakeDetector([ComponentProposal((200, 200, 240, 224), "r", .99, "fake", {})])
        output = recover_slice_components(np.zeros((160, 160, 3), np.uint8), original, detector)
        self.assertEqual(output.components[0], frozen[0])
        self.assertEqual(original.components, frozen)
        self.assertEqual(detector.calls, [(320, 320, 3)])
        self.assertEqual(output.components[1].key, "R2")
        self.assertEqual(output.components[1].bbox, (100, 100, 120, 112))
        self.assertEqual(output.components[1].value, "22k")

    def test_no_unclaimed_reference_skips_extra_detection(self):
        original = self.fixture(reference=False)
        detector = FakeDetector([])
        output = recover_slice_components(np.zeros((160, 160, 3), np.uint8), original, detector)
        self.assertEqual(output.components, original.components)
        self.assertEqual(detector.calls, [])

    def test_duplicate_body_is_not_exported(self):
        original = self.fixture()
        detector = FakeDetector([ComponentProposal((20, 20, 60, 44), "r", .99, "fake", {})])
        self.assertEqual(len(recover_slice_components(np.zeros((160, 160, 3), np.uint8),
                                                     original, detector).components), 1)

    def test_low_confidence_detection_is_not_exported(self):
        original = self.fixture()
        detector = FakeDetector([ComponentProposal((200, 200, 240, 224), "r", .10, "fake", {})])
        self.assertEqual(len(recover_slice_components(np.zeros((160, 160, 3), np.uint8),
                                                     original, detector).components), 1)

    def test_incompatible_type_is_not_assigned_reference(self):
        original = self.fixture()
        detector = FakeDetector([ComponentProposal((200, 200, 240, 224), "c", .99, "fake", {})])
        self.assertEqual(len(recover_slice_components(np.zeros((160, 160, 3), np.uint8),
                                                     original, detector).components), 1)


if __name__ == "__main__":
    unittest.main()
