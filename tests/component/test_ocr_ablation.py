from itertools import product
from types import SimpleNamespace
import math
import unittest

import numpy as np

from pcb.component.ocr_ablation import (
    BeamConfig, PairedCTCDecoder, prefix_beam_search, sequence_probability, role_anchor, valid_role,
    PairedPPOcrV5Backend, TextStagePPOcrV5Ablation, TextStagePPOcrV5Beam,
)


def collapse(path):
    return tuple(c for i, c in enumerate(path) if c and (i == 0 or c != path[i-1]))


class OCRDecoderTests(unittest.TestCase):
    def test_beam_and_exact_forward_match_exhaustive_ctc_paths(self):
        probs = np.array([[.2, .6, .2], [.5, .3, .2], [.2, .3, .5]])
        expected = {}
        for path in product(range(3), repeat=3):
            labels = collapse(path)
            expected[labels] = expected.get(labels, 0.) + math.prod(probs[t, c] for t, c in enumerate(path))
        actual = dict(prefix_beam_search(probs, BeamConfig(width=100, top_k=3)))
        for labels, probability in expected.items():
            self.assertAlmostEqual(math.exp(actual[labels]), probability)
            if labels:
                score, confidence = sequence_probability(probs, labels)
                self.assertAlmostEqual(math.exp(score), probability)
                self.assertGreaterEqual(confidence, 0.)
                self.assertLessEqual(confidence, 1.)

    def test_repeated_labels_require_blank(self):
        score, _ = sequence_probability(np.array([[.01, .99], [.01, .99]]), (1, 1))
        self.assertEqual(score, float("-inf"))
        score, _ = sequence_probability(np.array([[0., 1.], [1., 0.], [0., 1.]]), (1, 1))
        self.assertAlmostEqual(math.exp(score), 1.)

    def test_sequence_mass_can_beat_greedy_path_without_lexical_replacement(self):
        decoder = PairedCTCDecoder(SimpleNamespace(character=["blank", "A", "B"]), BeamConfig(confidence_gate=1.))
        probs = np.array([[.47, .03, .50], [.52, .11, .37], [.27, .16, .57]])
        chosen, confidence, event = decoder.choose(probs, "BB", .535)
        self.assertEqual(chosen, "B")
        self.assertIsNotNone(event)
        self.assertGreater(event["log_probability_delta"], 0.)
        self.assertLessEqual(confidence, .535)

    def test_high_confidence_and_nonlatin_are_preserved(self):
        decoder = PairedCTCDecoder(SimpleNamespace(character=["blank", "A"]))
        probs = np.array([[.1, .9], [.1, .9]])
        self.assertEqual(decoder.choose(probs, "R12", .99), ("R12", .99, None))
        self.assertEqual(decoder.choose(probs, "电阻", .7), ("电阻", .7, None))

    def test_anchor_cannot_change_device_prefix(self):
        anchor = role_anchor("Rl2")
        self.assertEqual(anchor, ("designator", "R"))
        self.assertTrue(valid_role("R12", anchor))
        self.assertFalse(valid_role("C12", anchor))
        self.assertTrue(valid_role("10k", role_anchor("10k")))

    def test_visual_probability_cannot_be_overruled_by_format_prior(self):
        decoder = PairedCTCDecoder(SimpleNamespace(character=["blank", "R", "l", "1", "2"]),
                                  BeamConfig(confidence_gate=1.))
        probs = np.array([[.005, .98, .005, .005, .005],
                          [.005, .005, .90, .085, .005],
                          [.005, .005, .005, .005, .98]])
        text, _, event = decoder.choose(probs, "Rl2", .9)
        self.assertEqual(text, "Rl2")
        self.assertIsNone(event)

    def test_paired_lookup_is_keyed_by_score_and_retains_greedy_api(self):
        class Greedy:
            character = ["blank", "A", "B"]
            def __call__(self, predictions, return_word_box=False, **kwargs):
                return [("BB", .535), ("BB", .99)], [None, None]
        decoder = PairedCTCDecoder(Greedy(), BeamConfig(confidence_gate=1.))
        probs = np.array([[.47, .03, .50], [.52, .11, .37], [.27, .16, .57]])
        lines, _ = decoder(np.stack([probs, probs]))
        self.assertEqual(lines, [("BB", .535), ("BB", .99)])
        self.assertIn(("BB", .535), decoder.readings)
        self.assertIn(("BB", .99), decoder.readings)
        self.assertEqual(decoder.readings[("BB", .535)][0], "B")

    def test_dev_selector_rejects_reserved_and_duplicate_ids_before_access(self):
        from tools.evaluate_component_local_joint import select_cases
        from pathlib import Path
        for ids in ("151", "200", "1,1", "0", "1,151"):
            with self.subTest(ids=ids), self.assertRaises(PermissionError):
                select_cases(Path("nonexistent/200_train_cases"), ids)

    def test_cuda_fallback_remains_an_error_on_repeated_calls(self):
        session = SimpleNamespace(session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
        backend = object.__new__(PairedPPOcrV5Backend)
        backend.engine = SimpleNamespace(text_det=SimpleNamespace(session=session),
                                         text_rec=SimpleNamespace(session=session, postprocess_op=SimpleNamespace(character=[])))
        backend.decoder = None
        backend.normalization = "legacy"
        backend.use_cuda = True
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "fell back"):
                backend._ensure_engine()

    def test_text_profiles_dispatch_to_the_selected_decoder_without_resource_mutation(self):
        context = SimpleNamespace(resources={})
        tokens = [SimpleNamespace(text="R1")]
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        stage = TextStagePPOcrV5Ablation()
        stage.backend = SimpleNamespace(recognize=lambda _: tokens, last_diagnostics={})
        self.assertIs(stage.run(image, context).texts, tokens)
        stage = TextStagePPOcrV5Beam()
        stage.backend = SimpleNamespace(recognize_pair=lambda _: {"greedy": [], "beam": tokens}, last_diagnostics={})
        self.assertIs(stage.run(image, context).texts, tokens)
        self.assertEqual(context.resources, {})


if __name__ == "__main__":
    unittest.main()
