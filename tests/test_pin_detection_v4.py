"""Synthetic pixels only: no development or held-out dataset reads."""
import copy
import unittest

import cv2
import numpy as np

from pcb.pin_detection import terminal_candidates_v3
from pcb.pin_detection_v4 import pixel_evidence, refine_terminal, terminal_candidates_v4
from pcb.schema import Component, Text


class PinLocalizationV4Tests(unittest.TestCase):
    def fixture(self, side="left", length=12):
        image = np.full((180, 180, 3), 255, np.uint8)
        component = Component("U1", "box", (50, 50, 110, 110))
        nx, ny = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}[side]
        base = {"left": (50, 80), "right": (110, 80), "top": (80, 50), "bottom": (80, 110)}[side]
        end = (base[0] + nx * length, base[1] + ny * length)
        cv2.line(image, base, end, (0, 0, 0), 1)
        terminal = {"tip": (base[0] + nx * 7, base[1] + ny * 7), "base": base,
                    "side": side, "method": "boundary_wire_support", "wire_support_score": 1}
        return image, component, terminal, end

    def test_four_directions_short_endpoint(self):
        for side in ("left", "right", "top", "bottom"):
            with self.subTest(side=side):
                image, component, terminal, end = self.fixture(side)
                result = refine_terminal(terminal, component, pixel_evidence(image))
                self.assertEqual(result["tip"], end)
                self.assertEqual(result["base"], terminal["base"])
                self.assertEqual(result["method"], "short_stub_endpoint_v4")

    def test_continuous_net_is_not_traced_to_search_limit(self):
        image, component, terminal, _ = self.fixture(length=40)
        result = refine_terminal(terminal, component, pixel_evidence(image))
        self.assertEqual(result["tip"], terminal["tip"])
        self.assertEqual(result["localization_evidence"], "continuous_wire_no_visible_pin_end")

    def test_parallel_wire_is_not_adopted(self):
        image, component, terminal, end = self.fixture()
        cv2.line(image, (50, 83), (15, 83), (0, 0, 0), 1)
        result = refine_terminal(terminal, component, pixel_evidence(image))
        self.assertEqual(result["tip"], end)

    def test_text_and_image_boundaries_are_not_endpoints(self):
        image, component, terminal, _ = self.fixture()
        token = Text("8", (37, 78, 41, 82))
        result = refine_terminal(terminal, component, pixel_evidence(image, [token]))
        self.assertEqual(result["tip"], terminal["tip"])
        image = np.full((180, 180, 3), 255, np.uint8)
        cv2.line(image, (10, 80), (0, 80), (0, 0, 0), 1)
        terminal = {**terminal, "base": (10, 80), "tip": (3, 80)}
        result = refine_terminal(terminal, component, pixel_evidence(image))
        self.assertEqual(result["tip"], terminal["tip"])

    def test_single_pixel_gap_does_not_appear_to_be_endpoint(self):
        image, component, terminal, _ = self.fixture()
        image[80, 42] = 255
        result = refine_terminal(terminal, component, pixel_evidence(image))
        self.assertEqual(result["tip"], terminal["tip"])
        self.assertEqual(result["localization_evidence"], "pixel_gap")

    def test_body_gap_is_not_skipped(self):
        image, component, terminal, _ = self.fixture()
        image[80, 49] = 255
        result = refine_terminal(terminal, component, pixel_evidence(image))
        self.assertEqual(result["tip"], terminal["tip"])
        self.assertEqual(result["localization_evidence"], "detached_from_body")

    def test_disconnected_zigzag_inside_corridor_is_not_a_stub(self):
        image, component, terminal, _ = self.fixture()
        image[:] = 255
        image[80, 50] = 0
        for distance in range(1, 12):
            image[80 + (1 if distance % 2 else -1), 50 - distance] = 0
        result = refine_terminal(terminal, component, pixel_evidence(image))
        self.assertEqual(result["tip"], terminal["tip"])
        self.assertEqual(result["localization_evidence"], "disconnected_corridor_ink")

    def test_L_corner_and_T_junction_are_distinguished(self):
        for junction in (False, True):
            image, component, terminal, end = self.fixture()
            cv2.line(image, (end[0], 80 if not junction else 72), (end[0], 90), (0, 0, 0), 1)
            result = refine_terminal(terminal, component, pixel_evidence(image))
            if junction:
                self.assertEqual(result["tip"], terminal["tip"])
            else:
                self.assertEqual(result["tip"], end)
                self.assertEqual(result["method"], "short_stub_corner_v4")

    def test_offset_corner_stays_on_real_ink(self):
        image, component, terminal, _ = self.fixture()
        image[:] = 255
        cv2.line(image, (50, 79), (38, 79), (0, 0, 0), 1)
        cv2.line(image, (38, 79), (38, 72), (0, 0, 0), 1)
        result = refine_terminal(terminal, component, pixel_evidence(image))
        self.assertEqual(result["tip"], (38, 79))

    def test_audited_simple_symbols_keep_all_original_fields_and_order(self):
        image = np.full((180, 180, 3), 255, np.uint8)
        for kind in ("r", "c", "l", "d", "crystal_4pin", "gnd"):
            with self.subTest(kind=kind):
                component = Component("X1", kind, (60, 65, 100, 90))
                original = terminal_candidates_v3(image, component)
                result = terminal_candidates_v4(image, component)
                self.assertEqual(len(original), len(result))
                for old, new in zip(original, result):
                    self.assertEqual(old, {key: new[key] for key in old})

    def test_inputs_unchanged_and_repeatable(self):
        image, component, terminal, _ = self.fixture()
        old_image, old_component, old_terminal = image.copy(), copy.deepcopy(component), copy.deepcopy(terminal)
        first = refine_terminal(terminal, component, pixel_evidence(image))
        self.assertEqual(first, refine_terminal(terminal, component, pixel_evidence(image)))
        np.testing.assert_array_equal(old_image, image)
        self.assertEqual(component, old_component)
        self.assertEqual(terminal, old_terminal)


if __name__ == "__main__":
    unittest.main()
