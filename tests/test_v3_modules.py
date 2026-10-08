import unittest
import numpy as np
from pcb.schema import Component
from pcb.pin_detection import simple_terminals, KeypointProvider
from pcb.pin_semantics import assign_pin_semantics
from pcb.component_detection import source_from_name


class V3Modules(unittest.TestCase):
    def test_vertical_two_terminal_contract(self):
        c = Component("C1", "c", (10, 20, 26, 40), body_bbox=(10, 20, 26, 40))
        terms = simple_terminals(c)
        pins, _ = assign_pin_semantics(c, terms, [])
        self.assertEqual([(p.number, p.name) for p in pins], [("1", "1"), ("2", "2")])
        self.assertEqual(pins[0].tip, (18, 50))
        self.assertEqual(pins[1].tip, (18, 10))

    def test_horizontal_two_terminal_contract(self):
        c = Component("R1", "r", (10, 20, 30, 36), body_bbox=(10, 20, 30, 36))
        pins, _ = assign_pin_semantics(c, simple_terminals(c), [])
        self.assertEqual(pins[0].tip, (0, 28))
        self.assertEqual(pins[1].tip, (40, 28))

    def test_hawp_provider_is_disabled_interface(self):
        provider = KeypointProvider()
        self.assertEqual(provider.name, "disabled")
        self.assertEqual(provider.candidates(np.zeros((10, 10, 3), np.uint8)), [])

    def test_source_style(self):
        self.assertEqual(source_from_name("0002-Datasheet.png"), "Datasheet")


if __name__ == "__main__":
    unittest.main()
