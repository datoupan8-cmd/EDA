import copy
import unittest

from pcb.component.recovery.power_recovery import recover_power_values, restore_voltage_symbols
from pcb.component.recovery.tokens import OcrToken


def scene(components):
    return {"components": components, "pins": {key: {} for key in components}, "nets": {}}


def component(kind="gnd", bbox=(40, 40, 50, 50), value=None):
    return {"Name": "original", "type": kind, "bbox": list(bbox), "value": value}


class PowerRecoveryTests(unittest.TestCase):
    def test_visible_label_and_coordinates_without_mutating_input(self):
        original = scene({"g": component()})
        saved = copy.deepcopy(original)
        result, events = recover_power_values(original, [OcrToken("GND", (38, 52, 52, 60), .99)], 100)
        self.assertEqual(result["components"]["g"]["value"], "GND")
        self.assertEqual(original, saved)
        self.assertEqual(len(events), 1)

    def test_absent_far_or_wrong_kind_label_keeps_null(self):
        for tokens in ([], [OcrToken("GND", (200, 200, 220, 210), .99)],
                       [OcrToken("3.3V", (38, 52, 52, 60), .99)]):
            result, events = recover_power_values(scene({"g": component()}), tokens, 100)
            self.assertIsNone(result["components"]["g"]["value"])
            self.assertEqual(events, [])

    def test_overlapping_ocr_alternatives_have_one_physical_owner(self):
        original = scene({"g1": component(), "g2": component(bbox=(53, 40, 63, 50))})
        tokens = [OcrToken("GND", (44, 52, 58, 60), .99), OcrToken("AGND", (44, 52, 58, 60), .98)]
        result, events = recover_power_values(original, tokens, 100)
        self.assertEqual(len(events), 1)
        self.assertEqual(sum(c["value"] is not None for c in result["components"].values()), 1)

    def test_existing_nonnull_field_reserves_its_text(self):
        original = scene({"g1": component(value="GND"), "g2": component(bbox=(53, 40, 63, 50))})
        result, events = recover_power_values(original, [OcrToken("GND", (38, 52, 52, 60), .99)], 100)
        self.assertEqual(result, original)
        self.assertEqual(events, [])

    def test_chip_pin_label_cannot_be_stolen_by_ground_symbol(self):
        original = scene({"g": component(), "U1": component("box", (30, 35, 70, 60))})
        result, _ = recover_power_values(original, [OcrToken("GND", (38, 52, 52, 60), .99)], 100)
        self.assertIsNone(result["components"]["g"]["value"])

    def test_restore_preserves_existing_pin_and_net_content(self):
        original = scene({"VREC_0001": component("box")})
        original["pins"]["VREC_0001"] = {"pin_1": {"pinname": "A", "point": {"x": 40, "y": 40}}}
        original["nets"] = {"n": {"hyperGraph": "(VREC_0001.1)", "edges": {}}}
        candidates = [{"type": "v", "bbox": [10, 10, 20, 20], "value": "3.3V"},
                      {"type": "r", "bbox": [30, 30, 40, 40], "value": "1k"}]
        result, events = restore_voltage_symbols(original, candidates, 100)
        self.assertEqual(len(events), 1)
        self.assertEqual(result["components"]["VREC_0002"]["bbox"], [10, 80, 20, 90])
        self.assertEqual(result["pins"]["VREC_0001"], original["pins"]["VREC_0001"])
        self.assertEqual(result["nets"], original["nets"])
        self.assertEqual(set(original["components"]), {"VREC_0001"})
        again, new = restore_voltage_symbols(result, candidates, 100)
        self.assertEqual(again, result)
        self.assertEqual(new, [])


if __name__ == "__main__":
    unittest.main()
