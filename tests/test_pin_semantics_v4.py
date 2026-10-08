import copy
import unittest
from types import SimpleNamespace

from pcb.pin_semantics import assign_pin_semantics
from pcb.pin_semantics_v4 import assign_pin_semantics_v4, assign_role_owners
from pcb.pin.semantics.v4 import PinSemanticsStageV4
from pcb.core.interfaces import ComponentStageOutput, PinLocalizationOutput
from pcb.schema import Component, Text
from pcb.text_detection import TokenRole, classify_tokens


def label(text, x, y, role="PIN_NUMBER", index=0, width=8, height=10):
    return TokenRole(index, Text(text, (x-width/2, y-height/2, x+width/2, y+height/2)),
                     role, text, confidence=.95)


def terminal(y, side="left", x=None):
    if side == "left": return {"tip": (80, y), "base": (100, y), "side": side}
    if side == "right": return {"tip": (220, y), "base": (200, y), "side": side}
    if side == "top": return {"tip": (x or 140, 80), "base": (x or 140, 100), "side": side}
    return {"tip": (x or 140, 220), "base": (x or 140, 200), "side": side}


class PinSemanticsV4Tests(unittest.TestCase):
    def setUp(self):
        self.component = Component("U1", "box", (100, 100, 200, 200))

    def run_semantics(self, terminals, roles):
        return assign_pin_semantics_v4(self.component, terminals, roles)

    def test_numeric_values_promoted_only_at_local_pin(self):
        roles = classify_tokens([Text("15", (88, 135, 96, 145))])
        self.assertEqual(roles[0].role, "VALUE")
        pins, events = self.run_semantics([terminal(140)], roles)
        self.assertEqual(pins[0].number, "15")
        self.assertTrue(pins[0].exportable)
        self.assertTrue(events[0]["promoted_numeric_value"])

    def test_inside_edge_number_accepted(self):
        pins, _ = self.run_semantics([terminal(140)], [label("2", 109, 140, "VALUE")])
        self.assertEqual(pins[0].observable_number, "2")

    def test_inside_letter_digit_is_name_outside_digit_is_number(self):
        for signal in ("A1", "A2", "P12"):
            roles = [label(signal, 109, 140, "PIN_NUMBER", index=0),
                     label("1", 90, 140, "VALUE", index=1)]
            pins, events = self.run_semantics([terminal(140)], roles)
            self.assertEqual(pins[0].number, "1")
            self.assertEqual(pins[0].name, signal)
            self.assertTrue(pins[0].exportable)
            self.assertTrue(events[0]["signal_label_reclassified"])

    def test_inside_letter_digit_without_number_remains_unknown_identity(self):
        pins, _ = self.run_semantics([terminal(140)], [label("A1", 109, 140)])
        self.assertEqual(pins[0].name, "A1")
        self.assertFalse(pins[0].exportable)
        self.assertIsNone(pins[0].observable_number)

    def test_outside_letter_digit_bga_number_still_accepted(self):
        pins, events = self.run_semantics([terminal(140)], [label("A1", 92, 140)])
        self.assertEqual(pins[0].number, "A1")
        self.assertTrue(pins[0].exportable)
        self.assertFalse(events[0]["signal_label_reclassified"])

    def test_far_number_not_stolen_from_same_row(self):
        pins, _ = self.run_semantics([terminal(140)], [label("2", 20, 140)])
        self.assertFalse(pins[0].exportable)
        self.assertIsNone(pins[0].observable_number)

    def test_central_model_number_rejected(self):
        pins, _ = self.run_semantics([terminal(140)], [label("123", 150, 140, "VALUE")])
        self.assertFalse(pins[0].exportable)

    def test_opposite_side_label_rejected(self):
        pins, _ = self.run_semantics([terminal(140)], [label("3", 208, 140)])
        self.assertFalse(pins[0].exportable)

    def test_orthogonal_side_and_dense_row_constraints(self):
        terms = [terminal(130), terminal(150), terminal(0, "top", 140)]
        roles = [label("1", 92, 130, index=1), label("2", 92, 150, index=2), label("3", 140, 92, index=3)]
        pins, _ = self.run_semantics(terms, roles)
        self.assertEqual([pin.number for pin in pins], ["1", "2", "3"])

    def test_duplicate_number_never_exported_twice(self):
        roles = [label("4", 92, 130, index=1), label("4", 92, 150, index=2)]
        pins, _ = self.run_semantics([terminal(130), terminal(150)], roles)
        self.assertEqual(sum(pin.exportable for pin in pins), 1)
        self.assertEqual(len({pin.number for pin in pins}), 2)

    def test_global_match_uses_second_candidate_not_drop(self):
        # The first row can use 4 or 5; the second row can only use 4.
        roles = [label("4", 92, 130, index=1), label("5", 89, 130, index=2),
                 label("4", 92, 150, index=3)]
        pins, _ = self.run_semantics([terminal(130), terminal(150)], roles)
        self.assertEqual([pin.number for pin in pins], ["5", "4"])
        self.assertTrue(all(pin.exportable for pin in pins))

    def test_unknowns_stable_under_terminal_order(self):
        terms = [terminal(160), terminal(120), terminal(140, "right")]
        pins, _ = self.run_semantics(terms, [])
        reverse, _ = self.run_semantics(list(reversed(terms)), [])
        self.assertEqual({pin.tip: pin.number for pin in pins}, {pin.tip: pin.number for pin in reverse})
        self.assertEqual([pin.number for pin in pins], ["UNK_left_2", "UNK_left_1", "UNK_right_1"])
        self.assertTrue(all(pin.inferred_number and not pin.exportable and pin.observable_number is None for pin in pins))

    def test_ground_identity_never_guessed(self):
        self.component.type = "gnd"
        pins, _ = self.run_semantics([terminal(140)], [label("17", 92, 140)])
        self.assertEqual(pins[0].number_source, "unobservable_internal_id")
        self.assertFalse(pins[0].exportable)

    def test_prior_for_two_terminal_component_unchanged(self):
        self.component.type = "r"
        terms = [terminal(140), terminal(140, "right")]
        self.assertEqual(self.run_semantics(terms, []), assign_pin_semantics(self.component, terms, []))

    def test_crystal_prior_unchanged(self):
        self.component.type = "crystal_4pin"
        terms = [terminal(130), terminal(150), terminal(130, "right"), terminal(150, "right")]
        self.assertEqual(self.run_semantics(terms, []), assign_pin_semantics(self.component, terms, []))

    def test_values_with_units_and_designators_not_numbers(self):
        for value, role in [("10K", "VALUE"), ("1uF", "VALUE"), ("R1", "DESIGNATOR")]:
            with self.subTest(value=value):
                pins, _ = self.run_semantics([terminal(140)], [label(value, 92, 140, role)])
                self.assertFalse(pins[0].exportable)

    def test_name_token_not_reused(self):
        pins, _ = self.run_semantics([terminal(139), terminal(145)], [label("VDD", 120, 142, "PIN_NAME", width=24)])
        self.assertLessEqual(sum(pin.name == "VDD" for pin in pins), 1)

    def test_input_not_mutated_and_empty_supported(self):
        terms, roles = [terminal(140)], [label("1", 92, 140)]
        before = copy.deepcopy((self.component, terms, roles))
        self.run_semantics(terms, roles)
        self.assertEqual((self.component, terms, roles), before)
        self.assertEqual(self.run_semantics([], roles), ([], []))

    def adjacent_components(self):
        first = Component("U1", "box", (100, 100, 200, 200))
        second = Component("U2", "box", (40, 100, 80, 200))
        return [(first, [terminal(140)]),
                (second, [{"tip": (100, 140), "base": (80, 140), "side": "right"}])]

    def test_same_token_has_only_one_component_owner_stable_under_reordering(self):
        localized = self.adjacent_components()
        roles = [label("10", 92, 140, "VALUE")]
        before = copy.deepcopy((localized, roles))
        owned, events = assign_role_owners(localized, roles)
        self.assertEqual(sum(map(len, owned)), 1)
        self.assertEqual(events[0]["owner_component"], "U1")
        self.assertEqual(events[0]["candidate_component_count"], 2)
        reverse_owned, reverse_events = assign_role_owners(list(reversed(localized)), roles)
        self.assertEqual(sum(map(len, reverse_owned)), 1)
        self.assertEqual(reverse_events[0]["owner_component"], "U1")
        self.assertEqual((localized, roles), before)

    def test_different_tokens_with_same_number_can_belong_to_different_chips(self):
        localized = self.adjacent_components()
        roles = [label("1", 96, 140, "VALUE", index=0), label("1", 84, 140, "VALUE", index=1)]
        owned, events = assign_role_owners(localized, roles)
        self.assertEqual([len(items) for items in owned], [1, 1])
        pins = [assign_pin_semantics_v4(c, ts, rs)[0][0] for (c, ts), rs in zip(localized, owned)]
        self.assertEqual([p.number for p in pins], ["1", "1"])
        self.assertTrue(all(p.exportable for p in pins))
        self.assertEqual({event["owner_component"] for event in events}, {"U1", "U2"})

    def test_duplicate_ocr_observation_stays_with_one_owner(self):
        localized = self.adjacent_components()
        roles = [label("10", 92, 140, "VALUE", index=0), label("10", 92, 140, "VALUE", index=1)]
        owned, events = assign_role_owners(localized, roles)
        self.assertEqual(sum(bool(items) for items in owned), 1)
        self.assertEqual(sum(map(len, owned)), 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["token_indices"], [0, 1])

    def test_wrapper_enforces_ownership_and_keeps_component_geometry(self):
        localized = self.adjacent_components()
        roles = [label("10", 92, 140, "VALUE")]
        components = [c for c, _ in localized]
        before = [(c.key, c.type, c.bbox, c.body_bbox, c.name, c.value) for c in components]
        context = SimpleNamespace(config=SimpleNamespace(legacy_pipeline_name="v3"), ocr=None, keypoint_provider=None)
        result = PinSemanticsStageV4().run(ComponentStageOutput(components, [r.token for r in roles], roles, {}),
                                         PinLocalizationOutput(localized), context)
        self.assertEqual(sum(p.exportable for c in result.components for p in c.pins), 1)
        self.assertEqual(sum(not p.exportable for c in result.components for p in c.pins), 1)
        self.assertEqual(result.diagnostics["shared_pin_token_candidates"], 1)
        self.assertEqual([(c.key, c.type, c.bbox, c.body_bbox, c.name, c.value) for c in result.components], before)

    def test_role_confidence_is_used_not_token_score(self):
        role = label("1", 92, 140, "VALUE")
        role.token.score = 0.0
        role.confidence = .93
        pins, _ = self.run_semantics([terminal(140)], [role])
        self.assertEqual(pins[0].number_confidence, .93)


if __name__ == "__main__":
    unittest.main()
