from __future__ import annotations

import unittest

from pcb.component.latest.text_roles import classify_tokens
from pcb.component.value_groups_v2 import (
    annotation_strength, filter_value_evidence, panel_edge_number,
    refine_guarded_values, select_literal,
)
from pcb.core.interfaces import ComponentStageOutput
from pcb.schema import Component, Text


def scene(components, texts):
    roles = classify_tokens(texts)
    accepted = [{"proposal_index": i, "token": role.token.text, "score": .5}
                for i, component in enumerate(components) for role in roles
                if role.role == "DESIGNATOR" and role.designator == component.key]
    diagnostics = {"component_provenance": [{"component": c.key} for c in components],
                   "designator_assignment": {"accepted": accepted}}
    return ComponentStageOutput(components, texts, roles, diagnostics, {})


class GuardedValuesTests(unittest.TestCase):
    def test_ohm_literal_beats_zero_alternative_without_generating_text(self):
        roles = classify_tokens([Text("3000", (10, 10, 40, 25), .99),
                                 Text("300Ω", (12, 11, 38, 23), .90)])
        component = Component("R1", "r", (10, 30, 40, 40))
        self.assertEqual(select_literal(component, roles).token.text, "300Ω")

    def test_complete_capacitor_unit_beats_truncated_same_region(self):
        roles = classify_tokens([Text("4.7u", (10, 10, 30, 25), .99),
                                 Text("4.7uF", (9, 10, 38, 25), .98)])
        self.assertEqual(select_literal(Component("C1", "c", (10, 30, 30, 40)), roles).token.text, "4.7uF")

    def test_number_column_is_negative_evidence_only_near_a_panel(self):
        roles = classify_tokens([Text("10", (30, 30, 38, 40)), Text("11", (30, 15, 38, 25)),
                                 Text("100", (130, 30, 149, 40))])
        panel = Component("U1", "box", (40, 10, 100, 90))
        self.assertTrue(panel_edge_number(roles[0], [panel], roles))
        self.assertFalse(panel_edge_number(roles[2], [panel], roles))

    def test_assigned_designator_text_cannot_also_supply_its_numeric_suffix(self):
        roles = classify_tokens([Text("R1", (10, 10, 30, 25), .9),
                                 Text("1", (15, 10, 28, 25), .99)])
        kept, rejected = filter_value_evidence(roles, [], {0: roles[0]})
        self.assertEqual(kept, [])
        self.assertEqual(rejected[0]["reason"], "physical_text_already_owned_by_designator")

    def test_reverse_layout_is_penalized_but_never_forbidden(self):
        anchor = Text("R1", (10, 30, 25, 40))
        above, below = Text("10k", (10, 12, 30, 22)), Text("10k", (10, 48, 30, 58))
        self.assertGreater(annotation_strength(anchor, below, ordered=True), annotation_strength(anchor, above, ordered=True))
        self.assertGreater(annotation_strength(anchor, above, ordered=True), 0)

    def test_overlapping_rectangles_of_separate_printed_lines_can_form_group(self):
        anchor = Text("R41", (12, 10, 33, 24))
        value = Text("1kΩ", (12, 19, 35, 34))
        self.assertGreater(annotation_strength(anchor, value, ordered=True), .8)
        # Alternate readings at one baseline remain mutually exclusive.
        self.assertEqual(annotation_strength(anchor, Text("1", anchor.bbox), ordered=True), 0)

    def test_shorthand_physical_region_is_not_filled_twice(self):
        original = scene([Component("C1", "c", (10, 35, 25, 50)), Component("C2", "c", (30, 35, 45, 50))],
                         [Text("4.7u", (10, 10, 38, 25), .98)])
        result = refine_guarded_values(original)
        self.assertEqual(sum(c.value is not None for c in result.components), 1)

    def test_glued_designator_value_is_kept_and_input_is_immutable(self):
        original = scene([Component("R1", "r", (10, 40, 25, 60), name="literal_name")],
                         [Text("R1 10k", (10, 10, 55, 25), .98)])
        result = refine_guarded_values(original)
        self.assertEqual(result.components[0].value, "10k")
        self.assertEqual(result.components[0].name, "literal_name")
        self.assertIsNone(original.components[0].value)
        self.assertIs(result.texts, original.texts)
        self.assertIs(result.roles, original.roles)

    def test_separate_ocr_read_of_designator_tail_cannot_fill_another_component(self):
        original = scene([Component("R1", "r", (10, 40, 25, 60)),
                          Component("R2", "r", (40, 40, 55, 60))],
                         [Text("R1 10k", (10, 10, 55, 25), .98),
                          Text("10k", (28, 10, 55, 25), .99)])
        result = refine_guarded_values(original)
        self.assertEqual(result.components[0].value, "10k")
        self.assertIsNone(result.components[1].value)


if __name__ == "__main__":
    unittest.main()
