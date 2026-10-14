from dataclasses import asdict
import copy
import unittest

from pcb.component.contextual_values import contextual_value_roles, refine_contextual_values
from pcb.component.latest.text_assignment import assign_values
from pcb.component.latest.text_roles import classify_tokens
from pcb.core.interfaces import ComponentStageOutput
from pcb.schema import Component, Text


class ContextualValueTests(unittest.TestCase):
    def original(self, components, texts, assignments):
        roles = classify_tokens(texts)
        provenance = [{"component": component.key} for component in components]
        designators = {index: next(role for role in roles if role.designator == key)
                       for index, key in assignments.items()}
        values = assign_values(components, roles, provenance, designators=designators)
        diagnostics = {"component_provenance": provenance, "value_assignment": values,
                       "designator_assignment": {"accepted": [
                           {"proposal_index": index, "token": role.token.text}
                           for index, role in designators.items()]}}
        return ComponentStageOutput(components, texts, roles, diagnostics)

    def test_numeric_alternative_to_designator_is_removed_from_value_assignment(self):
        original = self.original([Component("R3", "r", (100, 100, 120, 116), name="R3")],
            [Text("R3", (100, 87, 115, 97), .99), Text("33", (100, 87, 115, 97), .70),
             Text("10kΩ", (120, 82, 150, 94), .98)], {0: "R3"})
        self.assertEqual(original.components[0].value, "33")
        before = copy.deepcopy(asdict(original.components[0]))
        output = refine_contextual_values(original)
        self.assertEqual(output.components[0].value, "10kΩ")
        after = asdict(output.components[0])
        before.pop("value")
        after.pop("value")
        self.assertEqual(after, before)
        self.assertEqual(original.components[0].value, "33")
        self.assertIs(output.roles, original.roles)

    def test_separate_numeric_resistance_is_preserved(self):
        roles = classify_tokens([Text("R1", (10, 10, 22, 20), .99),
                                 Text("330", (10, 22, 30, 32), .95)])
        output, rejected = contextual_value_roles(roles, [])
        self.assertEqual(rejected, [])
        self.assertEqual(output, roles)

    def test_stronger_numeric_reading_is_not_overridden(self):
        roles = classify_tokens([Text("R1", (10, 10, 30, 20), .86),
                                 Text("31", (10, 10, 30, 20), .97)])
        self.assertEqual(contextual_value_roles(roles, [])[1], [])

    def test_low_confidence_text_cannot_discard_real_value(self):
        roles = classify_tokens([Text("R1", (10, 10, 30, 20), .70),
                                 Text("31", (10, 10, 30, 20), .60)])
        self.assertEqual(contextual_value_roles(roles, [])[1], [])

    def test_explicit_unit_value_is_not_rejected_by_numeric_conflict_rule(self):
        roles = classify_tokens([Text("R1", (10, 10, 30, 20), .99),
                                 Text("31k", (10, 10, 30, 20), .70)])
        self.assertEqual(contextual_value_roles(roles, [])[1], [])

    def test_signal_alternative_rejects_numeric_fragment(self):
        roles = classify_tokens([Text("GPIO1", (10, 10, 40, 20), .99),
                                 Text("1", (25, 10, 40, 20), .70)])
        output, rejected = contextual_value_roles(roles, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(output[1].role, "PIN_NUMBER")

    def test_pin_box_ablation_only_rejects_bare_interior_numbers(self):
        components = [Component("U1", "box", (100, 100, 200, 200))]
        roles = classify_tokens([Text("12", (105, 140, 120, 150), .99),
                                 Text("12k", (130, 140, 150, 150), .99),
                                 Text("330", (205, 140, 230, 150), .99)])
        output, rejected = contextual_value_roles(roles, components, conflicts=False, pin_boxes=True)
        self.assertEqual([row["text"] for row in rejected], ["12"])
        self.assertEqual([role.role for role in output], ["PIN_NUMBER", "VALUE", "VALUE"])

    def test_reassignment_preserves_explicit_designator_value_pair(self):
        original = self.original([
            Component("R1", "r", (10, 40, 30, 56), name="R1"),
            Component("R3", "r", (100, 100, 120, 116), name="R3")],
            [Text("R1 10kΩ", (10, 20, 55, 32), .99), Text("R3", (100, 87, 115, 97), .99),
             Text("33", (100, 87, 115, 97), .70), Text("22kΩ", (120, 82, 150, 94), .98)],
            {0: "R1", 1: "R3"})
        output = refine_contextual_values(original)
        self.assertEqual(output.components[0].value, "10kΩ")
        self.assertEqual(output.components[1].value, "22kΩ")


if __name__ == "__main__":
    unittest.main()
