"""L36 contracts and independent tiny brute-force optimality checks."""
import copy
from functools import lru_cache
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import numpy as np
import pin_side_joint_l36 as decoder
from pcb import pin_semantics_v6 as ordered
from pcb.schema import Component, Text
from pcb.text_detection import TokenRole


class ThreeSequenceTests(unittest.TestCase):
    def test_pair_scores_and_free_gaps(self):
        ns = np.array([[3., 0., 0.], [0., 0., 4.]])
        ss = np.array([[2., 0.], [0., 2.]])
        out, detail = decoder.align_three(ns, ss, np.ones((3, 2), bool))
        self.assertEqual(out, {0: (0, 0, 5.5), 1: (2, 1, 6.5)})
        self.assertEqual(detail['objective'], 12.)

    def test_missing_name_does_not_shift_following_pair(self):
        ns = np.eye(3) * 3
        ss = np.array([[0., 0.], [2., 0.], [0., 2.]])
        out, _ = decoder.align_three(ns, ss, np.ones((3, 2), bool))
        self.assertEqual(out, {0: (0, None, 3.), 1: (1, 0, 5.5), 2: (2, 1, 5.5)})

    def test_no_terminal_match_is_not_invented(self):
        out, _ = decoder.align_three(np.zeros((2, 1)), np.zeros((2, 1)), np.ones((1, 1), bool))
        self.assertEqual(out, {})

    def test_physical_words_consumed_once(self):
        out, _ = decoder.align_three(np.ones((3, 1)), np.ones((3, 1)), np.ones((1, 1), bool))
        self.assertEqual(len(out), 1)
        self.assertEqual(next(iter(out.values())), (0, 0, 2.5))

    def test_invalid_pair_can_only_match_one_role(self):
        out, _ = decoder.align_three(np.array([[3.]]), np.array([[2.]]), np.array([[False]]))
        self.assertEqual(out, {0: (0, None, 3.)})

    def test_prepaired_wrong_number_can_be_skipped(self):
        out, _ = decoder.align_three(np.array([[3., 0.]]), np.array([[2.]]), np.ones((2, 1), bool))
        self.assertEqual(out, {0: (0, 0, 5.5)})

    def test_order_not_numeric_string_order(self):
        out, _ = decoder.align_three(np.eye(2) * 3, np.eye(2) * 2, np.ones((2, 2), bool))
        self.assertEqual([out[i][:2] for i in range(2)], [(0, 0), (1, 1)])

    def test_name_only_and_number_only(self):
        out, _ = decoder.align_three(np.zeros((2, 0)), np.eye(2), np.zeros((0, 2), bool))
        self.assertEqual(out, {0: (None, 0, 1.), 1: (None, 1, 1.)})
        out, _ = decoder.align_three(np.eye(2), np.zeros((2, 0)), np.zeros((2, 0), bool))
        self.assertEqual(out, {0: (0, None, 1.), 1: (1, None, 1.)})

    def test_empty_axes(self):
        for n, m, k in ((0, 2, 3), (2, 0, 0), (0, 0, 0)):
            out, _ = decoder.align_three(np.zeros((n, m)), np.zeros((n, k)), np.zeros((m, k), bool))
            self.assertEqual(out, {})

    def test_invalid_dimensions_and_nonfinite(self):
        for ns, ss, pair in (([[1]], [[1], [2]], [[True]]),
                             ([[float('nan')]], [[1]], [[True]]),
                             ([[1]], [[1]], [[True, True]])):
            with self.assertRaises(ValueError):
                decoder.align_three(np.array(ns), np.array(ss), np.array(pair))

    def test_capacity_fallback_explicit(self):
        out, detail = decoder.align_three(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2), bool),
                                         decoder.DecodePolicy(max_states=1))
        self.assertEqual(out, {})
        self.assertEqual(detail['status'], 'capacity_fallback')

    def test_deterministic_and_input_immutable(self):
        ns, ss, pair = np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2), bool)
        originals = [x.copy() for x in (ns, ss, pair)]
        self.assertEqual(decoder.align_three(ns, ss, pair), decoder.align_three(ns, ss, pair))
        for x, original in zip((ns, ss, pair), originals):
            np.testing.assert_array_equal(x, original)

    def test_objective_matches_independent_exhaustive_search(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            ns, ss = rng.integers(0, 5, (2, 3)), rng.integers(0, 5, (2, 2))
            pair = rng.integers(0, 2, (3, 2)).astype(bool)
            @lru_cache(None)
            def solve(i, j, k):
                choices = [0.]
                if i < 2: choices.append(solve(i + 1, j, k))
                if j < 3: choices.append(solve(i, j + 1, k))
                if k < 2: choices.append(solve(i, j, k + 1))
                if i < 2 and j < 3 and ns[i, j] > 0:
                    choices.append(float(ns[i, j]) + solve(i + 1, j + 1, k))
                if i < 2 and k < 2 and ss[i, k] > 0:
                    choices.append(float(ss[i, k]) + solve(i + 1, j, k + 1))
                if i < 2 and j < 3 and k < 2 and pair[j, k] and ns[i, j] > 0 and ss[i, k] > 0:
                    choices.append(float(ns[i, j] + ss[i, k]) + .5 + solve(i + 1, j + 1, k + 1))
                return max(choices)
            _, detail = decoder.align_three(ns, ss, pair)
            self.assertEqual(detail['objective'], solve(0, 0, 0))


class AdapterTests(unittest.TestCase):
    def roles(self):
        return [TokenRole(0, Text('8', (87., 37., 94., 43.), .99, 'L3_word:n'), 'PIN_NUMBER', '8', confidence=.99),
                TokenRole(1, Text('VBAT', (105., 37., 130., 43.), .99, 'L3_word:s'), 'PIN_NAME', 'VBAT', confidence=.99)]

    def test_original_functions_restored_after_exception(self):
        old = (ordered._pair_number_names, ordered._assign_side, ordered._number_score)
        with self.assertRaises(RuntimeError):
            with decoder.joint_adapter([]):
                raise RuntimeError('deliberate')
        self.assertEqual(old, (ordered._pair_number_names, ordered._assign_side, ordered._number_score))

    def test_full_output_geometry_and_source_strings(self):
        c = Component('U1', 'box', (100., 20., 200., 80.), body_bbox=(100., 20., 200., 80.))
        terms = [{'tip': (90., 40.), 'base': (100., 40.), 'side': 'left', 'method': 'frozen'}]
        original = copy.deepcopy((c, terms, self.roles()))
        with decoder.joint_adapter(trace := []):
            pins, events = ordered.assign_pin_semantics_v6(c, terms, self.roles())
        self.assertEqual(len(pins), len(terms))
        self.assertEqual((pins[0].tip, pins[0].base, pins[0].side), ((90., 40.), (100., 40.), 'left'))
        self.assertEqual((pins[0].number, pins[0].name), ('8', 'VBAT'))
        self.assertEqual(events[0]['method'], 'frozen')
        self.assertEqual(trace[0]['selected'][0]['number_word_id'], 'L3_word:n')
        self.assertEqual((c, terms, self.roles()), original)

    def test_unresolved_abstention_unchanged(self):
        c = Component('UNRESOLVED_1', 'box', (100., 20., 200., 80.))
        terms = [{'tip': (90., 40.), 'base': (100., 40.), 'side': 'left'}]
        with decoder.joint_adapter([]):
            pins, _ = ordered.assign_pin_semantics_v6(c, terms, self.roles())
        self.assertFalse(pins[0].exportable)

    def test_nonbox_algorithm_bit_equivalent(self):
        c = Component('R1', 'r', (100., 20., 200., 80.))
        terms = [{'tip': (90., 40.), 'base': (100., 40.), 'side': 'left'},
                 {'tip': (210., 40.), 'base': (200., 40.), 'side': 'right'}]
        old = ordered.assign_pin_semantics_v6(c, terms, [])
        with decoder.joint_adapter([]):
            new = ordered.assign_pin_semantics_v6(c, terms, [])
        self.assertEqual(old, new)

    def test_capacity_fallback_preserves_original_side(self):
        c = Component('U1', 'box', (100., 20., 200., 80.))
        terms = [{'tip': (90., 40.), 'base': (100., 40.), 'side': 'left'}]
        old = ordered.assign_pin_semantics_v6(c, terms, self.roles())
        with decoder.joint_adapter(trace := [], decoder.DecodePolicy(max_states=1)):
            new = ordered.assign_pin_semantics_v6(c, terms, self.roles())
        self.assertEqual(old[0], new[0])
        self.assertEqual(trace[0]['status'], 'capacity_fallback')


class ProtocolTests(unittest.TestCase):
    def test_reserved_and_inference_targets_denied(self):
        from tools.run_pin_side_joint_l36 import AccessGuard
        guard = AccessGuard(install=False)
        for p in ('C:/data/200_train_cases/0151/a.png', 'C:/data/Golden/a.json',
                  'C:/EDA_Pin_Crossing_QuickTest/test.json', 'C:/data/200_train_cases/0014/a_target.json'):
            with self.assertRaises(PermissionError): guard.audit('open', (p, 'r'))
        guard.audit('open', ('C:/data/200_train_cases/0014/a.png', 'r'))

    def test_only_fixed_35_cases_accepted(self):
        from tools.run_pin_side_joint_l36 import source_paths, ALL
        self.assertEqual(len(set(ALL)), 35)
        for cid in ('0151', '0001', '0016'):
            with self.assertRaises(ValueError): source_paths(cid)

    def test_gate_requires_actual_gain_and_no_losses(self):
        from tools.run_pin_side_joint_l36 import value_gate
        a = {'tp': 100, 'f1': .2, 'macro_f1': .2}
        b = {'tp': 125, 'f1': .25, 'macro_f1': .21}
        rows = [{'before': {'f1': .2}, 'after': {'f1': .3}}] * 3
        self.assertTrue(value_gate(a, b, rows, [], {})['system_passed'])
        self.assertFalse(value_gate(a, b, rows, ['old'], {})['Pin_passed'])
        self.assertFalse(value_gate(a, b, rows, [], {'incorrect_added': 1})['system_passed'])
        self.assertFalse(value_gate(a, a, rows, [], {})['Pin_passed'])


if __name__ == '__main__':
    unittest.main()
