"""Acceptance runner accounting/boundary tests; no dataset, OCR or model run."""
import unittest
from tools.validate_pin_combined_l39 import check_ids, type_metrics
from tools.run_pin_combined_l38 import Guard


class FullAcceptanceTests(unittest.TestCase):
    def test_exact_150_ids(self):
        check_ids([{'case_id': f'{n:04d}'} for n in range(1, 151)])

    def test_partial_set_is_not_success(self):
        with self.assertRaises(AssertionError): check_ids([{'case_id': '0001'}])

    def test_duplicate_is_not_success(self):
        with self.assertRaises(AssertionError):
            check_ids([{'case_id': '0001'}, {'case_id': '0001'}], ('0001',))

    def test_sealed_is_never_permitted(self):
        with self.assertRaises(PermissionError):
            check_ids([{'case_id': '0151'}], ('0151',))

    def test_type_tp_is_gt_owner_type(self):
        pred = {'components': {'U1': {'type': 'box'}}, 'pins': {'U1': {'pin_1': {}}}}
        target = {'components': {'U1': {'type': 'box'}}, 'pins': {'U1': {'pin_1': {}, 'pin_2': {}}}}
        self.assertEqual(type_metrics(pred, target, {('U1', 'pin_1')})['box'],
                         {'tp': 1, 'pred': 1, 'gt': 2})

    def test_inference_gt_guard(self):
        guard = Guard(install=False); guard.phase = 'inference'
        with self.assertRaises(PermissionError):
            guard.audit('open', ('C:/data/200_train_cases/0001/0001_target.json', 'r'))


if __name__ == '__main__': unittest.main()
