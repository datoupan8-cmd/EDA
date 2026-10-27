import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from pcb.topology_stage.current150_freeze import FreezeGuard, valid_receipt, progress_callback
from pcb.topology_stage.power_comparison import sha


class CurrentFreezeSafety(unittest.TestCase):
    def test_progress_accepts_stage_and_message_keyword(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            progress_callback('0001')('recovery_text_progress', message='rotation views 1/10')
        event = json.loads(output.getvalue())
        self.assertEqual(event['stage'], 'recovery_text_progress')
        self.assertEqual(event['message'], 'rotation views 1/10')

    def test_inference_guard_blocks_target_and_reserved_case(self):
        guard = object.__new__(FreezeGuard)
        guard.actual_ids = set()
        guard.phase = 'inference'
        guard.allowed_target = None
        guard.target_opens = guard.blocked = 0
        guard.audit('open', ('C:/data/200_train_cases/0001/0001.png', 'rb'))
        self.assertEqual(guard.actual_ids, {'0001'})
        for path in ('C:/data/200_train_cases/0001/0001_target.json',
                     'C:/data/200_train_cases/0151/0151.png',
                     'C:/data/10_GTcase/sample.png'):
            with self.assertRaises(PermissionError):
                guard.audit('open', (path, 'rb'))
        self.assertEqual(guard.target_opens, 0)

    def test_resume_refuses_changed_version_image_or_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'input.png'
            image.write_bytes(b'image-v1')
            folder = root/'0001'
            folder.mkdir()
            names = {'scene': 'scene.json', 'mask': 'wire_mask.png', 'prediction': 'result.json'}
            for name in names.values():
                (folder/name).write_bytes(b'fixed-output')
            receipt = {'case_id': '0001', 'version_key': 'frozen', 'image_sha256': sha(image),
                       **{k: sha(folder/v) for k, v in names.items()}}
            (folder/'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
            self.assertEqual(valid_receipt(folder, '0001', 'frozen', image), receipt)
            with self.assertRaises(ValueError):
                valid_receipt(folder, '0001', 'other-version', image)
            (folder/'result.json').write_bytes(b'changed-output')
            with self.assertRaises(ValueError):
                valid_receipt(folder, '0001', 'frozen', image)
            (folder/'result.json').write_bytes(b'fixed-output')
            image.write_bytes(b'image-v2')
            with self.assertRaises(ValueError):
                valid_receipt(folder, '0001', 'frozen', image)


if __name__ == '__main__':
    unittest.main()
