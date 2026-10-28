import json
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pcb.topology_stage.current150_freeze import FreezeGuard, valid_receipt, progress_callback
from pcb.topology_stage import current150_freeze as freezer
from pcb.topology_stage.power_comparison import sha
from pcb.core.config import PipelineConfig


class CurrentFreezeSafety(unittest.TestCase):
    def test_missing_optional_seeds_allow_fresh_inference_but_unsafe_seeds_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(freezer, 'SEEDS', Path(tmp)/'seeds.json'):
            self.assertEqual(freezer.load_seed_manifest()['ACTUAL_CASE_IDS'], [])
            for ids, opens in ((['0001'], 1), (['0151'], 0), (['0001', '0001'], 0)):
                freezer.SEEDS.write_text(json.dumps({'ACTUAL_CASE_IDS': ids, 'GT_opens': opens}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    freezer.load_seed_manifest()

    def test_explicit_baseline_config_and_real_model_drift_are_checked(self):
        config = PipelineConfig(text='recovery_rotation_vertical_v1', component='rotation_vertical_n1_n2_v1',
                                pin_localization='l39', pin_semantics='l39', wire='v3_frame_guard', topology='v3')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root/'models/fixed.pt'
            model.parent.mkdir()
            model.write_bytes(b'original-model')
            digest = sha(model)
            with patch.object(freezer, 'ROOT', root), \
                 patch('pcb.core.team_config.load_team_config', return_value=config) as loader, \
                 patch('pcb.network_shared.selection.selected_versions', return_value={'net_hypergraph': 'legacy_v2', 'net_lines': 'legacy_v2'}), \
                 patch.object(freezer.subprocess, 'check_output', return_value=digest+' * models/fixed.pt\n'), \
                 patch.object(freezer, 'source_inventory', return_value={'source.py': 'fixed'}):
                _, lock, _ = freezer.version_lock('saved-baseline.json')
                loader.assert_called_once_with('saved-baseline.json')
                self.assertEqual(lock['models_sha256']['models/fixed.pt'], digest)
                model.write_bytes(b'changed-after-first-version-lock')
                with self.assertRaisesRegex(ValueError, 'Model changed'):
                    freezer.version_lock('saved-baseline.json')

    def test_merge_cannot_seal_a_full_manifest_without_receipts(self):
        args = SimpleNamespace(output=Path('unused-output'), shards=1, config='saved-baseline.json')
        v = {'version_key': 'fixed', 'GT_opens': 0}
        m = {'ACTUAL_CASE_IDS': list(freezer.IDS), 'GT_opens': 0,
             'cases': [{'case_id': cid} for cid in freezer.IDS]}
        inputs = {cid: (Path(cid+'.png'), None) for cid in freezer.IDS}
        with patch.object(freezer, 'read', side_effect=[v, m]), \
             patch.object(freezer, 'valid_receipt', return_value=None), \
             patch.object(freezer, 'version_lock', return_value=(None, None, 'fixed')), \
             patch.object(freezer, 'write') as writer:
            with self.assertRaisesRegex(ValueError, 'Missing frozen receipt'):
                freezer.merge(args, inputs, None)
            writer.assert_not_called()

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
