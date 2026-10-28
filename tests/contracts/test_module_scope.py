"""Keep the composed release explicit without relaxing single-owner branches."""
from pathlib import Path
import unittest

from tools.check_module_scope import check_paths


class ModuleScopeTests(unittest.TestCase):
    def test_release_accepts_both_owners_and_reviewed_integration_files(self):
        paths=[
            'pcb/component/recovery/profile.py','configs/component/current.json',
            'models/component/ocr_v5_server/model.onnx','tests/component/test_recovery.py',
            'pcb/pin/l45_release.py','configs/pin/l45_regression.json',
            'docs/pin/BASELINE.md','models/pin/l45_rc_pair/best.pt',
            'README.md','requirements.txt','models/SHA256SUMS.json',
            '.github/workflows/team-modules.yml','tools/check_module_scope.py',
        ]
        self.assertEqual(check_paths('component_pin_release',paths),[])

    def test_release_rejects_other_owners_and_unreviewed_shared_paths(self):
        paths=[
            'pcb/net_hypergraph/registry.py','pcb/net_lines/legacy_v2.py',
            'pcb/wire_stage/v3.py','pcb/topology_stage/v3.py','pcb/core/pipeline.py',
            'configs/net_hypergraph/current.json','configs/team.json',
            'tools/unreviewed.py','README.md.extra','tests/contracts/unreviewed.py',
            '.github/workflows/unreviewed.yml','200_train_cases/0001/image.png',
        ]
        self.assertEqual(check_paths('component_pin_release',paths),sorted(paths))

    def test_existing_owners_remain_strict(self):
        self.assertEqual(check_paths('pin',['pcb/component/recovery/profile.py','README.md']),
                         ['README.md','pcb/component/recovery/profile.py'])
        self.assertEqual(check_paths('component',['pcb/pin/l45_release.py','README.md']),
                         ['README.md','pcb/pin/l45_release.py'])

    def test_workflow_selects_only_the_named_release_before_generic_pin(self):
        root=Path(__file__).resolve().parents[2]
        workflow=(root/'.github/workflows/team-modules.yml').read_text('utf8')
        exact='feature/pin-l45-validated-20261012) owner=component_pin_release ;;'
        generic='feature/pin-*) owner=pin ;;'
        self.assertIn(exact,workflow)
        self.assertIn(generic,workflow)
        self.assertLess(workflow.index(exact),workflow.index(generic))
        self.assertIn('python tools/check_module_scope.py --owner "$owner" --base "$BASE_SHA" --branch',workflow)


if __name__=='__main__':unittest.main()
