"""Local integration glue: published Component + immutable L45 Pin + frozen T."""
from pathlib import Path
from dataclasses import asdict
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path('/submission')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 't_runtime')]
PROFILE = 'component_c82_ppocrv5_pin_l45_T_20261015'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    release = json.loads((HERE / 'integration_manifest.json').read_text('utf-8'))
    for name, record in release['files'].items():
        path = ROOT / name
        assert path.is_file() and path.stat().st_size == record['size'] and sha(path) == record['sha256'], name
    portable = json.loads((ROOT / 't_runtime/portable_manifest.json').read_text('utf-8'))
    for name, record in portable['files'].items():
        assert sha(ROOT / 't_runtime' / name) == record['sha256'], name
    algorithm = json.loads((ROOT / 't_runtime/algorithm.json').read_text('utf-8'))
    for name, digest in algorithm['shared_files'].items():
        assert sha(ROOT / name) == digest, name
    import ground_runtime as ground
    loader = importlib.machinery.SourceFileLoader('combined_l45_adapter', str(ROOT / 'agent_system'))
    adapter = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(adapter)
    adapter.PIPELINE_ID = PROFILE

    def checked_config(root):
        from pcb.core.team_config import load_team_config
        config = load_team_config('configs/team.json', root=root)
        expected = {'text': 'ppocrv5_mobile', 'component': 'contextual_values_v2',
                    'component_stage': 'BEST', 'text_rules': 'v4_1', 'pin_localization': 'l39',
                    'pin_semantics': 'l39', 'wire': 'v3_frame_guard', 'topology': 'v3',
                    'submission': 'official', 'variant': 'baseline', 'enable_flying_labels': False}
        actual = config.as_dict()
        assert all(actual.get(key) == value for key, value in expected.items()), actual
        return config

    adapter.checked_team_config = checked_config
    original_load = adapter.load_runtime

    def load(app_root, model_root, output_dir):
        run_one, validate, config, ocr, detector, metadata = original_load(app_root, model_root, output_dir)
        cells = dict(zip(run_one.__code__.co_freevars, run_one.__closure__))
        pipeline = cells['pipeline'].cell_contents
        component_text = pipeline.text

        class ComponentCache:
            def run(self, image, context):
                resources = context.resources.setdefault('component', {})
                resources['cache_dir'] = output_dir / 'runs/runtime/component'
                result = component_text.run(image, context)
                metadata['component_OCR_diagnostics'] = result.diagnostics
                return result

        pipeline.text = ComponentCache()

        class Submission(ground.Submission):
            def run(self, scene, context):
                self.source_scene = copy.deepcopy(scene)
                return super().run(scene, context)

        wrapper = Submission(pipeline.submission)
        pipeline.submission = wrapper
        assert type(pipeline.component).__name__ == 'ComponentStageContextualValues'
        assert type(pipeline.pin_semantics).__name__ == 'RCPointSemantics'
        metadata.update(component_source_commit=release['component_source_commit'],
                        component_source_tree=release['component_source_tree'],
                        component_release_files=release['component_files'],
                        component_text_stage='ppocrv5_mobile', component_stage='contextual_values_v2',
                        component_OCR_runtime='rapidocr==3.9.2',
                        component_code_unchanged=True, Pin_L45_algorithm_and_weights_unchanged=True,
                        T_algorithm_source_sha256=algorithm['frozen_policy_source_sha256'],
                        T_algorithm_blocks=algorithm['algorithm_blocks_sha256'],
                        T_portable_sha256=sha(ROOT / 't_runtime/ground_runtime.py'),
                        T_validation_mode='candidate', T_ground_rule_unchanged=True,
                        integration_profile=PROFILE,
                        control_definition='new Component + frozen L45 Pin without T; different from prior old-Component L45')
        from pcb.io import write_json

        def paired(image_path, case_dir, *args, **kwargs):
            result = run_one(image_path, case_dir, *args, **kwargs)
            write_json(case_dir / 'component_L45_without_T_result.json', wrapper.before)
            write_json(case_dir / 'raw_scene.json', asdict(wrapper.source_scene))
            write_json(case_dir / 'T_scene.json', asdict(wrapper.projected))
            write_json(case_dir / 'T_changes.json', {'accepted': wrapper.changes, 'rejected': wrapper.rejects})
            return result

        return paired, validate, config, ocr, detector, metadata

    adapter.load_runtime = load
    return adapter.main(sys.argv[1:])


if __name__ == '__main__':
    raise SystemExit(main())
