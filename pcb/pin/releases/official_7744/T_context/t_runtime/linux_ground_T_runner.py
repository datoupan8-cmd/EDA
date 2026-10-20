"""Offline validation shim in the immutable L45 image; no GT is mounted."""
from pathlib import Path
from dataclasses import asdict
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
ROOT = Path('/submission')
BUNDLE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(BUNDLE)]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    manifest = json.loads((BUNDLE/'portable_manifest.json').read_text('utf-8'))
    for name, record in manifest['files'].items():
        assert sha(BUNDLE/name) == record['sha256']
        assert (BUNDLE/name).stat().st_size == record['bytes']
    algorithm = json.loads((BUNDLE/'algorithm.json').read_text('utf-8'))
    for name, digest in algorithm['shared_files'].items():
        assert sha(ROOT/name) == digest, name
    import ground_runtime as ground
    loader = importlib.machinery.SourceFileLoader('accepted_l45_adapter', str(ROOT/'agent_system'))
    adapter = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(adapter)
    original_load = adapter.load_runtime

    def load(app_root, model_root, output_dir):
        run_one, validate, config, ocr, detector, metadata = original_load(app_root, model_root, output_dir)
        cells = dict(zip(run_one.__code__.co_freevars, run_one.__closure__))
        pipeline = cells['pipeline'].cell_contents
        mode = os.environ['T_VALIDATE_MODE']

        class Candidate(ground.Submission):
            def run(self, scene, context):
                self.source_scene = copy.deepcopy(scene)
                return super().run(scene, context)

        class Control:
            def __init__(self, baseline):
                self.baseline = baseline

            def run(self, scene, context):
                self.source_scene = copy.deepcopy(scene)
                result = self.baseline.run(scene, context)
                self.before = result.data
                self.projected = scene
                self.changes, self.rejects = [], []
                return result

        assert mode in ('candidate', 'control')
        wrapper = Candidate(pipeline.submission) if mode == 'candidate' else Control(pipeline.submission)
        pipeline.submission = wrapper
        metadata.update(T_validation_mode=mode,
                        T_algorithm_source_sha256=algorithm['frozen_policy_source_sha256'],
                        T_algorithm_blocks=algorithm['algorithm_blocks_sha256'],
                        T_portable_sha256=sha(BUNDLE/'ground_runtime.py'),
                        T_no_threshold_or_model_change=True)
        from pcb.io import write_json

        def paired(image_path, case_dir, *args, **kwargs):
            result = run_one(image_path, case_dir, *args, **kwargs)
            write_json(case_dir/'fresh_L45_result.json', wrapper.before)
            write_json(case_dir/'raw_scene.json', asdict(wrapper.source_scene))
            write_json(case_dir/'T_scene.json', asdict(wrapper.projected))
            write_json(case_dir/'T_changes.json', {'accepted': wrapper.changes, 'rejected': wrapper.rejects})
            return result

        return paired, validate, config, ocr, detector, metadata

    adapter.load_runtime = load
    return adapter.main(sys.argv[1:])


if __name__ == '__main__':
    raise SystemExit(main())
