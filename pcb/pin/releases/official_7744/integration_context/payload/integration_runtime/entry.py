"""Default competition entry for the complete local Component + L45 + T bundle."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
ROOT = Path('/submission')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-input', '--input', required=True, type=Path)
    parser.add_argument('-output', '--output', required=True, type=Path)
    args = parser.parse_args(argv)
    source, output = args.input.resolve(), args.output.resolve()
    if not source.is_dir() or source == output or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error('Separate input and output directories required')
    if (output / 'decision_log.json').exists() or any((output / 'final_results').glob('*_Pred.json')):
        parser.error('Use a fresh output directory')
    os.environ['EDA_PIN_VERSION'] = 'l45_regression'
    os.environ['EDA_PIN_SHARED_CACHE'] = str(output / 'runs/runtime')
    os.environ['YOLO_OFFLINE'] = 'true'
    import test_submission_adapter as adapter
    adapter.METHOD = 'YOLO + PP-OCRv5/contextual Component + frozen L45 Pin and T ground export'
    code = adapter.main(argv, agent_path=HERE / 'runner.py')
    if code == 0:
        log = json.loads((output / 'decision_log.json').read_text('utf-8'))
        log['runtime'].update(T_release_version='L46_T_20261015', T_self_contained=True,
                              T_external_algorithm_mounts_required=False,
                              integrated_component_release='c82f32e',
                              integration_release_version='component_L45_T_20261015')
        adapter.write_json(output / 'decision_log.json', log)
        adapter.write_json(output / 'final_results/decision_log.json', log)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
