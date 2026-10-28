"""Pin-private latest-tree migration and fixed full-development A/B entry."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT.parent / 'EDA_Team_Modular_Integration'
sys.path.insert(0, str(ROOT))
from pcb.runtime_paths import prepend_compatible_vendor_paths
VENDOR_PATHS = prepend_compatible_vendor_paths(ROOT)
DESIGN = {14, 17, 18, 87, 103}
CHECK = set(range(5, 151, 5))
OLD35 = DESIGN | CHECK
HEAD = 'dded9033fdd53faaef9a44d98821880741f071f4'
PRIVATE_CONFIG = ROOT / 'configs/pin/l42_control.json'
PLAN = ROOT / 'docs/pin/l42_migration_plan_20261012.md'
MIGRATION_OUTPUT = ROOT / 'runs/pin_l42_migration_35_20261012'
DATASET = Path('C:/Users/bluty/Desktop/PCB_Competition/赛题六公开数据集/200_train_cases')

def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        raise TypeError(type(value).__name__)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default), 'utf-8')
    temporary.replace(path)

def event(name, **values):
    print(json.dumps({'event': name, **values}, ensure_ascii=True), flush=True)

def signature():
    from pcb.core.team_config import load_team_config
    head = subprocess.check_output(['git', '--no-optional-locks', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if head != HEAD:
        raise ValueError('Latest validation checkout changed')
    original = read(ORIGINAL / 'docs/pin/baseline_20261011_l39/runtime_manifest.json')
    for entry in original['files'].values():
        if sha(ORIGINAL / entry['source_relative']) != entry['sha256']:
            raise ValueError('Original 382-file baseline changed')
    config = load_team_config(ROOT / 'configs/team.json').as_dict()
    if config != original['pipeline_config']:
        raise ValueError('Public default/frontend changed')
    selection = read(PRIVATE_CONFIG)
    if selection != {'pin_localization': 'l39', 'pin_semantics': 'l42_control'}:
        raise ValueError('Frozen private selection changed')
    migrated = read(ROOT / 'docs/pin/l42_migration_manifest_20261012.json')
    for entry in migrated['records']:
        if sha(ROOT / entry['path']) != entry['sha256']:
            raise ValueError(f'Migrated resource changed: {entry["path"]}')
    sources = {}
    for directory in ('pcb', 'configs', 'vendor', 'tools', 'assets'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.yml'}:
                sources[str(path.relative_to(ROOT))] = sha(path)
    return {'HEAD': HEAD, 'public_config': config, 'private_selection_B': selection,
            'runtime_hashes': sources, 'original_runtime_files_verified': len(original['files']),
            'migration_manifest_sha256': sha(ROOT / 'docs/pin/l42_migration_manifest_20261012.json'),
            'binary_and_effective_resources': migrated['records'],
            'plan_sha256': sha(PLAN), 'runner_sha256': sha(__file__),
            'python': sys.version, 'versions': {key: importlib.metadata.version(key) for key in (
                'torch', 'opencv-python', 'rapidocr-onnxruntime', 'onnxruntime')},
            'vendor_paths': [str(p) for p in VENDOR_PATHS], 'device': 'cuda', 'ocr_device': 'cuda',
            'cache': str(ORIGINAL / 'runs/team_cache'),
            'model_cache': str(ORIGINAL / 'runs/pin_identity_l42_20261012/r1_check/model_cache'),
            'actual_stage_A': 'pcb.pin.frozen_l39.stages.L39Semantics',
            'actual_stage_B': 'pcb.pin.model_control_l42.L42ControlSemantics',
            'selection_method': 'private_Pipeline_instance', 'expanded_assets': False, 'L43_used': False}

def reference(cid, arm):
    return ORIGINAL / 'runs/pin_l42_reproduction_20261012' / cid / arm / 'result.json'

def preflight(args):
    from tools.run_team_pipeline import select_inputs
    if args.output.exists():
        raise ValueError('Fresh output required')
    current = signature()
    if args.cohort == 'full150':
        complete = read(MIGRATION_OUTPUT / 'inference_complete.json')
        if complete['count'] != 35 or complete['GT_reads'] != 0 or not complete['all_reference_predictions_equal']:
            raise ValueError('Complete 35-image migration parity required first')
        if read(MIGRATION_OUTPUT / 'initial.json')['signature'] != current:
            raise ValueError('Migration runtime signature no longer matches')
    wanted = sorted(OLD35) if args.cohort == 'migration35' else list(range(1, 151))
    rows = select_inputs(None, args.dataset, ','.join(str(n) for n in wanted))
    inputs = []
    for cid, image, folder in rows:
        expected = {arm: {'path': str(reference(cid, arm)), 'sha256': sha(reference(cid, arm))}
                    for arm in ('A', 'B')} if int(cid) in OLD35 else None
        inputs.append({'case_id': cid, 'path': str(image), 'sha256': sha(image), 'reference': expected})
    write(args.output / 'initial.json', {'signature': current, 'inputs': inputs,
          'cohort': args.cohort, 'count': len(inputs), 'OFFICIAL_SCORE': False,
          'started_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'GT_free_inference': True, 'sealed_holdout_used': False})
    write(args.output / 'progress.json', {'records': [], 'expected': len(inputs), 'GT_reads': 0})
    event('preflight_pass', cohort=args.cohort, count=len(inputs), runtime_sources=len(current['runtime_hashes']))

def verify_saved(output, records):
    for row in records:
        for arm in ('A', 'B'):
            for name, digest in row[arm]['files'].items():
                if sha(output / row['case_id'] / arm / name) != digest:
                    raise ValueError('Saved predictions/diagnostics changed')
        if row.get('reference_parity') is False:
            raise ValueError('Unexplained migration mismatch; finite review required')

def infer(args):
    from tools.team_access_guard import Guard
    guard = Guard()
    network_attempts = []
    def offline(name, values):
        if name == 'socket.connect':
            network_attempts.append(str(values[1:]))
            raise PermissionError('Network use prohibited during inference')
    sys.addaudithook(offline)
    initial = read(args.output / 'initial.json')
    current = signature()
    if initial['signature'] != current or initial['cohort'] != args.cohort:
        raise ValueError('Frozen experiment signature changed')
    records = read(args.output / 'progress.json')['records']
    verify_saved(args.output, records)
    completed = {r['case_id'] for r in records}
    pending = [r for r in initial['inputs'] if r['case_id'] not in completed]
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Frozen CUDA execution required')
    from tools.run_team_pipeline import build_runtime
    from pcb.core.team_config import load_team_config
    from pcb.core.context import PipelineContext
    from pcb.pin.model_control_l42 import L42ControlSemantics
    from pcb.io import read_image
    from pcb.submission import validate_strict
    config = load_team_config(ROOT / 'configs/team.json')
    cache = Path(current['cache'])
    started = time.perf_counter()
    pipeline, ocr, detector, device = build_runtime(config, 'cuda', 'cuda', cache)
    stages = {'A': pipeline.pin_semantics, 'B': L42ControlSemantics()}
    event('inference_start', pending=len(pending), expected=initial['count'], device=device,
          actual_stages={arm: type(stage).__name__ for arm, stage in stages.items()})
    for item in pending:
        cid = item['case_id']
        if sha(item['path']) != item['sha256']:
            raise ValueError('Input changed')
        row = {'case_id': cid, 'image': item['path'], 'known35': int(cid) in OLD35}
        snapshots = {}
        torch.cuda.reset_peak_memory_stats()
        for arm in ('A', 'B'):
            folder = args.output / cid / arm
            if folder.exists():
                raise ValueError('Partial case preserved; review before retry')
            pipeline.pin_semantics = stages[arm]
            candidate = stages['B']
            before_calls = candidate.reader.calls if candidate.reader else 0
            before_hits = candidate.reader.cache_hits if candidate.reader else 0
            t0 = time.perf_counter()
            with guard.inferring():
                image = read_image(Path(item['path']))
                h, w = image.shape[:2]
                context = PipelineContext(Path(item['path']).name, w, h, config, ocr, detector,
                    resources={'pin': {'device': device, 'words_cache': cache / 'pin_words',
                              'model_cache': Path(current['model_cache'])}})
                artifacts = pipeline.run(image, context)
                validate_strict(artifacts.result, (w, h))
                write(folder / 'result.json', artifacts.result)
                scene = asdict(artifacts.scene)
                write(folder / 'scene.json', scene)
                write(folder / 'diagnostics.json', artifacts.scene.diagnostics)
                owners = []
                for component in scene['components']:
                    owners.append({k: v for k, v in component.items() if k != 'pins'})
                snapshots[arm] = {'components': artifacts.result['components'], 'texts': scene['texts'],
                                  'owners': owners}
            parity = None
            if item['reference']:
                expected = item['reference'][arm]
                if sha(expected['path']) != expected['sha256']:
                    raise ValueError('Reference prediction changed')
                saved = read(folder / 'result.json')
                old = read(expected['path'])
                parity = saved == old
                write(folder / 'comparison.json', {'full_prediction_equal': parity, 'reference': expected,
                                                  'compared_after_inference_saved': True})
                if not parity:
                    write(args.output / 'failure.json', {'case': cid, 'arm': arm,
                          'phase': 'reference_parity', 'GT_reads': guard.target_opens,
                          'changed_top_level': [k for k in saved if saved[k] != old.get(k)]})
                    raise AssertionError(f'Full prediction changed after migration: {cid}/{arm}')
            row[arm] = {'seconds': time.perf_counter() - t0, 'reference_parity': parity,
                        'files': {name: sha(folder / name) for name in (
                            'result.json', 'scene.json', 'diagnostics.json')}}
            if arm == 'B':
                row[arm].update({'model_OCR_calls': candidate.reader.calls - before_calls,
                                'model_cache_hits': candidate.reader.cache_hits - before_hits})
            event('arm_complete', case=cid, arm=arm, seconds=round(row[arm]['seconds'], 2), reference_parity=parity)
        if snapshots['A'] != snapshots['B']:
            write(args.output / 'failure.json', {'case': cid, 'phase': 'non_Pin_upstream_invariants',
                  'GT_reads': guard.target_opens})
            raise AssertionError('A/B Text or non-Pin Component data changed')
        row['reference_parity'] = True if item['reference'] else None
        row['upstream_equal'] = True
        row['GPU_peak_allocated'] = torch.cuda.max_memory_allocated()
        row['GPU_peak_reserved'] = torch.cuda.max_memory_reserved()
        records.append(row)
        write(args.output / 'progress.json', {'records': records, 'expected': initial['count'],
              'GT_reads': guard.target_opens, 'blocked_accesses': guard.blocked,
              'network_attempts': network_attempts})
        event('case_complete', case=cid, completed=len(records), expected=initial['count'])
    verify_saved(args.output, records)
    if signature() != current:
        raise ValueError('Runtime changed during inference')
    complete = {'count': len(records), 'records': records, 'GT_reads': guard.target_opens,
                'blocked_accesses': guard.blocked, 'network_attempts': network_attempts,
                'all_reference_predictions_equal': True, 'all_upstream_equal': True,
                'runtime_signature_verified': True, 'default_changed': False, 'algorithm_changed': False,
                'expanded_assets': False, 'L43_used': False, 'OFFICIAL_SCORE': False,
                'frontend_cache_used': True, 'cold_start_timing': False,
                'seconds_this_invocation': time.perf_counter() - started}
    if len(records) != initial['count']:
        raise ValueError('Incomplete cohort')
    write(args.output / 'inference_complete.json', complete)
    if args.cohort == 'migration35':
        write(args.output / 'final_verification.json', {'status': 'PASSED', 'HEAD': HEAD,
              'PNG_cases': 35, 'predictions': 70, 'all_full_predictions_equal': True,
              'GT_reads': 0, 'original_382_files_verified': True, 'default_changed': False,
              'new_runtime_source_count': len(current['runtime_hashes']), 'OFFICIAL_SCORE': False})
    event('inference_complete', count=len(records), GT_reads=guard.target_opens)

def evaluate(args):
    if args.cohort != 'full150':
        raise ValueError('Migration uses prediction parity; no repeated GT scoring')
    from tools.team_access_guard import Guard
    from evaluate_v2 import evaluate_case, aggregate, source_from_name, component_key_map, point, prf
    initial = read(args.output / 'initial.json')
    complete = read(args.output / 'inference_complete.json')
    if complete['count'] != 150 or complete['GT_reads'] != 0 or not complete['all_upstream_equal']:
        raise ValueError('All 300 GT-free predictions must be saved first')
    if (args.output / 'evaluation.json').exists():
        raise ValueError('Evaluation evidence never overwritten')
    if signature() != initial['signature']:
        raise ValueError('Frozen runtime signature changed')
    verify_saved(args.output, complete['records'])
    guard = Guard()
    records = {'A': [], 'B': []}
    differences, targets = [], []
    strata = {arm: defaultdict(lambda: [0, 0, 0]) for arm in ('A', 'B')}
    def true_pins(pred, gt):
        mapping = component_key_map(pred['components'], gt['components'])
        result = set()
        for owner, pins in pred['pins'].items():
            for key, pin in pins.items():
                g = gt['pins'].get(mapping.get(owner), {}).get(key)
                if g and pin.get('pinname', '') == g.get('pinname', '') and math.dist(point(pin['point']), point(g['point'])) <= 5:
                    result.add((mapping[owner], key))
        return result, mapping
    for item in complete['records']:
        cid = item['case_id']
        if not 1 <= int(cid) <= 150:
            raise PermissionError('Only allowed development cases')
        predictions = {arm: read(args.output / cid / arm / 'result.json') for arm in ('A', 'B')}
        saved = [args.output / cid / arm / 'result.json' for arm in ('A', 'B')]
        folder = Path(item['image']).parent
        target_paths = list(folder.glob('*_target.json'))
        if len(target_paths) != 1 or target_paths[0].resolve().parent != folder.resolve():
            raise ValueError('Require one safe target')
        with guard.evaluating(target_paths[0], saved):
            gt = read(target_paths[0])
            targets.append({'case': cid, 'sha256': sha(target_paths[0])})
        source = source_from_name(item['image'])
        true = {}
        for arm, pred in predictions.items():
            result = evaluate_case(pred, gt)
            records[arm].append({'case_id': cid, 'status': 'ok', 'source': source, 'evaluation': result})
            true[arm], mapping = true_pins(pred, gt)
            for owner, pins in pred['pins'].items():
                mapped = mapping.get(owner)
                owner_type = (gt['components'][mapped]['type'] if mapped in gt['components']
                              else pred['components'].get(owner, {}).get('type', 'unknown'))
                category = 'box' if owner_type == 'box' else 'nonbox'
                strata[arm][category][1] += len(pins)
            for owner, pins in gt['pins'].items():
                category = 'box' if gt['components'].get(owner, {}).get('type') == 'box' else 'nonbox'
                strata[arm][category][2] += len(pins)
            for owner, key in true[arm]:
                category = 'box' if gt['components'].get(owner, {}).get('type') == 'box' else 'nonbox'
                strata[arm][category][0] += 1
        a, b = true['A'], true['B']
        ea, eb = records['A'][-1]['evaluation'], records['B'][-1]['evaluation']
        score = lambda r: 100 * sum(weight * r['metrics'][family]['f1'] for weight, family in (
            (.30, 'Component'), (.25, 'Pin'), (.35, 'NetHypergraph'), (.10, 'NetLine')))
        differences.append({'case_id': cid, 'source': source, 'known35': int(cid) in OLD35,
                            'tp_gained': sorted(b - a), 'tp_lost': sorted(a - b),
                            'net_tp': len(b - a) - len(a - b),
                            'pin_macro_contribution_delta': eb['metrics']['Pin']['f1'] - ea['metrics']['Pin']['f1'],
                            'local_score_contribution_delta': score(eb) - score(ea)})
    summary = {arm: aggregate(rows) for arm, rows in records.items()}
    partitions = {}
    for name, include in (('old35', lambda n: n in OLD35), ('remaining115', lambda n: n not in OLD35)):
        partitions[name] = {arm: aggregate([r for r in rows if include(int(r['case_id']))])
                            for arm, rows in records.items()}
    largest = {'Pin_net_TP': max(differences, key=lambda r: (r['net_tp'], -int(r['case_id'])))['case_id'],
               'local_score': max(differences, key=lambda r: (r['local_score_contribution_delta'], -int(r['case_id'])))['case_id']}
    exclusions = {name: {'excluded_case': cid, 'summary': {arm: aggregate([r for r in rows if r['case_id'] != cid])
                   for arm, rows in records.items()}} for name, cid in largest.items()}
    a, b = (summary[arm]['overall'] for arm in ('A', 'B'))
    delta = {'Pin_TP': b['metrics']['Pin']['tp'] - a['metrics']['Pin']['tp'],
             'Pin_macro_F1': b['metrics']['Pin']['macro_f1'] - a['metrics']['Pin']['macro_f1'],
             'Pin_micro_F1': b['metrics']['Pin']['f1'] - a['metrics']['Pin']['f1'],
             'local_score': b['FinalScore'] - a['FinalScore']}
    rest = {arm: partitions['remaining115'][arm]['overall'] for arm in ('A', 'B')}
    rest_delta = {'Pin_TP': rest['B']['metrics']['Pin']['tp'] - rest['A']['metrics']['Pin']['tp'],
                  'Pin_macro_F1': rest['B']['metrics']['Pin']['macro_f1'] - rest['A']['metrics']['Pin']['macro_f1'],
                  'local_score': rest['B']['FinalScore'] - rest['A']['FinalScore']}
    positive = delta['Pin_TP'] > 0 and delta['Pin_macro_F1'] > 0 and delta['local_score'] > 0
    decision = ('POSITIVE_REVIEW_COVERAGE_AND_COST' if positive and rest_delta['Pin_TP'] > 0
                else 'NARROW_COVERAGE_FINITE_REVIEW' if positive else 'STOP_FINITE_REVIEW')
    data = {'OFFICIAL_SCORE': False, 'summary': summary, 'partitions': partitions, 'delta': delta,
            'remaining115_delta': rest_delta, 'differences': differences, 'excluded_largest': exclusions,
            'box_nonbox': {arm: {name: prf(*counts) for name, counts in groups.items()} for arm, groups in strata.items()},
            'stratum_assignment': 'mapped GT owner type; unmapped prediction uses predicted owner type',
            'decision': decision, 'cases': records, 'target_hashes': targets,
            'GT_reads_in_inference': 0, 'unique_evaluation_targets': len(targets),
            'GT_open_events_in_evaluation': guard.target_opens,
            'runtime_costs': {arm: {'seconds': sum(r[arm]['seconds'] for r in complete['records'])}
                              for arm in ('A', 'B')},
            'GPU_peak_allocated': max(r['GPU_peak_allocated'] for r in complete['records']),
            'GPU_peak_reserved': max(r['GPU_peak_reserved'] for r in complete['records']),
            'B_model_OCR_calls': sum(r['B']['model_OCR_calls'] for r in complete['records']),
            'B_model_cache_hits': sum(r['B']['model_cache_hits'] for r in complete['records']),
            'cold_start_timing': False, 'official_QA_difference': 'strict still checks nonbox key/name; not official score'}
    write(args.output / 'evaluation.json', data)
    if signature() != initial['signature']:
        raise ValueError('Frozen runtime changed during evaluation')
    write(args.output / 'final_verification.json', {'status': 'PASSED', 'HEAD': HEAD,
          'PNG_cases': 150, 'saved_predictions': 300, 'GT_reads_in_inference': 0,
          'evaluation_separate_process': True, 'unique_evaluation_targets': 150,
          'all_upstream_equal': True, 'old35_prediction_parity': True,
          'original_382_files_verified': True, 'decision': decision, 'OFFICIAL_SCORE': False})
    event('evaluation_complete', delta=delta, remaining115=rest_delta, decision=decision)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'infer', 'evaluate'), required=True)
    parser.add_argument('--cohort', choices=('migration35', 'full150'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to((ROOT / 'runs').resolve()):
        raise PermissionError('Fresh isolated runs output required')
    try:
        {'preflight': preflight, 'infer': infer, 'evaluate': evaluate}[args.phase](args)
    except Exception as exc:
        event('failed', kind=type(exc).__name__, message=str(exc))
        raise

if __name__ == '__main__':
    main()
