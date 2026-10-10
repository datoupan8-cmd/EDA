"""One frozen image-only L38/current acceptance run on development 0001..0150."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
from tools.run_pin_combined_l38 import (Guard, ALL, DESIGN, CHECK, read, git_state,
                                       verify, persist, compare_archive)
from pin_combined_l38 import build_runtime, digest
from pcb.core.context import PipelineContext
from pcb.core.pipeline import ModularPipeline
from pcb.core.registry import build_default_registry
from pcb.data_policy import discover_allowed_cases
from pcb.io import read_image, write_json
from pcb.submission import validate_strict
from tools.run_pin_side_joint_l36 import metrics, strict_ids, pair_changes, aggregate
from evaluate_v2 import source_from_name, prf

OUT = ROOT / 'reports/pin_full_validation_l39'
L38 = ROOT / 'reports/pin_combined_l38'
CASE_IDS = tuple(f'{i:04d}' for i in range(1, 151))
SOURCES = ('tools/validate_pin_combined_l39.py',
           'tests/pin/test_pin_combined_l39.py',
           'reports/pin_full_validation_l39/experiment_plan.md')


def check_ids(rows: list[dict], expected=CASE_IDS) -> None:
    """A missing, duplicate or sealed case never becomes a successful macro mean."""
    ids = [r['case_id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise AssertionError('Case identity/count mismatch')
    if any(not x.isdigit() or not 1 <= int(x) <= 150 for x in ids):
        raise PermissionError('Only 0001..0150')


def type_metrics(pred: dict, target: dict, ids: set) -> dict:
    """Partition existing strict TP by GT owner type, without new match rules."""
    gt, proposed, tp = Counter(), Counter(), Counter()
    for owner, pins in target['pins'].items():
        gt[target['components'][owner]['type']] += len(pins)
    for owner, pins in pred['pins'].items():
        proposed[pred['components'][owner]['type']] += len(pins)
    for owner, _ in ids:
        tp[target['components'][owner]['type']] += 1
    return {kind: {'tp': tp[kind], 'pred': proposed[kind], 'gt': gt[kind]}
            for kind in sorted(set(gt) | set(proposed))}


def prepare(dataset: Path) -> dict:
    """Discover only permitted folders; inference receives no target contents."""
    if (OUT / 'initial_state.json').exists():
        state = read(OUT / 'initial_state.json'); verify(state)
        if str(dataset.resolve()) != read(OUT / 'input_manifest.json')['dataset_root']:
            raise AssertionError('Prepared dataset path changed')
        return state
    old = read(L38 / 'initial_state.json'); verify(old)
    seal = read(L38 / 'delivery_verification.json')
    hashes = dict(old['sha256'])
    for relative, expected in seal['artifact_sha256'].items():
        p = ROOT / relative
        if digest(p) != expected:
            raise AssertionError('L38 delivery changed: ' + relative)
        hashes[str(p)] = expected
    hashes[str(L38 / 'delivery_verification.json')] = digest(L38 / 'delivery_verification.json')
    train = read(ROOT / 'reports/pin_learned_locator_l1/training_manifest.json')
    expected_targets = {f"{r['case_id']:04d}": r['target_sha256'] for r in train['records']}
    for case in ALL:
        phase = 'design' if case in DESIGN else 'check'
        meta = read(ROOT / f'reports/pin_library_semantics_l37/{phase}/cases/{case}.json')
        if case in expected_targets and expected_targets[case] != meta['target_sha256']:
            raise AssertionError('Historical target signatures disagree')
        expected_targets[case] = meta['target_sha256']
    trained = {f"{r['case_id']:04d}" for r in train['records']}
    records = []
    for files in discover_allowed_cases(dataset):
        case = f'{files.case_id:04d}'
        image_sha = digest(files.image_path)
        if image_sha != train['image_sha256'][str(files.case_id)]:
            raise AssertionError('Image version changed; version audit first: ' + case)
        hashes[str(files.image_path)] = image_sha
        records.append({'case_id': case, 'image_path': str(files.image_path.resolve()),
                        'target_path': str(files.target_path.resolve()),
                        'image_sha256': image_sha,
                        'expected_target_sha256': expected_targets.get(case),
                        'source': source_from_name(files.image_path.name),
                        'L1_training_image': case in trained,
                        'fixed_historical_check': case in CHECK,
                        'previous_L38_case': case in ALL})
    check_ids(records)
    manifest = {'dataset_root': str(dataset.resolve()), 'cases': records,
                'known_target_signature_count': len(expected_targets),
                'target_content_read_in_preparation': False,
                'unique_L1_training_images': len(trained),
                'OFFICIAL_SCORE': False, 'sealed_holdout_used': False}
    write_json(OUT / 'input_manifest.json', manifest)
    hashes[str(OUT / 'input_manifest.json')] = digest(OUT / 'input_manifest.json')
    for relative in SOURCES:
        hashes[str(ROOT / relative)] = digest(ROOT / relative)
    state = {'sha256': hashes, 'git': git_state(), 'single_variable': 'current vs frozen L38 combination',
             'OFFICIAL_SCORE': False, 'sealed_holdout_used': False}
    verify(state); write_json(OUT / 'initial_state.json', state)
    return state


def target_after_saved(meta: dict, guard: Guard, saved: list[Path]):
    """Target is read by this evaluator only, after both image predictions exist."""
    with guard.evaluating(meta['target_path'], saved):
        value = digest(Path(meta['target_path']))
        if meta['expected_target_sha256'] and value != meta['expected_target_sha256']:
            raise AssertionError('LATEST TARGET VERSION DRIFT: ' + meta['case_id'])
        target = read(meta['target_path'])
    return target, value


def run(args, state: dict, guard: Guard) -> None:
    if (OUT / 'complete.json').exists():
        raise RuntimeError('Completed full run preserved; use summarize/verify')
    manifest = read(OUT / 'input_manifest.json')
    started = time.perf_counter()
    with guard.inferring():
        pipeline, ocr, detector = build_runtime(ROOT, args.library, ROOT / 'runs/ocr_cache_v4_1',
                                               ROOT / 'reports/pin_word_decoder_l3/ocr_cache')
    rows = []
    for meta in manifest['cases']:
        case = meta['case_id']; record = OUT / f'cases/{case}.json'
        saved = [OUT / f'predictions/{v}/{case}/result.json' for v in ('A', 'B')]
        if record.exists():
            row = read(record)
            for v, p in zip(('A', 'B'), saved):
                if digest(p) != row['prediction_sha256'][v]:
                    raise AssertionError('Resume prediction changed')
            _, target_sha = target_after_saved(meta, guard, saved)
            if target_sha != row['target_sha256']:
                raise AssertionError('Resume target changed')
            rows.append(row)
            continue
        if (OUT / f'failures/{case}.json').exists():
            raise RuntimeError('Preserved failed case; diagnose before any rerun')
        try:
            if digest(Path(meta['image_path'])) != meta['image_sha256']:
                raise AssertionError('Prepared image changed')
            old_calls = (pipeline.reader.cache_hits, pipeline.reader.calls, pipeline.reader.direct_calls)
            with guard.inferring():
                image = read_image(meta['image_path'])
                context = PipelineContext(Path(meta['image_path']).name, image.shape[1], image.shape[0],
                                          pipeline.config, ocr, detector)
                outputs = pipeline.run(image, context)
            persist(outputs, OUT, case)
            parity = compare_archive(outputs, case) if case in ALL else None
            if parity and not all(x['exact'] for x in parity.values()):
                raise AssertionError('L38 reproduction changed')
            formal_parity = None
            if case in ('0001', '0002'):
                with guard.inferring():
                    reference = ModularPipeline(pipeline.config, build_default_registry()).run(image, context)
                formal_parity = reference.result == outputs.baseline.result
                if not formal_parity:
                    raise AssertionError('A differs from formal ModularPipeline')
            target, target_sha = target_after_saved(meta, guard, saved)
            evaluated = {label: metrics(art.result, target, art.scene.diagnostics, case, meta['source'])
                         for label, art in (('before', outputs.baseline), ('after', outputs.candidate))}
            ids = {label: strict_ids(art.result, target)
                   for label, art in (('before', outputs.baseline), ('after', outputs.candidate))}
            pairs = pair_changes(outputs.baseline.result, outputs.candidate.result, target)
            row = {**meta, **evaluated, 'target_sha256': target_sha,
                   'historical_target_hash_confirmed': meta['expected_target_sha256'] is not None,
                   'prediction_sha256': {v: digest(p) for v, p in zip(('A', 'B'), saved)},
                   'frontend_unchanged': True, 'nonbox_pins_unchanged': True, 'schema_passed': True,
                   'image_only': True, 'L38_parity': parity, 'formal_A_parity': formal_parity,
                   'gained': sorted(ids['after'] - ids['before']), 'lost': sorted(ids['before'] - ids['after']),
                   'pair_changes': {k: len(v) for k, v in pairs.items()},
                   'pair_examples': {k: v[:5] for k, v in pairs.items()},
                   'pin_by_type': {label: type_metrics(art.result, target, ids[label])
                                   for label, art in (('before', outputs.baseline), ('after', outputs.candidate))},
                   'runtime_seconds_shared_frontend_A_plus_B': outputs.trace['total_seconds'],
                   'frontend_seconds': outputs.trace['frontend_seconds'],
                   'local_word_cache': dict(zip(('hits', 'detector_calls', 'direct_crops'),
                       (now - old for now, old in zip((pipeline.reader.cache_hits, pipeline.reader.calls,
                                                       pipeline.reader.direct_calls), old_calls)))),
                   'frontend_diagnostics': outputs.baseline.scene.diagnostics.get('ocr', {})}
            write_json(record, row); rows.append(row)
            write_json(OUT / 'progress.json', {'done': len(rows), 'total': 150,
                       'case_ids': [r['case_id'] for r in rows], 'elapsed_seconds': time.perf_counter()-started,
                       'target_opens': guard.target_opens, 'blocked_access_attempts': guard.blocked})
            print(json.dumps({'done': len(rows), 'case': case,
                  'score_A': evaluated['before']['evaluation']['FinalScore'],
                  'score_B': evaluated['after']['evaluation']['FinalScore'],
                  'pin_gained': len(row['gained']), 'pin_lost': len(row['lost']),
                  'seconds': round(outputs.trace['total_seconds'], 3)}, ensure_ascii=False), flush=True)
        except Exception as exc:
            write_json(OUT / f'failures/{case}.json', {'case_id': case, 'error': repr(exc),
                       'traceback': traceback.format_exc(), 'completed_cases': len(rows),
                       'algorithm_modified': False, 'OFFICIAL_SCORE': False})
            raise
    verify(state)
    check_ids(rows)
    summarize(rows, {'elapsed_seconds_this_process': time.perf_counter()-started,
                     'target_opens_this_process': guard.target_opens,
                     'blocked_access_attempts': guard.blocked, 'full_150_run': True})


def build_summary(rows: list[dict]) -> dict:
    check_ids(rows)
    if not all(r['schema_passed'] and r['frontend_unchanged'] and r['nonbox_pins_unchanged'] for r in rows):
        raise AssertionError('Acceptance requires all stage invariants')
    summary = {label: aggregate([r[label] for r in rows]) for label in ('before', 'after')}
    delta = summary['after']['overall']['FinalScore'] - summary['before']['overall']['FinalScore']
    groups = {
        'fixed_historical_check30': [r for r in rows if r['fixed_historical_check']],
        'L1_training119': [r for r in rows if r['L1_training_image']],
        'excluded_duplicate1': [r for r in rows if not r['L1_training_image'] and not r['fixed_historical_check']],
        'exclude_design5_145': [r for r in rows if r['case_id'] not in DESIGN],
        'outside_previous_L38_35': [r for r in rows if not r['previous_L38_case']],
    }
    stratified = {name: {label: aggregate([r[label] for r in group]) for label in ('before', 'after')}
                  for name, group in groups.items()}
    changes = Counter(); types = {k: defaultdict(Counter) for k in ('before', 'after')}
    for row in rows:
        changes.update(row['pair_changes'])
        for label in types:
            for kind, counts in row['pin_by_type'][label].items():
                types[label][kind].update(counts)
    types = {label: {kind: prf(c['tp'], c['pred'], c['gt']) for kind, c in values.items()}
             for label, values in types.items()}
    shifts = [{'case_id': r['case_id'], 'source': r['source'],
               'score_delta': r['after']['evaluation']['FinalScore']-r['before']['evaluation']['FinalScore'],
               'pin_gained': len(r['gained']), 'pin_lost': len(r['lost']), 'pair_changes': r['pair_changes']}
              for r in rows]
    gained = [{'case_id': r['case_id'], 'pin': p} for r in rows for p in r['gained']]
    lost = [{'case_id': r['case_id'], 'pin': p} for r in rows for p in r['lost']]
    if summary['after']['overall']['metrics']['Pin']['tp']-summary['before']['overall']['metrics']['Pin']['tp'] != len(gained)-len(lost):
        raise AssertionError('TP delta/set accounting mismatch')
    decisions = {'improved': sum(s['score_delta'] > 1e-9 for s in shifts),
                 'regressed': sum(s['score_delta'] < -1e-9 for s in shifts),
                 'unchanged': sum(abs(s['score_delta']) <= 1e-9 for s in shifts)}
    return {'summary': summary, 'score_delta_points': delta, 'stratified': stratified,
            'pin_by_type': types, 'gained': gained, 'lost': lost, 'pair_changes': dict(changes),
            'case_score_changes': shifts, 'case_change_counts': decisions,
            'candidate_retained_for_competition': delta > 1e-9,
            'formal_current_changed': False, 'official_environment_verified': False,
            'OFFICIAL_SCORE': False, 'sealed_holdout_used': False}


def summarize(rows: list[dict], runtime: dict | None = None) -> dict:
    result = build_summary(rows)
    result['metadata'] = {'case_range': '0001-0150', 'success_count': 150, 'failure_count': 0,
                          'A': 'formal current', 'B': 'unchanged L38 combined image-only pipeline',
                          'no_threshold_tuning': True, 'model_training_images_present': 119,
                          'training_data_result_is_not_generalization': True,
                          'runtime_note': 'shared frontend A+B with content caches; not standalone cold A/B runtime',
                          'full_150_run': True, **(runtime or {})}
    # On the 35 previously signed cases, all metric fields must reproduce L38.
    by = {r['case_id']: r for r in rows}
    for phase in ('design', 'check'):
        old = read(L38 / phase / 'complete.json')
        for r in old['cases']:
            for label in ('before', 'after'):
                if by[r['case_id']][label]['evaluation'] != r[label]['evaluation']:
                    raise AssertionError('L38 scoring drift: ' + r['case_id'])
    result['previous_35_predictions_and_metrics_reproduced'] = True
    write_json(OUT / 'complete.json', result)
    columns = ('case_id', 'source', 'L1_training_image', 'score_A', 'score_B', 'score_delta',
               'pin_TP_A', 'pin_TP_B', 'pin_gained', 'pin_lost',
               'correct_pair_gained', 'correct_pair_lost', 'wrong_pair_added', 'wrong_pair_removed',
               'runtime_shared_A_B_seconds')
    with (OUT / 'per_case.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for r in rows:
            a, b = r['before']['evaluation'], r['after']['evaluation']; p = r['pair_changes']
            writer.writerow(dict(zip(columns, (r['case_id'], r['source'], r['L1_training_image'],
                a['FinalScore'], b['FinalScore'], b['FinalScore']-a['FinalScore'],
                a['metrics']['Pin']['tp'], b['metrics']['Pin']['tp'], len(r['gained']), len(r['lost']),
                p['correct_gained'], p['correct_lost'], p['incorrect_added'], p['incorrect_removed'],
                r['runtime_seconds_shared_frontend_A_plus_B']))))
    write_json(OUT / 'dataset_version_check.json', {
        'all_150_images_match_frozen_L1_hash': True,
        'known_targets_unchanged': sum(r['historical_target_hash_confirmed'] for r in rows),
        'first_hash_capture_cases': [r['case_id'] for r in rows if not r['historical_target_hash_confirmed']],
        'targets': {r['case_id']: r['target_sha256'] for r in rows},
        'latest_target_only': True, 'sealed_holdout_used': False})
    print(json.dumps({'complete': True, 'count': 150, 'score_delta_points': result['score_delta_points'],
                     'gained': len(result['gained']), 'lost': len(result['lost']),
                     'case_changes': result['case_change_counts'], 'pair_changes': result['pair_changes']}, ensure_ascii=False), flush=True)
    return result


def verify_outputs(state: dict, guard: Guard) -> None:
    verify(state); rows = [read(OUT / f'cases/{case}.json') for case in CASE_IDS]
    check_ids(rows)
    for row in rows:
        paths = [OUT / f'predictions/{v}/{row["case_id"]}/result.json' for v in ('A', 'B')]
        for v, path in zip(('A', 'B'), paths):
            if digest(path) != row['prediction_sha256'][v]:
                raise AssertionError('Prediction changed after evaluation')
            pred = read(path); validate_strict(pred)
        _, current_target_sha = target_after_saved(row, guard, paths)
        if current_target_sha != row['target_sha256']:
            raise AssertionError('Target changed since evaluation')
    current = build_summary(rows); old = read(OUT / 'complete.json')
    if any(current[k] != old[k] for k in current):
        raise AssertionError('Aggregate no longer reproduces case records')
    write_json(OUT / 'verification.json', {'protected_files': len(state['sha256']),
        'unchanged': True, 'prediction_files_verified': 300, 'all_150_latest_targets_unchanged': True,
        'aggregate_reproduced': True, 'OFFICIAL_SCORE': False, 'sealed_holdout_used': False,
        'formal_config_changed': False, 'commit': False, 'push': False})


def main():
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('prepare', 'run', 'summarize', 'verify'), required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--library', type=Path)
    args = parser.parse_args(); guard = Guard(); state = prepare(args.dataset)
    if args.task == 'run':
        if not args.library: parser.error('--run requires --library')
        run(args, state, guard)
    elif args.task == 'summarize':
        summarize([read(OUT / f'cases/{case}.json') for case in CASE_IDS])
    elif args.task == 'verify':
        verify_outputs(state, guard)
    verify(state)
    print(json.dumps({'task': args.task, 'protected_files_unchanged': len(state['sha256']),
                     'blocked_access_attempts': guard.blocked}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
