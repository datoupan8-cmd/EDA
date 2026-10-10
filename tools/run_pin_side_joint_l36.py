"""Frozen L3 B replay and a gated L36 experiment; targets are evaluation-only."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]

from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.registry import build_default_registry
from pcb.io import read_image, write_json
from pcb.schema import Scene, Text
from pcb.text_detection import TokenRole
from pcb.submission import validate_strict
from pin_side_joint_l36 import POLICY, stage
from pin_word_reader_l3 import stage_b, canonical_tokens, same_instance
from pin_nonbox_export_l26_check import archived_components
from pin_contextual_roles_e6 import assert_invariants, local_roles
from pin_skeleton_followup import strict_ids, number_bbox_score
from pin_terminal_reader_l2_run import predict
from pin_g_wire_e1 import pair_changes
from tools.evaluate_pin_qa_diagnostic import digest, read, git_state, evaluate_pins, METHODS
from tools.analyze_pin_localization import strict_component_pairs, coordinate_assignment
from pcb.coordinates import target_to_opencv
from pcb.text_detection import classify_tokens
from pcb import pin_semantics_v6 as ordered
from evaluate_v2 import evaluate_case, aggregate, component_key_map

OUT = ROOT / 'reports/pin_side_joint_l36'
DESIGN = ('0014', '0017', '0018', '0087', '0103')
CHECK = tuple(f'{n:04d}' for n in range(5, 151, 5))
ALL = DESIGN + CHECK
SOURCES = ('experiments/pin_side_joint_l36.py', 'tools/run_pin_side_joint_l36.py',
           'tests/pin/test_pin_side_joint_l36.py', 'reports/pin_side_joint_l36/experiment_plan.md')
L1 = ROOT / 'reports/pin_learned_locator_l1'
L3 = ROOT / 'reports/pin_word_decoder_l3'


class AccessGuard:
    """Deny nonselected development images, all reserved data, and inference GT."""
    def __init__(self, install=True):
        self.allowed_target = None
        self.target_opens = self.blocked = 0
        if install:
            sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        value = str(args[0]).replace('\\', '/').lower()
        parts = value.split('/')
        denied = any(p in {'golden', '10gt', '10_gtcase', 'sealed',
                          'eda_pin_crossing_quicktest'} for p in parts)
        if '200_train_cases' in parts:
            pos = parts.index('200_train_cases')
            denied |= pos + 1 >= len(parts) or parts[pos + 1] not in ALL
        is_target = value.endswith('_target.json')
        if denied or (is_target and value != self.allowed_target):
            self.blocked += 1
            raise PermissionError('L36 refuses reserved data or an inference target read')
        if is_target:
            self.target_opens += 1

    @contextmanager
    def evaluating(self, target: Path, prediction_path: Path):
        if self.allowed_target is not None or not prediction_path.is_file():
            raise AssertionError('Save image-only prediction before evaluation')
        self.allowed_target = str(target.resolve()).replace('\\', '/').lower()
        try:
            yield
        finally:
            self.allowed_target = None


def verify(initial):
    for path, sha in initial['sha256'].items():
        if digest(Path(path)) != sha:
            raise AssertionError('Frozen algorithm/input changed: ' + path)
    now = git_state()
    if any(now[k] != initial['git'][k] for k in ('head', 'branch', 'tracked_diff_sha256')):
        raise AssertionError('Existing tracked changes were altered')


def freeze():
    path = OUT / 'initial_state.json'
    if path.exists():
        initial = read(path); verify(initial); return initial
    previous = read(ROOT / 'reports/pin_owner_headroom_l35/initial_state.json')
    hashes = dict(previous['sha256'])
    for folder in ('pcb', 'configs'):
        for fp in (ROOT / folder).rglob('*'):
            if fp.is_file() and '__pycache__' not in fp.parts and fp.suffix != '.pyc':
                hashes[str(fp)] = digest(fp)
    for fp in (ROOT / 'evaluate_v2.py', L1 / 'best.pt', L1 / 'training_manifest.json',
               L3 / 'read_design/complete.json', L3 / 'check/complete.json'):
        hashes[str(fp)] = digest(fp)
    for source in SOURCES:
        hashes[str(ROOT / source)] = digest(ROOT / source)
    initial = {'sha256': hashes, 'git': git_state(), 'policy': asdict(POLICY),
               'baseline': 'L3 B (both groups), not production current or mixed L13/L28',
               'Design': DESIGN, 'Check': CHECK, 'OFFICIAL_SCORE': False,
               'SEALED_HOLDOUT_USED_FOR_DEVELOPMENT': False,
               'single_variable': 'simultaneous three-sequence side decoding'}
    verify(initial); write_json(path, initial); return initial


def source_paths(case):
    if case not in ALL:
        raise ValueError('Only frozen 35 development cases')
    design = case in DESIGN
    reference = L3 / ('read_design' if design else 'check')
    raw = (ROOT / f'reports/pin_terminal_reader_l2/design/raw_terminals/{case}.json'
           if design else L1 / f'validation/predictions/L1/{case}/raw_terminals.json')
    return reference, raw


def hydrate(case):
    """Actual raw candidates + events, never substitute exported Pins for raw."""
    reference, raw_path = source_paths(case)
    complete = read(reference / 'complete.json')
    old = next(r for r in complete['cases'] if r['case_id'] == case)
    signature = complete['metadata']['inputs'][case]
    paths = {'front': ROOT.parent / f'tmp/pin_sina_frontend/{case}.json',
             'raw': raw_path, 'pool': reference / f'words/{case}.json',
             'diagnostics': reference / f'predictions/B/{case}/diagnostics.json',
             'prediction': reference / f'predictions/B/{case}/result.json'}
    inputs = {str(p): digest(p) for p in paths.values()}
    front, raw, pool, diagnostics, prediction = (read(paths[k]) for k in paths)
    if inputs[str(raw_path)] != signature['raw_sha256'] or inputs[str(paths['pool'])] != signature['word_pool_sha256']:
        raise AssertionError('Frozen raw/word input drift')
    manifest = read(L1 / 'training_manifest.json')
    if tuple(f'{n:04d}' for n in manifest['validation_ids']) != CHECK or set(map(int, CHECK)) & set(manifest['train_ids']):
        raise AssertionError('Historical split drift')
    for p, h in front['signature']['frontend_source_sha256'].items():
        if digest(ROOT / p.replace('\\', '/')) != h:
            raise AssertionError('Signed frontend source changed')
    images = list((Path(manifest['dataset_root']) / case).glob('*.png'))
    if len(images) != 1:
        raise AssertionError('Unique PNG required within exact permitted case')
    image_path = images[0]
    image_sha = digest(image_path)
    if any(image_sha != h for h in (signature['image_sha256'], front['signature']['image_sha256'],
                                    manifest['image_sha256'][str(int(case))])):
        raise AssertionError('Image version changed; dataset version analysis needed first')
    image = read_image(image_path); height, width = image.shape[:2]
    config = PipelineConfig.load(ROOT / 'configs/current.json')
    if asdict(config) != front['signature']['config']:
        raise AssertionError('Frontend pipeline config changed')
    scene = Scene(width, height, archived_components(front, raw, diagnostics),
                  [Text(**{**t, 'bbox': tuple(t['bbox'])}) for t in front['texts']],
                  diagnostics=copy.deepcopy(diagnostics))
    roles = [TokenRole(**{**r, 'token': Text(**{**r['token'], 'bbox': tuple(r['token']['bbox'])})})
             for r in front['roles']]
    context = PipelineContext(image_path.name, width, height, config, None)
    target = image_path.with_name(image_path.stem + '_target.json')
    inputs[str(image_path)] = image_sha
    meta = {'case_id': case, 'source': old['source'], 'width': width, 'height': height,
            'target_path': str(target), 'target_sha256': old['target_sha256'],
            'historical_prediction_path': str(paths['prediction']), 'inputs': inputs}
    return scene, raw, pool, roles, image, context, prediction, meta


def metrics(prediction, target, diagnostics, case, source):
    value = evaluate_case(prediction, target, diagnostics)
    if value['metrics']['Pin']['tp'] != len(strict_ids(prediction, target)):
        raise AssertionError('Strict TP set/evaluator mismatch')
    return {'case_id': case, 'source': source, 'status': 'ok', 'evaluation': value}


def field_review(scene, raw, pool, roles, prediction, target):
    """Finite evaluator-only field coverage; no claim about visual observability."""
    lookup = {r['component']: r['terminals'] for r in raw}
    mapping = dict(strict_component_pairs(prediction['components'], target['components']))
    details = []
    for c in scene.components:
        if c.type != 'box' or c.key not in mapping:
            continue
        words = canonical_tokens(pool[c.key]['words'])
        retained = [r.token for r in roles if not any(same_instance(r.token.bbox, w.bbox) for w in words)]
        adapted, _ = local_roles(c, lookup[c.key], classify_tokens([*retained, *words], [c.body_bbox or c.bbox]))
        nr = [r for r in adapted if (r.role == 'PIN_NUMBER' or (r.role == 'VALUE' and r.normalized.isdigit()))
              and ordered.PIN_NUMBER.fullmatch(r.normalized)]
        sr = [r for r in adapted if r.role in ('PIN_NAME', 'MODEL_TEXT') and ordered.SIGNAL.fullmatch(r.normalized)]
        gowner = mapping[c.key]
        gt = list(target['pins'].get(gowner, {}).items())
        points = [target_to_opencv(p['point']['x'], p['point']['y'], scene.height) for _, p in gt]
        for match in coordinate_assignment(lookup[c.key], points):
            if match['distance'] > 5:
                continue
            # Reuse the P0 assignment record schema (checked by unit/smoke tests).
            ri, gi = match['pred_index'], match['gt_index']
            terminal, pin = lookup[c.key][ri], c.pins[ri]
            key, g = gt[gi]
            suffix = key[4:] if key.startswith('pin_') else key
            number_support = [r for r in nr if r.normalized == suffix and number_bbox_score(r, terminal, c) > 0]
            name_support = [r for r in sr if r.token.text.strip() == g.get('pinname', '')
                            and ordered._name_score(r, terminal, c) > 0]
            legal = any(abs(ordered._role_tangent(a, terminal['side']) - ordered._role_tangent(b, terminal['side']))
                        <= ordered._row_pair_limit(a, b, terminal['side'], c.body_bbox or c.bbox)
                        for a in number_support for b in name_support)
            details.append({'component': c.key, 'terminal_index': ri, 'gt_owner': gowner, 'gt_key': key,
                            'gt_name': g.get('pinname', ''), 'distance': match['distance'],
                            'number_available': bool(number_support), 'name_available': bool(name_support),
                            'legal_pair_available': legal, 'selected_number': pin.number, 'selected_name': pin.name,
                            'exportable': pin.exportable,
                            'exact_fields': pin.number == suffix and pin.name == g.get('pinname', '')})
    return details


def value_gate(a, b, cases, lost, pairs):
    guards = {'TP_gain_at_least_20pct': b['tp'] > a['tp'] and b['tp'] >= 1.2 * a['tp'],
              'micro_F1_gain_at_least_20pct': b['f1'] > a['f1'] and b['f1'] >= 1.2 * a['f1'],
              'macro_F1_not_lower': b['macro_f1'] >= a['macro_f1'],
              'at_least_three_improved_cases': sum(r['after']['f1'] > r['before']['f1'] for r in cases) >= 3,
              'no_old_TP_loss': not lost}
    guards['Pin_passed'] = all(guards.values())
    guards['system_passed'] = guards['Pin_passed'] and not pairs.get('incorrect_added', 0) and not pairs.get('correct_lost', 0)
    return guards


def baseline(initial, guard):
    dest = OUT / 'baseline_complete.json'
    if dest.exists():
        data = read(dest)
        if len(data['cases']) != 35:
            raise AssertionError('Incomplete baseline seal')
        return data
    registry = build_default_registry(); rows = []; evaluations = {'design': [], 'check': []}
    started = time.perf_counter()
    for case in ALL:
        scene, raw, pool, roles, image, context, historical, meta = hydrate(case)
        baseline_scene, _ = stage_b(scene, raw, pool, roles)
        assert_invariants(scene, baseline_scene, raw)
        pred = predict(baseline_scene, image, context, registry)
        if pred != historical:
            raise AssertionError('Full baseline JSON parity failed: ' + case)
        # Reference prediction already exists and its hash is sealed here;
        # avoid 35 large duplicate result files.
        record = {**meta, 'full_JSON_parity': True, 'prediction_sha256': digest(Path(meta['historical_prediction_path']))}
        write_json(OUT / f'baseline/{case}.json', record)
        with guard.evaluating(Path(meta['target_path']), OUT / f'baseline/{case}.json'):
            if digest(Path(meta['target_path'])) != meta['target_sha256']:
                raise AssertionError('Target version drift; do not mix old and latest GT')
            gt = read(Path(meta['target_path']))
        ev = metrics(pred, gt, baseline_scene.diagnostics, case, meta['source'])
        phase = 'design' if case in DESIGN else 'check'
        evaluations[phase].append(ev)
        record['metrics'] = ev['evaluation']['metrics']
        rows.append(record)
        print(json.dumps({'baseline_case': case, 'JSON_parity': True, 'Pin': record['metrics']['Pin']}), flush=True)
    summaries = {k: aggregate(v) for k, v in evaluations.items()}
    for phase, original in (('design', 'read_design'), ('check', 'check')):
        expected = read(L3 / original / 'complete.json')['end_to_end']['B']['overall']['metrics']
        if summaries[phase]['overall']['metrics'] != expected:
            raise AssertionError('Same JSON but historical metric parity changed')
    result = {'cases': rows, 'summary': summaries, 'all_35_JSON_parity': True,
              'same_algorithm_both_groups': True, 'seconds': time.perf_counter() - started,
              'OFFICIAL_SCORE': False, 'sealed_holdout_used': False}
    verify(initial); write_json(dest, result); return result


def experiment(phase, initial, guard):
    if phase == 'check' and not read(OUT / 'design/complete.json')['guards']['system_passed']:
        raise RuntimeError('Design did not pass both gates; new Check inference prohibited')
    if not (OUT / 'baseline_complete.json').is_file():
        raise RuntimeError('Freeze unified baseline first')
    dest = OUT / phase
    if (dest / 'complete.json').exists():
        return read(dest / 'complete.json')
    registry = build_default_registry(); cases = DESIGN if phase == 'design' else CHECK
    evaluations = {'before': [], 'after': []}; rows = []; gained = []; lost = []; pairs = Counter()
    started = time.perf_counter()
    for case in cases:
        original, raw, pool, roles, image, context, historical, meta = hydrate(case)
        before, _ = stage_b(original, raw, pool, roles)
        saved = json.dumps([asdict(before), raw, pool, [asdict(r) for r in roles]], sort_keys=True)
        after, trace = stage(before, raw, pool, roles)
        assert_invariants(before, after, raw)
        if saved != json.dumps([asdict(before), raw, pool, [asdict(r) for r in roles]], sort_keys=True):
            raise AssertionError('Frozen decoder inputs mutated')
        pred = predict(after, image, context, registry)
        validate_strict(pred, (context.width, context.height))
        if pred['components'] != historical['components']:
            raise AssertionError('Component export changed')
        for c in before.components:
            if c.type != 'box' and pred['pins'][c.key] != historical['pins'][c.key]:
                raise AssertionError('Nonbox export changed')
        path = dest / f'predictions/{case}/result.json'
        write_json(path, pred)
        write_json(dest / f'predictions/{case}/diagnostics.json', after.diagnostics)
        write_json(dest / f'decode_trace/{case}.json', trace)
        with guard.evaluating(Path(meta['target_path']), path):
            if digest(Path(meta['target_path'])) != meta['target_sha256']:
                raise AssertionError('GT version drift')
            target = read(Path(meta['target_path']))
        old_ids, new_ids = strict_ids(historical, target), strict_ids(pred, target)
        gained.extend({'case_id': case, 'pin': p} for p in sorted(new_ids - old_ids))
        lost.extend({'case_id': case, 'pin': p} for p in sorted(old_ids - new_ids))
        change = {k: len(v) for k, v in pair_changes(historical, pred, target).items()}
        pairs.update(change)
        a = metrics(historical, target, before.diagnostics, case, meta['source'])
        b = metrics(pred, target, after.diagnostics, case, meta['source'])
        evaluations['before'].append(a); evaluations['after'].append(b)
        field_a = field_review(before, raw, pool, roles, historical, target)
        field_b = field_review(after, raw, pool, roles, pred, target)
        frozen_fields = ('component', 'terminal_index', 'gt_owner', 'gt_key', 'distance',
                         'number_available', 'name_available', 'legal_pair_available')
        if [[v[k] for k in frozen_fields] for v in field_a] != [[v[k] for k in frozen_fields] for v in field_b]:
            raise AssertionError('Candidate/geometry capacity changed in decoder-only experiment')
        qa = {key: {metric: evaluate_pins(value, target, 'qa_fields_only', METHODS[0])[metric]
                    for metric in ('overall', 'box')}
              for key, value in (('before', historical), ('after', pred))}
        row = {'case_id': case, 'source': meta['source'], 'before': a['evaluation']['metrics']['Pin'],
               'after': b['evaluation']['metrics']['Pin'], 'gained': len(new_ids - old_ids),
               'lost': len(old_ids - new_ids), 'pair_changes': change, 'qa_diagnostic': qa,
               'changed_box_terminals': sum((p.number, p.name, p.exportable) != (q.number, q.name, q.exportable)
                   for c, d in zip(before.components, after.components) if c.type == 'box'
                   for p, q in zip(c.pins, d.pins)), 'invariants_passed': True}
        rows.append(row)
        write_json(dest / f'field_review/{case}.json', {'before': field_a, 'after': field_b})
        write_json(dest / 'progress.json', {'cases': rows, 'gained': gained, 'lost': lost})
        print(json.dumps(row, ensure_ascii=True), flush=True)
    summaries = {k: aggregate(v) for k, v in evaluations.items()}
    a, b = (summaries[k]['overall']['metrics']['Pin'] for k in ('before', 'after'))
    guards = value_gate(a, b, rows, lost, pairs)
    result = {'phase': phase, 'cases': rows, 'summary': summaries, 'guards': guards,
              'gained': gained, 'lost': lost, 'pair_changes': dict(pairs),
              'metadata': {'seconds': time.perf_counter() - started, 'OFFICIAL_SCORE': False,
                           'sealed_holdout_used': False, 'OCR_calls': 0, 'model_calls': 0,
                           'GT_read_in_inference': False, 'blocked_access_attempts': guard.blocked}}
    verify(initial); write_json(dest / 'complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('baseline', 'design', 'check', 'verify'), required=True)
    args = parser.parse_args()
    guard = AccessGuard(); initial = freeze()
    if args.task == 'baseline':
        baseline(initial, guard)
    elif args.task in ('design', 'check'):
        experiment(args.task, initial, guard)
    verify(initial)
    write_json(OUT / f'{args.task}_verification.json', {'protected_files': len(initial['sha256']),
               'unchanged': True, 'git': git_state(), 'target_opens': guard.target_opens,
               'blocked_access_attempts': guard.blocked, 'sealed_holdout_used': False})


if __name__ == '__main__':
    main()
