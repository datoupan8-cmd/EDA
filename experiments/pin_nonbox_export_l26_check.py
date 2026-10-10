"""Finite Check replay of frozen L25; prepare never opens official targets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
from pcb.io import write_json
from tools.evaluate_pin_qa_diagnostic import (
    METHODS, POLICIES, aggregate_scenarios, digest, evaluate_pins, git_state, read)
from pin_nonbox_export_l25 import POLICY, admit, old_network_projection
from pin_nonbox_export_l25_run import mapped_pair_audit, predicted_refs

CHECK = tuple(f'{n:04d}' for n in range(5, 151, 5))
OUT = ROOT / 'reports/pin_nonbox_export_l26_check'
REFERENCE = ROOT / 'reports/pin_word_decoder_l3/check'
L1 = ROOT / 'reports/pin_learned_locator_l1'
SOURCES = ('experiments/pin_nonbox_export_l26_check.py',
           'tests/pin/test_pin_nonbox_export_l26_check.py',
           'reports/pin_nonbox_export_l26_check/experiment_plan.md')


def assert_check_read_allowed(path: str | Path, allowed_targets: set[Path]) -> None:
    """Exact 30-case allowlist, default-deny official target access."""
    p = Path(path)
    parts = str(p).replace('\\', '/').lower().split('/')
    if any(x in {'golden', 'sealed', '10gt', '10_gtcase', '10_gt'} or
           x.startswith(('golden', '10gt.', '10_gtcase.')) or
           'eda_pin_crossing_quicktest' in x for x in parts):
        raise PermissionError('Sealed/Golden/QuickTest access prohibited')
    if '200_train_cases' in parts:
        i = parts.index('200_train_cases')
        if i + 1 >= len(parts) or parts[i + 1] not in CHECK:
            raise PermissionError('Dataset access limited to fixed 30 Check cases')
    if '_target' in p.name.lower() and p.suffix.lower() == '.json':
        if p.resolve() not in allowed_targets:
            raise PermissionError('Only separate evaluation may open official targets')


class CheckReadGuard:
    def __init__(self):
        self.allowed_targets: set[Path] = set()
        self.target_opens: list[str] = []
        self.blocked: list[str] = []
        sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        try:
            assert_check_read_allowed(path, self.allowed_targets)
        except PermissionError:
            self.blocked.append(str(path))
            raise
        if '_target' in path.name.lower() and path.suffix.lower() == '.json':
            self.target_opens.append(str(path.resolve()))

    @contextmanager
    def evaluating(self, path: Path):
        assert_check_read_allowed(path, {path.resolve()})
        self.allowed_targets.add(path.resolve())
        try:
            yield
        finally:
            self.allowed_targets.remove(path.resolve())


def archived_components(front: dict, raw: list[dict], diagnostics: dict):
    """Rebuild complete archived candidates; never synthesize from exported Pins."""
    from pcb.schema import Component, Pin
    blocks = {b['component']: b for b in raw}
    events = defaultdict(list)
    for event in diagnostics['pin_events']:
        events[event['component']].append(event)
    keys = [c['key'] for c in front['components']]
    if len(keys) != len(set(keys)) or set(keys) != set(blocks):
        raise AssertionError('Raw/frontend owner mismatch or duplicate owner')
    if set(events) - set(keys):
        raise AssertionError('Events outside frontend')
    output = []
    for record in front['components']:
        c = Component(**{**record, 'pins': [], 'bbox': tuple(record['bbox']),
                         'body_bbox': tuple(record['body_bbox']) if record.get('body_bbox') else None})
        block, rows = blocks[c.key], events[c.key]
        if record['pins'] or block['type'] != c.type or len(rows) != len(block['terminals']):
            raise AssertionError('Incomplete archived terminal/event metadata')
        for event, terminal in zip(rows, block['terminals']):
            for field in ('tip', 'base', 'side', 'method'):
                if event.get(field) != terminal.get(field):
                    raise AssertionError(f'Archived event geometry/order differs: {c.key}.{field}')
            # Event omits exportable for audited simple branches: historical default is True.
            # Confidence/source fields not archived as complete dataclasses remain defaults.
            # Exact final A prediction parity is compulsory; no algorithm is re-invoked here.
            c.pins.append(Pin(str(event['number']), event['pinname'], tuple(event['tip']),
                              base=tuple(event['base']) if event.get('base') is not None else None,
                              side=event['side'], exportable=event.get('exportable', True)))
        output.append(c)
    return output


def verify(initial: dict) -> None:
    for p, h in {**initial['protected_hashes'], **initial['experiment_sources']}.items():
        if digest(Path(p)) != h:
            raise AssertionError(f'Frozen source changed: {p}')
    current = git_state()
    if any(current[k] != initial['git'][k] for k in ('head', 'branch', 'tracked_diff_sha256')):
        raise AssertionError('Tracked Git state changed')


def initial_state() -> dict:
    path = OUT / 'initial_state.json'
    if path.exists():
        value = read(path)
        verify(value)
        return value
    old = read(ROOT / 'reports/pin_nonbox_export_l25/initial_state.json')
    hashes = {**old['protected_hashes'], **old['experiment_sources']}
    for folder in ('pcb', 'configs'):
        for p in (ROOT / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                hashes[str(p)] = digest(p)
    for p in (REFERENCE / 'complete.json', L1 / 'training_manifest.json',
              ROOT / 'reports/pin_nonbox_export_l25/complete.json',
              ROOT / 'tools/evaluate_pin_qa_diagnostic.py',
              ROOT / 'experiments/pin_pair_score_l11_run.py',
              ROOT / 'experiments/wire_frame_guard.py'):
        hashes[str(p)] = digest(p)
    sources = {str(ROOT / p): digest(ROOT / p) for p in SOURCES}
    value = {'protected_hashes': hashes, 'experiment_sources': sources, 'git': git_state(),
             'policy': POLICY, 'case_ids': CHECK,
             'baseline': 'archived L3 Check C (not L13, not production current)',
             'OFFICIAL_SCORE': False, 'SEALED_HOLDOUT_USED_FOR_DEVELOPMENT': False}
    verify(value)
    write_json(path, value)
    return value


def hydrate(case: str, inputs: dict):
    from pcb.schema import Scene, Text
    from pcb.io import read_image
    from pcb.core.context import PipelineContext
    from pcb.core.config import PipelineConfig
    if case not in CHECK:
        raise ValueError('Fixed Check cases only')
    manifest = read(L1 / 'training_manifest.json')
    if tuple(f'{n:04d}' for n in manifest['validation_ids']) != CHECK or set(map(int, CHECK)) & set(manifest['train_ids']):
        raise AssertionError('Frozen train/Check split mismatch')
    complete = read(REFERENCE / 'complete.json')
    if tuple(r['case_id'] for r in complete['cases']) != CHECK:
        raise AssertionError('Historical Check manifest mismatch')
    row = next(r for r in complete['cases'] if r['case_id'] == case)
    paths = [ROOT.parent / f'tmp/pin_sina_frontend/{case}.json',
             L1 / f'validation/predictions/L1/{case}/raw_terminals.json',
             REFERENCE / f'predictions/C/{case}/diagnostics.json',
             REFERENCE / f'predictions/C/{case}/result.json']
    for p in paths:
        inputs[str(p)] = digest(p)
    front, raw, diagnostics, pred = (read(p) for p in paths)
    if inputs[str(paths[1])] != complete['metadata']['inputs'][case]['raw_sha256']:
        raise AssertionError('Historical raw terminals changed')
    for p, h in front['signature']['frontend_source_sha256'].items():
        if digest(ROOT / p.replace('\\', '/')) != h:
            raise AssertionError('Signed frontend source changed')
    folder = Path(manifest['dataset_root']) / case
    images = list(folder.glob('*.png'))
    if len(images) != 1:
        raise AssertionError('Unique PNG not resolved within exact Check folder')
    image_path = images[0]
    inputs[str(image_path)] = digest(image_path)
    if (inputs[str(image_path)] != manifest['image_sha256'][str(int(case))] or
        inputs[str(image_path)] != front['signature']['image_sha256'] or
        inputs[str(image_path)] != complete['metadata']['inputs'][case]['image_sha256']):
        raise AssertionError('Image version drift')
    image = read_image(image_path)
    h, w = image.shape[:2]
    target_path = image_path.with_name(image_path.stem + '_target.json')
    if not target_path.is_file():
        raise AssertionError('Expected target path absent (content not read)')
    scene = Scene(w, h, archived_components(front, raw, diagnostics),
                  [Text(**{**t, 'bbox': tuple(t['bbox'])}) for t in front['texts']],
                  diagnostics=copy.deepcopy(diagnostics))
    config = PipelineConfig.load(ROOT / 'configs/current.json')
    if asdict(config) != front['signature']['config']:
        raise AssertionError('Frontend config no longer matches current baseline')
    context = PipelineContext(image_path.name, w, h, config, None)
    meta = {'case_id': case, 'width': w, 'height': h, 'image_name': image_path.name,
            'image_sha256': inputs[str(image_path)], 'source': row['source'],
            'target_path': str(target_path), 'target_sha256': row['target_sha256']}
    return scene, diagnostics['pin_events'], raw, image, context, pred, meta


def generate(case: str, initial: dict) -> dict:
    from pin_pair_score_l11_run import json_hash, predict
    from pcb.core.registry import build_default_registry
    from pcb.submission import validate_strict
    inputs = {}
    scene, events, raw, image, context, historical, meta = hydrate(case, inputs)
    signature = json_hash([asdict(scene), events, raw])
    a_scene = copy.deepcopy(scene)
    b_scene, trace = admit(scene, events)
    if signature != json_hash([asdict(scene), events, raw]):
        raise AssertionError('Frozen inference inputs mutated')
    registry = build_default_registry()
    a, a_geo = predict(a_scene, image, context, registry)
    if a != historical:
        raise AssertionError(f'Historical complete A prediction replay mismatch: {case}')
    b, b_geo = predict(b_scene, image, context, registry)
    if a_geo != b_geo or a['components'] != b['components']:
        raise AssertionError('Component/downstream geometry changed')
    for ck, pp in a['pins'].items():
        if any(b['pins'][ck].get(pk) != p for pk, p in pp.items()):
            raise AssertionError('Old output Pin changed/lost')
        if a['components'][ck]['type'] == 'box' and pp != b['pins'][ck]:
            raise AssertionError('Box output changed')
    if old_network_projection(a, predicted_refs(a)) != old_network_projection(b, predicted_refs(a)):
        raise AssertionError('Old-reference network members or edges changed')
    files = {}
    for variant, pred, current in [('A', a, a_scene), ('B', b, b_scene)]:
        validate_strict(pred, (context.width, context.height))
        for filename, data in [('result.json', pred), ('diagnostics.json', current.diagnostics),
                               ('internal_pins.json', {c.key: [asdict(p) for p in c.pins] for c in current.components})]:
            p = OUT / f'predictions/{variant}/{case}/{filename}'
            write_json(p, data)
            files[str(p)] = digest(p)
    value = {'case_id': case, 'meta': meta, 'trace': trace, 'input_sha256': inputs,
             'prediction_sha256': files, 'geometry': a_geo,
             'A_full_prediction_equal': True, 'candidate_geometry_invariant': True,
             'downstream_geometry_equal': True, 'old_ref_subgraph_equal': True,
             'admitted': sum(t['admitted'] for t in trace),
             'admitted_by_type': dict(Counter(t['type'] for t in trace if t['admitted'])),
             'rejected_reasons': dict(Counter(t['reason'] for t in trace if not t['admitted']))}
    verify(initial)
    for p, h in inputs.items():
        if digest(Path(p)) != h:
            raise AssertionError('Input changed during generation')
    write_json(OUT / f'prepared_cases/{case}.json', value)
    print(json.dumps({k: value[k] for k in ('case_id', 'admitted', 'admitted_by_type', 'A_full_prediction_equal')}, ensure_ascii=False), flush=True)
    return value


def prepare(smoke=False):
    started = time.perf_counter()
    guard = CheckReadGuard()
    initial = initial_state()
    if (OUT / 'prepared.json').exists():
        raise RuntimeError('Check already sealed; no repeated selection')
    if not read(ROOT / 'reports/pin_nonbox_export_l25/complete.json')['mechanism_and_safety_passed']:
        raise RuntimeError('L25 Design gate did not pass')
    records = []
    for case in CHECK[:2] if smoke else CHECK:
        path = OUT / f'prepared_cases/{case}.json'
        if path.exists():
            row = read(path)
            for p, h in {**row['input_sha256'], **row['prediction_sha256']}.items():
                if digest(Path(p)) != h:
                    raise AssertionError('Saved smoke/prepare changed')
        else:
            row = generate(case, initial)
        records.append(row)
    if guard.target_opens or guard.blocked:
        raise AssertionError('Unexpected inference target/forbidden read')
    write_json(OUT / ('smoke.json' if smoke else 'prepared.json'),
               {'cases': records, 'all_predictions_saved_before_target_reads': True,
                'target_reads': 0, 'new_OCR_YOLO_VLM_solver_localization_calls': 0,
                'seconds': time.perf_counter() - started, 'OFFICIAL_SCORE': False})
    print(json.dumps({'phase': 'smoke' if smoke else 'prepare', 'cases': len(records), 'target_reads': 0}, ensure_ascii=False), flush=True)


def evaluate():
    from pin_g_wire_e1 import pair_changes
    from pcb.coordinates import opencv_to_target
    from pcb.submission import validate_strict
    started = time.perf_counter()
    guard = CheckReadGuard()
    initial = read(OUT / 'initial_state.json')
    verify(initial)
    if (OUT / 'complete.json').exists():
        raise RuntimeError('One frozen Check evaluation only; no re-tuning')
    prepared = read(OUT / 'prepared.json')
    if tuple(r['case_id'] for r in prepared['cases']) != CHECK:
        raise AssertionError('All 30 predictions must be sealed before target access')
    preds = {'A': {}, 'B': {}}
    for row in prepared['cases']:
        for p, h in {**row['input_sha256'], **row['prediction_sha256']}.items():
            if digest(Path(p)) != h:
                raise AssertionError('Sealed input/prediction changed')
        for variant in preds:
            pred = read(OUT / f'predictions/{variant}/{row["case_id"]}/result.json')
            validate_strict(pred, (row['meta']['width'], row['meta']['height']))
            preds[variant][row['case_id']] = pred
    targets = {}
    for row in prepared['cases']:
        path = Path(row['meta']['target_path'])
        with guard.evaluating(path):
            raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['meta']['target_sha256']:
            raise AssertionError('Official target version drift: stop for version analysis')
        targets[row['case_id']] = json.loads(raw.decode('utf-8-sig'))
    aggregate, details, admitted_rows, network_audits, legacy_pairs = {}, [], [], {}, {}
    old_correspondence_losses = []
    for policy in POLICIES:
        for method in METHODS if policy != 'legacy_strict' else METHODS[:1]:
            label = policy if method == METHODS[0] else policy + '_min_distance_sensitivity'
            grouped = {v: [] for v in preds}
            network_audits[label] = []
            for row in prepared['cases']:
                case = row['case_id']
                a, b, target = preds['A'][case], preds['B'][case], targets[case]
                qa = {v: evaluate_pins(preds[v][case], target, policy, method) for v in preds}
                for variant, score in qa.items():
                    score.update(case_id=case, source=row['meta']['source'])
                    grouped[variant].append(score)
                if qa['A']['box'] != qa['B']['box']:
                    raise AssertionError('Box metrics changed')
                old_gt = {tuple(m['gt']) for m in qa['A']['matches']}
                new_gt = {tuple(m['gt']) for m in qa['B']['matches']}
                old_correspondence_losses.extend({'scenario': label, 'case_id': case, 'pin': x} for x in sorted(old_gt - new_gt))
                if policy != 'legacy_strict':
                    network_audits[label].append({'case_id': case, **mapped_pair_audit(a, b, target, qa['A'], qa['B'])})
                if label == 'legacy_strict':
                    legacy_pairs[case] = pair_changes(a, b, target)
                if label == 'qa_fields_only':
                    by_pred = {tuple(m['pred']): m for m in qa['B']['matches']}
                    for item in row['trace']:
                        if not item['admitted']:
                            continue
                        owner, pin = item['owner'], item['new_pin']
                        gk = qa['B']['component_key_map'].get(owner)
                        xy = opencv_to_target(*pin['tip'], row['meta']['height'])
                        possible = target['pins'].get(gk, {}) if gk else {}
                        nearest = min(((math.dist(xy, (p['point']['x'], p['point']['y'])), pk) for pk, p in possible.items()), default=None, key=lambda x: x[0])
                        match = by_pred.get((owner, 'pin_' + pin['number']))
                        bucket = ('OWNER_NOT_MAPPED' if not gk else 'NO_GT_PIN_ON_OWNER' if not nearest else
                                  'CORRECT_5PX_ONE_TO_ONE' if match else
                                  'NEAR_5PX_BUT_NOT_ASSIGNED' if nearest[0] <= 5 else
                                  'OFFSET_5_TO_10PX' if nearest[0] <= 10 else
                                  'OFFSET_10_TO_20PX' if nearest[0] <= 20 else 'FARTHER_THAN_20PX')
                        admitted_rows.append({'case_id': case, 'owner': owner, 'type': item['type'],
                                              'terminal': item['terminal'], 'local_number': pin['number'],
                                              'point': xy, 'GT_owner': gk, 'bucket': bucket,
                                              'nearest_distance_unconstrained': nearest[0] if nearest else None,
                                              'nearest_GT_pin': nearest[1] if nearest else None, 'match': match})
            summary = {v: aggregate_scenarios(rows) for v, rows in grouped.items()}
            if label == 'legacy_strict':
                historical = read(REFERENCE / 'complete.json')['metrics']['C']
                for field in ('tp', 'pred', 'gt', 'f1', 'macro_f1'):
                    if abs(summary['A']['overall'][field] - historical[field]) > 1e-12:
                        raise AssertionError('Historical strict Check metrics not reproduced')
            aggregate[label] = summary
            details.append({'scenario': label, 'variants': grouped})
    principal = aggregate['qa_fields_only']
    a, b = principal['A']['nonbox'], principal['B']['nonbox']
    primary = next(d for d in details if d['scenario'] == 'qa_fields_only')['variants']
    improved, regressed, unchanged = [], [], []
    for x, y in zip(primary['A'], primary['B']):
        dest = improved if y['nonbox']['f1'] > x['nonbox']['f1'] else regressed if y['nonbox']['f1'] < x['nonbox']['f1'] else unchanged
        dest.append(x['case_id'])
    audits = [r for rows in network_audits.values() for r in rows]
    guards = {'nonbox_TP_and_micro_F1_net_gain': b['tp'] > a['tp'] and b['f1'] > a['f1'],
              'nonbox_relative_gain_20pct': b['tp'] >= 1.2 * a['tp'] or b['f1'] >= 1.2 * a['f1'],
              'at_least_3_nonbox_cases_improved': len(improved) >= 3,
              'nonbox_macro_not_lower': b['macro_f1'] >= a['macro_f1'],
              'box_and_old_Pin_correspondence_no_loss': not old_correspondence_losses,
              'old_ref_subgraph_and_downstream_geometry_equal': True,
              'no_new_verifiable_wrong_Pair_under_assumptions': not any(r['incorrect_added'] for r in audits),
              'no_verifiable_correct_Pair_loss_under_assumptions': not any(r['correct_lost'] for r in audits)}
    if guard.blocked or len(guard.target_opens) != len(CHECK):
        raise AssertionError('Unexpected target/forbidden access')
    verify(initial)
    payload = {'stage': 'L26 frozen L25 30-case Check regression', 'aggregate': aggregate,
               'baseline': initial['baseline'], 'details': details, 'admitted_pin_attribution': admitted_rows,
               'admitted_buckets': dict(Counter(r['bucket'] for r in admitted_rows)),
               'admitted_by_type': dict(Counter(r['type'] for r in admitted_rows)),
               'legacy_raw_reference_Pair_changes': legacy_pairs,
               'evaluator_only_mapped_Pair_audits': network_audits, 'guards': guards,
               'old_correspondence_losses': old_correspondence_losses,
               'mechanism_and_safety_passed': all(guards.values()),
               'improved_nonbox_cases': improved, 'regressed_nonbox_cases': regressed,
               'unchanged_nonbox_cases': unchanged, 'adopt_current': False, 'official_reply_pending': True,
               'OFFICIAL_SCORE': False, 'FinalScore': 'NOT_COMPUTED',
               'SEALED_HOLDOUT_USED_FOR_DEVELOPMENT': False, 'Check_run': True, 'full150_run': False,
               'new_OCR_YOLO_VLM_solver_localization_calls': 0, 'inference_target_reads': 0,
               'evaluation_target_reads': len(guard.target_opens), 'blocked_access_attempts': guard.blocked,
               'historical_check_used_in_past_development': True, 'Check_used_for_parameter_tuning_this_run': False,
               'candidate_geometry_and_box_outputs_unchanged': True,
               'seconds': time.perf_counter() - started, 'source_and_input_hashes_unchanged': True}
    write_json(OUT / 'complete.json', payload)
    print(json.dumps({'L26_complete': True, 'nonbox_A': a, 'nonbox_B': b,
                      'overall_A': principal['A']['overall'], 'overall_B': principal['B']['overall'],
                      'guards': guards, 'admitted_buckets': payload['admitted_buckets'],
                      'seconds': payload['seconds']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('smoke', 'prepare', 'evaluate'))
    args = parser.parse_args()
    if args.phase == 'evaluate':
        evaluate()
    else:
        prepare(smoke=args.phase == 'smoke')
