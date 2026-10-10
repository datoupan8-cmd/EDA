"""Finite cached A30 admission audit; no inference, solver, or target access.

The existing pure candidate functions are the reference, not a new decoder.
Image review is separate from string equality; no counterfactual predictions
or new accuracy scores are produced by this utility.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT)]
OUT = ROOT / 'reports/pin_pair_admission_l10'
DESIGN = ('0014', '0017', '0018', '0087', '0103')
A_CATEGORY = 'both_role_fields_available_pair_incompatible_or_rejected'


class ReadGuard:
    """Block all targets and reserved images, even accidental helper access."""
    def __init__(self):
        self.blocked = 0
        sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        value = str(args[0]).replace('\\', '/').lower()
        parts = value.split('/')
        forbidden = any(p in {'10gt', '10_gtcase', 'golden', 'sealed',
                              'eda_pin_crossing_quicktest'} for p in parts)
        if '200_train_cases' in parts:
            i = parts.index('200_train_cases')
            if i + 1 < len(parts):
                part = parts[i + 1]
                forbidden |= not (part.isdigit() and 1 <= int(part) <= 150)
        if forbidden or value.endswith('_target.json'):
            self.blocked += 1
            raise PermissionError('L10 blocks target / sealed / QuickTest reads')


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def git_state() -> dict:
    def query(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                              text=True, encoding='utf-8', check=True).stdout
    return {k: query(*args) for k, args in {
        'head': ('rev-parse', 'HEAD'), 'branch': ('branch', '--show-current'),
        'status': ('status', '--short'), 'diff_names': ('diff', '--name-only'),
        'diff_stat': ('diff', '--stat'), 'diff': ('diff',)}.items()}


def protected_snapshot() -> dict:
    files = [p for folder in ('pcb', 'configs') for p in (ROOT / folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    files += [ROOT / p for p in ('evaluate_v2.py', 'experiments/pin_word_decoder_l3.py',
        'experiments/pin_word_reader_l3.py', 'experiments/pin_joint_abstention_l7_2.py',
        'experiments/pin_joint_abstention_l7_4.py', 'experiments/pin_word_resources_l7_3.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def pair_checks(num: dict, name: dict, old, old_score: float, policy) -> dict:
    """Trace the original short-circuit order, plus independent violations.

    Replacement checks behind a geometry failure are explicitly hypothetical.
    They use the frozen original old_score, not a refitted counterfactual score.
    """
    ng, sg = num['geometry'], name['geometry']
    quality = (num['score'] + name['score']) / 2
    tangent_delta = abs(ng['tangent'] - sg['tangent'])
    tangent_limit = max(8., .75 * max(ng['height'], sg['height']))
    checks = [
        ('same_physical_word', num['word'] == name['word']),
        ('number_not_outward_of_name', ng['normal'] > sg['normal'] - 1.),
        ('number_name_tangent_separation', tangent_delta > tangent_limit),
        ('minimum_pair_quality', quality < policy.minimum_pair_score),
    ]
    import pin_word_decoder_l3 as decoder
    accepted, reason = decoder.replacement_allowed(old, num, name, quality, old_score, policy)
    # Independent replacement predicates are diagnostic only; first failure
    # still comes from the exact existing replacement_allowed function.
    replacement = []
    if old.exportable:
        cn = num['text'] != str(old.number)
        sn = name['text'] != str(old.name or '')
        if cn and not {'strip', 'direct'} <= set(num['families']):
            replacement.append('new_number_lacks_strip_direct_agreement')
        if sn:
            before = decoder.normalize(old.name or '')
            if before and name['text'] != before and name['text'] in before:
                replacement.append('proper_substring_cannot_replace_full_name')
            if not name['corroborated']:
                replacement.append('new_name_lacks_complete_word_agreement')
        if (cn or sn) and quality <= old_score + policy.replacement_margin:
            replacement.append('insufficient_joint_improvement')
    preliminary = [k for k, failed in checks if failed]
    first = preliminary[0] if preliminary else (None if accepted else reason)
    violations = preliminary + replacement
    return {'first_blocker': first, 'preliminary_checks_pass': not preliminary,
        'all_independent_violations': violations,
        'sole_violation_at_frozen_old_score': violations[0] if len(violations) == 1 else None,
        'replacement_check': {'accepted': accepted, 'reason': reason,
                              'hypothetical_after_preliminary_failure': bool(preliminary)},
        'quality': quality, 'minimum_quality': policy.minimum_pair_score,
        'old_score': old_score, 'replacement_margin': policy.replacement_margin,
        'number_normal_inside': ng['normal'], 'name_normal_inside': sg['normal'],
        'required_number_normal_max': sg['normal'] - 1.,
        'tangent_delta': tangent_delta, 'tangent_limit': tangent_limit}


def evidence(candidate: dict, words: dict) -> dict:
    import pin_word_decoder_l3 as decoder
    result = copy.deepcopy(candidate)
    result['exact_readings'] = [copy.deepcopy(r) for r in words[candidate['word']]['readings']
        if decoder.normalize(r['text']) == candidate['text']]
    result['all_readings'] = copy.deepcopy(words[candidate['word']]['readings'])
    return result


def short_option(option: dict) -> dict:
    fields = ('terminal', 'side', 'number', 'name', 'number_word', 'name_word',
              'score', 'quality', 'fallback', 'abstain', 'reason',
              'number_tangent', 'name_tangent')
    return {k: option.get(k) for k in fields}


def hydrate_case(key: str, inputs: dict):
    """Deserialize saved image-only data; never call a Stage or load_inputs."""
    from pcb.schema import Component, Pin
    paths = {'frontend': ROOT.parent / f'tmp/pin_sina_frontend/{key}.json',
        'raw': ROOT / f'reports/pin_terminal_reader_l2/design/raw_terminals/{key}.json',
        'baseline': ROOT / f'reports/pin_terminal_reader_l2/design/predictions/L1/{key}/diagnostics.json',
        'pool': ROOT / f'reports/pin_word_resources_l7_3/mechanism/pools/{key}.json',
        'trace': ROOT / f'reports/pin_joint_abstention_l7_4/design/trace/{key}.json'}
    for p in paths.values():
        inputs[str(p)] = digest(p)
    front, raw, baseline, pool, trace = (read(paths[k]) for k in paths)
    components = {}
    events = defaultdict(list)
    for ev in baseline['pin_events']:
        events[ev['component']].append(ev)
    terms = {r['component']: r['terminals'] for r in raw}
    for row in front['components']:
        c = Component(**{**row, 'bbox': tuple(row['bbox']),
            'body_bbox': tuple(row['body_bbox']) if row.get('body_bbox') else None})
        evs = events[c.key]
        if len(evs) != len(terms[c.key]):
            raise AssertionError('Baseline event / raw terminal count mismatch')
        for ev, t in zip(evs, terms[c.key]):
            if ev['side'] != t['side'] or math.dist(ev['tip'], t['tip']) > 1e-7:
                raise AssertionError('Baseline events / terminals differ')
        c.pins = [Pin(str(ev.get('number')), ev.get('pinname'), tuple(ev['tip']),
            base=tuple(ev['base']), side=ev['side'], exportable=ev.get('exportable', True))
            for ev in evs]
        components[c.key] = c
    # Compare the exact saved image-only frontend sources, no cache fallback.
    for name, expected in front['signature']['frontend_source_sha256'].items():
        if digest(ROOT / name.replace('\\', '/')) != expected:
            raise AssertionError('Saved frontend source changed')
    return components, terms, pool, {r['component']: r for r in trace}, front['signature']


def rebuild_records(records: list, inputs: dict) -> tuple[list, dict, dict]:
    import pin_word_decoder_l3 as decoder
    # Prohibit accidental optimizer use even if the audit is later edited.
    def prohibited(*args, **kwargs):
        raise RuntimeError('L10 forbids optimizer execution')
    decoder.solve_options = prohibited
    decoder.milp = prohibited
    contexts, parity, rows = {}, Counter(), []
    for key in DESIGN:
        components, terms, pool, trace, signature = hydrate_case(key, inputs)
        contexts[key] = signature
        owners = {r['owner'] for r in records if r['case_id'] == key}
        for owner in sorted(owners):
            c, ts, words = components[owner], terms[owner], pool[owner]['words']
            original = json.dumps([asdict(c), ts, words], sort_keys=True)
            options, rejected = decoder.pair_options(words, c, ts, c.pins)
            wordmap = {w['id']: w for w in words}
            selected = {o['terminal']: o for o in trace[owner]['selected']}
            for record in [r for r in records if r['case_id'] == key and r['owner'] == owner]:
                i = record['terminal']; t = ts[i]; old = c.pins[i]
                nums = decoder.role_options(words, c, t, ts, 'number')
                names = decoder.role_options(words, c, t, ts, 'name')
                number, name = record['GT_pin'].removeprefix('pin_'), record['GT_pinname']
                nr = [r for r in nums if r['text'] == number]
                sr = [r for r in names if r['text'] == name]
                exact = [o for o in options if o['terminal'] == i and o['number'] == number and o['name'] == name]
                event = record['D_event']
                if t['side'] != event['side'] or math.dist(t['tip'], event['tip']) > 1e-7:
                    raise AssertionError('Saved evaluated terminal geometry differs')
                se = event.get('exportable', True) and str(event.get('number')) == number and event.get('pinname') == name
                category = ('selected_exact_fields' if se else
                    'exact_pair_available_not_selected' if exact else
                    A_CATEGORY if nr and sr else
                    'number_role_candidate_missing' if not nr and sr else
                    'name_role_candidate_missing' if nr and not sr else 'both_role_candidates_missing')
                if (category != record['category'] or bool(nr) != record['number_role_candidate']
                    or bool(sr) != record['name_role_candidate']
                    or bool(exact) != record['exact_pair_available']
                    or [short_option(o) for o in exact] != record['exact_options']):
                    raise AssertionError(f'Frozen category/options replay mismatch: {key}/{owner}/{i}')
                parity[category] += 1
                if category != A_CATEGORY:
                    continue
                fallback = next(o for o in options if o['terminal'] == i and o['fallback'])
                combos = []
                for num in nr:
                    for nm in sr:
                        checks = pair_checks(num, nm, old, fallback['quality'], decoder.POLICY)
                        if checks['first_blocker'] is None:
                            raise AssertionError('A30 exact pair unexpectedly admitted')
                        combos.append({'id': f'{key}_{owner}_{i}_p{len(combos)}',
                            'number': evidence(num, wordmap), 'name': evidence(nm, wordmap), **checks})
                rows.append({'case_id': key, 'owner': owner, 'GT_owner': record['GT_owner'],
                    'terminal': i, 'GT_pin': record['GT_pin'], 'GT_pinname': name,
                    'point_error': record['point_error'], 'terminal_geometry': copy.deepcopy(t),
                    'body_bbox': list(c.body_bbox or c.bbox), 'old_pin': asdict(old),
                    'saved_D_event': event, 'saved_selected': short_option(selected[i]),
                    'number_role_options': nums, 'name_role_options': names,
                    'exact_combinations': combos,
                    'saved_rejected_replacements': [r for r in rejected if r['terminal'] == i]})
            if json.dumps([asdict(c), ts, words], sort_keys=True) != original:
                raise AssertionError('Read-only reconstruction mutated frozen input')
    if sum(parity.values()) != 314 or len(rows) != 30:
        raise AssertionError('Fixed audit population changed')
    return rows, dict(parity), contexts


def make_review_cards(rows: list, signatures: dict, inputs: dict) -> list:
    """Render actual source pixels and word extents; annotations are not OCR."""
    import cv2
    import numpy as np
    manifest_path = ROOT / 'reports/pin_learned_locator_l1/training_manifest.json'
    inputs[str(manifest_path)] = digest(manifest_path)
    manifest = read(manifest_path)
    dataset = Path(manifest['dataset_root'])
    cards, image_cache, review = [], {}, []
    for row in rows:
        key = row['case_id']
        if key not in image_cache:
            # Only open the four preselected development case directories.
            paths = list((dataset / key).glob('*.png'))
            if len(paths) != 1:
                raise RuntimeError(f'Image path not unambiguous for {key}')
            path = paths[0]
            sha = digest(path)
            if sha != signatures[key]['image_sha256'] or sha != manifest['image_sha256'][str(int(key))]:
                raise AssertionError('Image hash differs from frozen frontend / training manifest')
            inputs[str(path)] = sha
            image_cache[key] = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = image_cache[key]; h, w = image.shape[:2]
        for p in row['exact_combinations']:
            term = row['terminal_geometry']
            boxes = [p[k]['bbox'] for k in ('number', 'name')]
            pts = [term['tip'], term['base']]
            x1 = max(0, math.floor(min([b[0] for b in boxes] + [q[0] for q in pts]) - 14))
            y1 = max(0, math.floor(min([b[1] for b in boxes] + [q[1] for q in pts]) - 14))
            x2 = min(w, math.ceil(max([b[2] for b in boxes] + [q[0] for q in pts]) + 14))
            y2 = min(h, math.ceil(max([b[3] for b in boxes] + [q[1] for q in pts]) + 14))
            crop = image[y1:y2, x1:x2].copy()
            overlay = crop.copy()
            for kind, color in (('number', (30, 180, 30)), ('name', (240, 80, 30))):
                a, b, c, d = p[kind]['bbox']
                cv2.rectangle(overlay, (round(a-x1), round(b-y1)), (round(c-x1), round(d-y1)), color, 1)
            bx = row['body_bbox']
            cv2.rectangle(overlay, (round(bx[0]-x1), round(bx[1]-y1)),
                          (round(bx[2]-x1), round(bx[3]-y1)), (160, 160, 160), 1)
            for point, color in ((term['tip'], (0, 0, 255)), (term['base'], (200, 0, 200))):
                cv2.circle(overlay, (round(point[0]-x1), round(point[1]-y1)), 2, color, -1)
            scale = min(5., 570 / max(1, crop.shape[1]), 205 / max(1, crop.shape[0]))
            # Each row shows unmarked pixels on the left and annotations on the right.
            card = np.full((310, 1200, 3), 255, np.uint8)
            title = f"{p['id']}  {term['side']}  GT:{row['GT_pin']}/{row['GT_pinname']}  D:{row['saved_D_event'].get('number')}/{row['saved_D_event'].get('pinname')}"
            cv2.putText(card, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(card, 'FIRST: ' + p['first_blocker'], (12, 46), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
            note = f"normal num/name={p['number_normal_inside']:.2f}/{p['name_normal_inside']:.2f}; tangent gap/limit={p['tangent_delta']:.2f}/{p['tangent_limit']:.2f}; q={p['quality']:.3f}; old={p['old_score']:.3f}"
            cv2.putText(card, note, (12, 69), cv2.FONT_HERSHEY_SIMPLEX, .47, (0, 0, 0), 1, cv2.LINE_AA)
            for panel, left in ((crop, 10), (overlay, 610)):
                resized = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                rh, rw = resized.shape[:2]
                card[90:90+rh, left:left+rw] = resized
            path = OUT / 'review' / (p['id'] + '.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode('.png', card)[1].tofile(str(path))
            cards.append(card)
            p['review_image'] = str(path)
            p['review_roi'] = [x1, y1, x2, y2]
            review.append({'id': p['id'], 'case_id': key, 'owner': row['owner'],
                'terminal': row['terminal'], 'GT_pin': row['GT_pin'], 'GT_pinname': row['GT_pinname'],
                'review_image': str(path), 'review_roi': [x1, y1, x2, y2],
                'physical_ownership': 'REVIEW_REQUIRED', 'visible_number': None,
                'visible_name': None, 'notes': ''})
    for start in range(0, len(cards), 5):
        sheet = np.concatenate(cards[start:start+5])
        cv2.imencode('.png', sheet)[1].tofile(str(OUT / 'review' / f'sheet_{start//5+1:02d}.png'))
    return review


def run() -> None:
    if (OUT / 'admission_audit.json').exists():
        raise RuntimeError('Finite audit already saved; do not rerun for selection')
    guard = ReadGuard()
    initial, protected = git_state(), protected_snapshot()
    inputs = {}
    source = ROOT / 'reports/pin_joint_abstention_l7_4/finite_attribution.json'
    inputs[str(source)] = digest(source)
    frozen = read(source)
    records = frozen['candidate_coverage']['records']
    rows, parity, signatures = rebuild_records(records, inputs)
    if parity != {k: v for k, v in frozen['candidate_coverage']['counts'].items()
                  if k != 'geometrically_accurate_pairs'}:
        raise AssertionError('Fixed category totals differ')
    review = make_review_cards(rows, signatures, inputs)
    first = Counter(p['first_blocker'] for r in rows for p in r['exact_combinations'])
    record_first = Counter()
    for r in rows:
        blockers = sorted({p['first_blocker'] for p in r['exact_combinations']})
        r['record_first_blockers'] = blockers
        record_first[blockers[0] if len(blockers) == 1 else 'MIXED_FIRST_BLOCKERS'] += 1
    import pin_word_decoder_l3 as decoder
    write(OUT / 'admission_audit.json', {'metadata': {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'case_ids': sorted({r['case_id'] for r in rows}),
        'candidate_policy': asdict(decoder.POLICY), 'OCR_runs': 0, 'VLM_runs': 0,
        'solver_runs': 0, 'stage_runs': 0, 'new_predictions': 0, 'target_reads': 0,
        'OFFICIAL_SCORE': False, 'sealed_holdout_used': False,
        'coordinate_system': 'Original image pixels, top-left; cached terminal tip/base and OCR boxes',
        'caveat': 'Exact local strings and sole violations are not attainable TP forecasts; no relaxation/solver run.'},
        'parity': {'replayed_records': 314, 'category_counts': parity, 'all_categories_and_exact_options_identical': True},
        'counts': {'records': len(rows), 'exact_string_combinations': sum(len(r['exact_combinations']) for r in rows),
                   'record_first_blocker': dict(record_first), 'combination_first_blocker': dict(first)},
        'records': rows})
    write(OUT / 'physical_review_template.json', review)
    # Record a local user whitespace edit without restoring or rewriting it.
    head_source = subprocess.run(['git', 'show', 'HEAD:pcb/pin_semantics.py'], cwd=ROOT,
        capture_output=True, text=True, encoding='utf-8', check=True).stdout
    now_source = (ROOT / 'pcb/pin_semantics.py').read_text(encoding='utf-8')
    ast_equal = ast.dump(ast.parse(head_source)) == ast.dump(ast.parse(now_source))
    final = protected_snapshot()
    if final != protected or any(digest(Path(p)) != sha for p, sha in inputs.items()):
        raise AssertionError('Protected algorithm/config or frozen evidence changed')
    write(OUT / 'verification.json', {'initial_git_state': initial, 'final_git_state': git_state(),
        'protected_sha256_before': protected, 'protected_sha256_after': final,
        'protected_byte_identical_this_audit': True, 'frozen_input_sha256': inputs,
        'preexisting_pin_semantics_edit_ast_equal_HEAD': ast_equal,
        'historical_hash_warning': 'Current pin_semantics.py has a preexisting trailing-empty-line deletion. '
            'This audit preserves it and does not claim byte parity with older freezes.',
        'blocked_access_attempts': guard.blocked, 'target_reads': 0,
        'formal_config_changed': False, 'Check_run': False, 'full_150_run': False,
        'commit': False, 'push': False, 'OFFICIAL_SCORE': False, 'sealed_holdout_used': False,
        'audit_script_sha256': digest(Path(__file__))})
    print(json.dumps({'parity': parity, 'record_first_blocker': dict(record_first),
        'combination_first_blocker': dict(first), 'review_cards': len(review),
        'protected_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
