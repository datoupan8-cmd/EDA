"""Read-only identity diagnostics on sealed predictions for cases 0001..0150.

GT is used only to classify saved observations. No predictions, OCR, model
parameters, identities, or inference rules are generated or changed here.
Geometry matching ignores names/types, maximizes admissible cardinality and
then IoU, and keeps competing matches explicitly ambiguous.
"""
import argparse
from collections import Counter, defaultdict
import importlib.metadata
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from pcb.component_proposal_fusion import ComponentProposal
from pcb.component.latest.text_assignment import designator_weight, _same_family
from pcb.component.latest.text_roles import DESIGNATOR_RE, TokenRole, prefix_type
from pcb.schema import Text
from tools.team_access_guard import Guard
from .power_comparison import IDS, discover, read, write, sha
from . import peer_protocol2055 as peer


IOU_THRESHOLD = .5  # Diagnostic convention, not selected against the scores.
ASSIGNMENT_THRESHOLD = .055  # Existing production rule, not a new threshold.


def canonical(text):
    """Diagnostic comparison only; never used to rewrite an output key."""
    return re.sub(r'\s+', '', str(text)).casefold()


def boxes_iou(left, right):
    if not left or not right:
        return np.zeros((len(left), len(right)))
    a = np.asarray([r['bbox'] for r in left], float)
    b = np.asarray([r['bbox'] for r in right], float)
    assert all(peer.valid_bbox(r['bbox']) for r in left+right)
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(high-low, 0).prod(axis=2)
    union = (a[:, 2:]-a[:, :2]).prod(axis=1)[:, None] + (b[:, 2:]-b[:, :2]).prod(axis=1)[None, :] - intersection
    return intersection/union


def geometry_matches(targets, observations):
    """One-to-one, name/type-blind matching; competition is not a hard label."""
    overlaps = boxes_iou(targets, observations)
    eligible = overlaps >= IOU_THRESHOLD
    assigned = {}
    if overlaps.size:
        cardinality_weight = min(overlaps.shape)+1
        rows, cols = linear_sum_assignment(-(eligible*(cardinality_weight+overlaps)))
        assigned = {int(i): int(j) for i, j in zip(rows, cols) if eligible[i, j]}
    result = []
    for i in range(len(targets)):
        candidates = [int(j) for j in np.flatnonzero(eligible[i])]
        j = assigned.get(i)
        competing = len(candidates) > 1 or any(int(eligible[:, k].sum()) > 1 for k in candidates)
        state = 'ambiguous' if competing else 'matched' if j is not None else 'no_strict_match'
        result.append({'status': state, 'index': j, 'candidate_count': len(candidates),
                       'iou': float(overlaps[i, j]) if j is not None else None,
                       'best_any_iou': float(overlaps[i].max()) if observations else 0.,
                       'candidates': [{'index': k, 'key': observations[k]['key'],
                                       'type': observations[k]['type'], 'iou': float(overlaps[i, k])}
                                      for k in candidates]})
    return result


def top_rows(components, height, native=False):
    rows = []
    source = ((c['key'], c) for c in components) if native else components.items()
    for key, c in source:
        b = c['bbox']
        rows.append({**c, 'key': key, 'bbox': list(b) if native else [b[0], height-b[3], b[2], height-b[1]]})
    return rows


def ocr_evidence(expected, scene, body):
    """Report literal/role evidence, not a claim that a glyph was never read."""
    diagnostic = scene['diagnostics']
    pool = [{**t, 'source': t.get('source_id', 'shared')} for t in scene['texts']]
    pool += [{**t, 'source': 'private_parseq'}
             for t in diagnostic.get('component_local_recognition', {}).get('accepted', [])]
    literal = [t for t in pool if canonical(t['text']) == canonical(expected)]
    literal_roles = Counter(r['role']+':'+r.get('reason', '') for r in diagnostic['text_roles']
                            if canonical(r['text']) == canonical(expected))
    role_rows = []
    for r in diagnostic['text_roles']:
        if r['role'] != 'DESIGNATOR':
            continue
        m = DESIGNATOR_RE.fullmatch(re.sub(r'\s+', '', r['text']))
        if not m or canonical(m.group(1)) != canonical(expected):
            continue
        text = Text(r['text'], tuple(r['bbox']), r['confidence'])
        role = TokenRole(r['index'], text, 'DESIGNATOR', r['normalized'],
                         m.group(1).upper(), prefix_type(m.group(1)), m.group(2), r['confidence'])
        proposal = ComponentProposal(tuple(body['bbox']), body['type'], body.get('confidence', 1.), 'diagnostic')
        role_rows.append({**r, 'existing_rule_weight': designator_weight(proposal, role)})
    return {'literal_count': len(literal),
            'exact_case_literal_count': sum(t['text'].strip() == expected for t in literal),
            'raw_variants': sorted({t['text'] for t in literal}),
            'literal_role_counts': dict(literal_roles),
            'literal_observations': [{'text': t['text'], 'bbox': t['bbox'], 'source': t['source']} for t in literal],
            'designator_candidate_count': len(role_rows),
            'best_existing_rule_weight': max((r['existing_rule_weight'] for r in role_rows), default=0.),
            'designator_candidates': role_rows}


def body_cause(target, final, before, fm, bm, scene, power=False):
    """Evidence-based situation categories. Ambiguity is never called a miss."""
    out = {'final_geometry': fm, 'before_filter_geometry': bm}
    if fm['status'] == 'ambiguous':
        return {**out, 'cause': 'geometry_ambiguous'}
    if fm['status'] != 'matched':
        if bm['status'] == 'ambiguous':
            return {**out, 'cause': 'geometry_ambiguous'}
        if bm['status'] == 'matched':
            b = before[bm['index']]
            correct = peer.rail(b) == peer.rail(target) if power else b['key'] == target['key']
            return {**out, 'cause': 'correct_identity_filtered' if correct else 'body_filtered_with_wrong_or_unresolved_identity',
                    'before_filter_component': b, 'ocr': ocr_evidence(str(target.get('value')) if power else target['key'], scene, b)}
        rejected = scene['diagnostics'].get('fusion_rejected', [])
        raw = [r['proposal'] for r in rejected if isinstance(r, dict) and 'proposal' in r]
        hit = float(boxes_iou([target], raw).max()) if raw else 0.
        return {**out, 'cause': 'fusion_rejected_candidate' if hit >= IOU_THRESHOLD else 'no_prefilter_box_at_diagnostic_IoU',
                'best_rejected_iou': hit,
                'ocr': ocr_evidence(str(target.get('value')) if power else target['key'], scene, target)}
    p = final[fm['index']]
    out['final_component'] = p
    if power:
        expected, actual = peer.rail(target), peer.rail(p)
        if actual == expected:
            return {**out, 'cause': 'correct_identity_on_body'}
        if actual is None or actual[:2] != expected[:2]:
            return {**out, 'cause': 'component_type_mismatch'}
        evidence = ocr_evidence(str(target.get('value')), scene, p)
        value = p.get('value')
        cause = ('power_value_empty' if value in (None, '') else
                 'power_value_case_or_space_only' if canonical(value) == canonical(target.get('value')) else
                 'power_value_wrong_nonempty')
        return {**out, 'cause': cause, 'ocr': evidence}
    if not _same_family(target['type'], p['type']):
        return {**out, 'cause': 'component_type_mismatch', 'ocr': ocr_evidence(target['key'], scene, p)}
    expected, key = target['key'], p['key']
    if key == expected:
        return {**out, 'cause': 'correct_identity_on_body'}
    evidence = ocr_evidence(expected, scene, p)
    out['ocr'] = evidence
    if key.casefold() == expected.casefold():
        return {**out, 'cause': 'case_only_key'}
    if canonical(key) == canonical(expected):
        return {**out, 'cause': 'space_or_case_and_space_key'}
    if bm['status'] == 'matched' and before[bm['index']]['key'] == expected:
        return {**out, 'cause': 'correct_identity_lost_after_prefilter'}
    if evidence['designator_candidate_count']:
        cause = ('candidate_assignment_or_key_collision' if evidence['best_existing_rule_weight'] >= ASSIGNMENT_THRESHOLD
                 else 'candidate_blocked_by_existing_weight_or_gate')
    else:
        cause = 'literal_seen_but_not_designator_candidate' if evidence['literal_count'] else 'correct_literal_not_in_archived_OCR'
    return {**out, 'cause': cause}


def compact_observation(row):
    # Pins and per-source metadata are not needed in component evidence rows.
    return {k: row[k] for k in ('key', 'type', 'bbox', 'value', 'confidence', 'observable_designator', 'source_id') if k in row}


def inspect_case(scene, pred, gt, previous):
    height = scene['height']
    targets = top_rows(gt['components'], height)
    final = top_rows(pred['components'], height)
    before = top_rows(scene['diagnostics']['candidate_audit_before_filter']['components'], height, native=True)
    fm, bm = geometry_matches(targets, final), geometry_matches(targets, before)
    target_index = {t['key']: i for i, t in enumerate(targets)}
    gt_vertices = set().union(set(), *(n['vertices'] for n in peer.logical_nets(gt)[0]))
    weights = Counter(v[1] for v in gt_vertices if v[0] != 'rail')
    pred_rails = {peer.rail(c) for c in pred['components'].values()}-{None}
    missing_rails = {v for v in gt_vertices if v[0] == 'rail'}-pred_rails
    # Check the export boundary directly; no corrected result is generated.
    native = {c['key']: c for c in scene['components']}
    export_missing = set(native)-set(pred['components'])
    export_extra = set(pred['components'])-set(native)
    assert not export_missing and not export_extra, 'Scene/export component keys differ'
    for key, c in native.items():
        p = pred['components'][key]
        assert c['type'] == p['type'] and c['value'] == p['value'], 'Scene/export identity fields differ'
    owners, primary, present_wrong, by_type = [], Counter(), Counter(), defaultdict(Counter)
    for owner, weight in sorted(weights.items()):
        i = target_index[owner]
        t = targets[i]
        missing = owner not in pred['components'] or peer.rail(pred['components'][owner]) is not None
        evidence = body_cause(t, final, before, fm[i], bm[i], scene)
        for key in ('final_component', 'before_filter_component'):
            if key in evidence:
                evidence[key] = compact_observation(evidence[key])
        evidence.update(GT_owner=owner, GT_type=t['type'], GT_bbox=t['bbox'],
                        missing_in_original_owner_bucket=missing, member_weight=weight)
        # A name exists globally but demonstrably belongs to another retained body.
        named_iou = 0.
        if owner in pred['components']:
            named = next(p for p in final if p['key'] == owner)
            named_iou = float(boxes_iou([t], [named])[0, 0])
        evidence['same_name_body_iou'] = named_iou
        if missing:
            primary[evidence['cause']] += weight
            by_type['ic' if peer.is_ic(t) else 'component'][evidence['cause']] += weight
        elif fm[i]['status'] == 'matched' and final[fm[i]['index']]['key'] != owner and named_iou < IOU_THRESHOLD:
            present_wrong[evidence['cause']] += weight
            evidence['existing_key_but_other_body_evidence'] = True
        owners.append(evidence)
    power_rows = []
    rail_counts = Counter()
    for rail in sorted(missing_rails):
        instances = []
        for i, t in enumerate(targets):
            if peer.rail(t) != rail:
                continue
            e = body_cause(t, final, before, fm[i], bm[i], scene, power=True)
            for key in ('final_component', 'before_filter_component'):
                if key in e:
                    e[key] = compact_observation(e[key])
            instances.append({**e, 'GT_symbol_key': t['key'], 'GT_bbox': t['bbox'], 'GT_value': t.get('value')})
        assert instances
        causes = {e['cause'] for e in instances}
        assert 'correct_identity_on_body' not in causes, 'A globally missing rail cannot have an exact observed instance'
        cause = next(iter(causes)) if len(causes) == 1 else 'mixed_symbol_instance_causes'
        rail_counts[cause] += 1
        power_rows.append({'rail': rail, 'cause': cause, 'symbol_count': len(instances),
                           'symbol_causes': dict(Counter(e['cause'] for e in instances)), 'instances': instances})
    prior = previous['identity']['exclusive_losses']['component_identity_missing']
    actual = {k: sum(v.values()) for k, v in by_type.items()}
    if missing_rails:
        actual['rail'] = len(missing_rails)
    assert {k:v for k,v in actual.items() if v} == {k:v for k,v in prior.items() if v}, (actual, prior)
    case_rows = [r for r in owners if r['missing_in_original_owner_bucket'] and r['cause'] == 'case_only_key']
    case_evidence = Counter()
    for r in case_rows:
        o = r['ocr']
        case_evidence['members_with_exact_case_literal' if o['exact_case_literal_count'] else
                      'members_with_casefold_only_literal' if o['literal_count'] else
                      'members_without_separate_literal'] += r['member_weight']
    global_alias = Counter()
    for r in owners:
        if r['missing_in_original_owner_bucket']:
            matches = [k for k,c in pred['components'].items() if peer.rail(c) is None and k.casefold() == r['GT_owner'].casefold()]
            if matches:
                global_alias['members_with_global_casefold_alias'] += r['member_weight']
                if r['cause'] != 'case_only_key':
                    global_alias['members_with_alias_but_geometry_not_confirmed'] += r['member_weight']
    return {'original_missing_member_count': sum(primary.values())+len(missing_rails),
            'missing_owner_count': sum(r['missing_in_original_owner_bucket'] for r in owners),
            'missing_nonrail_members': dict(primary), 'missing_nonrail_by_type': {k:dict(v) for k,v in by_type.items()},
            'missing_rail_count': len(missing_rails), 'missing_rail_causes': dict(rail_counts),
            'case_only_raw_evidence': dict(case_evidence), 'global_casefold_alias_evidence': dict(global_alias),
            'existing_name_on_other_body_members': dict(present_wrong),
            'scene_export_missing_keys': [], 'scene_export_extra_keys': [],
            'owners': owners, 'missing_rails': power_rows}


def summed(dicts):
    result = Counter()
    for d in dicts:
        result.update(d)
    return dict(result)


def filtered_situation(o):
    b, e = o['before_filter_component'], o['ocr']
    if not _same_family(o['GT_type'], b['type']):
        return 'component_type_mismatch'
    if b['key'].casefold() == o['GT_owner'].casefold():
        return 'case_only_key'
    if e['best_existing_rule_weight'] >= ASSIGNMENT_THRESHOLD:
        return 'candidate_assignment_or_key_collision'
    if e['designator_candidate_count']:
        return 'candidate_blocked_by_existing_weight_or_gate'
    return 'literal_seen_but_not_designator_candidate' if e['literal_count'] else 'correct_literal_not_in_archived_OCR'


def aggregate(rows):
    out = {key: summed(r[key] for r in rows.values()) for key in (
        'missing_nonrail_members', 'missing_rail_causes', 'case_only_raw_evidence',
        'global_casefold_alias_evidence', 'existing_name_on_other_body_members')}
    out.update({key: sum(r[key] for r in rows.values()) for key in (
        'original_missing_member_count', 'missing_owner_count', 'missing_rail_count')})
    out['missing_nonrail_by_type'] = {k: summed(r['missing_nonrail_by_type'].get(k, {}) for r in rows.values())
                                       for k in ('ic', 'component')}
    missing_owners = [o for r in rows.values() for o in r['owners'] if o['missing_in_original_owner_bucket']]
    out['missing_owner_causes_unweighted'] = dict(Counter(o['cause'] for o in missing_owners))
    filtered = [o for o in missing_owners if o['cause'] == 'body_filtered_with_wrong_or_unresolved_identity']
    evidence, identity_state = defaultdict(Counter), defaultdict(Counter)
    role_reasons = Counter()
    grammar = defaultdict(Counter)
    root_by_type = defaultdict(Counter)
    for o in filtered:
        b, e, weight = o['before_filter_component'], o['ocr'], o['member_weight']
        state = ('unresolved' if b['key'].startswith('UNRESOLVED_') else
                 'case_only' if b['key'].casefold() == o['GT_owner'].casefold() else 'other_wrong_key')
        kind = filtered_situation(o)
        for unit, amount in (('members', weight), ('owners', 1)):
            evidence[unit][kind] += amount
            identity_state[unit][state] += amount
        for reason, count in e['literal_role_counts'].items():
            # Presence per owner, not number of duplicate OCR readings.
            if count:
                role_reasons[reason] += weight
    out['filtered_body_text_evidence'] = {k: dict(v) for k,v in evidence.items()}
    out['filtered_body_identity_state'] = {k: dict(v) for k,v in identity_state.items()}
    out['filtered_literal_role_evidence_not_exclusive'] = dict(role_reasons)
    confidences = [o['before_filter_component'].get('confidence', 0.) for o in filtered]
    out['filtered_body_detector_confidence'] = ({'minimum': min(confidences), 'median': float(np.median(confidences)),
                                                'maximum': max(confidences)} if confidences else {})
    assert sum(evidence['members'].values()) == out['missing_nonrail_members'].get('body_filtered_with_wrong_or_unresolved_identity', 0)
    # Partition the same missing identities by the earliest observable identity
    # obstacle, looking inside filtered bodies rather than calling filtering the
    # sole cause of their already unresolved identities.
    for o in missing_owners:
        kind = filtered_situation(o) if o['cause'] == 'body_filtered_with_wrong_or_unresolved_identity' else o['cause']
        root_by_type['ic' if o['GT_type'].strip().lower() in ('ic', 'box') else 'component'][kind] += o['member_weight']
        m = DESIGNATOR_RE.fullmatch(o['GT_owner'])
        supported = bool(m and canonical(m.group(1)) == canonical(o['GT_owner']) and not m.group(2))
        grammar['members']['supported_full_key' if supported else 'unsupported_full_key'] += o['member_weight']
        if not supported and o.get('ocr', {}).get('literal_count', 0):
            grammar['members']['unsupported_full_key_with_literal_observed'] += o['member_weight']
    out['missing_nonrail_root_situations_by_type'] = {k:dict(v) for k,v in root_by_type.items()}
    out['missing_nonrail_root_situations'] = summed(root_by_type.values())
    out['missing_owner_grammar_flags_not_exclusive'] = {k:dict(v) for k,v in grammar.items()}
    assert sum(out['missing_nonrail_root_situations'].values()) == sum(out['missing_nonrail_members'].values())
    symbols = [i for r in rows.values() for p in r['missing_rails'] for i in p['instances']]
    out['missing_rail_symbol_causes'] = dict(Counter(i['cause'] for i in symbols))
    out['missing_rail_evidence_flags_not_exclusive'] = {
        'any_nonempty_wrong_value': sum(any(i['cause'] in ('power_value_wrong_nonempty', 'power_value_case_or_space_only') for i in p['instances']) for r in rows.values() for p in r['missing_rails']),
        'any_empty_value': sum(any(i['cause'] == 'power_value_empty' for i in p['instances']) for r in rows.values() for p in r['missing_rails']),
        'any_literal_value_observation': sum(any(i.get('ocr', {}).get('literal_count', 0) for i in p['instances']) for r in rows.values() for p in r['missing_rails']),
        'all_GT_instances_use_empty_value_VCC_fallback': sum(all(i['GT_value'] in (None, '') for i in p['instances']) for r in rows.values() for p in r['missing_rails']),
    }
    assert sum(out['missing_nonrail_members'].values())+out['missing_rail_count'] == out['original_missing_member_count']
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--frozen', type=Path, default=Path('runs/current150_power_20261027'))
    p.add_argument('--previous', type=Path, default=Path('runs/third_stage_audit_detail_20261027/report.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', choices=('all', 'smoke'), default='all')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior evidence: use a fresh output directory')
    guard = Guard()
    inputs = discover(args.dataset)  # Verify all whitelist directories even for smoke.
    ids = IDS if args.cases == 'all' else IDS[:2]
    previous = read(args.previous)
    sealed = read(args.frozen/'comparison/predictions_sealed.json')
    manifest = read(args.frozen/'inputs_serial/manifest.json')
    assert previous['ACTUAL_CASE_IDS'] == sealed['ACTUAL_CASE_IDS'] == manifest['ACTUAL_CASE_IDS'] == list(IDS)
    assert manifest['GT_opens'] == sealed['GT_opens'] == 0
    assert previous['sealed_predictions_sha256'] == sha(args.frozen/'comparison/predictions_sealed.json')
    source_hash = sha(Path(__file__))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    rows = {}
    print({'ACTUAL_CASE_IDS': list(ids), 'DIAGNOSTIC_ONLY': True, 'reserved_used': False}, flush=True)
    for cid in ids:
        folder = args.frozen/'inputs_serial'/cid
        prediction = args.frozen/'comparison/v2'/cid/'result.json'
        assert sha(prediction) == sealed['sha256'][cid]['v2']
        assert sha(folder/'scene.json') == sealed['input_provenance'][cid]['scene.json']
        assert sha(inputs[cid][0]) == sealed['input_provenance'][cid]['image_sha256']
        scene, pred = read(folder/'scene.json'), read(prediction)
        with guard.evaluating(inputs[cid][1], [prediction, folder/'scene.json']):
            gt = read(inputs[cid][1])
        rows[cid] = inspect_case(scene, pred, gt, previous['cases'][cid])
        if int(cid)%10 == 0 or cid == ids[-1]:
            print({'completed': cid, 'GT_opens': guard.target_opens}, flush=True)
    assert source_hash == sha(Path(__file__)), 'Diagnostic source changed during run'
    totals = aggregate(rows)
    write(args.output/'report.json', {'ACTUAL_CASE_IDS': list(ids), 'DIAGNOSTIC_ONLY': True, 'OFFICIAL_SCORE': False,
          'GT_opens': guard.target_opens, 'reserved_used': False, 'baseline_commit': commit,
          'input_version_key': manifest['version_key'], 'audit_source_sha256': source_hash,
          'previous_report_sha256': sha(args.previous), 'sealed_predictions_sha256': sha(args.frozen/'comparison/predictions_sealed.json'),
          'geometry': {'IoU_threshold': IOU_THRESHOLD, 'type_or_identity_used_for_matching': False,
                       'one_to_one': True, 'competing_matches_kept_ambiguous': True, 'threshold_search': False},
          'environment': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__,
                          'scipy': importlib.metadata.version('scipy')},
          'aggregate': totals, 'cases': rows})
    print(totals, flush=True)


if __name__ == '__main__':
    main()
