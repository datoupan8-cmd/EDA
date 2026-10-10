"""Independent Q&A Pin diagnostics with explicit assumptions, never an official scorer.

Consumes frozen predictions. No inference, coordinate conversion, target rewriting,
network remapping, or pipeline registration is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import evaluate_v2 as legacy
from pcb.assignment import maximum_weight_assignment
from pcb.data_policy import assert_allowed_case_id
from pcb.submission import validate_strict

DESIGN = ('0014', '0017', '0018', '0087', '0103')
PIN_RADIUS = 5.0
SPECIAL_BBOX_RADIUS = 20.0
POLICIES = ('legacy_strict', 'qa_fields_only', 'qa_fields_plus_vcc')
METHODS = ('max_cardinality_then_distance', 'min_distance_then_gate')


def point(pin: dict) -> tuple[float, float]:
    value = tuple(float(pin['point'][axis]) for axis in ('x', 'y'))
    if not all(math.isfinite(x) for x in value):
        raise ValueError('Pin coordinates must be finite')
    return value


def geometry_pairs(pred: list[dict], gt: list[dict], method: str = METHODS[0]) -> list[tuple[int, int, float]]:
    """One-to-one geometry diagnostic; the official assignment policy is unknown."""
    if method not in METHODS:
        raise ValueError(f'Unknown assignment assumption: {method}')
    if not pred or not gt:
        return []
    distances = [[math.dist(point(p), point(g)) for g in gt] for p in pred]
    if method == METHODS[0]:
        # One additional valid pair outweighs all possible distance tie-breaks.
        bonus = min(len(pred), len(gt)) + 1.0
        weights = [[bonus + 1.0 - d / PIN_RADIUS if d <= PIN_RADIUS else 0.0 for d in row] for row in distances]
        pairs = maximum_weight_assignment(weights)
    else:
        ceiling = max(max(row) for row in distances) + 1.0
        pairs = maximum_weight_assignment([[ceiling - d for d in row] for row in distances])
    return [(i, j, distances[i][j]) for i, j in pairs if distances[i][j] <= PIN_RADIUS]


def component_map(pred: dict, gt: dict, policy: str) -> dict[str, str]:
    """Keep legacy mapping; only the final scenario extends the GND-style rule to v."""
    mapping = legacy.component_key_map(pred, gt)
    if policy != 'qa_fields_plus_vcc':
        return mapping
    mapping = {p: g for p, g in mapping.items() if pred[p]['type'] != 'v' and gt[g]['type'] != 'v'}
    pp = sorted(k for k, c in pred.items() if c['type'] == 'v')
    gg = sorted(k for k, c in gt.items() if c['type'] == 'v')
    if pp and gg:
        cost = [[legacy.bbox_error(pred[p]['bbox'], gt[g]['bbox']) for g in gg] for p in pp]
        ceiling = max(max(row) for row in cost) + 1.0
        for i, j in maximum_weight_assignment([[ceiling - d for d in row] for row in cost]):
            if cost[i][j] <= SPECIAL_BBOX_RADIUS:
                mapping[pp[i]] = gg[j]
    if len(set(mapping.values())) != len(mapping):
        raise AssertionError('Component ownership must remain one-to-one')
    return mapping


def pin_entries(data: dict) -> dict[tuple[str, str], dict]:
    entries = {}
    for c in data['components'].values():
        if not isinstance(c.get('type'), str):
            raise ValueError('Component type must be a string')
        bbox = c['bbox']
        if len(bbox) != 4 or not all(math.isfinite(float(x)) for x in bbox):
            raise ValueError('Component bbox must contain four finite coordinates')
    for ck, pins in data['pins'].items():
        if ck not in data['components']:
            raise ValueError(f'Pin owner absent from components: {ck}')
        if not isinstance(data['components'][ck].get('type'), str):
            raise ValueError('Component type must be a string')
        for pk, pin in pins.items():
            point(pin)
            entries[(ck, pk)] = pin
    return entries


def matched_pins(pred: dict, gt: dict, mapping: dict, policy: str, method: str) -> list[dict]:
    matches = []
    for pk, gk in mapping.items():
        pp, gg = pred['pins'].get(pk, {}), gt['pins'].get(gk, {})
        exact = policy == 'legacy_strict' or gt['components'][gk]['type'] == 'box'
        if exact:
            pairs = [(pkey, pkey, math.dist(point(p), point(gg[pkey]))) for pkey, p in pp.items()
                     if pkey in gg and p.get('pinname', '') == gg[pkey].get('pinname', '')
                     and math.dist(point(p), point(gg[pkey])) <= PIN_RADIUS]
        else:
            pkeys, gkeys = sorted(pp), sorted(gg)
            pairs = [(pkeys[i], gkeys[j], distance) for i, j, distance in
                     geometry_pairs([pp[k] for k in pkeys], [gg[k] for k in gkeys], method)]
        for pkey, gkey, distance in pairs:
            matches.append({'pred': [pk, pkey], 'gt': [gk, gkey], 'distance': distance,
                            'GT_type': gt['components'][gk]['type'],
                            'match_rule': 'exact_number_name_5px' if exact else method})
    if len({tuple(m['pred']) for m in matches}) != len(matches) or len({tuple(m['gt']) for m in matches}) != len(matches):
        raise AssertionError('A Pin has been scored more than once')
    return matches


def evaluate_pins(pred: dict, gt: dict, policy: str, method: str = METHODS[0]) -> dict:
    """Score immutable exported Pins. GT type determines the semantic exemption."""
    if policy not in POLICIES or method not in METHODS:
        raise ValueError('Unknown diagnostic policy')
    pp, gg = pin_entries(pred), pin_entries(gt)
    mapping = component_map(pred['components'], gt['components'], policy)
    matches = matched_pins(pred, gt, mapping, policy, method)
    matched_gt = set(mapping.values())
    strict_components = {p for p, g in mapping.items()
                         if legacy.component_counts({p: pred['components'][p]}, gt['components'], {p: g})[0] == 1}
    strict_gt = {mapping[p] for p in strict_components}

    def pred_type(key):
        ck = key[0]
        return gt['components'][mapping[ck]]['type'] if ck in mapping else pred['components'][ck]['type']

    def gt_type(key):
        return gt['components'][key[0]]['type']

    def score(types=None, condition=None):
        if condition == 'key_map':
            powners, gowners = set(mapping), matched_gt
        elif condition == 'strict_component':
            powners, gowners = strict_components, strict_gt
        else:
            powners, gowners = set(pred['components']), set(gt['components'])
        pkeys = {key for key in pp if key[0] in powners and (types is None or pred_type(key) in types)}
        gkeys = {key for key in gg if key[0] in gowners and (types is None or gt_type(key) in types)}
        tp = sum(tuple(m['pred']) in pkeys and tuple(m['gt']) in gkeys for m in matches)
        return legacy.prf(tp, len(pkeys), len(gkeys))

    all_types = {gt_type(key) for key in gg} | {pred_type(key) for key in pp}
    nonbox = all_types - {'box'}
    def strata(condition=None):
        return {'overall': score(condition=condition), 'box': score({'box'}, condition), 'nonbox': score(nonbox, condition)}

    geometry = []
    for pk, gk in mapping.items():
        if gt['components'][gk]['type'] != 'box':
            continue
        pkeys, gkeys = sorted(pred['pins'].get(pk, {})), sorted(gt['pins'].get(gk, {}))
        for i, j, distance in geometry_pairs([pp[(pk, k)] for k in pkeys], [gg[(gk, k)] for k in gkeys], method):
            geometry.append({'pred': [pk, pkeys[i]], 'gt': [gk, gkeys[j]], 'distance': distance})
    duplicate_points = {}
    for label, entries in [('pred', pp), ('gt', gg)]:
        counts = Counter((key[0], point(p)) for key, p in entries.items())
        duplicate_points[label] = sum(n - 1 for n in counts.values() if n > 1)
    return {'policy': policy, 'nonbox_assignment': method, **strata(),
            'conditional_on_component_key_map': strata('key_map'),
            'conditional_on_strict_component_tp': strata('strict_component'),
            'by_owner_type': {t: score({t}) for t in sorted(all_types)},
            'component_key_map': mapping, 'strict_component_count': len(strict_components),
            'matches': matches, 'same_owner_duplicate_point_excess': duplicate_points,
            'box_exported_pin_geometry_only': {
                **legacy.prf(len(geometry), score({'box'})['pred'], score({'box'})['gt']),
                'conditional_recall': legacy.div(len(geometry), score({'box'}, 'key_map')['gt']),
                'matches': geometry, 'NOT_RAW_LOCALIZATION_TERMINALS': True}}


def aggregate_counts(rows: list[dict]) -> dict:
    """Micro counts; omit truly empty strata from macro rather than inflate them."""
    total = legacy.prf(sum(r['tp'] for r in rows), sum(r['pred'] for r in rows), sum(r['gt'] for r in rows))
    populated = [r for r in rows if r['pred'] or r['gt']]
    return {**total, 'macro_f1': sum(r['f1'] for r in populated) / len(populated) if populated else None,
            'macro_cases': len(populated)}


def assert_read_allowed(path: str | Path, allowed_targets: set[Path]) -> None:
    value = str(path).replace('\\', '/').lower()
    parts = value.split('/')
    if any(p in {'golden', 'sealed', '10gt', '10_gtcase', '10_gt'} or
           p.startswith(('golden', '10gt.', '10_gtcase.')) or 'eda_pin_crossing_quicktest' in p for p in parts):
        raise PermissionError('Sealed/Golden/QuickTest access is prohibited')
    if '200_train_cases' in parts:
        i = parts.index('200_train_cases')
        if i + 1 >= len(parts) or not parts[i + 1].isdigit():
            raise PermissionError('No dataset-wide reads')
        assert_allowed_case_id(int(parts[i + 1]))
        if parts[i + 1] not in DESIGN:
            raise PermissionError('This finite replay is limited to the fixed five Design cases')
    if '_target' in parts[-1] and parts[-1].endswith('.json'):
        if Path(path).resolve() not in allowed_targets:
            raise PermissionError('Target reads are only allowed in the evaluation context')


class ReadGuard:
    def __init__(self):
        self.allowed_targets = set()
        self.target_opens = []
        self.blocked = []
        sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        try:
            assert_read_allowed(path, self.allowed_targets)
        except PermissionError:
            self.blocked.append(str(path))
            raise
        if '_target' in path.name.lower() and path.suffix.lower() == '.json':
            self.target_opens.append(str(path.resolve()))

    @contextmanager
    def evaluating(self, path: Path):
        self.allowed_targets.add(path.resolve())
        try:
            yield
        finally:
            self.allowed_targets.remove(path.resolve())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def git_state() -> dict:
    return {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
            'tracked_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)).hexdigest(),
            'status_short': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)}


def aggregate_scenarios(rows: list[dict]) -> dict:
    output = {k: aggregate_counts([r[k] for r in rows]) for k in ('overall', 'box', 'nonbox')}
    for condition in ('conditional_on_component_key_map', 'conditional_on_strict_component_tp'):
        output[condition] = {k: aggregate_counts([r[condition][k] for r in rows]) for k in ('overall', 'box', 'nonbox')}
    types = sorted(set().union(*(r['by_owner_type'] for r in rows)))
    output['by_owner_type'] = {t: aggregate_counts([r['by_owner_type'][t] for r in rows if t in r['by_owner_type']]) for t in types}
    geom = [r['box_exported_pin_geometry_only'] for r in rows]
    output['box_exported_pin_geometry_only'] = aggregate_counts(geom)
    output['box_exported_pin_geometry_only']['conditional_recall'] = legacy.div(sum(g['tp'] for g in geom), sum(r['conditional_on_component_key_map']['box']['gt'] for r in rows))
    output['box_exported_pin_geometry_only']['NOT_RAW_LOCALIZATION_TERMINALS'] = True
    return output


def run(output: Path, dataset_root: Path | None = None) -> dict:
    if (output / 'complete.json').exists() or (output / 'input_seal.json').exists():
        raise RuntimeError('This finite replay is already sealed; use a new output directory')
    started = time.perf_counter()
    guard = ReadGuard()
    output.mkdir(parents=True, exist_ok=True)
    meta_path = ROOT / 'reports/pin_visible_semantics_l22/prepared.json'
    l13_path = ROOT / 'reports/pin_reader_l13/design/complete.json'
    l22_path = ROOT / 'reports/pin_visible_semantics_l22/complete.json'
    qa_path = ROOT / 'reports/pin_identity_contract_l23/evidence.json'
    metadata, old_l13, old_l22, qa = (read(p) for p in (meta_path, l13_path, l22_path, qa_path))
    if tuple(row['case_id'] for row in metadata['cases']) != DESIGN:
        raise AssertionError('Fixed Design manifest changed')
    if dataset_root is not None and dataset_root.name != '200_train_cases':
        raise ValueError('Dataset override must name the 200_train_cases directory')
    # L13 A reuses L7.4 D in the original runner; it was not copied into L13.
    variants = {'L13_A': (ROOT/'reports/pin_joint_abstention_l7_4/design/predictions/D', old_l13['end_to_end']['A']['overall']['metrics']['Pin']),
                'L13_B': (ROOT/'reports/pin_reader_l13/design/predictions/B', old_l13['end_to_end']['B']['overall']['metrics']['Pin']),
                'L22_ORACLE_B': (ROOT/'reports/pin_visible_semantics_l22/predictions/B_ORACLE', old_l22['end_to_end']['B_ORACLE']['overall']['metrics']['Pin'])}
    predictions, paths, protected = {}, {}, {}
    for folder in ('pcb', 'configs'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                protected[str(p)] = digest(p)
    provenance_paths = (ROOT/'reports/pin_reader_l13/design/A_parity.json',
                        ROOT/'experiments/pin_reader_l12_design.py',
                        ROOT/'experiments/pin_reader_l13_design.py',
                        ROOT/'experiments/pin_pair_score_l11_run.py')
    for p in (ROOT/'evaluate_v2.py', meta_path, l13_path, l22_path, qa_path, *provenance_paths):
        protected[str(p)] = digest(p)
    for source in qa['sources']:
        if digest(Path(source['path'])) != source['sha256']:
            raise AssertionError('Audited scoring source changed; re-audit rules first')
        protected[source['path']] = source['sha256']
    for name, (base, _) in variants.items():
        predictions[name] = {}
        for row in metadata['cases']:
            case = row['case_id']
            p = base/case/'result.json'
            protected[str(p)] = digest(p)
            predictions[name][case] = read(p)
            validate_strict(predictions[name][case], (row['width'], row['height']))
    for path, signature in metadata['prediction_sha256'].items():
        p = Path(path)
        if not p.resolve().is_relative_to((ROOT/'reports/pin_visible_semantics_l22/predictions').resolve()):
            raise ValueError('L22 sealed path outside its prediction directory')
        if digest(p) != signature:
            raise AssertionError(f'Sealed L22 input changed: {p}')
        protected[str(p)] = signature
    for row in metadata['cases']:
        case = row['case_id']
        saved_a = read(ROOT/f'reports/pin_visible_semantics_l22/predictions/A/{case}/result.json')
        if predictions['L13_B'][case] != saved_a:
            raise AssertionError('L13 B versus L22 A parity lost')
        target = Path(row['target_path'])
        if dataset_root is not None:
            target = dataset_root/case/target.name
        if target.parent.name != case or target.parent.parent.name != '200_train_cases' or '_target' not in target.name:
            raise ValueError('Unexpected target path')
        paths[case] = target
    if guard.target_opens:
        raise AssertionError('Target opened before predictions were loaded and sealed')
    initial_git = git_state()
    protocol = {'prediction_provenance': {name: str(base) for name, (base, _) in variants.items()},
                'L13_A_provenance': 'L13 -> L12 -> L11 reference L7.4 D; archived A_parity verified by original runner',
                'box_rule': 'exact GT type box; number/name exact; point <=5',
                'nonbox_rule': 'within mapped component, one-to-one geometry <=5; semantic fields ignored',
                'power_ground_extension': 'v only; gnd behavior unchanged; legacy-style bbox Hungarian then <=20 (ASSUMPTION)',
                'assignment_sensitivity': list(METHODS), 'network_ref_remap': 'NOT_IMPLEMENTED_PENDING_OFFICIAL_REPLY',
                'FinalScore': 'NOT_COMPUTED_FOR_MIXED_OR_UNCONFIRMED_RULES',
                'macro_empty_strata': 'omit cases with both zero pred and zero GT',
                'unmapped_prediction_strata': 'declared prediction type; mapped predictions use GT owner type',
                'all_new_rules_are_diagnostic': True}
    experiment_sources = [Path(__file__), ROOT/'tests/pin/test_pin_qa_diagnostic.py', output/'experiment_plan.md']
    experiment_sources = {str(p): digest(p) for p in experiment_sources if p.is_file()}
    write(output/'input_seal.json', {'case_ids': DESIGN, 'all_predictions_loaded_before_target_read': True,
          'protected_hashes': protected, 'experiment_sources': experiment_sources, 'git': initial_git, 'protocol': protocol})
    raw_targets = {}
    for row in metadata['cases']:
        case, target = row['case_id'], paths[row['case_id']]
        with guard.evaluating(target):
            raw_targets[case] = target.read_bytes()
        if hashlib.sha256(raw_targets[case]).hexdigest() != row['target_sha256']:
            raise AssertionError('Latest target version changed; stop for version diff before scoring')
    targets = {case: json.loads(raw.decode('utf-8-sig')) for case, raw in raw_targets.items()}
    scenarios, cases = {}, []
    for name, (_, expected) in variants.items():
        scenarios[name] = {}
        for policy in POLICIES:
            for method in METHODS if policy != 'legacy_strict' else METHODS[:1]:
                label = policy if method == METHODS[0] else policy+'_min_distance_sensitivity'
                rows = []
                for row in metadata['cases']:
                    case = row['case_id']
                    data = evaluate_pins(predictions[name][case], targets[case], policy, method)
                    data['case_id'] = case
                    if policy == 'legacy_strict':
                        count = legacy.pin_counts(predictions[name][case]['pins'], targets[case]['pins'], data['component_key_map'])
                        if tuple(data['overall'][k] for k in ('tp', 'pred', 'gt')) != count:
                            raise AssertionError('Legacy match trace versus pin_counts parity failed')
                    rows.append(data)
                aggregated = aggregate_scenarios(rows)
                scenarios[name][label] = aggregated
                cases.append({'variant': name, 'scenario': label, 'cases': rows})
                if policy == 'legacy_strict':
                    for key in ('tp', 'pred', 'gt', 'precision', 'recall', 'f1', 'macro_f1'):
                        if not math.isclose(aggregated['overall'][key], expected[key], abs_tol=1e-12, rel_tol=0):
                            raise AssertionError(f'Historical strict parity failed: {name}/{key}')
    if guard.blocked:
        raise AssertionError('Unexpected forbidden read attempt')
    changed = [p for p, signature in protected.items() if digest(Path(p)) != signature]
    changed_experiment = [p for p, signature in experiment_sources.items() if digest(Path(p)) != signature]
    after_git = git_state()
    if changed or changed_experiment or any(after_git[k] != initial_git[k] for k in ('head', 'branch', 'tracked_diff_sha256')):
        raise AssertionError('Frozen engineering/input/Git state changed')
    result = {'metadata': {'stage': 'L24 Q&A Pin diagnostic replay', 'case_ids': DESIGN,
              'OFFICIAL_SCORE': False, 'SEALED_HOLDOUT_USED_FOR_DEVELOPMENT': False,
              'new_inference_calls': 0, 'Check_run': False, 'full150_run': False,
              'current_modified': False, 'algorithm_modified': False,
              'scoring_change_is_NOT_algorithm_improvement': True,
              'legacy_replay_passed': 3, 'target_open_count': len(guard.target_opens),
              'blocked_reads': guard.blocked, 'seconds': time.perf_counter()-started},
              'protocol': protocol, 'aggregate': scenarios, 'details': cases,
              'targets': {case: {'path': str(paths[case]), 'sha256': hashlib.sha256(raw).hexdigest()} for case, raw in raw_targets.items()},
              'verification': {'protected_files': len(protected), 'changed_files': changed,
                               'changed_experiment_sources': changed_experiment, 'git_after': after_git}}
    write(output/'complete.json', result)
    print(json.dumps({'legacy_replay_passed': 3, 'protected_files': len(protected), 'seconds': result['metadata']['seconds'],
                     'L13_B': {name: {k: row[k] for k in ('overall', 'box', 'nonbox', 'box_exported_pin_geometry_only')}
                               for name, row in scenarios['L13_B'].items()}}, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/pin_qa_diagnostic_l24')
    parser.add_argument('--dataset-root', type=Path)
    args = parser.parse_args()
    run(args.output, args.dataset_root)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
