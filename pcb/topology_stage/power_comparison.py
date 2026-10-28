"""Frozen-input paired power experiment; default whitelist is exactly 0001-0150.

Run from the project root with python -m pcb.topology_stage.power_comparison.
All candidate predictions are sealed before any target is opened. No fitting.
"""
import argparse
import copy
from dataclasses import asdict, fields
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from pcb.schema import Component, Pin, Scene, Text
from pcb.network_shared.contracts import snapshot
from pcb.network_shared.graph import build_graph
from pcb.network_shared.composite import assemble, validate_plan
from pcb.net_hypergraph.legacy_v2 import LegacyHypergraph
from pcb.net_lines.legacy_v2 import LegacyLines
from pcb.submission import export, validate_strict
from tools.team_access_guard import Guard
from .power_terminals import attach_power_terminals
from . import protocol_202610 as clarified
from . import peer_adapter as peer

ROOT = Path(__file__).resolve().parents[2]
IDS = tuple(f'{i:04d}' for i in range(1, 151))
TRAIN = tuple(c for c in IDS if int(c) % 5)
DEV = tuple(c for c in IDS if not int(c) % 5)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def discover(dataset, ids=IDS):
    root = Path(dataset).resolve()
    if root.name != '200_train_cases' or len(ids) != len(set(ids)) or set(ids)-set(IDS):
        raise PermissionError('Require unique exact whitelist IDs and 200_train_cases root')
    result = {}
    for cid in ids:
        folder = root/cid
        if not folder.is_dir() or folder.resolve() != folder:
            raise ValueError('Missing or redirected case: '+cid)
        images, targets = list(folder.glob('*.png')), list(folder.glob('*_target*.json'))
        if len(images) != 1 or len(targets) != 1:
            raise ValueError('Missing/duplicate image or target: '+cid)
        for p in (*images, *targets):
            if p.resolve().parent != folder or p.name[:4] != cid:
                raise ValueError('Mismatched or redirected input: '+cid)
        result[cid] = images[0], targets[0]
    return result


def scene_from(row):
    def pick(cls, d):
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in d.items() if k in allowed}
    components = []
    for r in row['components']:
        c = pick(Component, r)
        c['pins'] = [Pin(**pick(Pin, p)) for p in r['pins']]
        components.append(Component(**c))
    return Scene(row['width'], row['height'], components,
                 [Text(**pick(Text, t)) for t in row.get('texts', [])],
                 diagnostics=copy.deepcopy(row.get('diagnostics', {})))


def source_inventory():
    paths = sorted(set([*ROOT.glob('pcb/**/*.py'), *ROOT.glob('tools/*.py'),
                        *ROOT.glob('configs/**/*.json'), ROOT/'requirements.txt']))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def common_pair_counts(a, b, gt):
    aa, bb = [clarified.logical_networks(x, gt['components']) for x in (a, b)]
    gg = clarified.logical_networks(gt)
    sets = [[r['vertices'] for r in rows] for rows in (aa, bb, gg)]
    known = set.intersection(*(set().union(set(), *ss) for ss in sets))
    ap, bp, gp = [clarified.pairs([s & known for s in ss]) for ss in sets]
    return {'baseline_false': len(ap-gp), 'candidate_false': len(bp-gp),
            'baseline_missed': len(gp-ap), 'candidate_missed': len(gp-bp)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', choices=('all', 'smoke'), default='all')
    parser.add_argument('--predict-only', action='store_true', help='Seal paired predictions without opening any target')
    args = parser.parse_args()
    guard = Guard()
    ids = IDS if args.cases == 'all' else ('0001', '0002')
    inputs = discover(args.dataset)  # always verify all 150 whitelist directories
    if args.output.exists():
        raise ValueError('Preserve previous evidence: output already exists')
    args.output.mkdir(parents=True)
    print(json.dumps({'ACTUAL_CASE_IDS': ids, 'GT_opens': 0, 'reserved_used': False}), flush=True)
    config_report = read(args.frozen/'report.json') if (args.frozen/'report.json').is_file() else None
    if config_report is not None and config_report.get('target_opens', 0):
        raise ValueError('Baseline inference must not open GT')
    source = source_inventory()
    models = {}
    lfs = subprocess.check_output(['git', 'lfs', 'ls-files', '--long'], cwd=ROOT, text=True)
    for line in lfs.splitlines():
        digest, status, name = line.split(' ', 2)
        if status == '*' and (name.startswith('models/') or name.endswith('/best.pt')):
            path = ROOT/name
            if sha(path) != digest:
                raise ValueError('Model checksum changed: '+name)
            models[name] = digest
    environment = {'python': sys.version, 'executable': sys.executable,
                   'packages': {}}
    for name in ('torch', 'numpy', 'scipy', 'opencv-python', 'onnxruntime-gpu', 'rapidocr', 'timm'):
        try:
            environment['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment['packages'][name] = None
    write(args.output/'plan.json', {'ACTUAL_CASE_IDS': ids, 'config': config_report,
          'source_sha256': source, 'models_sha256': models, 'environment': environment,
          'protocol': peer.VERSION, 'secondary_protocol': clarified.VERSION,
          'protocol_assumptions': clarified.__doc__,
          'train_ids': [c for c in ids if c in TRAIN], 'internal_ids': [c for c in ids if c in DEV],
          'threshold_search': False, 'independent_validation': False})
    seals, provenance, traces = {}, {}, {}
    for cid in ids:
        folder = args.frozen/cid
        with guard.inferring():
            original = read(folder/'result.json')
            row = read(folder/'scene.json')
            scene = scene_from(row)
            mask = cv2.imdecode(np.fromfile(folder/'wire_mask.png', np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError('Missing mask: '+cid)
            graph = build_graph(mask)
            logical = LegacyHypergraph().run(snapshot(scene), graph, mode='directional', merge_labels=False)
            scene.nets = assemble(logical.plan, LegacyLines().run(graph, logical.plan))
            if export(scene) != original:
                raise AssertionError('Frozen baseline replay mismatch: '+cid)
            ordinary = {c.key: asdict(c) for c in scene.components if c.type not in {'v', 'gnd'}}
            plan, trace = attach_power_terminals(scene, graph, logical.plan, logical.diagnostics)
            validate_plan(plan, snapshot(scene), graph)
            for old, new in zip(logical.plan.groups, plan.groups):
                if (old.group_id, old.path_ids, old.contact_points) != (new.group_id, new.path_ids, new.contact_points):
                    raise AssertionError('Existing wire grouping/geometry changed')
            scene.nets = assemble(plan, LegacyLines().run(graph, plan))
            candidate = export(scene)
            validate_strict(candidate, (scene.width, scene.height))
            if original['components'] != candidate['components']:
                raise AssertionError('Component fields changed')
            if ordinary != {c.key: asdict(c) for c in scene.components if c.type not in {'v', 'gnd'}}:
                raise AssertionError('Ordinary pins changed')
            saved = args.output/'power_v1'/cid/'result.json'
            write(saved, candidate)
            write(saved.with_name('diagnostics.json'), trace)
            seals[cid] = sha(saved)
            provenance[cid] = {name: sha(folder/name) for name in ('result.json', 'scene.json', 'wire_mask.png')}
            provenance[cid]['image_sha256'] = sha(inputs[cid][0])
            traces[cid] = {'added_refs': len(trace['events']), 'rejected': len(trace['rejected'])}
        if int(cid) % 10 == 0 or cid == ids[-1]:
            print(json.dumps({'event': 'paired_predictions', 'case_id': cid, 'GT_opens': guard.target_opens}), flush=True)
    if source_inventory() != source:
        raise ValueError('Source/config changed before prediction seal')
    write(args.output/'predictions_sealed.json', {'ACTUAL_CASE_IDS': ids, 'GT_opens': guard.target_opens,
          'candidate_sha256': seals, 'frozen_input_sha256': provenance, 'power_counts': traces})
    if args.predict_only:
        print(json.dumps({'event': 'predictions_only_complete', 'GT_opens': guard.target_opens,
                          'ACTUAL_CASE_IDS': ids, 'output': str(args.output)}), flush=True)
        return
    records = {protocol: {v: [] for v in ('baseline', 'power_v1')} for protocol in ('peer2055', 'clarified', 'legacy')}
    pair_rows = {}
    from evaluate_v2 import evaluate_case as old_eval, aggregate as old_aggregate, source_from_name
    for split in ('train', 'internal'):
        cohort = [c for c in ids if (c in TRAIN) == (split == 'train')]
        for cid in cohort:
            saved = args.output/'power_v1'/cid/'result.json'
            for name, digest in provenance[cid].items():
                path = inputs[cid][0] if name == 'image_sha256' else args.frozen/cid/name
                if sha(path) != digest:
                    raise ValueError('Frozen input changed: '+cid+'/'+name)
            if sha(saved) != seals[cid]:
                raise ValueError('Candidate changed after seal: '+cid)
            with guard.evaluating(inputs[cid][1], [saved, args.frozen/cid/'result.json']):
                gt = read(inputs[cid][1])
            baseline, candidate = read(args.frozen/cid/'result.json'), read(saved)
            pair_rows[cid] = common_pair_counts(baseline, candidate, gt)
            for variant, pred in (('baseline', baseline), ('power_v1', candidate)):
                for protocol, evaluator in (('peer2055', peer.evaluate_case),
                                             ('clarified', clarified.evaluate_case), ('legacy', old_eval)):
                    records[protocol][variant].append({'case_id': cid, 'source': source_from_name(inputs[cid][0].name),
                        'status': 'ok', 'split': split, 'evaluation': evaluator(pred, gt)})
        print(json.dumps({'event': 'evaluated', 'split': split, 'case_ids': cohort,
                          'GT_opens': guard.target_opens}), flush=True)
    report = {'OFFICIAL_SCORE': False, 'protocol': peer.VERSION, 'ACTUAL_CASE_IDS': ids,
              'reserved_used': False, 'target_opens': guard.target_opens, 'common_identity_pairs': pair_rows,
              'records': records, 'aggregate': {}}
    for protocol, aggregator in (('peer2055', peer.aggregate),
                                 ('clarified', clarified.aggregate), ('legacy', old_aggregate)):
        report['aggregate'][protocol] = {}
        for split in ('train', 'internal', 'all'):
            report['aggregate'][protocol][split] = {v: aggregator([r for r in rr if split == 'all' or r['split'] == split])['overall']
                                                    for v, rr in records[protocol].items()
                                                    if any(split == 'all' or r['split'] == split for r in rr)}
    decisions = {}
    for split in ('train', 'internal'):
        block = report['aggregate']['peer2055'][split]
        if not block:
            continue
        a, b = block['baseline'], block['power_v1']
        rows = [v for c, v in pair_rows.items() if (c in TRAIN) == (split == 'train')]
        delta_false = sum(r['candidate_false']-r['baseline_false'] for r in rows)
        repaired = sum(r['baseline_missed']-r['candidate_missed'] for r in rows)
        delta_net = b['metrics']['NetHypergraph']['macro_f1']-a['metrics']['NetHypergraph']['macro_f1']
        delta_score = b['FinalScore']-a['FinalScore']
        decisions[split] = {'delta_net_f1': delta_net, 'delta_score_points': delta_score,
                            'added_false_pairs': delta_false, 'repaired_missed_pairs': repaired,
                            'pass': delta_net >= (.005 if split == 'train' else 0) and
                                    delta_score >= 0 and (delta_score > 0 if split == 'train' else True) and
                                    delta_false <= repaired}
    report['decision'] = {'cohorts': decisions, 'adopt': args.cases == 'all' and len(decisions) == 2 and all(r['pass'] for r in decisions.values())}
    if source_inventory() != source:
        raise ValueError('Source/config changed during evaluation')
    write(args.output/'report.json', report)
    print(json.dumps({'event': 'complete', 'decision': report['decision'], 'report': str(args.output/'report.json')}), flush=True)


if __name__ == '__main__':
    main()
