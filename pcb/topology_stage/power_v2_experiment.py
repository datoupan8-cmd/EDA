"""Prepare sealed baseline/V1/V2 predictions, then score the frozen variants.

Default cohort is exactly 0001-0150. The current7 cohort is separately named
and reported; it cannot silently replace the historical frozen full cohort.
"""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np

from tools.team_access_guard import Guard
from pcb.network_shared.contracts import snapshot
from pcb.network_shared.graph import build_graph
from pcb.network_shared.composite import assemble, validate_plan
from pcb.net_hypergraph.legacy_v2 import LegacyHypergraph
from pcb.net_lines.legacy_v2 import LegacyLines
from pcb.submission import export, validate_strict
from .power_comparison import IDS, TRAIN, DEV, discover, read, write, sha, scene_from, source_inventory
from .power_terminals import attach_power_terminals
from .power_v2 import attach_power_v2, power_crops
from . import peer_protocol2055 as peer

ROOT = Path(__file__).resolve().parents[2]
CURRENT_IDS = ('0001', '0002', '0005', '0011', '0013', '0040', '0095')
LEGACY_ROOT = Path('E:/EDA/EDA/runs/net_proposals_20261026/current_inputs_parallel')


def current_folder(cid):
    if cid in ('0001', '0002'):
        return ROOT/'runs/power_rails_20261027/frozen_inputs'/cid
    if cid == '0040':
        return ROOT/'runs/power_rails_v2_20261027/current_input0040'/cid
    return ROOT/'runs/power_rails_v2_20261027/current_inputs_extra'/cid


def common_pairs(preds, gt):
    networks = [peer.logical_nets(p, gt['components'])[0] for p in preds]
    networks.append(peer.logical_nets(gt)[0])
    sets = [[r['vertices'] for r in rows] for rows in networks]
    known = set.intersection(*(set().union(set(), *ss) for ss in sets))
    pairs = []
    for ss in sets:
        result = set()
        for s in ss:
            vv = sorted(s & known)
            result.update((a, b) for i, a in enumerate(vv) for b in vv[i+1:])
        pairs.append(result)
    return {v: {'false': len(p-pairs[-1]), 'missed': len(pairs[-1]-p)}
            for v, p in zip(('baseline', 'v1', 'v2'), pairs)}


def prepare(args, ids, inputs, guard):
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    args.output.mkdir(parents=True)
    old_manifest = read(LEGACY_ROOT/'manifest.json') if args.cohort == 'frozen150' else None
    if old_manifest is not None:
        if old_manifest['ACTUAL_CASE_IDS'] != list(IDS) or old_manifest['GT_opens'] or old_manifest.get('blocked_accesses', 0):
            raise ValueError('Require the complete image-only 0001-0150 manifest')
        original_records = {r['case_id']: r for r in old_manifest['cases']}
        if len(original_records) != 150:
            raise ValueError('Missing/duplicate source cases')
    source = source_inventory()
    plan = {'ACTUAL_CASE_IDS': ids, 'cohort': args.cohort, 'OFFICIAL_SCORE': False,
            'protocol_sha256': sha(Path(peer.__file__)), 'protocol': peer.PROTOCOL_VERSION,
            'baseline_manifest': old_manifest, 'source_sha256': source,
            'threshold_search': False, 'GT_opens': 0, 'reserved_used': False}
    write(args.output/'plan.json', plan)
    seals, stats, provenance = {}, {}, {}
    print(json.dumps({'event': 'prepare', 'ACTUAL_CASE_IDS': ids, 'GT_opens': 0, 'cohort': args.cohort}), flush=True)
    for cid in ids:
        start = time.perf_counter()
        with guard.inferring():
            folder = LEGACY_ROOT/cid if args.cohort == 'frozen150' else current_folder(cid)
            if old_manifest is not None:
                for key, filename in (('scene', 'scene.json'), ('mask', 'wire_mask.png'), ('prediction', 'result.json')):
                    if sha(folder/filename) != original_records[cid][key]:
                        raise ValueError('Source manifest hash mismatch: '+cid+'/'+filename)
            original = read(folder/'result.json')
            scene = scene_from(read(folder/'scene.json'))
            image = cv2.imdecode(np.fromfile(inputs[cid][0], np.uint8), 1)
            mask = cv2.imdecode(np.fromfile(folder/'wire_mask.png', np.uint8), 0)
            if image is None or mask is None or image.shape[:2] != (scene.height, scene.width) or mask.shape != image.shape[:2]:
                raise ValueError('Input dimensions mismatch: '+cid)
            graph = build_graph(mask)
            logical = LegacyHypergraph().run(snapshot(scene), graph, mode='directional', merge_labels=False)
            old_geometry = LegacyLines().run(graph, logical.plan)
            scene.nets = assemble(logical.plan, old_geometry)
            if export(scene) != original:
                raise ValueError('Baseline replay mismatch: '+cid)
            ordinary = {c.key: asdict(c) for c in scene.components if c.type not in {'gnd', 'v'}}
            rows, count = {'baseline': original}, {}
            for variant in ('v1', 'v2'):
                candidate_scene = copy.deepcopy(scene)
                if variant == 'v1':
                    p, trace = attach_power_terminals(candidate_scene, graph, logical.plan, logical.diagnostics)
                else:
                    p, trace = attach_power_v2(candidate_scene, graph, logical.plan, logical.diagnostics,
                                               power_crops(image, candidate_scene))
                validate_plan(p, snapshot(candidate_scene), graph)
                if [(g.group_id, g.path_ids, g.contact_points) for g in p.groups] != [(g.group_id, g.path_ids, g.contact_points) for g in logical.plan.groups]:
                    raise AssertionError('Physical group assignment changed')
                geometry = LegacyLines().run(graph, p)
                if geometry != old_geometry:
                    raise AssertionError('Existing group geometry changed')
                candidate_scene.nets = assemble(p, geometry)
                result = export(candidate_scene)
                validate_strict(result, (scene.width, scene.height))
                if result['components'] != original['components'] or ordinary != {c.key: asdict(c) for c in candidate_scene.components if c.type not in {'gnd', 'v'}}:
                    raise AssertionError('Component/ordinary-pin changes')
                rows[variant] = result
                count[variant] = {'added_refs': len(trace['events']), 'extra_v2_refs': trace.get('v2_extra_refs', 0),
                                  'rejected': len(trace['rejected'])}
                write(args.output/variant/cid/'diagnostics.json', trace)
            seals[cid] = {}
            for variant, data in rows.items():
                out = args.output/variant/cid/'result.json'
                write(out, data)
                seals[cid][variant] = sha(out)
            stats[cid] = count
            provenance[cid] = {'image_sha256': sha(inputs[cid][0]), 'source_folder': str(folder),
                                **{name: sha(folder/name) for name in ('scene.json', 'wire_mask.png', 'result.json')}}
        if int(cid) % 10 == 0 or cid == ids[-1]:
            print(json.dumps({'event': 'prepared_case', 'case_id': cid, 'seconds': time.perf_counter()-start,
                              'GT_opens': guard.target_opens}), flush=True)
    if source_inventory() != source:
        raise ValueError('Source changed before prediction seal')
    write(args.output/'predictions_sealed.json', {'ACTUAL_CASE_IDS': ids, 'GT_opens': guard.target_opens,
           'sha256': seals, 'input_provenance': provenance, 'counts': stats})
    print(json.dumps({'event': 'sealed', 'case_count': len(ids), 'GT_opens': guard.target_opens}), flush=True)


def score(args, ids, inputs, guard):
    plan, sealed = read(args.output/'plan.json'), read(args.output/'predictions_sealed.json')
    if plan['ACTUAL_CASE_IDS'] != list(ids) or sealed['ACTUAL_CASE_IDS'] != list(ids) or sealed['GT_opens']:
        raise ValueError('Incomplete prediction seal')
    if source_inventory() != plan['source_sha256']:
        raise ValueError('Source changed after prediction seal')
    rows = {v: [] for v in ('baseline', 'v1', 'v2')}
    pair_rows = {}
    print(json.dumps({'event': 'score', 'ACTUAL_CASE_IDS': ids, 'cohort': args.cohort}), flush=True)
    for cid in ids:
        paths = [args.output/v/cid/'result.json' for v in rows]
        for v, path in zip(rows, paths):
            if sha(path) != sealed['sha256'][cid][v]:
                raise ValueError('Prediction changed after seal')
        if sha(inputs[cid][0]) != sealed['input_provenance'][cid]['image_sha256']:
            raise ValueError('Image changed after seal')
        with guard.evaluating(inputs[cid][1], paths):
            target = read(inputs[cid][1])
        predictions = [read(p) for p in paths]
        pair_rows[cid] = common_pairs(predictions, target)
        for variant, data in zip(rows, predictions):
            rows[variant].append({'case_id': cid, 'evaluation': peer.evaluate(data, target)})
        if int(cid) % 10 == 0 or cid == ids[-1]:
            print(json.dumps({'event': 'scored_case', 'case_id': cid, 'GT_opens': guard.target_opens}), flush=True)
    report = {'cohort': args.cohort, 'ACTUAL_CASE_IDS': ids, 'OFFICIAL_SCORE': False,
              'target_opens': guard.target_opens, 'reserved_used': False, 'rows': rows,
              'common_identity_pairs': pair_rows, 'aggregate': {}}
    splits = {'all': ids}
    if args.cohort == 'frozen150':
        splits.update(train=TRAIN, internal=DEV)
    for split, chosen in splits.items():
        report['aggregate'][split] = {v: peer.aggregate([r['evaluation'] for r in rr if r['case_id'] in chosen])
                                       for v, rr in rows.items()}
    write(args.output/'report.json', report)
    print(json.dumps({'event': 'complete', 'aggregate': report['aggregate'], 'report': str(args.output/'report.json')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cohort', choices=('frozen150', 'current7'), default='frozen150')
    p.add_argument('--stage', choices=('prepare', 'score'), required=True)
    args = p.parse_args()
    guard = Guard()
    inputs = discover(args.dataset)
    ids = IDS if args.cohort == 'frozen150' else CURRENT_IDS
    (prepare if args.stage == 'prepare' else score)(args, ids, inputs, guard)


if __name__ == '__main__':
    main()
