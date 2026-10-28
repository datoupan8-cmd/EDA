"""GT-only diagnostics for sealed 0001-0150 predictions; never a predictor.

Original topology statements are retained verbatim in AST form. Inserted hooks
only copy DSU parents and selected snap indices. Each replay must equal the
unobserved implementation and the sealed baseline membership.
"""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import inspect
import itertools
from pathlib import Path

import cv2
import numpy as np

from tools.team_access_guard import Guard
from pcb.coordinates import opencv_to_target
from pcb.network_shared.contracts import snapshot
from pcb.network_shared.graph import build_graph
from pcb.net_hypergraph import legacy_v2 as legacy
from . import peer_protocol2055 as peer
from .power_comparison import IDS, discover, read, write, sha, scene_from


def vertex(owner, port, components, gt_components):
    own = components[owner]
    r = peer.rail(own)
    if r is not None:
        return r
    return ('ic', owner, port) if peer.is_ic(gt_components.get(owner, own)) else ('component', owner)


def legal_vertices(pred, gt_components):
    return {f'{o}.{p[4:]}': vertex(o, p[4:], pred['components'], gt_components)
            for o, pins in pred['pins'].items() for p in pins}


def physical_vertices(pred, gc):
    cs = pred['components']
    return [{vertex(*peer.parse_ref(ref, cs), cs, gc) for ref in peer.references(n)}
            for n in pred['nets'].values()]


def merge_rails(groups):
    parents = list(range(len(groups)))
    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i
    first = {}
    for i, group in enumerate(groups):
        for v in group:
            if v[0] == 'rail':
                if v in first:
                    parents[root(i)] = root(first[v])
                else:
                    first[v] = i
    result = defaultdict(set)
    for i, group in enumerate(groups):
        result[root(i)].update(group)
    return list(result.values())


def universe(groups):
    return set().union(set(), *groups)


def pairs(groups, known):
    return set().union(set(), *(set(itertools.combinations(sorted(s & known), 2)) for s in groups))


def pair_counts(found, target):
    return {'tp': len(found & target), 'fp': len(found-target), 'fn': len(target-found)}


def identity_ceiling(gt_groups, available):
    """Rigorous relaxed F1 bound: no new correct identities, free net count.

Each GT network's overlap weight cannot exceed 2*a/(a+g). For k matched
networks the largest k weights bound their sum; unmatched predicted networks
only increase the denominator. Dropping extras/reusing identities is allowed
in this relaxation, so this is deliberately optimistic.
"""
    m = len(gt_groups)
    if not m:
        return 0.
    weights = sorted((2*len(g & available)/(len(g & available)+len(g)) if g else 0.
                      for g in gt_groups), reverse=True)
    cumulative, best = 0., 0.
    for k, weight in enumerate(weights, 1):
        cumulative += weight
        best = max(best, 2*cumulative/(k+m))
    return best


def identity_stages(scene, pred, gt, gg):
    g = universe(gg)
    pc, gc = pred['components'], gt['components']
    rails = {peer.rail(c) for c in pc.values()}-{None}
    owners = {v for v in g if (v[0] == 'rail' and v in rails) or
              (v[0] != 'rail' and v[1] in pc and peer.rail(pc[v[1]]) is None)}
    raw = {(c.key, p.number) for c in scene.components for p in c.pins}
    numbered = {v for v in owners if v[0] != 'ic' or (v[1], v[2]) in raw}
    legal = set(legal_vertices(pred, gc).values())
    net = universe([n['vertices'] for n in peer.logical_nets(pred, gc)[0]])
    available = [owners, numbered, legal & g, net & g]
    assert all(b <= a for a, b in zip([g]+available, available))
    losses, pair_losses = {}, {}
    before = g
    before_pairs = pairs(gg, g)
    for name, current in zip(('component_identity_missing', 'ic_number_missing',
                              'legal_terminal_missing', 'legal_identity_not_in_net'), available):
        losses[name] = dict(Counter(v[0] for v in before-current))
        after_pairs = pairs(gg, current)
        pair_losses[name] = len(before_pairs-after_pairs)
        before_pairs = after_pairs
        before = current
    return {'GT_unique': len(g), 'GT_by_type': dict(Counter(v[0] for v in g)),
            'correct_in_net': len(g & net), 'missing_in_net': len(g-net),
            'wrong_in_net': len(net-g), 'wrong_in_net_by_type': dict(Counter(v[0] for v in net-g)),
            'legal_tp': len(legal & g), 'legal_fp': len(legal-g), 'legal_fn': len(g-legal),
            'exclusive_losses': losses,
            'exclusive_pair_losses': pair_losses,
            'ceilings': {name: identity_ceiling(gg, a) for name, a in zip(
                ('component_identity', 'raw_ic_numbers', 'legal_terminals', 'in_net'), available)}}


def observed_topology(scene, graph):
    """Copy-only observation of the unchanged legacy algorithm."""
    original = ast.parse(inspect.getsource(legacy.build_hypergraph))
    tree = copy.deepcopy(original)
    marks = {'stroke': 'junction', 'seglists': 'bridge', 'contacts': 'snap',
             'label_hits': 'contact', 'groups': 'duplicate'}
    body = []
    for node in tree.body[0].body:
        names = {n.id for target in getattr(node, 'targets', ()) for n in ast.walk(target) if isinstance(n, ast.Name)}
        hit = names & marks.keys()
        if hit:
            assert len(hit) == 1
            body += ast.parse("_audit_stage(%r, locals())" % marks[next(iter(hit))]).body
        body.append(node)
    tree.body[0].body = body
    class AddSnaps(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'snaps' and call.func.attr == 'append':
                return [node, *ast.parse('_audit_snap(pi, wi, q)').body]
            return node
    tree = AddSnaps().visit(tree)
    class RemoveAudit(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id.startswith('_audit_'):
                return None
            return node
    assert ast.dump(RemoveAudit().visit(copy.deepcopy(tree))) == ast.dump(original)
    captured, snapped = {}, {}
    def observe(name, state):
        captured[name] = tuple(state['dsu' if name in ('junction', 'bridge') else 'pdsu'].p)
        if name == 'duplicate':
            captured['pins'] = list(state['pins'])
            captured['contact_decisions'] = len(state['contacts'])
    def snap(pi, wi, q):
        snapped[pi] = {'path': wi, 'point': tuple(float(x) for x in q)}
    namespace = {**legacy.__dict__, '_audit_stage': observe, '_audit_snap': snap}
    exec(compile(ast.fix_missing_locations(tree), '<third-stage copy-only audit>', 'exec'), namespace)
    result = namespace['build_hypergraph'](snapshot(scene), graph, mode='directional', merge_labels=False)
    normal = legacy.build_hypergraph(snapshot(scene), graph, mode='directional', merge_labels=False)
    assert result == normal, 'Observation changed the original topology'
    assert set(marks.values()) <= captured.keys()
    captured['snapped'] = snapped
    return result, captured


def root(parents, i):
    while parents[i] != i:
        i = parents[i]
    return i


def replay_groups(captured, npaths, refs, phase):
    rows = defaultdict(set)
    for i, (_, _, ref) in enumerate(captured['pins']):
        if ref not in refs:
            continue
        if phase in ('path', 'junction', 'bridge'):
            if i in captured['snapped']:
                path = captured['snapped'][i]['path']
                group = ('path', path if phase == 'path' else root(captured[phase], path))
            else:
                group = ('unattached', i)
        else:
            group = ('group', root(captured[phase], npaths+i))
        rows[group].add(refs[ref])
    return list(rows.values())


def gt_edge_index(gn):
    a, b, owners = [], [], []
    for i, net in enumerate(gn):
        for e in net['edges']:
            if peer.valid_edge(e):
                a.append(peer.xy(e[0])); b.append(peer.xy(e[1])); owners.append(i)
    return np.asarray(a, float).reshape(-1, 2), np.asarray(b, float).reshape(-1, 2), np.asarray(owners, int)


def destination_status(v, point, gn, index, tolerance=5.):
    own = {i for i, n in enumerate(gn) if v in n['vertices']}
    if not own:
        return 'wrong_identity'
    a, b, owners = index
    if not len(a):
        return 'no_GT_line_nearby'
    delta = b-a
    t = np.clip(np.sum((np.asarray(point)-a)*delta, axis=1)/np.maximum(np.sum(delta*delta, axis=1), 1e-20), 0, 1)
    distances = np.linalg.norm(a+t[:, None]*delta-np.asarray(point), axis=1)
    near = set(owners[distances <= tolerance].tolist())
    if not near:
        return 'no_GT_line_nearby'
    if not near & own:
        return 'wrong_net_geometry'
    return 'supported_correct_net' if near <= own else 'ambiguous_GT_nets'


def wire_coverage(mask, gn, height):
    distance = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    count, lengths = Counter(), Counter()
    h, w = mask.shape
    for net in gn:
        for e in net['edges']:
            if not peer.valid_edge(e):
                continue
            a, b = np.asarray(peer.xy(e[0])), np.asarray(peer.xy(e[1]))
            length = float(np.linalg.norm(b-a))
            points = a+np.linspace(0, 1, max(2, int(np.ceil(length))+1))[:, None]*(b-a)
            x = np.rint(points[:, 0]).astype(int)
            y = np.rint(height-points[:, 1]).astype(int)
            valid = (x >= 0) & (x <= w) & (y >= 0) & (y <= h)
            supported = np.zeros(len(x), bool)
            supported[valid] = distance[np.clip(y[valid], 0, h-1), np.clip(x[valid], 0, w-1)] <= 5.
            label = 'full' if supported.all() else 'none' if not supported.any() else 'partial'
            count[label] += 1
            lengths['GT_length_px'] += length
            lengths['supported_length_px'] += length*float(supported.mean())
    return {'segments': dict(count), **lengths}


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh diagnostic output directory')
    guard = Guard()
    inputs = discover(args.dataset)
    source = args.frozen
    comparison = source/'comparison'
    sealed, manifest = read(comparison/'predictions_sealed.json'), read(source/'inputs_serial/manifest.json')
    old_report = read(comparison/'report.json')
    assert sealed['ACTUAL_CASE_IDS'] == manifest['ACTUAL_CASE_IDS'] == list(IDS)
    assert sealed['GT_opens'] == manifest['GT_opens'] == 0
    rows = {}
    audit_hash = sha(Path(__file__))
    print({'ACTUAL_CASE_IDS': list(IDS), 'DIAGNOSTIC_ONLY': True}, flush=True)
    for cid in IDS:
        paths = [comparison/v/cid/'result.json' for v in ('baseline', 'v1', 'v2')]
        predictions = [read(p) for p in paths]
        for v, p in zip(('baseline', 'v1', 'v2'), paths):
            assert sha(p) == sealed['sha256'][cid][v]
        assert sha(inputs[cid][0]) == sealed['input_provenance'][cid]['image_sha256']
        folder = source/'inputs_serial'/cid
        for name in ('scene.json', 'wire_mask.png'):
            assert sha(folder/name) == sealed['input_provenance'][cid][name]
        scene = scene_from(read(folder/'scene.json'))
        mask = cv2.imdecode(np.fromfile(folder/'wire_mask.png', np.uint8), 0)
        graph = build_graph(mask)
        result, captured = observed_topology(scene, graph)
        with guard.evaluating(inputs[cid][1], paths):
            gt = read(inputs[cid][1])
        gn = peer.logical_nets(gt)[0]
        gg = [n['vertices'] for n in gn]
        physical = [physical_vertices(p, gt['components']) for p in predictions]
        logical = [[n['vertices'] for n in peer.logical_nets(p, gt['components'])[0]] for p in predictions]
        known = set.intersection(*(universe(groups) for groups in logical+[gg]))
        target = pairs(gg, known)
        refs = legal_vertices(predictions[0], gt['components'])
        expected_refs = Counter(frozenset(peer.references(n)) for n in predictions[0]['nets'].values())
        replayed_refs = Counter(frozenset(r for r in g.pins if r in refs) for g in result.plan.groups if any(r in refs for r in g.pins))
        assert expected_refs == replayed_refs, 'Sealed baseline references differ'
        groups = {phase: replay_groups(captured, len(graph.paths), refs, phase)
                  for phase in ('path', 'junction', 'bridge', 'snap', 'contact', 'duplicate')}
        assert Counter(map(frozenset, groups['duplicate'])) == Counter(map(frozenset, physical[0])), 'Sealed baseline grouping differs'
        stage_groups = {**{k: merge_rails(v) for k, v in groups.items()},
                        'baseline': logical[0], 'v1': logical[1], 'v2': logical[2]}
        stage_pairs = {k: pairs(v, known) for k, v in stage_groups.items()}
        assert stage_pairs['duplicate'] == stage_pairs['baseline']
        assert stage_pairs['bridge'] == stage_pairs['snap']
        counts = {k: pair_counts(v, target) for k, v in stage_pairs.items()}
        physical_stages = {**groups, 'v1_physical': physical[1], 'v2_physical': physical[2], 'v2_logical': logical[2]}
        physical_pairs = {k: pairs(v, known) for k, v in physical_stages.items()}
        for v in ('baseline', 'v1', 'v2'):
            expected = old_report['common_identity_pairs'][cid][v]
            assert counts[v]['fp'] == expected['false'] and counts[v]['fn'] == expected['missed']
        changes = {}
        order = ['path', 'junction', 'bridge', 'snap', 'contact', 'duplicate', 'baseline', 'v1', 'v2']
        for previous, following in zip(order, order[1:]):
            added, removed = stage_pairs[following]-stage_pairs[previous], stage_pairs[previous]-stage_pairs[following]
            changes[following] = {'added_correct_pairs': len(added & target), 'added_wrong_pairs': len(added-target),
                                  'removed_correct_pairs': len(removed & target), 'removed_wrong_pairs': len(removed-target)}
        rail_stages = {}
        for v, pp, ll in zip(('baseline', 'v1', 'v2'), physical, logical):
            rail_stages[v] = {'before': pair_counts(pairs(pp, known), target), 'after': pair_counts(pairs(ll, known), target)}
        index = gt_edge_index(gn)
        snap_counts = Counter()
        for i, (_, pin, ref) in enumerate(captured['pins']):
            snap_counts['all_raw_instances'] += 1
            snap_counts['all_snapped' if i in captured['snapped'] else 'all_unattached'] += 1
            if ref not in refs:
                continue
            snap_counts['legal_instances'] += 1
            if i not in captured['snapped']:
                snap_counts['legal_unattached'] += 1
                continue
            q = opencv_to_target(*captured['snapped'][i]['point'], scene.height)
            snap_counts['legal_'+destination_status(refs[ref], q, gn, index)] += 1
        pd = read(comparison/'v2'/cid/'diagnostics.json')
        power_counts = Counter()
        for e in pd['events']:
            own, port = peer.parse_ref(e['ref'], predictions[2]['components'])
            v = vertex(own, port, predictions[2]['components'], gt['components'])
            status = destination_status(v, opencv_to_target(*e['tip'], scene.height), gn, index)
            power_counts[e['method']+'|'+status] += 1
        target_all = pairs(gg, universe(gg))
        current_known = universe(logical[2]) & universe(gg)
        target_known = pairs(gg, current_known)
        predicted_all = pairs(logical[2], universe(logical[2]))
        predicted_known = pairs(logical[2], current_known)
        full_pairs = pair_counts(predicted_all, target_all)
        full_pairs.update(GT_pairs=len(target_all),
                          missing_due_to_identity=len(target_all-target_known),
                          missing_despite_known_identity=len(target_known-predicted_known),
                          false_involving_wrong_identity=len(predicted_all-predicted_known),
                          false_between_known_identities=len(predicted_known-target_all))
        assert full_pairs['fn'] == full_pairs['missing_due_to_identity']+full_pairs['missing_despite_known_identity']
        assert full_pairs['fp'] == full_pairs['false_involving_wrong_identity']+full_pairs['false_between_known_identities']
        wire_access = {refs[r] for g in result.plan.groups if g.path_ids for r in g.pins if r in refs}
        for e in pd['events']:
            owner, port = peer.parse_ref(e['ref'], predictions[2]['components'])
            wire_access.add(vertex(owner, port, predictions[2]['components'], gt['components']))
        residual = {}
        for label, missed in [('fixed_common', target-stage_pairs['v2']), ('v2_all_known', target_known-predicted_known)]:
            without_wire = sum(a not in wire_access or b not in wire_access for a, b in missed)
            residual[label] = {'at_least_one_without_wire_path': without_wire,
                               'both_have_wire_paths_but_disconnected': len(missed)-without_wire}
        identity = identity_stages(scene, predictions[2], gt, gg)
        assert sum(identity['exclusive_pair_losses'].values()) == full_pairs['missing_due_to_identity']
        rows[cid] = {'identity': identity, 'pair_stages': counts,
                     'physical_pair_stages': {k: pair_counts(v, target) for k, v in physical_pairs.items()},
                     'end_to_end_pairs': full_pairs, 'residual_missing_pairs': residual,
                     'pair_changes': changes, 'rail_merge': rail_stages, 'snap_geometry': dict(snap_counts),
                     'power_geometry': dict(power_counts), 'wire_coverage': wire_coverage(mask, gn, scene.height),
                     'decisions': {'junctions': len(result.diagnostics['junctions']),
                                   'split_crossings': len(result.diagnostics['nonconnecting_crossings']),
                                   'short_bridges': len(result.diagnostics['short_bridges']),
                                   'contacts': captured['contact_decisions'],
                                   'duplicate_ref_groups': len(result.diagnostics['duplicate_export_pin_refs'])}}
        if int(cid)%10 == 0:
            print({'completed': cid, 'GT_opens': guard.target_opens}, flush=True)
    def summed(values):
        out = Counter()
        for value in values:
            out.update(value)
        return dict(out)
    aggregate = {key: summed(r[key] for r in rows.values()) for key in ('snap_geometry', 'power_geometry', 'decisions')}
    aggregate['identity'] = summed({k: v for k, v in r['identity'].items() if isinstance(v, int)} for r in rows.values())
    aggregate['identity_by_type'] = summed(r['identity']['GT_by_type'] for r in rows.values())
    aggregate['wrong_identity_by_type'] = summed(r['identity']['wrong_in_net_by_type'] for r in rows.values())
    aggregate['exclusive_identity_losses'] = {k: summed(r['identity']['exclusive_losses'][k] for r in rows.values()) for k in next(iter(rows.values()))['identity']['exclusive_losses']}
    aggregate['exclusive_identity_pair_losses'] = summed(r['identity']['exclusive_pair_losses'] for r in rows.values())
    aggregate['end_to_end_pairs'] = summed(r['end_to_end_pairs'] for r in rows.values())
    aggregate['residual_missing_pairs'] = {k: summed(r['residual_missing_pairs'][k] for r in rows.values()) for k in ('fixed_common', 'v2_all_known')}
    aggregate['identity_ceilings'] = {k: sum(r['identity']['ceilings'][k] for r in rows.values())/len(rows) for k in next(iter(rows.values()))['identity']['ceilings']}
    for field in ('pair_stages', 'pair_changes', 'physical_pair_stages'):
        aggregate[field] = {k: summed(r[field][k] for r in rows.values()) for k in next(iter(rows.values()))[field]}
    aggregate['rail_merge'] = {v: {k: summed(r['rail_merge'][v][k] for r in rows.values()) for k in ('before', 'after')} for v in ('baseline', 'v1', 'v2')}
    aggregate['wire_coverage'] = {**summed({k: r['wire_coverage'][k] for k in ('GT_length_px', 'supported_length_px')} for r in rows.values()), 'segments': summed(r['wire_coverage']['segments'] for r in rows.values())}
    pin_rows = [r['evaluation']['pin_diagnostics'] for r in old_report['rows']['baseline']]
    aggregate['IC_pins'] = {'predicted': sum(r['identity']['pred'] for r in pin_rows),
        'GT': sum(r['identity']['gt'] for r in pin_rows),
        'identity_matched': sum(r['identity']['tp'] for r in pin_rows),
        'name_wrong_on_known_identity': sum(len(r['name_wrong_on_identity']) for r in pin_rows),
        'location_wrong_on_known_identity': sum(len(r['distance_wrong_on_identity']) for r in pin_rows),
        'both_wrong_on_known_identity': sum(len(r['name_and_distance_wrong_overlap']) for r in pin_rows),
        'fully_correct': sum(len(r['matched']) for r in pin_rows)}
    assert sum(aggregate['wire_coverage']['segments'].values()) == old_report['aggregate']['all']['v2']['metrics']['NetLine']['gt']
    assert audit_hash == sha(Path(__file__)), 'Audit source changed during diagnosis'
    write(args.output/'report.json', {'ACTUAL_CASE_IDS': IDS, 'DIAGNOSTIC_ONLY': True, 'OFFICIAL_SCORE': False,
          'GT_opens': guard.target_opens, 'reserved_used': False, 'input_version_key': manifest['version_key'],
          'audit_source_sha256': audit_hash, 'original_report_sha256': sha(comparison/'report.json'),
          'sealed_predictions_sha256': sha(comparison/'predictions_sealed.json'),
          'pair_universe': 'same fixed identities as baseline/v1/v2 comparison',
          'geometry_tolerance_px': 5, 'aggregate': aggregate, 'cases': rows})
    print(aggregate, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--frozen', type=Path, default=Path('runs/current150_power_20261027'))
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
