"""Independent implementation of the supplied 7075 protocol; NOT official source.

No disk/GT access occurs here. Inputs are saved prediction and allowed evaluation
target objects. Existing evaluate_v2.py remains unchanged.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import json
import math
import sys
from pathlib import Path

from ..assignment import maximum_weight_assignment

PROTOCOL_VERSION = '2026-10-18-independent-thresholds-only'
OFFICIAL_SCORE = False
COMPONENT_THRESHOLD = 20.0
PIN_THRESHOLD = 5.0
WIRE_THRESHOLD = 5.0
WEIGHTS = {'Component': .30, 'Pin': .25, 'NetHypergraph': .35, 'NetLine': .10}
UNVERIFIED = [
    'Official assignment tie-breaking and malformed-record details are unavailable.',
    'Non-Pin empty-set F1 is implemented as zero; only IC-empty zero is explicit in protocol.',
    'Power alias classification uses stripped lowercase type; component fields still compare raw exact values.',
    'Connectivity known IC classification uses GT type, as interpreted from protocol; power rail classification uses own type/value.',
    'Edges are pooled without geometric deduplication across merged source nets.',
]


def metric(tp, pred, gt):
    return {'tp': tp, 'pred': pred, 'gt': gt, 'fp': pred - tp, 'fn': gt - tp,
            'precision': tp / pred if pred else 0., 'recall': tp / gt if gt else 0.,
            'f1': 2 * tp / (pred + gt) if pred + gt else 0.}


def finite(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)


def valid_point(p):
    return isinstance(p, dict) and finite(p.get('x')) and finite(p.get('y'))


def xy(p):
    return float(p['x']), float(p['y'])


def valid_bbox(b):
    return isinstance(b, (list, tuple)) and len(b) == 4 and all(map(finite, b)) and b[0] < b[2] and b[1] < b[3]


def valid_component(c):
    return isinstance(c, dict) and isinstance(c.get('type'), str) and valid_bbox(c.get('bbox'))


def section(data, name):
    v = data.get(name, {}) if isinstance(data, dict) else {}
    return v if isinstance(v, dict) else {}


def typ(c):
    return str(c.get('type', '')).strip().lower() if isinstance(c, dict) else ''


def is_ic(c):
    return typ(c) in ('box', 'ic')


def rail(c):
    t = typ(c)
    if t == 'gnd':
        return ('rail', 'ground', 'GND')
    if t in ('v', 'vcc', 'vdd'):
        value = c.get('value')
        return ('rail', 'supply', json.dumps('VCC' if value in (None, '') else value, ensure_ascii=False, sort_keys=True))
    return None


def components_score(pred, gt):
    pc, gc = section(pred, 'components'), section(gt, 'components')
    tp = 0
    for key, p in pc.items():
        g = gc.get(key)
        if rail(p) is not None or rail(g) is not None:
            continue
        if not valid_component(p) or not valid_component(g):
            continue
        if any(p.get(k) != g.get(k) for k in ('type', 'Name', 'value')):
            continue
        if max(abs(a-b) for a, b in zip(p['bbox'], g['bbox'])) <= COMPONENT_THRESHOLD:
            tp += 1
    pp = [c for c in pc.values() if rail(c) is not None]
    gg = [c for c in gc.values() if rail(c) is not None]
    # Max cardinality of admissible geometry/type/value pairs, not nearest-first.
    if pp and gg:
        weights = [[int(valid_component(p) and valid_component(g)
                        and p.get('type') == g.get('type') and p.get('value') == g.get('value')
                        and max(abs(a-b) for a, b in zip(p['bbox'], g['bbox'])) <= COMPONENT_THRESHOLD)
                    for g in gg] for p in pp]
        tp += sum(weights[i][j] for i, j in maximum_weight_assignment(weights))
    return metric(tp, len(pc), len(gc))


def scoped_pins(data, gt_components, prediction=False):
    cs = section(data, 'components')
    rows = {}
    for key, pins in section(data, 'pins').items():
        owner = gt_components.get(key, cs.get(key, {})) if prediction else cs.get(key, {})
        if not is_ic(owner) or not isinstance(pins, dict):
            continue
        for pk, p in pins.items():
            rows[key, pk] = p
    return rows


def valid_pin(key, p):
    return isinstance(key, str) and key.startswith('pin_') and len(key) > 4 and isinstance(p, dict) and isinstance(p.get('pinname'), str) and valid_point(p.get('point'))


def pins_score(pred, gt):
    pc, gc = section(pred, 'components'), section(gt, 'components')
    pp, gg = scoped_pins(pred, gc, True), scoped_pins(gt, gc)
    matched, identities, names_wrong, distance_wrong = [], [], [], []
    for (owner, key), p in pp.items():
        g = gg.get((owner, key))
        if owner not in pc or owner not in gc or not valid_pin(key, p) or not valid_pin(key, g):
            continue
        identities.append((owner, key))
        good_name = p['pinname'] == g['pinname']
        good_distance = math.dist(xy(p['point']), xy(g['point'])) <= PIN_THRESHOLD
        if not good_name:
            names_wrong.append((owner, key))
        if not good_distance:
            distance_wrong.append((owner, key))
        if good_name and good_distance:
            matched.append((owner, key))
    return metric(len(matched), len(pp), len(gg)), {
        'identity': metric(len(identities), len(pp), len(gg)), 'matched': matched,
        'name_wrong_on_identity': names_wrong, 'distance_wrong_on_identity': distance_wrong,
        'name_and_distance_wrong_overlap': sorted(set(names_wrong) & set(distance_wrong)),
        'gt_identity_missing_count': len(set(gg) - set(identities)),
    }


def references(net):
    text = net.get('hyperGraph', '') if isinstance(net, dict) else ''
    if not isinstance(text, str):
        return set()
    text = text.strip()
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]
    return {v.strip() for v in text.split(',') if v.strip()}


def parse_ref(ref, components):
    for key in sorted(components, key=lambda k: (-len(k), k)):
        prefix = key + '.'
        if ref.startswith(prefix):
            return key, ref[len(prefix):]
    if '.' in ref:
        return tuple(ref.split('.', 1))
    return ref, None


def logical_nets(data, gt_components=None):
    cs = section(data, 'components')
    raw = list(section(data, 'nets').items())
    vertices, power_sets, edges = [], [], []
    unknown = 0
    for _, n in raw:
        vs, rails = set(), set()
        for ref in references(n):
            key, port = parse_ref(ref, cs)
            own = cs.get(key)
            if own is None:
                unknown += 1
                vs.add(('unknown', ref))
                continue
            r = rail(own)
            if r is not None:
                vs.add(r); rails.add(r)
            elif is_ic((gt_components or {}).get(key, own)):
                vs.add(('ic', key, port))
            else:
                vs.add(('component', key))
        vertices.append(vs); power_sets.append(rails)
        ed = n.get('edges', {}) if isinstance(n, dict) else {}
        edges.append(list(ed.values()) if isinstance(ed, dict) else (list(ed) if isinstance(ed, list) else []))
    parent = list(range(len(raw)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    first = {}
    for i, rails in enumerate(power_sets):
        for r in rails:
            if r in first:
                a, b = find(i), find(first[r]); parent[a] = b
            else:
                first[r] = i
    groups = defaultdict(list)
    for i in range(len(raw)):
        groups[find(i)].append(i)
    groups = sorted(groups.values(), key=lambda ids: min(ids))
    out = []
    for ids in groups:
        out.append({'source_keys': [raw[i][0] for i in ids],
                    'vertices': set().union(*(vertices[i] for i in ids)),
                    'rails': set().union(*(power_sets[i] for i in ids)),
                    'edges': [e for i in ids for e in edges[i]]})
    return out, {'raw_net_count': len(raw), 'logical_net_count': len(out),
                 'power_groups': sum(bool(n['rails']) for n in out), 'unknown_references': unknown}


def valid_edge(e):
    return isinstance(e, (list, tuple)) and len(e) == 2 and all(map(valid_point, e))


def edge_tp(pp, gg):
    if not pp or not gg:
        return 0
    weights = []
    for p in pp:
        row = []
        for g in gg:
            good = False
            if valid_edge(p) and valid_edge(g):
                direct = max(math.dist(xy(p[0]), xy(g[0])), math.dist(xy(p[1]), xy(g[1])))
                reverse = max(math.dist(xy(p[0]), xy(g[1])), math.dist(xy(p[1]), xy(g[0])))
                good = min(direct, reverse) <= WIRE_THRESHOLD
            row.append(int(good))
        weights.append(row)
    return sum(weights[i][j] for i, j in maximum_weight_assignment(weights))


def graph_wire_score(pred, gt):
    pn, pd = logical_nets(pred, section(gt, 'components'))
    gn, gd = logical_nets(gt)
    weights = [[2 * len(p['vertices'] & g['vertices']) / (len(p['vertices']) + len(g['vertices']))
                if p['vertices'] or g['vertices'] else 0. for g in gn] for p in pn]
    pairs = maximum_weight_assignment(weights) if pn and gn else []
    pairs = [(i, j) for i, j in pairs if weights[i][j] > 0]
    soft_tp = sum(weights[i][j] for i, j in pairs)
    wire_tp = sum(edge_tp(pn[i]['edges'], gn[j]['edges']) for i, j in pairs)
    return metric(soft_tp, len(pn), len(gn)), metric(wire_tp, sum(len(n['edges']) for n in pn), sum(len(n['edges']) for n in gn)), {
        'prediction': pd, 'target': gd, 'assigned_pairs': len(pairs),
        'assignment_soft_f1': [weights[i][j] for i, j in pairs]}


@lru_cache(maxsize=2048)
def cached_graph(pred_serialized, gt_serialized):
    return graph_wire_score(json.loads(pred_serialized), json.loads(gt_serialized))


def evaluate(pred, gt):
    c = components_score(pred, gt)
    p, pin_diag = pins_score(pred, gt)
    def graph_input(x):
        return json.dumps({'components': section(x, 'components'), 'nets': section(x, 'nets')}, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    n, w, graph_diag = cached_graph(graph_input(pred), graph_input(gt))
    metrics = {'Component': c, 'Pin': p, 'NetHypergraph': n, 'NetLine': w}
    return {'metrics': metrics, 'FinalScore': 100 * sum(metrics[k]['f1'] * v for k, v in WEIGHTS.items()),
            'pin_diagnostics': pin_diag, 'graph_diagnostics': graph_diag,
            'OFFICIAL_SCORE': False, 'protocol': PROTOCOL_VERSION,
            'spatial_match_thresholds_px': {'component': COMPONENT_THRESHOLD, 'pin': PIN_THRESHOLD, 'wire': WIRE_THRESHOLD}}


def aggregate(rows):
    if not rows:
        return None
    m = {}
    for key in WEIGHTS:
        vals = [r['metrics'][key] for r in rows]
        m[key] = {**metric(sum(v['tp'] for v in vals), sum(v['pred'] for v in vals), sum(v['gt'] for v in vals)),
                  'macro_f1': sum(v['f1'] for v in vals) / len(vals)}
    return {'count': len(rows), 'metrics': m, 'FinalScore': sum(r['FinalScore'] for r in rows) / len(rows)}
