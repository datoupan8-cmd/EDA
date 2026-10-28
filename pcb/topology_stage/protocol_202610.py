"""Versioned local reading of the supplied October scoring note.

Not organizer code. Text is exact: no case/unit/whitespace normalization.
Explicit global-identity assumptions: gnd -> GND, v -> nonempty value,
net_* -> nonempty value. Missing labels retain separate instance identities.
"""
from collections import Counter
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

VERSION = 'clarified_202610_v1'
OFFICIAL_SCORE = False
IC = frozenset({'box', 'ic', 'IC'})
AUTO = frozenset({'gnd', 'v', 'net_input', 'net_output', 'net_short', 'net_bidirection'})


def prf(tp, pred, gt):
    return {'tp': int(tp), 'pred': int(pred), 'gt': int(gt),
            'precision': tp/pred if pred else float(not gt),
            'recall': tp/gt if gt else float(not pred),
            'f1': 2*tp/(pred+gt) if pred+gt else 1.0}


def assignment(weights):
    a = np.asarray(weights, dtype=float)
    if not a.size:
        return []
    rows, cols = linear_sum_assignment(-a)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if a[i, j] > 0]


def name(c):
    return c.get('name', c.get('Name'))


def bbox_error(a, b):
    return max(abs(x-y) for x, y in zip(a, b))


def component_counts(pred, gt):
    p, g = pred['components'], gt['components']
    def match(a, b, automatic=False):
        return (a['type'] == b['type'] and bbox_error(a['bbox'], b['bbox']) <= 20
                and (automatic or name(a) == name(b))
                and (a['type'] in IC or a.get('value') == b.get('value')))
    tp = sum(k in p and match(p[k], c) for k, c in g.items() if c['type'] not in AUTO)
    for kind in AUTO:
        pp = [c for c in p.values() if c['type'] == kind]
        gg = [c for c in g.values() if c['type'] == kind]
        tp += len(assignment([[int(match(a, b, True)) for b in gg] for a in pp]))
    return prf(tp, len(p), len(g))


def pin_counts(pred, gt):
    # A wrong predicted type cannot remove a GT IC from the scored scope.
    gc, pc = gt['components'], pred['components']
    gp = {(k, pk): row for k, pins in gt['pins'].items() if gc[k]['type'] in IC
          for pk, row in pins.items()}
    pp = {(k, pk): row for k, pins in pred['pins'].items()
          if (gc[k]['type'] in IC if k in gc else pc[k]['type'] in IC)
          for pk, row in pins.items()}
    def xy(row):
        return row['point']['x'], row['point']['y']
    tp = sum(k in gp and row.get('pinname') == gp[k].get('pinname')
             and math.dist(xy(row), xy(gp[k])) <= 5 for k, row in pp.items())
    return prf(tp, len(pp), len(gp))


def parse_ref(ref, components):
    owners = [k for k in components if ref.startswith(k+'.')]
    if not owners:
        return None, ref
    owner = max(owners, key=len)
    return owner, ref[len(owner)+1:]


def global_identity(c):
    if c['type'] == 'gnd':
        return ('rail', 'GND')
    value = c.get('value')
    if c['type'] in AUTO and isinstance(value, str) and value.strip():
        return ('rail', value)
    return None


def logical_networks(data, reference_components=None):
    """Return normalized vertex sets plus all original branch segments."""
    components = data['components']
    rows = []
    for net in data['nets'].values():
        vertices = set()
        for ref in net['hyperGraph'][1:-1].split(','):
            if not ref:
                continue
            owner, pin = parse_ref(ref, components)
            if owner is None:
                vertices.add(('invalid', ref))
                continue
            c = components[owner]
            rail = global_identity(c)
            # Classification follows GT for a known ordinary/IC identity.
            kind = (reference_components or {}).get(owner, c)['type']
            if rail is not None:
                vertices.add(rail)
            elif kind in IC:
                vertices.add(('pin', owner, pin))
            else:
                vertices.add(('component', owner))
        rows.append({'vertices': vertices, 'edges': list(net['edges'].values())})
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    first = {}
    for i, row in enumerate(rows):
        for v in row['vertices']:
            if v[0] == 'rail':
                if v in first:
                    parents[find(i)] = find(first[v])
                else:
                    first[v] = i
    groups = {}
    for i, row in enumerate(rows):
        group = groups.setdefault(find(i), {'vertices': set(), 'edges': []})
        group['vertices'].update(row['vertices'])
        group['edges'].extend(row['edges'])
    return list(groups.values())


def segment_count(pred, gt):
    def pt(v):
        return v['x'], v['y']
    def valid(p, g):
        a, b = map(pt, p)
        c, d = map(pt, g)
        return min(max(math.dist(a, c), math.dist(b, d)),
                   max(math.dist(a, d), math.dist(b, c))) <= 5
    return len(assignment([[int(valid(p, g)) for g in gt] for p in pred]))


def pairs(sets):
    result = set()
    for vertices in sets:
        vv = sorted(vertices)
        result.update((a, b) for i, a in enumerate(vv) for b in vv[i+1:])
    return result


def evaluate_case(pred, gt, diagnostics=None):
    pp = logical_networks(pred, gt['components'])
    gg = logical_networks(gt)
    ps, gs = [r['vertices'] for r in pp], [r['vertices'] for r in gg]
    matches = assignment([[2*len(p & g)/(len(p)+len(g)) if p or g else 0
                           for g in gs] for p in ps])
    ntp = sum(len(ps[i] & gs[j]) for i, j in matches)
    ltp = sum(segment_count(pp[i]['edges'], gg[j]['edges']) for i, j in matches)
    pv = set().union(set(), *ps)
    gv = set().union(set(), *gs)
    known = pv & gv
    pk, gk = pairs([s & known for s in ps]), pairs([s & known for s in gs])
    metrics = {'Component': component_counts(pred, gt), 'Pin': pin_counts(pred, gt),
               'NetHypergraph': prf(ntp, sum(map(len, ps)), sum(map(len, gs))),
               'NetLine': prf(ltp, sum(len(r['edges']) for r in pp), sum(len(r['edges']) for r in gg)),
               'PinPair': prf(len(pk & gk), len(pk), len(gk))}
    pc, gc = Counter(v for s in ps for v in s), Counter(v for s in gs for v in s)
    cap = sum(min(n, gc[v]) for v, n in pc.items())
    def power_refs(d):
        refs = [r for n in d['nets'].values() for r in n['hyperGraph'][1:-1].split(',') if r]
        return sum(parse_ref(r, d['components'])[0] is not None and
                   d['components'][parse_ref(r, d['components'])[0]]['type'] in {'gnd', 'v'} for r in refs)
    counts = {'pred_raw_nets': len(pred['nets']), 'gt_raw_nets': len(gt['nets']),
              'pred_logical_nets': len(pp), 'gt_logical_nets': len(gg),
              'pred_merged_branches': len(pred['nets'])-len(pp),
              'gt_merged_branches': len(gt['nets'])-len(gg),
              'pred_power_refs': power_refs(pred), 'gt_power_refs': power_refs(gt),
              'false_known_pairs': len(pk-gk), 'missed_known_pairs': len(gk-pk)}
    return {'protocol': VERSION, 'OFFICIAL_SCORE': False, 'metrics': metrics,
            'FinalScore': 100*sum(w*metrics[k]['f1'] for w, k in
                                  ((.30, 'Component'), (.25, 'Pin'), (.35, 'NetHypergraph'), (.10, 'NetLine'))),
            'counts': counts, 'identity_membership_cap': prf(cap, sum(map(len, ps)), sum(map(len, gs)))['f1']}


def aggregate(records):
    rows = [r['evaluation'] for r in records if r.get('status') == 'ok']
    if not rows:
        return {}
    return {'overall': {'case_count': len(rows),
            'metrics': {k: {**prf(sum(r['metrics'][k]['tp'] for r in rows),
                                   sum(r['metrics'][k]['pred'] for r in rows),
                                   sum(r['metrics'][k]['gt'] for r in rows)),
                              'macro_f1': sum(r['metrics'][k]['f1'] for r in rows)/len(rows)}
                        for k in rows[0]['metrics']},
            'FinalScore': sum(r['FinalScore'] for r in rows)/len(rows),
            'counts': {k: sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}}}
