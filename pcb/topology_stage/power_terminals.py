"""Image-evidence power terminals added to a frozen logical network plan.

No union of existing groups, no new path and no change to ordinary pins.
The schema suffix is a local single-terminal identifier, not an OCR pin number.
"""
from dataclasses import replace
import math

import numpy as np

from ..network_shared.contracts import NetworkPlan
from ..schema import Pin

VERSION = 'power_terminals_v1'


def attach_power_terminals(scene, graph, plan, diagnostics):
    groups = {g.group_id: g for g in plan.groups}
    ref_groups = {r: g.group_id for g in plan.groups for r in g.pins}
    path_groups = {i: g.group_id for g in plan.groups for i in g.path_ids}
    snaps = {r['pin']: r for r in diagnostics.get('snaps', [])}
    stroke = float(scene.diagnostics.get('wire_stroke_width', 1.0))
    radius = max(2.5, min(12., 2.2*stroke+2))  # unchanged legacy snap distance
    replacements, additions, events, rejected = {}, {}, [], []
    endpoints = []
    for i, path in enumerate(graph.paths):
        for points in (path, path[::-1]):
            if len(points) < 3:
                continue
            # Only physical loose ends; junctions are not symbol attachment tips.
            y, x = points[0]
            if len(graph.adjacency.get((y, x), ())) != 1:
                continue
            yy, xx = points[min(len(points)-1, max(2, int(round(2*stroke))))]
            tangent = np.asarray((xx-x, yy-y), float)
            tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
            endpoints.append((i, (float(x), float(y)), tangent))

    for c in scene.components:
        if c.type not in {'gnd', 'v'}:
            continue
        if c.type == 'v' and (not isinstance(c.value, str) or not c.value.strip()):
            rejected.append({'owner': c.key, 'reason': 'missing_supply_identity'})
            continue
        if any(p.exportable for p in c.pins):
            rejected.append({'owner': c.key, 'reason': 'existing_export_protected'})
            continue
        candidates = []
        for index, pin in enumerate(c.pins):
            ref = c.key+'.'+pin.number
            hit = snaps.get(ref)
            gid = ref_groups.get(ref)
            if hit is not None and gid is not None and groups[gid].path_ids:
                candidates.append((float(hit['distance']), gid, index, pin.tip, pin.base, pin.side))
        method = 'existing_directional_snap'
        if not candidates:
            method = 'wire_endpoint_at_symbol_boundary'
            x1, y1, x2, y2 = c.body_bbox or c.bbox
            cx, cy = (x1+x2)/2, (y1+y2)/2
            sides = [('left', (x1, cy), (-1., 0.), y2-y1),
                     ('right', (x2, cy), (1., 0.), y2-y1),
                     ('top', (cx, y1), (0., -1.), x2-x1),
                     ('bottom', (cx, y2), (0., 1.), x2-x1)]
            for path_id, point, tangent in endpoints:
                for side, base, outward, span in sides:
                    delta = np.asarray(point)-np.asarray(base)
                    normal = float(np.dot(delta, outward))
                    lateral = abs(float(delta[1] if side in {'left', 'right'} else delta[0]))
                    if not -stroke <= normal <= radius:
                        continue
                    if lateral > max(2*stroke, .30*span):
                        continue
                    if abs(float(np.dot(tangent, outward))) < .70:
                        continue
                    distance = math.dist(point, base)
                    if distance <= radius:
                        candidates.append((distance, path_groups[path_id], None, point, base, side))
        if not candidates:
            rejected.append({'owner': c.key, 'reason': 'no_supported_terminal'})
            continue
        candidates.sort(key=lambda r: (r[0], r[1], -1 if r[2] is None else r[2]))
        best = candidates[0]
        if any(row[1] != best[1] and row[0] <= best[0]+stroke for row in candidates[1:]):
            rejected.append({'owner': c.key, 'reason': 'ambiguous_wire_groups'})
            continue
        distance, gid, index, tip, base, side = best
        if not all(math.isfinite(v) for v in tip) or not (0 <= tip[0] <= scene.width and 0 <= tip[1] <= scene.height):
            rejected.append({'owner': c.key, 'reason': 'invalid_terminal'})
            continue
        used = {p.number for i, p in enumerate(c.pins) if i != index}
        suffix = '1' if '1' not in used else 'PWR1'
        while suffix in used:
            suffix += '_'
        ref = c.key+'.'+suffix
        if index is None:
            pin = Pin(suffix, '', tuple(tip), base=tuple(base), side=side,
                      inferred_number=True, exportable=True, number_source='single_terminal_power_schema')
            c.pins.append(pin)
            additions.setdefault(gid, []).append(ref)
        else:
            pin = c.pins[index]
            old_ref = c.key+'.'+pin.number
            replacements[old_ref] = ref
            pin.number = suffix
            pin.exportable = True
            pin.inferred_number = True
            pin.observable_number = None
            pin.number_confidence = 0.
            pin.number_source = 'single_terminal_power_schema'
        events.append({'owner': c.key, 'type': c.type, 'value': c.value, 'ref': ref,
                       'group_id': gid, 'distance': round(distance, 4), 'tip': tuple(tip), 'method': method})
    rows = tuple(replace(g, pins=tuple(replacements.get(r, r) for r in g.pins)+tuple(additions.get(g.group_id, ())))
                 for g in plan.groups)
    return NetworkPlan(rows), {'version': VERSION, 'events': events, 'rejected': rejected,
                              'snap_radius': radius, 'existing_group_merges': 0}
