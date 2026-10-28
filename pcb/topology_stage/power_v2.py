"""Power recovery using local image evidence, with frozen physical wire groups."""
from dataclasses import replace
import math
from types import MappingProxyType

import cv2
import numpy as np

from ..core.interfaces import TopologyStageOutput
from ..network_shared.composite import assemble, validate_plan
from ..network_shared.contracts import snapshot, NetworkPlan
from ..network_shared.graph import build_graph
from ..schema import Pin
from .power_terminals import attach_power_terminals
from .versions import TopologyStageV3

VERSION = 'power_terminals_v2_image_evidence'


def power_crops(image, scene):
    """Read-only local observations carried explicitly by WireStageOutput.debug."""
    rows = {}
    h, w = image.shape[:2]
    for c in scene.components:
        if c.type not in {'gnd', 'v'}:
            continue
        x1, y1, x2, y2 = c.body_bbox or c.bbox
        a, b = max(0, int(math.floor(x1))-26), max(0, int(math.floor(y1))-26)
        z, q = min(w, int(math.ceil(x2))+27), min(h, int(math.ceil(y2))+27)
        crop = image[b:q, a:z].copy()
        crop.setflags(write=False)
        rows[c.key] = {'origin': (a, b), 'image': crop}
    return MappingProxyType(rows)


class PowerEvidenceWire:
    """Wrap the selected Wire without changing its masks or inference rules."""
    def __init__(self, baseline):
        self.baseline = baseline

    def run(self, image, scene, context):
        wire = self.baseline.run(image, scene, context)
        return replace(wire, debug={**wire.debug, 'power_symbol_crops': power_crops(image, scene)})


def line_evidence(crop, base, tip, stroke, obstacles):
    """Test real image ink along the proposed stem; never synthesize wire pixels."""
    image = crop['image']
    if not image.size:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ink = (gray < 205) | ((hsv[:, :, 1] >= 45) & (hsv[:, :, 2] < 248))
    delta = np.asarray(tip, float)-np.asarray(base, float)
    distance = float(np.linalg.norm(delta))
    if distance < 1:
        return None
    direction = delta/distance
    normal = np.asarray((-direction[1], direction[0]))
    start = min(distance, max(1., .5*stroke))
    points = np.asarray(base)+np.linspace(start, distance, max(3, int(math.ceil(distance))+1))[:, None]*direction
    for box in obstacles:
        a, b, z, q = box
        inside = (points[:, 0] > a+.5*stroke) & (points[:, 0] < z-.5*stroke) & (points[:, 1] > b+.5*stroke) & (points[:, 1] < q-.5*stroke)
        if inside.any():
            return None
    origin = np.asarray(crop['origin'])
    supported = np.zeros(len(points), bool)
    half_width = max(1, int(math.ceil(.5*stroke)))
    for offset in range(-half_width, half_width+1):
        sample = np.rint(points+offset*normal-origin).astype(int)
        valid = ((sample[:, 0] >= 0) & (sample[:, 0] < ink.shape[1]) &
                 (sample[:, 1] >= 0) & (sample[:, 1] < ink.shape[0]))
        supported[valid] |= ink[sample[valid, 1], sample[valid, 0]]
    longest = current = 0
    for v in supported:
        current = 0 if v else current+1
        longest = max(longest, current)
    empty_run = longest*max(0., distance-start)/max(1, len(points)-1)
    ratio = float(supported.mean())
    if ratio < .90 or empty_run > stroke:
        return None
    return {'ink_ratio': ratio, 'max_empty_run_px': empty_run}


def attach_power_v2(scene, graph, plan, diagnostics, crops):
    # V1 stays byte-for-byte unchanged and is always the first decision layer.
    plan, old_trace = attach_power_terminals(scene, graph, plan, diagnostics)
    recoverable = {r['owner'] for r in old_trace['rejected'] if r['reason'] == 'no_supported_terminal'}
    path_groups = {i: g.group_id for g in plan.groups for i in g.path_ids}
    stroke = float(scene.diagnostics.get('wire_stroke_width', 1.))
    old_radius = max(2.5, min(12., 2.2*stroke+2))
    endpoints = []
    for i, path in enumerate(graph.paths):
        for points in (path, path[::-1]):
            if len(points) < 3 or len(graph.adjacency.get(points[0], ())) != 1:
                continue
            y, x = points[0]
            yy, xx = points[min(len(points)-1, max(2, int(round(2*stroke))))]
            vector = np.asarray((xx-x, yy-y), float)
            vector /= max(float(np.linalg.norm(vector)), 1e-6)
            endpoints.append((path_groups[i], (float(x), float(y)), vector))
    added, events, recovered = {}, [], set()
    for c in scene.components:
        if c.key not in recoverable or c.key not in crops:
            continue
        x1, y1, x2, y2 = c.body_bbox or c.bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        reach = min(24., old_radius+max(0., min(x2-x1, y2-y1)))
        obstacles = [d.body_bbox or d.bbox for d in scene.components if d.key != c.key]
        sides = [('left', (x1, cy), (-1., 0.), y2-y1), ('right', (x2, cy), (1., 0.), y2-y1),
                 ('top', (cx, y1), (0., -1.), x2-x1), ('bottom', (cx, y2), (0., 1.), x2-x1)]
        candidates = []
        for gid, point, tangent in endpoints:
            for side, center, outward, span in sides:
                delta = np.asarray(point)-np.asarray(center)
                normal = float(np.dot(delta, outward))
                lateral = abs(float(delta[1] if side in {'left', 'right'} else delta[0]))
                if not 0 <= normal <= reach or lateral > max(2*stroke, .30*span):
                    continue
                if float(np.dot(tangent, outward)) < .70:
                    continue
                base = (center[0], point[1]) if side in {'left', 'right'} else (point[0], center[1])
                evidence = line_evidence(crops[c.key], base, point, stroke, obstacles)
                if evidence is not None:
                    candidates.append((normal, gid, point, base, side, evidence))
        candidates.sort(key=lambda r: (r[0], r[1]))
        if not candidates:
            continue
        best = candidates[0]
        if any(r[1] != best[1] and r[0] <= best[0]+stroke for r in candidates[1:]):
            continue
        distance, gid, point, base, side, evidence = best
        used = {p.number for p in c.pins}
        suffix = '1'
        while suffix in used:
            suffix += '_'
        ref = c.key+'.'+suffix
        c.pins.append(Pin(suffix, '', tuple(point), base=tuple(base), side=side,
                          exportable=True, inferred_number=True, number_source='image_supported_power_schema'))
        added.setdefault(gid, []).append(ref)
        recovered.add(c.key)
        events.append({'owner': c.key, 'type': c.type, 'value': c.value, 'ref': ref, 'group_id': gid,
                       'tip': point, 'base': base, 'distance': distance, 'search_reach': reach,
                       'method': 'image_supported_suppressed_stem', **evidence})
    result = NetworkPlan(tuple(replace(g, pins=g.pins+tuple(added.get(g.group_id, ()))) for g in plan.groups))
    trace = {**old_trace, 'version': VERSION, 'events': old_trace['events']+events,
             'v2_extra_refs': len(events), 'evidence_crop_count': len(crops),
             'rejected': [r for r in old_trace['rejected'] if r['owner'] not in recovered]}
    return result, trace


class PowerTerminalTopologyV2(TopologyStageV3):
    def run(self, scene, wire, context):
        crops = wire.debug.get('power_symbol_crops')
        if crops is None:
            raise ValueError('Power V2 requires explicit local image evidence from PowerEvidenceWire')
        graph = build_graph(wire.mask)
        logical = self.hypergraph.run(snapshot(scene), graph, mode='directional',
                                      merge_labels=context.config.enable_flying_labels)
        plan, trace = attach_power_v2(scene, graph, logical.plan, logical.diagnostics, crops)
        validate_plan(plan, snapshot(scene), graph)
        scene.nets = assemble(plan, self.lines.run(graph, plan))
        scene.diagnostics.update(logical.diagnostics)
        scene.diagnostics.update(topology='v3_power_terminals_v2', power_terminals=trace,
                                 nets=len(scene.nets), net_edges=sum(len(n.segments) for n in scene.nets))
        return TopologyStageOutput(graph.skeleton.copy())
