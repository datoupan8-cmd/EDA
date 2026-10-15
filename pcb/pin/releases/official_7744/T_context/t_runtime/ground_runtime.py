import copy, math
from pcb.submission import export, validate_strict

def segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy else 0.
    return math.dist(p, (a[0]+t*dx, a[1]+t*dy))

def project(scene):
    out = copy.deepcopy(scene); changes = []; rejects = []; remap = {}
    for owner in out.components:
        if owner.type != 'gnd':
            continue
        if len(owner.pins) != 1:
            rejects.append({'owner': owner.key, 'reason': 'not_exactly_one_raw_terminal'}); continue
        p = owner.pins[0]
        if p.exportable:
            rejects.append({'owner': owner.key, 'reason': 'already_exported_protected'}); continue
        if p.number_source != 'unobservable_internal_id':
            rejects.append({'owner': owner.key, 'reason': 'unqualified_number_source'}); continue
        ref = owner.key+'.'+p.number
        networks = [n for n in out.nets if ref in n.pins]
        if len(networks) != 1:
            rejects.append({'owner': owner.key, 'reason': 'nonunique_or_absent_existing_net'}); continue
        if not all(math.isfinite(x) for x in p.tip) or not (0 <= p.tip[0] <= scene.width and 0 <= p.tip[1] <= scene.height):
            rejects.append({'owner': owner.key, 'reason': 'invalid_tip'}); continue
        segments = networks[0].segments
        distance = min((segment_distance(p.tip, a, b) for a, b in segments), default=math.inf)
        if distance > 5.:
            rejects.append({'owner': owner.key, 'reason': 'no_local_wire_support', 'distance': distance if math.isfinite(distance) else None}); continue
        remap[ref] = owner.key+'.1'
        changes.append({'owner': owner.key, 'old_schema_ref': ref, 'new_schema_ref': remap[ref],
                        'wire_distance_px': distance, 'tip': p.tip, 'printed_number_inferred': False})
        p.number = '1'; p.exportable = True; p.inferred_number = True
        p.observable_number = None; p.number_confidence = 0.
        p.number_source = 'single_terminal_ground_schema'
    for n in out.nets:
        n.pins = [remap.get(ref, ref) for ref in n.pins]
    before, after = export(scene), export(out)
    assert before['components'] == after['components']
    for key, pins in before['pins'].items():
        owner = next(c for c in scene.components if c.key == key)
        if owner.type != 'gnd':
            assert after['pins'][key] == pins
        else:
            assert all(after['pins'][key].get(k) == v for k, v in pins.items())
    assert [n.segments for n in scene.nets] == [n.segments for n in out.nets]
    validate_strict(after)
    return out, after, changes, rejects

class Submission:
    def __init__(self, baseline):
        self.baseline = baseline

    def run(self, scene, context):
        self.before = self.baseline.run(scene, context).data
        self.projected, expected, self.changes, self.rejects = project(scene)
        out = self.baseline.run(self.projected, context)
        assert out.data == expected
        return out
