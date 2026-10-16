"""L40 public-symbol identity proposal; never creates/moves a terminal.

No OCR, network or annotation access. Assets are checked only when selected.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from .frozen_l39.library_semantics import name_key, model_key
from ..assignment import maximum_weight_assignment


@dataclass(frozen=True)
class StructurePolicy:
    row_fraction: float = .35
    min_template_coverage: float = .60
    confidence: float = .80
    min_anchor_slots: int = 2
    min_anchor_values: int = 2
    orientation_margin: int = 2
    normal_min_pixels: float = 30.0
    normal_body_fraction: float = .50


POLICY = StructurePolicy()
ASSET_DIRECTORY = Path(__file__).resolve().parents[2] / 'models/pin/l40_public_symbols'


def normalized(text):
    return re.sub(r'[^A-Z0-9]', '', unicodedata.normalize('NFKC', str(text)).upper())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compile_symbol(record, response, library):
    """Parse independent EasyEDA symbols, not footprint pads or source IDs."""
    result = response.get('result', {})
    data = result.get('dataStr', {})
    data = json.loads(data) if isinstance(data, str) else data
    if str(result.get('docType')) != '2' or str(data.get('head', {}).get('docType')) != '2':
        raise ValueError('not_schematic_symbol')
    model = record['model']
    meta = data['head'].get('c_para', {})
    # Exact public MPN corroboration; no fuzzy family/supplier-code guessing.
    mpn = meta.get('Manufacturer Part') or result.get('title', '')
    if normalized(mpn) != normalized(model):
        raise ValueError('public_model_identity_mismatch')
    rectangles, pins = [], []
    for shape in data.get('shape', []):
        if shape.startswith('R~'):
            r = shape.split('~')
            x, y, w, h = map(float, (r[1], r[2], r[5], r[6]))
            if w > 0 and h > 0:
                rectangles.append([x, y, x+w, y+h])
        elif shape.startswith('P~'):
            sections = shape.split('^^')
            h = sections[0].split('~')
            if h[1] != 'show':
                continue
            name = sections[3].split('~')[4].strip()
            number = sections[4].split('~')[4].strip().upper()
            x, y = map(float, sections[1].split('~')[:2])
            if not name or not number:
                raise ValueError('incomplete_symbol_pin')
            pins.append({'number': number, 'name': name, 'point': [x, y]})
    if not rectangles or len(pins) < 2:
        raise ValueError('no_body_or_visible_pins')
    body = max(rectangles, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
    if len({p['number'] for p in pins}) != len(pins):
        raise ValueError('multiple_positions_for_one_number')
    public_pairs = {(str(p.get('number', '')).strip().upper(), normalized(p.get('name', '')))
                    for p in library[model]['pins']}
    if any((p['number'], normalized(p['name'])) not in public_pairs for p in pins):
        raise ValueError('symbol_pairs_not_supported_by_original_public_library')
    for p in pins:
        p['uv'] = [(p['point'][0]-body[0])/(body[2]-body[0]),
                   (p['point'][1]-body[1])/(body[3]-body[1])]
    return {'model': model, 'source': record['url'], 'source_sha256': record['sha256'],
            'uuid': result.get('uuid'), 'body': body, 'pins': pins}


def load_templates(directory=ASSET_DIRECTORY, library=None):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text('utf-8'))
    if library is None:
        library = json.loads((Path(__file__).resolve().parents[2]/'assets/cpntLibrary.json').read_text('utf-8'))
    accepted, events = {}, []
    for row in manifest['sources']:
        if row['status'] != 'downloaded':
            events.append({'model': row['model'], 'status': row['status']}); continue
        path = directory / row['file']
        if path.parent != directory or digest(path) != row['sha256']:
            raise ValueError('Public asset path/hash mismatch')
        try:
            template = compile_symbol(row, json.loads(path.read_text('utf-8')), library)
            accepted[row['model']] = template
            events.append({'model': row['model'], 'status': 'accepted', 'pin_count': len(template['pins'])})
        except (ValueError, KeyError, IndexError) as exc:
            events.append({'model': row['model'], 'status': 'rejected', 'reason': str(exc)})
    return accepted, events


def transform(u, v, turns, mirror):
    if mirror:
        u = 1-u
    for _ in range(turns):
        u, v = 1-v, u
    return u, v


def outward_side(u, v):
    distances = {'left': -u, 'right': u-1, 'top': -v, 'bottom': v-1}
    if max(distances.values()) >= 0:
        return max(distances, key=distances.get)
    distances = {'left': u, 'right': 1-u, 'top': v, 'bottom': 1-v}
    return min(distances, key=distances.get)


def slots_for(template, turns, mirror):
    slots = []
    for p in template['pins']:
        u, v = transform(*p['uv'], turns, mirror)
        side = outward_side(u, v)
        slots.append({**p, 'side': side, 'tangent': v if side in ('left','right') else u})
    for slot in slots:
        gaps = [abs(slot['tangent']-s['tangent']) for s in slots
                if s['side'] == slot['side'] and abs(slot['tangent']-s['tangent']) > 1e-8]
        slot['pitch'] = min(gaps) if gaps else .5
    return slots


def align_side(terminals, slots, body, fraction):
    """Maximum-cardinality then minimum-error monotone geometric alignment."""
    if not terminals or not slots:
        return []
    side = slots[0]['side']
    axis = 1 if side in ('left','right') else 0
    lo, hi = (body[1],body[3]) if axis == 1 else (body[0],body[2])
    if hi <= lo:
        return []
    ts = sorted(terminals, key=lambda v:(v[1].base[axis] if v[1].base else v[1].tip[axis], v[0]))
    ss = sorted(slots, key=lambda s:(s['tangent'], s['number']))
    n, m = len(ts),len(ss)
    bonus = n+m+1
    dp = [[0.]*(m+1) for _ in range(n+1)]
    actions = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):
            best = max((dp[i-1][j],1),(dp[i][j-1],2))
            t = ((ts[i-1][1].base or ts[i-1][1].tip)[axis]-lo)/(hi-lo)
            error = abs(t-ss[j-1]['tangent'])/ss[j-1]['pitch']
            if error <= fraction:
                best = max(best,(dp[i-1][j-1]+bonus-error,3))
            dp[i][j],actions[i][j] = best
    pairs = []
    i,j=n,m
    while i and j:
        action = actions[i][j]
        if action == 3:
            pairs.append((ts[i-1][0],ss[j-1]));i-=1;j-=1
        elif action == 1:i-=1
        else:j-=1
    return list(reversed(pairs))


def token_pool(scene, component):
    preparation = scene.diagnostics.get('pin_l39_stage_trace', {}).get('joint', {}).get('role_preparation', [])
    local = next((r.get('canonical_tokens', []) for r in preparation if r['component'] == component.key), [])
    raw = [{'text': t.text, 'bbox': t.bbox, 'score': t.score, 'source_id': t.source_id} for t in scene.texts] + local
    # Deduplicate witnesses only; do not alter the frozen OCR/candidate pool.
    selected = []
    for token in sorted(raw,key=lambda t:-t['score']):
        b=token['bbox'];cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
        if any(abs(cx-(s['bbox'][0]+s['bbox'][2])/2)<.5*min(b[2]-b[0],s['bbox'][2]-s['bbox'][0]) and
               abs(cy-(s['bbox'][1]+s['bbox'][3])/2)<.5*min(b[3]-b[1],s['bbox'][3]-s['bbox'][1])
               for s in selected):
            continue
        selected.append(token)
    return [{**t, '_physical_id':str(i)} for i,t in enumerate(selected)]


def witnesses(slot, tokens, body, policy):
    x1,y1,x2,y2=body
    vertical=slot['side'] in ('left','right')
    span = y2-y1 if vertical else x2-x1
    normal_span = x2-x1 if vertical else y2-y1
    edge = {'left':x1,'right':x2,'top':y1,'bottom':y2}[slot['side']]
    normal_limit = max(policy.normal_min_pixels, policy.normal_body_fraction*normal_span)
    allowed = {normalized(slot['name']), normalized(slot['number'])}
    hits=[];conflicts=[]
    for token in tokens:
        if token['score'] < policy.confidence:
            continue
        b=token['bbox'];x,y=(b[0]+b[2])/2,(b[1]+b[3])/2
        tangent = (y-y1)/(y2-y1) if vertical else (x-x1)/(x2-x1)
        normal = abs((x if vertical else y)-edge)
        if normal > normal_limit or abs(tangent-slot['tangent']) > policy.row_fraction*slot['pitch']:
            continue
        value=normalized(token['text'])
        if value in allowed:
            hits.append({'text':token['text'],'bbox':b,'confidence':token['score'],'value':value,
                         'physical_id':token.get('_physical_id',str(id(token)))})
        elif re.fullmatch(r'\d{1,4}',token['text'].strip()):
            conflicts.append({'text':token['text'],'bbox':b,'confidence':token['score']})
    return hits,conflicts


def independent_anchors(evidence):
    """One physical witness can support at most one anchor slot."""
    slots=list(evidence)
    tokens=sorted({h['physical_id'] for hits in evidence.values() for h in hits})
    if not slots or not tokens:
        return []
    lookup={t:i for i,t in enumerate(tokens)}
    weights=[[0.]*(len(tokens)+len(slots)) for _ in slots]
    hits_by_pair={}
    for i,slot in enumerate(slots):
        for h in evidence[slot]:
            j=lookup[h['physical_id']]
            score=len(slots)+1+h['confidence']
            if score>weights[i][j]:weights[i][j]=score;hits_by_pair[i,j]=h
    return [{'terminal_index':slots[i],'hit':hits_by_pair[i,j]}
            for i,j in maximum_weight_assignment(weights) if (i,j) in hits_by_pair]


def apply(scene, templates, policy=POLICY, conflict_resolver=None):
    candidate=copy.deepcopy(scene)
    models=scene.diagnostics.get('L37_public_library', {}).get('models', {})
    trace=[];changes=[]
    for c in candidate.components:
        if c.type != 'box' or c.key not in models:
            continue
        keys=tuple(models[c.key]['library_keys'])
        if len(keys)!=1 or keys[0] not in templates:
            trace.append({'component':c.key,'status':'no_unambiguous_template','models':keys});continue
        template=templates[keys[0]];body=c.body_bbox or c.bbox
        tokens=token_pool(scene,c);directions=[]
        for mirror in (False,True):
            for turns in range(4):
                slots=slots_for(template,turns,mirror);pairs=[]
                for side in ('left','right','top','bottom'):
                    pairs+=align_side([(i,p) for i,p in enumerate(c.pins) if p.side==side],
                                      [s for s in slots if s['side']==side],body,policy.row_fraction)
                evidence={};conflict_map={}
                for i,slot in pairs:
                    hits,conflicts=witnesses(slot,tokens,body,policy)
                    evidence[i]=hits;conflict_map[i]=conflicts
                anchored=independent_anchors(evidence)
                values={a['hit']['value'] for a in anchored}
                coverage=len(pairs)/len(slots)
                viable=coverage >= policy.min_template_coverage
                directions.append({'turns':turns,'mirror':mirror,'pairs':pairs,'evidence':evidence,
                                   'conflicts':conflict_map,'anchor_count':len(anchored),
                                   'anchor_values':len(values),'coverage':coverage,'viable':viable})
        directions.sort(key=lambda d:(d['viable'],d['anchor_count'],d['anchor_values'],d['coverage']),reverse=True)
        best=directions[0]
        runner=max((d['anchor_count'] for d in directions[1:] if d['viable']),default=0)
        info={'component':c.key,'model':keys[0],'source_sha256':template['source_sha256'],
              'orientations':[{k:d[k] for k in ('turns','mirror','anchor_count','anchor_values','coverage','viable')} for d in directions]}
        if (not best['viable'] or best['anchor_count']<policy.min_anchor_slots or
                best['anchor_values']<policy.min_anchor_values or best['anchor_count']-runner<policy.orientation_margin):
            trace.append({**info,'status':'layout_or_anchor_rejected'});continue
        proposals={}
        for i,slot in best['pairs']:
            hits=best['evidence'][i]
            if best['conflicts'][i] and not hits:
                continue
            p=c.pins[i]
            # Preserve literal image spelling if semantically equal; do not
            # change an underscore/slash/case convention unnecessarily.
            name=p.name if p.name and normalized(p.name)==normalized(slot['name']) else slot['name']
            proposals[i]=(slot['number'],name,slot,hits)
        numbers=[proposals[i][0] if i in proposals else p.number for i,p in enumerate(c.pins)
                 if i in proposals or p.exportable]
        if any(count>1 for count in Counter(numbers).values()):
            if conflict_resolver is None:
                trace.append({**info,'status':'whole_component_number_collision'});continue
            proposals,blocked=conflict_resolver(c.pins,proposals)
            info['blocked_proposals']=blocked
            info['collision_strategy']='fixed_point_safe_transactions'
            final=[proposals[i][0] if i in proposals else p.number for i,p in enumerate(c.pins)
                   if i in proposals or p.exportable]
            if any(count>1 for count in Counter(final).values()):
                raise AssertionError('Conflict resolution produced duplicate numbers')
        anchor_confidence=min(h['confidence'] for hits in best['evidence'].values() for h in hits)
        confidence=min(models[c.key]['model_confidence'],anchor_confidence)
        for i,(number,name,slot,hits) in proposals.items():
            p=c.pins[i]
            if p.exportable and p.number==number and p.name==name:
                continue
            row={'component':c.key,'terminal_index':i,'old_number':p.number,'old_name':p.name,
                 'old_exportable':p.exportable,'number':number,'name':name,'tip':p.tip,
                 'side':p.side,'model':keys[0],'source_sha256':template['source_sha256'],
                 'turns':best['turns'],'mirror':best['mirror'],'local_witnesses':hits,
                 'anchor_count':best['anchor_count'],'confidence':confidence}
            changes.append(row)
            p.number,p.name,p.exportable=number,name,True
            p.inferred_number=True;p.observable_number=None
            p.number_confidence=confidence;p.name_confidence=confidence
            p.number_source='L40_public_symbol_structure'
        trace.append({**info,'status':'accepted','proposal_count':len(proposals),
                      'turns':best['turns'],'mirror':best['mirror']})
    events=defaultdict(list)
    for e in candidate.diagnostics.get('pin_events',[]):events[e['component']].append(e)
    for change in changes:
        if events[change['component']]:
            e=events[change['component']][change['terminal_index']]
            e.update(number=change['number'],pinname=change['name'],exportable=True,
                     semantic_method='L40_public_symbol_structure',structure_evidence=copy.deepcopy(change))
    candidate.diagnostics['L40_structure']={'policy':asdict(policy),'components':trace,'changes':changes,
                                           'GT_used':False,'OCR_modified':False,'terminal_geometry_modified':False}
    return candidate


class L40Semantics:
    """Selected-only wrapper; original L39 stays the default."""
    conflict_resolver=None
    def __init__(self):
        self.baseline=None;self.templates=None

    def run_image(self,image,component,localization,context):
        from .frozen_l39.stages import L39Semantics
        from ..schema import Scene
        from ..core.interfaces import PinSemanticsOutput
        if self.baseline is None:self.baseline=L39Semantics()
        if self.templates is None:self.templates,_=load_templates()
        original=self.baseline.run_image(image,component,localization,context)
        result=apply(Scene(context.width,context.height,original.components,component.texts,
                           diagnostics=original.diagnostics),self.templates,
                     conflict_resolver=self.conflict_resolver)
        return PinSemanticsOutput(result.components,result.diagnostics)
