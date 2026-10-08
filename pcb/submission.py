"""Official-raw-schema export and strict fail-closed validation."""
from __future__ import annotations
import math
from .coordinates import opencv_bbox_to_target, opencv_to_target, assert_bbox_in_image, assert_point_in_image
from .official_types import OFFICIAL_COMPONENT_TYPES, to_official_type

def export(scene):
    def point(p):
        x,y=opencv_to_target(float(p[0]),float(p[1]),scene.height)
        return {'x':round(x,3),'y':round(y,3)}
    result={'components':{},'pins':{},'nets':{}}
    for c in scene.components:
        typ=to_official_type(c.type)
        bbox=[round(v,3) for v in opencv_bbox_to_target(c.bbox,scene.height)]
        # Component V4 keeps the visible designator in the object key while a
        # chip/model name is stored independently in ``name``.
        display=c.name or c.observable_designator or c.net_label or c.key
        result['components'][c.key]={'Name':display,'type':typ,'value':c.value,'bbox':bbox}
        result['pins'][c.key]={}
        for p in c.pins:
            if not p.exportable:continue
            result['pins'][c.key][f'pin_{p.number}']={'pinname':p.name or '','point':point(p.tip)}
    valid_refs={f'{key}.{pk[4:]}' for key,pins in result['pins'].items() for pk in pins}
    assigned=set();net_index=0
    for net in scene.nets:
        refs=[]
        for ref in net.pins:
            if ref not in valid_refs:continue
            if ref in assigned:
                raise ValueError(f'Pin assigned to multiple scene nets: {ref}')
            refs.append(ref);assigned.add(ref)
        if not refs:continue
        net_index+=1
        result['nets'][f'net_{net_index}']={'hyperGraph':'('+','.join(refs)+')',
            'edges':{f'edge_{j}':[point(a),point(b)] for j,(a,b) in enumerate(net.segments,1)}}
    validate_strict(result,(scene.width,scene.height));return result

def validate_strict(data,size=None):
    if not isinstance(data,dict) or set(data)!={'components','pins','nets'}:raise ValueError('Top level must contain exactly components, pins, nets')
    if not all(isinstance(data[k],dict) for k in data):raise ValueError('Top-level values must be objects')
    width,height=size if size else (None,None)
    def number(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
    def check_point(p,where):
        if not isinstance(p,dict) or set(p)!={'x','y'} or not all(number(p[k]) for k in ('x','y')):raise ValueError(f'{where}: invalid point')
        if size:assert_point_in_image((p['x'],p['y']),width,height)
    components,pins,nets=data['components'],data['pins'],data['nets']
    if set(pins)!=set(components):raise ValueError('pins must contain exactly every component key')
    valid_refs=set()
    for key,c in components.items():
        if not isinstance(key,str) or not key or ',' in key:raise ValueError(f'Unsafe component key: {key!r}')
        if not isinstance(c,dict) or set(c)!={'Name','type','value','bbox'}:raise ValueError(f'{key}: component fields must be Name/type/value/bbox')
        if c['type'] not in OFFICIAL_COMPONENT_TYPES:raise ValueError(f'{key}: non-official type {c["type"]!r}')
        if c['Name'] is not None and not isinstance(c['Name'],str):raise ValueError(f'{key}: Name must be string/null')
        if c['value'] is not None and not isinstance(c['value'],str):raise ValueError(f'{key}: value must be string/null')
        b=c['bbox']
        if not isinstance(b,list) or len(b)!=4 or not all(number(v) for v in b):raise ValueError(f'{key}: invalid bbox')
        if size:assert_bbox_in_image(b,width,height)
        if not isinstance(pins[key],dict):raise ValueError(f'{key}: pins must be object')
        for pk,p in pins[key].items():
            if not isinstance(pk,str) or not pk.startswith('pin_') or not pk[4:]:raise ValueError(f'{key}: invalid pin key {pk!r}')
            # Parentheses occur in official physical labels such as R(SENSE.
            # Comma remains forbidden because it is the hyperGraph delimiter.
            if ',' in pk[4:]:raise ValueError(f'{key}.{pk}: unsafe pin suffix')
            if not isinstance(p,dict) or set(p)!={'pinname','point'} or not isinstance(p['pinname'],str):raise ValueError(f'{key}.{pk}: invalid pin fields')
            check_point(p['point'],f'{key}.{pk}');valid_refs.add(f'{key}.{pk[4:]}')
    assigned=set()
    for nk,n in nets.items():
        if not isinstance(nk,str) or not nk:raise ValueError('Empty net key')
        if not isinstance(n,dict) or set(n)!={'hyperGraph','edges'}:raise ValueError(f'{nk}: invalid net fields')
        text=n['hyperGraph']
        if not isinstance(text,str) or not(text.startswith('(') and text.endswith(')')):raise ValueError(f'{nk}: invalid hyperGraph')
        refs=text[1:-1].split(',') if text[1:-1] else []
        if not refs:raise ValueError(f'{nk}: empty hyperGraph')
        if len(refs)!=len(set(refs)):raise ValueError(f'{nk}: duplicate hyperGraph pin')
        missing=[r for r in refs if r not in valid_refs]
        if missing:raise ValueError(f'{nk}: unknown hyperGraph references {missing[:3]}')
        overlap=assigned.intersection(refs)
        if overlap:raise ValueError(f'{nk}: pins assigned to multiple nets {sorted(overlap)[:3]}')
        assigned.update(refs)
        if not isinstance(n['edges'],dict):raise ValueError(f'{nk}: edges must be object')
        for ek,line in n['edges'].items():
            if not isinstance(ek,str) or not isinstance(line,list) or len(line)!=2:raise ValueError(f'{nk}.{ek}: edge must contain exactly two points')
            check_point(line[0],f'{nk}.{ek}[0]');check_point(line[1],f'{nk}.{ek}[1]')
    return True

validate=validate_strict
