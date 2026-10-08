"""Explicit annotation adapter. Never called by the default image-only inference."""
import json,re
from collections import Counter,defaultdict
from .schema import Component,Pin,Text,Scene,Net,DSU
from .io import natural

TYPE_MAP={'box':'ic','boxic':'ic','r':'resistor','c':'capacitor','l':'inductor','gnd':'gnd','v':'vcc',
          'pin':'connector','net_input':'vbus_in','net_output':'vbus_out','net_bidirection':'vbus_bi',
          'd':'diode','led':'led','mosfet':'mosfet','bjt':'bjt','crystal':'crystal','switch':'switch',
          'testpoint':'testpoint','fuse':'fuse','battary':'battery','amp':'amplifier'}
DESIG=re.compile(r'(?<![A-Za-z0-9])((?:USB|U|IC|R|C|L|Q|D|J|P|Y|X|SW|TP|F|CN|LED)(?:\d+[A-Za-z]?|\?))(?![A-Za-z0-9])',re.I)
# Numeric pins plus common one-letter/AA-AZ BGA rows.  Tokens such as RM2 are
# far more likely net/pin names than physical pin numbers in this dataset.
PINNO=re.compile(r'(?:\d{1,4}|[A-Za-z]\d{1,3}|A[A-Za-z]\d{1,3}|EP|PAD)\Z')

def pin_numbers(value):
    if value is None:return []
    s=str(value).strip()
    parts=re.split(r'\s*[,，;]\s*',s)
    return parts if parts and all(PINNO.fullmatch(p) for p in parts) else []

def unique_keys(comps):
    counts=Counter(c.key for c in comps);used=set();index=Counter()
    for c in comps:
        base=c.key
        if not c.observable_designator and not base.startswith('UNRESOLVED_'):
            c.observable_designator=base
        if counts[base]>1:
            index[base]+=1;c.key=f'{base}_{index[base]}'
        if c.key in used:
            j=1
            while f'{c.key}__{j}' in used:j+=1
            c.key=f'{c.key}__{j}'
        used.add(c.key)

def load_annotation(path,origin='top-left'):
    d=json.loads(path.read_text(encoding='utf-8-sig'))
    w,h=d['canvas']['width'],d['canvas']['height']
    if origin not in ('top-left','bottom-left'):raise ValueError('Unknown annotation origin')
    def box(b):
        return (float(b['x1']),float(b['y1']),float(b['x2']),float(b['y2'])) if origin=='top-left' else (float(b['x1']),h-float(b['y2']),float(b['x2']),h-float(b['y1']))
    def point(p):return float(p[0]),float(p[1]) if origin=='top-left' else h-float(p[1])
    texts=d.get('texts',{})
    def lookup(v):
        if v is None:return None
        return texts.get(str(v),{}).get('content',v)
    comps=[];warnings=[]
    for cid in sorted(d.get('components',{}),key=natural):
        c=d['components'][cid]
        if c.get('state','active') not in ('active',None):continue
        typ=TYPE_MAP.get(str(c.get('category','other')).lower(),str(c.get('category','other')).lower())
        associated=list(c.get('associatedTexts',{}).values())
        values=[str(t['content']).strip() for t in associated if t.get('content') not in (None,'')]
        key=next((str(t['content']).strip() for t in associated if t.get('details')=='designator' and t.get('content')),None)
        if not key:
            key=next((m.group(1) for s in values if (m:=DESIG.search(s))),None)
        label=None
        if typ in ('vcc','gnd','vbus_in','vbus_out','vbus_bi'):
            label=next((str(t['content']).strip() for t in associated if t.get('details') in ('netlabel','name','value') and t.get('content')),None)
            label=label or (values[0] if values else ('GND' if typ=='gnd' else None))
            key=key or label
        if not key:
            key=f'UNRESOLVED_{cid}';warnings.append({'component':cid,'issue':'unresolved_designator'})
        value=next((str(t['content']) for t in associated if t.get('details')=='value' and t.get('content')),None)
        name=next((str(t['content']) for t in associated if t.get('details') in ('name','model') and t.get('content')),None)
        cp=Component(key,typ,box(c['location']),name or label,value,source_id=cid,body_bbox=box(c.get('realBBox') or c['location']),net_label=label)
        rawpins=c.get('pins',{});raw=[];used=set()
        for pid in sorted(rawpins,key=natural):
            pin=rawpins[pid]
            if not isinstance(pin.get('tip'),list) or len(pin['tip'])!=2:
                warnings.append({'pin':pid,'issue':'missing_tip'});continue
            number=pin.get('pinNumber') or lookup(pin.get('pinNumberByOCR'))
            numbers=pin_numbers(number);used.update(numbers)
            raw.append((pid,pin,numbers,number))
        fallback=1
        for pid,pin,numbers,rawnumber in raw:
            inferred=not numbers
            if inferred:
                while str(fallback) in used:fallback+=1
                numbers=[str(fallback)];used.add(str(fallback));fallback+=1
                warnings.append({'pin':pid,'issue':'fallback_pin_number','raw':rawnumber})
            for number in numbers:
                if any(p.number==number for p in cp.pins):
                    warnings.append({'pin':pid,'issue':'duplicate_pin_number','number':number})
                    while str(fallback) in used:fallback+=1
                    number=str(fallback);used.add(number);fallback+=1;inferred=True
                cp.pins.append(Pin(number,lookup(pin.get('pinNameByOCR')),point(pin['tip']),pid,point(pin['base']) if pin.get('base') else None,inferred))
        comps.append(cp)
    unique_keys(comps)
    textlist=[Text(str(t.get('content') or ''),box(t['location']),source_id=tid) for tid,t in texts.items() if isinstance(t.get('location'),dict)]
    scene=Scene(int(w),int(h),comps,textlist,diagnostics={'mode':'annotation-assisted','annotation_origin':origin,'warnings':warnings,'reference_is_official':False})
    return scene,d

def reference_nets(scene,data):
    """Derived diagnostic reference, NOT official converter output or independent ground truth."""
    pinrefs=defaultdict(list)
    for c in scene.components:
        for p in c.pins:pinrefs[p.source_id].append(f'{c.key}.{p.number}')
    edges=list(data.get('edges',{}).items());dsu=DSU(len(edges));by_label={};by_pin={}
    texts=data.get('texts',{})
    for i,(eid,e) in enumerate(edges):
        lab=e.get('netlabel')
        lab=texts.get(str(lab),{}).get('content',lab)
        if lab:
            if lab in by_label:dsu.union(i,by_label[lab])
            by_label[lab]=i
        for pid in e.get('connect',{}):
            if pid in by_pin:dsu.union(i,by_pin[pid])
            by_pin[pid]=i
    groups=defaultdict(lambda:{'pins':set(),'segments':[]})
    for i,(_,e) in enumerate(edges):
        group=groups[dsu.find(i)]
        for pid in e.get('connect',{}):group['pins'].update(pinrefs.get(pid,[]))
        for path in e.get('keyPointPaths',{}).get('wire',[]) or []:
            for a,b in zip(path,path[1:]):
                if len(a)==len(b)==2:
                    def pt(p):return (float(p[0]),float(p[1]) if scene.diagnostics.get('annotation_origin')=='top-left' else scene.height-float(p[1]))
                    group['segments'].append((pt(a),pt(b)))
    return [Net(sorted(g['pins']),g['segments']) for g in groups.values() if g['pins']]
