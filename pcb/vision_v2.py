"""Accuracy-oriented image frontend: global text matching and explicit pin stages."""
from __future__ import annotations
import math,re
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .schema import Component,Pin,Scene,Text
from .annotation import DESIG,unique_keys,pin_numbers
from .vision import symbol_candidates

PREFIX_TYPE={'U':'ic','IC':'ic','R':'resistor','C':'capacitor','L':'inductor','Q':'mosfet','D':'diode',
             'J':'connector','P':'connector','USB':'connector','Y':'crystal','X':'crystal','SW':'switch',
             'TP':'testpoint','F':'fuse','CN':'connector','LED':'led'}
VALUE_RE=re.compile(r'^(?:\d+(?:\.\d+)?|\.\d+)\s*(?:[pnumkKM]?(?:F|H|Ω|ohm)?|R)$',re.I)
SIGNAL_RE=re.compile(r'^[A-Za-z][A-Za-z0-9_./+\-]{1,31}$')

def _value_compatible(typ,s):
    s=re.sub(r'\s+','',str(s));up=s.upper()
    if not VALUE_RE.fullmatch(s):return False
    if typ=='resistor':return not bool(re.search(r'(?:F|H|V)$',up))
    if typ=='capacitor':return bool(re.search(r'(?:F|^[0-9.]+[PNUΜU]$)',up)) or bool(re.fullmatch(r'\d{2,4}',up))
    if typ=='inductor':return bool(re.search(r'H$',up))
    return True

def rect_distance(p,b):
    x,y=p;x1,y1,x2,y2=b
    return math.hypot(max(x1-x,0,x-x2),max(y1-y,0,y-y2))

def bbox_gap(a,b):
    return math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0))

def prefix_type(key):
    p=re.match('[A-Za-z]+',key).group().upper();return PREFIX_TYPE.get(p,'other')

def _candidate_body(c):
    x1,y1,x2,y2=c['bbox'];return (float(x1),float(y1),float(x2+1),float(y2+1))

def _fused_visual_split(ocr,image,t):
    s=t.text.strip();m=re.fullmatch(r'((?:R|C|L)\d+)([0-9].*)',s,re.I)
    options=[]
    # Enumerate numeric boundary choices; visual OCR and a projection valley decide.
    p=re.match(r'[A-Za-z]+',s)
    if not p:return None
    start=p.end()
    for cut in range(start+1,len(s)):
        left,right=s[:cut],s[cut:]
        if not DESIG.fullmatch(left) or not _value_compatible(prefix_type(left),right):continue
        # A bare trailing digit is more likely part of C10/R15 than a value.
        # Unit-bearing values (10k, 15pF) are unambiguous; bare numeric values
        # need at least two digits and may not start with zero.
        unit_bearing=bool(re.search(r'[A-Za-zΩ]',right))
        if not unit_bearing and (len(right)<2 or right.startswith('0')):continue
        if unit_bearing and re.match(r'^0\d',right):continue
        x1,y1,x2,y2=map(int,t.bbox);crop=image[max(0,y1-2):min(image.shape[0],y2+2),max(0,x1-2):min(image.shape[1],x2+2)]
        if crop.size==0:continue
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);ink=(gray<210).astype(np.uint8);proj=ink.sum(0)
        expected=int(round(cut/len(s)*crop.shape[1]));lo=max(2,expected-10);hi=min(crop.shape[1]-2,expected+11)
        if lo>=hi:continue
        peak=max(1,int(np.percentile(proj,75)))
        for px in sorted(range(lo,hi),key=lambda x:int(proj[x]))[:3]:
            valley=int(proj[px])
            if valley>max(1,.55*peak):continue
            l=max(ocr.recognize_crop(crop[:,:px+1]),key=lambda x:x[1],default=('',0));r=max(ocr.recognize_crop(crop[:,px+1:]),key=lambda x:x[1],default=('',0))
            ln=re.sub(r'\s+','',l[0]).upper();rn=re.sub(r'\s+','',r[0]).upper()
            evidence=ln==left.upper() and rn==re.sub(r'\s+','',right).upper()
            if evidence and min(l[1],r[1])>=.35:options.append((len(left)*2+l[1]+r[1]-valley/peak,left.upper(),r[0].strip(),px,valley/peak,l[1],r[1]))
    if not options:return None
    _,left,right,px,ratio,lc,rc=max(options)
    return {'designator':left,'value':right,'split_x':int(t.bbox[0]-2+px),'valley_ratio':ratio,'left_conf':lc,'right_conf':rc}

def _designator_tokens(texts,image,ocr):
    out=[]
    for i,t in enumerate(texts):
        s=t.text.strip();visual=_fused_visual_split(ocr,image,t)
        if visual:out.append((i,visual['designator'],prefix_type(visual['designator']),len(visual['designator']),visual['value'],visual));continue
        m=DESIG.match(s)
        if m:
            key=m.group(1).upper();out.append((i,key,prefix_type(key),m.end(),s[m.end():].strip(),None));continue
        up=s.upper()
        supply=re.fullmatch(r'(?:GND|AGND|DGND|PGND|VCC|VDD|VSS|VBUS|\+\d+(?:\.\d+)?V|[0-5](?:\.\d+)?V)',up)
        if supply:out.append((i,up,'gnd' if 'GND' in up or up=='VSS' else 'vcc',len(s),'',None))
    return out

def _match_designators(candidates,texts,image,ocr):
    anchors=_designator_tokens(texts,image,ocr)
    if not anchors or not candidates:return [],anchors
    cost=np.full((len(anchors),len(candidates)),1e6,float);features={}
    for ai,(ti,key,typ,_,_,_) in enumerate(anchors):
        t=texts[ti];th=max(4.,t.bbox[3]-t.bbox[1])
        for ci,c in enumerate(candidates):
            b=_candidate_body(c);bw=b[2]-b[0];bh=b[3]-b[1];dist=bbox_gap(t.bbox,b)
            if dist>max(40.,7*th):continue
            large=min(bw,bh)>=20
            if typ=='ic' and not large:continue
            if typ in ('resistor','capacitor','inductor','diode') and max(bw,bh)>max(100,10*th):continue
            inside=b[0]<t.center[0]<b[2] and b[1]<t.center[1]<b[3]
            # Designator normally sits above/below a symbol; IC labels can sit inside.
            direction_penalty=0 if (t.center[1]<=b[1]+th or t.center[1]>=b[3]-th or typ in ('ic','connector')) else 2.5
            type_penalty=0 if ((typ=='ic')==large) else 2.0
            conf_penalty=(1-max(0,min(1,t.score)))*2
            score=dist/th+direction_penalty+type_penalty+conf_penalty+0.02*math.dist(t.center,((b[0]+b[2])/2,(b[1]+b[3])/2))/th
            cost[ai,ci]=score;features[(ai,ci)]={'distance':dist,'height_norm':dist/th,'direction_penalty':direction_penalty,'ocr_confidence':t.score,'score':score}
    rr,cc=linear_sum_assignment(cost);matches=[]
    for ai,ci in zip(rr,cc):
        if cost[ai,ci]>=12:continue
        matches.append((ai,ci,features[(ai,ci)]))
    return matches,anchors

def _read_rotations(ocr,image,box,kind):
    x1,y1,x2,y2=map(lambda v:int(round(v)),box);pad=2
    crop=image[max(0,y1-pad):min(image.shape[0],y2+pad),max(0,x1-pad):min(image.shape[1],x2+pad)]
    if crop.size==0:return None,0.,0
    best=(None,0.,0)
    for turns in range(4):
        patch=np.rot90(crop,turns).copy()
        raw=ocr.recognize_crop(patch)
        for s,conf in raw:
            s=s.strip()
            valid=bool(pin_numbers(s)) if kind=='number' else bool(SIGNAL_RE.fullmatch(s)) and not pin_numbers(s)
            if valid and conf>best[1]:best=(s,conf,turns*90)
    return best

def _projection_split(ocr,image,t,key,tail):
    """Accept fused designator/value only with an image valley and two crop OCR readings."""
    if not tail or not VALUE_RE.fullmatch(tail):return None
    x1,y1,x2,y2=map(int,t.bbox);crop=image[max(0,y1-2):min(image.shape[0],y2+2),max(0,x1-2):min(image.shape[1],x2+2)]
    if crop.size==0:return None
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);ink=(gray<210).astype(np.uint8);proj=ink.sum(axis=0)
    expected=int(round(len(key)/max(1,len(key)+len(tail))*crop.shape[1]));lo=max(2,expected-5);hi=min(crop.shape[1]-2,expected+6)
    if lo>=hi:return None
    split=lo+int(np.argmin(proj[lo:hi]));valley=int(proj[split]);peak=max(1,int(np.percentile(proj,75)))
    if valley>max(1,.45*peak):return None
    left=ocr.recognize_crop(crop[:,:split+1]);right=ocr.recognize_crop(crop[:,split+1:])
    l=max(left,key=lambda x:x[1],default=('',0));r=max(right,key=lambda x:x[1],default=('',0))
    if DESIG.fullmatch(l[0].strip()) and VALUE_RE.fullmatch(r[0].strip()) and min(l[1],r[1])>=.45:
        return {'designator':l[0].strip().upper(),'value':r[0].strip(),'split_x':x1-2+split,'valley_ratio':valley/peak,'left_conf':l[1],'right_conf':r[1]}
    return None

def _body_bbox(local,origin,typ,outer):
    if typ not in ('ic','connector'):return outer
    x1,y1=origin;hh,ww=local.shape
    vert=cv2.morphologyEx(local,cv2.MORPH_OPEN,np.ones((max(12,int(hh*.45)),1),np.uint8))
    horiz=cv2.morphologyEx(local,cv2.MORPH_OPEN,np.ones((1,max(12,int(ww*.45))),np.uint8))
    cols=np.where(vert.max(0)>0)[0];rows=np.where(horiz.max(1)>0)[0]
    return (x1+int(cols.min()),y1+int(rows.min()),x1+int(cols.max()),y1+int(rows.max())) if len(cols)>1 and len(rows)>1 else outer

def terminal_candidates(image,component,labels,ids,wire_hint):
    x1,y1,x2,y2=map(int,component.bbox);local=np.isin(labels[y1:y2,x1:x2],ids).astype(np.uint8)*255
    bx1,by1,bx2,by2=component.body_bbox or component.bbox
    contact=local & cv2.dilate(wire_hint[y1:y2,x1:x2],np.ones((5,5),np.uint8))
    if component.type in ('ic','connector'):
        cv2.rectangle(contact,(max(0,int(bx1-x1+2)),max(0,int(by1-y1+2))),(min(local.shape[1]-1,int(bx2-x1-2)),min(local.shape[0]-1,int(by2-y1-2))),0,-1)
    n,lab,_,_=cv2.connectedComponentsWithStats(contact);points=[]
    for k in range(1,n):
        yy,xx=np.where(lab==k)
        if not len(xx):continue
        pts=np.c_[xx+x1,yy+y1];m=pts.mean(0)
        distances={'left':abs(m[0]-bx1),'right':abs(m[0]-bx2),'top':abs(m[1]-by1),'bottom':abs(m[1]-by2)};side=min(distances,key=distances.get)
        if side=='left':tip=pts[pts[:,0].argmin()]
        elif side=='right':tip=pts[pts[:,0].argmax()]
        elif side=='top':tip=pts[pts[:,1].argmin()]
        else:tip=pts[pts[:,1].argmax()]
        points.append((tuple(map(float,tip)),side))
    if component.type in ('resistor','capacitor','inductor','diode','led','crystal') and len(points)!=2:
        vertical=(y2-y1)>(x2-x1);points=[(((x1+x2)/2.,float(y1)),'top'),(((x1+x2)/2.,float(y2)),'bottom')] if vertical else [((float(x1),(y1+y2)/2.),'left'),((float(x2),(y1+y2)/2.),'right')]
    if component.type in ('gnd','vcc') and not points:
        points=[(((x1+x2)/2.,float(y1)),'top')] if component.type=='gnd' else [(((x1+x2)/2.,float(y2)),'bottom')]
    unique=[]
    for p,s in points:
        if not any(math.dist(p,q[0])<2.5 for q in unique):unique.append((p,s))
    return unique

def _base_for(p,side,b):
    x,y=p;x1,y1,x2,y2=b
    return {'left':(x1,y),'right':(x2,y),'top':(x,y1),'bottom':(x,y2)}[side]

def recognize_pin_semantics(ocr,image,component,terminals,texts):
    bx1,by1,bx2,by2=component.body_bbox or component.bbox;staged=[]
    simple=component.type in ('resistor','capacitor','inductor','diode','led','crystal','gnd','vcc','testpoint')
    if simple:
        ordered=sorted(terminals,key=lambda x:(x[0][0],x[0][1]))
        pins=[];low=[]
        for i,(p,side) in enumerate(ordered,1):
            pins.append(Pin(str(i),'',p,base=_base_for(p,side,(bx1,by1,bx2,by2)),inferred_number=True,number_confidence=.35,name_confidence=0,side=side))
            low.append({'tip':p,'reason':'geometry_assigned_two_terminal_number','assigned':str(i),'ocr_confidence':0.0})
        return pins,low,[{'tip':p,'side':s,'number':str(i+1),'number_conf':.35,'name':'','name_conf':0,'number_rotation':0,'name_rotation':0} for i,(p,s) in enumerate(ordered)]
    for p,side in terminals:
        if side=='left':nbox=(p[0],p[1]-13,bx1+1,p[1]+3);namebox=(bx1,p[1]-10,bx2,p[1]+10)
        elif side=='right':nbox=(bx2-1,p[1]-13,p[0]+1,p[1]+3);namebox=(bx1,p[1]-10,bx2,p[1]+10)
        elif side=='top':nbox=(p[0]-10,p[1],p[0]+10,by1+1);namebox=(p[0]-12,by1,p[0]+12,by2)
        else:nbox=(p[0]-10,by2-1,p[0]+10,p[1]+1);namebox=(p[0]-12,by1,p[0]+12,by2)
        number,nconf,nrot=_read_rotations(ocr,image,nbox,'number')
        name,nmconf,nmrot=_read_rotations(ocr,image,namebox,'name')
        staged.append({'tip':p,'side':side,'base':_base_for(p,side,(bx1,by1,bx2,by2)),'number':number,'number_conf':nconf,'number_rotation':nrot,'name':name,'name_conf':nmconf,'name_rotation':nmrot})
    # Confident physical numbers are reserved before assigning explicit low-confidence IDs.
    used=set();next_unknown=1;pins=[];low=[]
    for item in staged:
        n=item['number']
        if not n or n in used:
            while f'UNK{next_unknown}' in used:next_unknown+=1
            n=f'UNK{next_unknown}';next_unknown+=1;low.append({'tip':item['tip'],'reason':'unrecognized_or_duplicate_number','assigned':n,'ocr_confidence':item['number_conf']})
        used.add(n);pins.append(Pin(n,item['name'],item['tip'],base=item['base'],inferred_number=n.startswith('UNK'),number_confidence=item['number_conf'],name_confidence=item['name_conf'],side=item['side']))
    return pins,low,staged

def _assign_values_models(image,ocr,components,texts,anchor_info):
    used=set();events=[];eligible=[]
    for ci,c in enumerate(components):
        ai=anchor_info[ci];t=texts[ai['text_index']];tail=ai['tail'];split=ai.get('visual_split') or _projection_split(ocr,image,t,c.key,tail)
        if split:
            c.key=split['designator'];c.value=split['value'];events.append({'component':c.key,'kind':'visual_fused_split',**split});continue
        candidates=[]
        for ti,tt in enumerate(texts):
            if ti==ai['text_index'] or ti in used:continue
            dist=bbox_gap(c.bbox,tt.bbox);h=max(4,tt.bbox[3]-tt.bbox[1])
            if dist>max(45,6*h):continue
            if c.type in ('resistor','capacitor','inductor'):
                role=_value_compatible(c.type,tt.text.strip())
            else:role=bool(re.search('[A-Za-z]',tt.text) and re.search(r'\d',tt.text) and not DESIG.fullmatch(tt.text.strip()))
            if not role:continue
            direction=0 if tt.center[1]>=c.bbox[1] else .5
            candidates.append((dist/h+direction+(1-tt.score),ti,tt))
        eligible.append((ci,candidates))
    # Global one-to-one role assignment prevents one nearby value/model serving two parts.
    comps=[x for x in eligible if x[1]];text_ids=sorted({ti for _,cand in comps for _,ti,_ in cand})
    if comps and text_ids:
        ti_col={t:i for i,t in enumerate(text_ids)};cost=np.full((len(comps),len(text_ids)),1e5)
        for r,(_,cand) in enumerate(comps):
            for score,ti,_ in cand:cost[r,ti_col[ti]]=score
        rr,cc=linear_sum_assignment(cost)
        for r,col in zip(rr,cc):
            if cost[r,col]>8:continue
            ci=comps[r][0];ti=text_ids[col];tt=texts[ti];c=components[ci]
            if c.type in ('resistor','capacitor','inductor'):c.value=tt.text.strip()
            elif c.type=='ic':c.name=tt.text.strip()
            events.append({'component':c.key,'kind':'global_role_assignment','text':tt.text,'score':float(cost[r,col]),'ocr_confidence':tt.score})
    return events

def detect_scene_v2(image,ocr):
    h,w=image.shape[:2];long=max(h,w);scale=min(1.5,1800/long) if long else 1
    work=cv2.resize(image,None,fx=scale,fy=scale) if abs(scale-1)>1e-3 else image
    texts=ocr.recognize(work);cands,labels,symbol_mask,colored=symbol_candidates(work,texts)
    matches,anchors=_match_designators(cands,texts,work,ocr);components=[];anchor_info={};bg,gg,rr=[x.astype(np.int16) for x in cv2.split(work)]
    # Hint only; final wire extractor is independent and adaptive.
    dark=(cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)<215).astype(np.uint8)*255
    for ai,ci,feat in matches:
        ti,key,typ,end,tail,visual_split=anchors[ai];b=_candidate_body(cands[ci]);x1,y1,x2,y2=map(int,b)
        local=np.isin(labels[y1:y2,x1:x2],cands[ci]['ids']).astype(np.uint8)*255
        c=Component(key,typ,b,source_id=f'visual_{ci}',confidence=max(.01,min(1,texts[ti].score/(1+feat['score']/8))))
        if typ in ('gnd','vcc'):c.net_label=key;c.name=key
        c.body_bbox=_body_bbox(local,(x1,y1),typ,b);components.append(c);anchor_info[len(components)-1]={'text_index':ti,'tail':tail,'visual_split':visual_split,'match_features':feat,'candidate':ci}
    split_events=_assign_values_models(work,ocr,components,texts,anchor_info)
    low=[];pin_events=[]
    for ci,c in enumerate(components):
        cand=cands[anchor_info[ci]['candidate']];terms=terminal_candidates(work,c,labels,cand['ids'],dark)
        pins,lows,events=recognize_pin_semantics(ocr,work,c,terms,texts);c.pins=pins;low.extend({'component':c.key,**x} for x in lows);pin_events.extend({'component':c.key,**x} for x in events)
    # Scale every output back to original coordinates.
    sx=w/work.shape[1];sy=h/work.shape[0]
    def sb(b):return (b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy)
    for c in components:
        c.bbox=sb(c.bbox);c.body_bbox=sb(c.body_bbox)
        for p in c.pins:p.tip=(p.tip[0]*sx,p.tip[1]*sy);p.base=(p.base[0]*sx,p.base[1]*sy) if p.base else None
    for t in texts:t.bbox=sb(t.bbox)
    unique_keys(components)
    matched_ai={x[0] for x in matches}
    scene=Scene(w,h,components,texts,diagnostics={'mode':'image-only','pipeline':'v2','frontend_scale':scale,'token_count':0,
        'ocr_regions':len(texts),'symbol_candidates':len(cands),'component_matches':len(matches),'component_match_features':[{'key':anchors[ai][1],**feat} for ai,_,feat in matches],
        'unmatched_designators':[anchors[i][1] for i in range(len(anchors)) if i not in matched_ai],
        'visual_fused_splits':split_events,'pin_recognition':pin_events,'low_confidence_pins':low,'colored_symbols':colored})
    return scene

def sanitize_low_confidence_pins(scene):
    events=[]
    for c in scene.components:
        complex_part=c.type in ('ic','connector','mosfet','bjt');used=set()
        for idx,p in enumerate(c.pins,1):
            numeric_outlier=complex_part and p.number.isdigit() and int(p.number)>max(128,2*len(c.pins))
            if (p.inferred_number and complex_part) or numeric_outlier:
                old=p.number;p.number=f'UNK{idx}';p.exportable=False;p.number_confidence=0.0
                reason='numeric_outlier_for_component_pin_count' if numeric_outlier else 'no_physical_pin_number_evidence'
                events.append({'component':c.key,'old_placeholder':old,'terminal_id':p.number,'reason':reason})
            else:
                p.number_confidence=.35 if p.inferred_number else max(.7,p.number_confidence)
                if p.number in used:p.exportable=False;events.append({'component':c.key,'terminal_id':p.number,'reason':'duplicate_number'})
            used.add(p.number)
    scene.diagnostics.update({'pin_sanitation_enabled':True,'nonexported_low_confidence_terminals':events,
        'exportable_pins':sum(p.exportable for c in scene.components for p in c.pins),'terminal_candidates':sum(len(c.pins) for c in scene.components)})
    return scene

def refine_v1_scene(image,ocr,scene,safe_fused_only=False,sanitize_pins=True,global_roles=True):
    """Evidence-gated V2 text/pin cleanup while preserving V1 geometry."""
    anchor_info={};splits=[];sanitized=[]
    for ci,c in enumerate(scene.components):
        candidates=[]
        for ti,t in enumerate(scene.texts):
            s=t.text.strip()
            if s.upper()==c.key.upper() or s.upper().startswith(c.key.upper()):
                candidates.append((bbox_gap(c.bbox,t.bbox)/max(4,t.bbox[3]-t.bbox[1]),ti,t))
        if candidates:
            _,ti,t=min(candidates);visual=_fused_visual_split(ocr,image,t);tail=t.text.strip()[len(c.key):].strip()
            if safe_fused_only and visual and not re.search(r'[A-Za-zΩ]',visual['value']):visual=None
            if visual and (c.key.upper()==t.text.strip().upper() or c.key.upper().startswith(visual['designator'])):
                old=c.key;c.key=visual['designator'];c.value=visual['value'];splits.append({'old_key':old,'new_key':c.key,**visual})
            anchor_info[ci]={'text_index':ti,'tail':tail,'visual_split':visual}
        for p in c.pins:p.number_confidence=.35 if p.inferred_number else max(.7,p.number_confidence)
    mapped=sorted(anchor_info)
    if mapped and global_roles:_assign_values_models(image,ocr,[scene.components[i] for i in mapped],scene.texts,{j:anchor_info[i] for j,i in enumerate(mapped)})
    # Value assignment can perform a second evidence-gated fused split, so key
    # uniqueness must be enforced after that operation.
    unique_keys(scene.components)
    if sanitize_pins:
        sanitize_low_confidence_pins(scene);sanitized=scene.diagnostics['nonexported_low_confidence_terminals']
    scene.diagnostics.update({'pipeline':'v2_refined_v1_geometry','safe_fused_only':safe_fused_only,'pin_sanitation_enabled':sanitize_pins,'global_text_roles_enabled':global_roles,'visual_fused_splits':splits,'nonexported_low_confidence_terminals':sanitized,
        'exportable_pins':sum(p.exportable for c in scene.components for p in c.pins),'terminal_candidates':sum(len(c.pins) for c in scene.components)})
    return scene

def parse_pins_for_components(image,ocr,components,texts=None,known_tips=False):
    """Oracle experiment helper; caller owns the component/tip source provenance."""
    texts=texts or ocr.recognize(image);events=[];low=[]
    for c in components:
        if known_tips:terms=[(p.tip,p.side or min({'left':abs(p.tip[0]-c.bbox[0]),'right':abs(p.tip[0]-c.bbox[2]),'top':abs(p.tip[1]-c.bbox[1]),'bottom':abs(p.tip[1]-c.bbox[3])},key=lambda x:{'left':abs(p.tip[0]-c.bbox[0]),'right':abs(p.tip[0]-c.bbox[2]),'top':abs(p.tip[1]-c.bbox[1]),'bottom':abs(p.tip[1]-c.bbox[3])}[x])) for p in c.pins]
        else:
            # Geometry-only fallback around known bbox; deliberately does not read annotation pin tips.
            x1,y1,x2,y2=c.bbox;vertical=y2-y1>x2-x1
            terms=[(((x1+x2)/2,y1),'top'),(((x1+x2)/2,y2),'bottom')] if c.type in ('resistor','capacitor','inductor','diode') and vertical else [((x1,(y1+y2)/2),'left'),((x2,(y1+y2)/2),'right')]
        pins,lows,ev=recognize_pin_semantics(ocr,image,c,terms,texts);c.pins=pins;events.extend({'component':c.key,**x} for x in ev);low.extend({'component':c.key,**x} for x in lows)
    return {'pin_recognition':events,'low_confidence_pins':low}

def associate_text_for_known_components(image,ocr,components,texts=None):
    """Predict Name/value for oracle-bbox experiments; component keys remain oracle inputs."""
    texts=texts or ocr.recognize(image);anchor_info={};events=[]
    for ci,c in enumerate(components):
        candidates=[]
        for ti,t in enumerate(texts):
            s=t.text.strip().upper()
            if s==c.key.upper() or s.startswith(c.key.upper()):
                candidates.append((bbox_gap(c.bbox,t.bbox)/max(4,t.bbox[3]-t.bbox[1]),ti,t))
        if candidates:
            _,ti,t=min(candidates);tail=t.text.strip()[len(c.key):].strip();visual=_fused_visual_split(ocr,image,t)
        else:ti=0 if texts else -1;tail='';visual=None
        if ti>=0:anchor_info[ci]={'text_index':ti,'tail':tail,'visual_split':visual}
    eligible_components=[components[i] for i in sorted(anchor_info)]
    remap={j:anchor_info[i] for j,i in enumerate(sorted(anchor_info))}
    events=_assign_values_models(image,ocr,eligible_components,texts,remap) if eligible_components else []
    return {'text_role_events':events,'matched_designators':len(anchor_info)}
