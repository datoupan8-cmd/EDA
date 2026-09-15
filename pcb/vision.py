"""CPU image-only front end. No Dataset300 JSON, EDS, or netlist is read here."""
import re,math,hashlib,json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize
from .schema import Component,Pin,Text,Scene
from .annotation import DESIG,unique_keys,pin_numbers
from .io import write_json

def _create_rapidocr(det_limit_side_len=None):
    """Support both RapidOCR 1.4.x (Python <=3.12) and 1.2.3 (Python 3.14)."""
    from rapidocr_onnxruntime import RapidOCR
    kwargs={'intra_op_num_threads':4,'inter_op_num_threads':2}
    if det_limit_side_len is not None:kwargs['det_limit_side_len']=det_limit_side_len
    try:return RapidOCR(**kwargs)
    except (KeyError,TypeError):return RapidOCR()

def rect_distance(p,b):
    x,y=p;a,c,d,e=b
    return math.hypot(max(a-x,0,x-d),max(c-y,0,y-e))

class OCR:
    def __init__(self,cache_dir=None):self.engine=None;self.cache=Path(cache_dir) if cache_dir else None
    def recognize(self,image):
        try:package_version=version('rapidocr-onnxruntime')
        except PackageNotFoundError:package_version='unknown'
        signature=f'rapidocr-full-v4.1-package-{package_version}-d2048-cls0'.encode()
        key=hashlib.sha256(image.tobytes()+signature).hexdigest()
        path=self.cache/(key+'.json') if self.cache else None
        if path and path.exists():raw=json.loads(path.read_text(encoding='utf8'))
        else:
            if self.engine is None:
                self.engine=_create_rapidocr(2048)
            scale=min(2.0,3000/max(image.shape[:2]))
            scaled=cv2.resize(image,None,fx=scale,fy=scale)
            result,_=self.engine(scaled,use_cls=False)
            raw=[{'text':t,'bbox':[min(p[0] for p in b)/scale,min(p[1] for p in b)/scale,max(p[0] for p in b)/scale,max(p[1] for p in b)/scale],'score':float(s)} for b,t,s in (result or [])]
            if path:write_json(path,raw)
        return [Text(t['text'],tuple(t['bbox']),t['score']) for t in raw]

    def read_pin_number(self,image,box):
        x1,y1,x2,y2=[int(round(v)) for v in box]
        patch=image[max(0,y1):min(image.shape[0],y2),max(0,x1):min(image.shape[1],x2)]
        if min(patch.shape[:2])<3:return None
        b,g,r=[v.astype(np.int16) for v in cv2.split(patch)]
        ink=(r>g+35)&(r>b+35)&(g<180)
        if ink.sum()<3:return None
        patch=np.where(ink,0,255).astype(np.uint8)
        patch=cv2.copyMakeBorder(patch,5,5,5,5,cv2.BORDER_CONSTANT,value=255)
        patch=cv2.cvtColor(cv2.resize(patch,None,fx=4,fy=4),cv2.COLOR_GRAY2BGR)
        try:package_version=version('rapidocr-onnxruntime')
        except PackageNotFoundError:package_version='unknown'
        key=hashlib.sha256(patch.tobytes()+f'pin-rec-v4.1-{package_version}'.encode()).hexdigest()
        path=self.cache/(key+'.json') if self.cache else None
        if path and path.exists():result=json.loads(path.read_text(encoding='utf8'))
        else:
            if self.engine is None:
                self.engine=_create_rapidocr()
            result,_=self.engine(patch,use_det=False,use_cls=False)
            if path:write_json(path,result or [])
        if result and float(result[0][1])>=.65 and re.fullmatch(r'\d{1,3}',result[0][0]):return result[0][0]
        return None

    def recognize_crop(self,image):
        """Recognize one local text crop and return (text, confidence) candidates."""
        if image is None or image.size==0:return []
        patch=image
        if min(patch.shape[:2])<20:patch=cv2.resize(patch,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        try:package_version=version('rapidocr-onnxruntime')
        except PackageNotFoundError:package_version='unknown'
        key=hashlib.sha256(patch.tobytes()+f'rapidocr-local-v4.1-{package_version}'.encode()).hexdigest()
        path=self.cache/(key+'.local.json') if self.cache else None
        if path and path.exists():return [tuple(x) for x in json.loads(path.read_text(encoding='utf8'))]
        if self.engine is None:
            self.engine=_create_rapidocr(2048)
        result,_=self.engine(patch,use_det=False,use_cls=False)
        out=[]
        for item in result or []:
            if isinstance(item,(list,tuple)) and len(item)>=2:out.append((str(item[0]),float(item[1])))
        if path:write_json(path,out)
        return out

def symbol_candidates(image,texts):
    b,g,r=[x.astype(np.int16) for x in cv2.split(image)]
    red=((r>g+35)&(r>b+35)&(g<180)).astype(np.uint8)*255
    colored=red.sum()/255>40
    if colored:mask=red.copy()
    else:
        _,mask=cv2.threshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        for t in texts:
            a,b,c,d=map(int,t.bbox);cv2.rectangle(mask,(a,b),(c,d),0,-1)
        # Remove long straight wire interiors for a rough monochrome shape proposal.
        long=cv2.bitwise_or(cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,35),np.uint8)),cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((35,1),np.uint8)))
        mask=cv2.subtract(mask,long)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    rects=[]
    for i in range(1,count):
        x,y,w,h,area=map(int,stats[i])
        if area<10 or max(w,h)<7 or w>image.shape[1]*.85 or h>image.shape[0]*.9:continue
        rects.append({'bbox':(x,y,x+w-1,y+h-1),'ids':[i],'area':area})
    # Capacitor plates may be two separate colored components. Pair only aligned short bars.
    consumed=set();merged=[]
    for i,a in enumerate(rects):
        if i in consumed:continue
        x1,y1,x2,y2=a['bbox'];wa,ha=x2-x1,y2-y1
        pair=None
        if min(wa,ha)<=18 and max(wa,ha)>=10 and max(wa,ha)<=30:
            for j,b in enumerate(rects):
                if j<=i or j in consumed:continue
                u1,v1,u2,v2=b['bbox'];wb,hb=u2-u1,v2-v1
                horizontal=ha<=18 and hb<=18 and wa>=ha and wb>=hb and abs(x1-u1)<=2 and abs(x2-u2)<=2 and 1<max(v1-y2,y1-v2)<=8
                vertical=wa<=18 and wb<=18 and ha>=wa and hb>=wb and abs(y1-v1)<=2 and abs(y2-v2)<=2 and 1<max(u1-x2,x1-u2)<=8
                if horizontal or vertical:pair=j;break
        if pair is not None:
            b=rects[pair];u1,v1,u2,v2=b['bbox'];consumed.add(pair)
            merged.append({'bbox':(min(x1,u1),min(y1,v1),max(x2,u2),max(y2,v2)),'ids':a['ids']+b['ids'],'area':a['area']+b['area']})
        else:merged.append(a)
    return merged,labels,mask,colored

def detect_scene(image,ocr):
    h,w=image.shape[:2];long_side=max(h,w)
    scale=1800/long_side if long_side>1800 else 1200/long_side if long_side<1200 else 1.0
    working=cv2.resize(image,None,fx=scale,fy=scale) if scale!=1 else image
    scene=detect_scene_native(working,ocr)
    sx=w/working.shape[1];sy=h/working.shape[0]
    def box(b):return b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy
    for c in scene.components:
        c.bbox=box(c.bbox)
        if c.body_bbox:c.body_bbox=box(c.body_bbox)
        for p in c.pins:
            p.tip=(p.tip[0]*sx,p.tip[1]*sy)
            if p.base:p.base=(p.base[0]*sx,p.base[1]*sy)
    for t in scene.texts:t.bbox=box(t.bbox)
    scene.width=w;scene.height=h;scene.diagnostics['frontend_working_scale']=scale
    return scene

def detect_scene_native(image,ocr):
    h,w=image.shape[:2];texts=ocr.recognize(image)
    candidates,labels,symbol_mask,colored=symbol_candidates(image,texts)
    prefixes={'U':'ic','IC':'ic','R':'resistor','C':'capacitor','L':'inductor','Q':'mosfet','D':'diode','J':'connector','USB':'connector','P':'connector','Y':'crystal','X':'crystal','SW':'switch','TP':'testpoint','F':'fuse','CN':'connector','LED':'led'}
    anchors=[]
    enclosures=[c['bbox'] for c in candidates if c['bbox'][2]-c['bbox'][0]>40 and c['bbox'][3]-c['bbox'][1]>40]
    for ti,t in enumerate(texts):
        inside=any(a<t.center[0]<c and b<t.center[1]<d for a,b,c,d in enclosures)
        if inside and not re.fullmatch(r'(?:U|IC|J|USB)(?:\d+|\?)',t.text.strip(),re.I):continue
        # Whole component designators or a designator followed by a value; not pin strings.
        m=DESIG.match(t.text.strip())
        if m:
            key=m.group(1);prefix=re.match('[A-Za-z]+',key).group().upper()
            key=prefix+key[len(prefix):]
            anchors.append((ti,key,prefixes.get(prefix,'other'),None))
        elif re.fullmatch(r'(?:GND|AGND|DGND|VCC|VDD|[+\-]?\d+(?:\.\d+)?V)',t.text.strip(),re.I):
            key=t.text.strip();anchors.append((ti,key,'gnd' if 'GND' in key.upper() else 'vcc',key))
    choices=[]
    for ai,(ti,key,typ,label) in enumerate(anchors):
        t=texts[ti];th=max(5,t.bbox[3]-t.bbox[1])
        for ci,cand in enumerate(candidates):
            b=cand['bbox'];bw,bh=b[2]-b[0],b[3]-b[1]
            if min(bw,bh)<1:continue
            if typ=='ic' and (bw<25 or bh<25):continue
            if typ in ('resistor','capacitor','gnd','vcc') and max(bw,bh)>max(90,th*8):continue
            if typ in ('gnd','vcc') and (max(bw,bh)>30 or len(cand['ids'])>1):continue
            dist=rect_distance(t.center,b)
            if dist==0 and typ not in ('ic','connector'):continue
            if dist>max(28,th*5):continue
            # Text usually lies close to, but outside, the symbol.
            center=((b[0]+b[2])/2,(b[1]+b[3])/2)
            score=dist+.025*math.dist(t.center,center)
            if typ=='ic':score-=100  # Allocate enclosure candidates before short pin-like labels.
            elif typ in ('gnd','vcc'):score+=100
            choices.append((score,ai,ci))
    used_a=set();used_c=set();comps=[];warnings=[]
    bg,gg,rr=[x.astype(np.int16) for x in cv2.split(image)]
    green=((gg>rr+25)&(gg>bg+20)&(gg<235)).astype(np.uint8)*255
    near_wire=cv2.dilate(green,np.ones((5,5),np.uint8))
    for score,ai,ci in sorted(choices):
        if ai in used_a or ci in used_c:continue
        used_a.add(ai);used_c.add(ci)
        ti,key,typ,label=anchors[ai];t=texts[ti];cand=candidates[ci]
        x1,y1,x2,y2=cand['bbox']
        cp=Component(key,typ,(float(x1),float(y1),float(x2+1),float(y2+1)),source_id=f'visual_{ci}',net_label=label,name=label,confidence=min(1.0,max(0.1,t.score-max(score,0)/100)))
        local=(np.isin(labels[y1:y2+1,x1:x2+1],cand['ids']).astype(np.uint8)*255)
        # The filled interior contour approximates an IC body; pin stubs stay outside.
        contours,_=cv2.findContours(local,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        bodies=[]
        for contour in contours:
            a,b,c,d=cv2.boundingRect(contour)
            if c>20 and d>20 and cv2.contourArea(contour)>c*d*.65:bodies.append((c*d,(x1+a,y1+b,x1+a+c-1,y1+b+d-1)))
        cp.body_bbox=max(bodies)[1] if bodies else cp.bbox
        if typ in ('ic','connector'):
            vertical=cv2.morphologyEx(local,cv2.MORPH_OPEN,np.ones((max(20,int((y2-y1)*.5)),1),np.uint8))
            horizontal=cv2.morphologyEx(local,cv2.MORPH_OPEN,np.ones((1,max(20,int((x2-x1)*.5))),np.uint8))
            columns=np.where(vertical.max(axis=0)>0)[0]
            rows=np.where(horizontal.max(axis=1)>0)[0]
            if len(columns)>=2 and len(rows)>=2:
                cp.body_bbox=(x1+int(columns.min()),y1+int(rows.min()),x1+int(columns.max()),y1+int(rows.max()))
        points=[]
        contact=local & near_wire[y1:y2+1,x1:x2+1]
        # Ignore contact inside a component body (e.g. filled polygons).
        if typ in ('ic','connector'):
            bx1,by1,bx2,by2=map(int,cp.body_bbox)
            cv2.rectangle(contact,(bx1-x1+2,by1-y1+2),(bx2-x1-2,by2-y1-2),0,-1)
        n,lab,stats,centers=cv2.connectedComponentsWithStats(contact)
        for k in range(1,n):
            ys,xs=np.where(lab==k)
            pts=np.stack([xs+x1,ys+y1],axis=1)
            # Choose the extreme symbol pixel pointing toward the nearest wire.
            center=pts.mean(axis=0);cx,cy=(x1+x2)/2,(y1+y2)/2
            axis=0 if abs(center[0]-cx)/max(1,x2-x1)>abs(center[1]-cy)/max(1,y2-y1) else 1
            extreme=pts[:,axis].max() if center[axis]>(cx,cy)[axis] else pts[:,axis].min()
            tip=pts[pts[:,axis]==extreme].mean(axis=0)
            points.append(tuple(map(float,tip)))
        if len(points)<2 and typ not in ('gnd','vcc'):
            sk=skeletonize(local>0).astype(np.uint8)
            degrees=cv2.filter2D(sk,cv2.CV_16S,np.ones((3,3),np.int16))-sk
            ys,xs=np.where((sk>0)&(degrees==1))
            points.extend((float(x+x1),float(y+y1)) for x,y in zip(xs,ys))
        if not points:
            points=[((x1+x2)/2,float(y1)),((x1+x2)/2,float(y2))] if y2-y1>x2-x1 else [(float(x1),(y1+y2)/2),(float(x2),(y1+y2)/2)]
            warnings.append({'component':key,'issue':'geometric_pin_fallback'})
        unique=[]
        for p in points:
            if not any(math.dist(p,q)<3 for q in unique):unique.append(p)
        if typ in ('resistor','capacitor','inductor') and len(unique)<2:
            # Capacitor plates include long horizontal bars but pins leave vertically.
            vertical=(y2-y1>x2-x1) if typ!='capacitor' else (x2-x1>=y2-y1)
            unique=[((x1+x2)/2,float(y1)),((x1+x2)/2,float(y2))] if vertical else [(float(x1),(y1+y2)/2),(float(x2),(y1+y2)/2)]
            warnings.append({'component':key,'issue':'two_terminal_geometry'})
        if typ in ('gnd','vcc'):unique=sorted(unique,key=lambda p:min(abs(p[0]-x1),abs(p[0]-x2),abs(p[1]-y1),abs(p[1]-y2)))[:1]
        elif typ in ('resistor','capacitor','inductor','diode','led','crystal') and len(unique)>2:
            unique=list(max(((a,b) for i,a in enumerate(unique) for b in unique[i+1:]),key=lambda ab:math.dist(*ab)))
        staged=[]
        for i,p in enumerate(sorted(unique,key=lambda p:(round(p[0],1),p[1])),1):
            nearby=[(math.dist(p,tt.center),tt) for tt in texts if tt is not t and math.dist(p,tt.center)<24]
            nearby.sort(key=lambda v:v[0]);number=None;pinname=None
            if typ in ('ic','connector','mosfet','bjt'):
                bx1,by1,bx2,by2=cp.body_bbox
                if typ in ('ic','connector') and colored and (p[0]<bx1 or p[0]>bx2):
                    box=(p[0],p[1]-12,bx1,p[1]-1) if p[0]<bx1 else (bx2,p[1]-12,p[0]+1,p[1]-1)
                    number=ocr.read_pin_number(image,box)
                if number is None:
                    number=next((ns[0] for _,tt in nearby if (ns:=pin_numbers(tt.text)) and len(ns)==1 and len(ns[0])<=3 and not (bx1<tt.center[0]<bx2 and by1<tt.center[1]<by2)),None)
                # Pin names live inside the body, aligned with the pin row/column.
                horizontal=p[0]<=bx1 or p[0]>=bx2
                aligned=[tt for tt in texts if bx1<=tt.center[0]<=bx2 and by1<=tt.center[1]<=by2 and re.search('[A-Za-z]',tt.text)
                         and (abs(tt.center[1]-p[1])<=5 if horizontal else abs(tt.center[0]-p[0])<=5)]
                if aligned:pinname=min(aligned,key=lambda tt:rect_distance(p,tt.bbox)).text
            staged.append((p,number,pinname))
        used_numbers={number for _,number,_ in staged if number};emitted=set()
        for i,(p,number,pinname) in enumerate(staged,1):
            fallback=number is None or number in emitted
            if fallback:
                number=str(i)
                while number in used_numbers:number=str(int(number)+1)
            used_numbers.add(number);emitted.add(number)
            cp.pins.append(Pin(number,pinname,p,inferred_number=fallback))
        # Read value/model from text near the designator; preserve spelling/case.
        others=sorted((tt for tt in texts if tt is not t and not DESIG.fullmatch(tt.text.strip())),key=lambda tt:math.dist(t.center,tt.center))
        if typ in ('resistor','capacitor','inductor'):
            tail=t.text[len(key):].strip()
            def is_value(s):return bool(re.fullmatch(r'\d+(?:\.\d+)?\s*(?:[pnumkKM]?(?:F|H|[oO]hm|Ω)?|[Rr])',s))
            cp.value=tail if is_value(tail) else next((tt.text for tt in others if is_value(tt.text) and math.dist(t.center,tt.center)<45),None)
        elif typ=='ic':
            cp.name=next((tt.text for tt in others if re.search('[A-Za-z]',tt.text) and re.search(r'\d',tt.text) and math.dist(t.center,tt.center)<50),None)
        comps.append(cp)
    # Stable image order only; official annotation-ID order is unavailable in PNG-only mode.
    comps.sort(key=lambda c:(c.bbox[1],c.bbox[0]));unique_keys(comps)
    scene=Scene(w,h,comps,texts,diagnostics={'mode':'image-only','frontend':'opencv_symbols+rapidocr_cpu','token_count':0,
        'ocr_regions':len(texts),'symbol_candidates':len(candidates),'colored_symbols':colored,'warnings':warnings,
        'unmatched_designators':[anchors[i][1] for i in range(len(anchors)) if i not in used_a],
        'limitations':['Heuristic symbol/pin detection; not a trained PCB detector','Unknown pin numbers use deterministic placeholders, not recovered physical numbers','No official score claimed']})
    return scene
