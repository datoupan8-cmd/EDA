"""PCB-adapted CC topology with directional pin snapping and logical labels."""
from collections import defaultdict
import math,re
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .schema import DSU,Net
from .wire import trace_paths,simplify,projection
from .annotation import DESIG

SIGNAL=re.compile(r'^[A-Za-z][A-Za-z0-9_./+\-]{1,31}$')
VALUE=re.compile(r'^\d+(?:\.\d+)?(?:[pnumkKM]?(?:F|H|V|Ω|ohm)?|R)$',re.I)
def norm_label(s):return re.sub(r'\s+','',s).upper().replace('−','-')

def _direction(pin,component):
    if pin.base and math.dist(pin.tip,pin.base)>.5:v=np.array(pin.tip)-np.array(pin.base)
    else:
        cx=(component.bbox[0]+component.bbox[2])/2;cy=(component.bbox[1]+component.bbox[3])/2;v=np.array(pin.tip)-np.array((cx,cy))
    n=np.linalg.norm(v);return v/max(n,1e-6)

def _is_label(t,scene):
    s=t.text.strip()
    if not SIGNAL.fullmatch(s) or DESIG.fullmatch(s) or VALUE.fullmatch(s):return False
    if len(s)==1:return False
    for c in scene.components:
        b=c.body_bbox or c.bbox
        if b[0]-1<=t.center[0]<=b[2]+1 and b[1]-1<=t.center[1]<=b[3]+1:return False
    return True

def _bbox_point_dist(p,b):
    x,y=p;return math.hypot(max(b[0]-x,0,x-b[2]),max(b[1]-y,0,y-b[3]))

def build_topology_v2(scene,mask,mode='directional',merge_labels=True,traced=None):
    paths,sk,adj=traced if traced is not None else trace_paths(mask);npaths=len(paths);dsu=DSU(npaths);at=defaultdict(list)
    for i,path in enumerate(paths):at[path[0]].append((i,path[1]));at[path[-1]].append((i,path[-2]))
    thickness=cv2.distanceTransform(mask,cv2.DIST_L2,3);junctions=[];crossings=[]
    for node,arms in at.items():
        if len(arms)<2:continue
        y,x=node;split=False
        if len(arms)==4:
            local=[]
            for idx,_ in arms:
                p=paths[idx] if paths[idx][0]==node else paths[idx][::-1];local.append((idx,p[min(5,len(p)-1)]))
            widths=[thickness[q] for _,q in local];dot=thickness[y,x]>max(1.7,float(np.median(widths))*1.55)
            horizontal=[i for i,q in local if abs(q[1]-x)>abs(q[0]-y)];vertical=[i for i,q in local if abs(q[1]-x)<abs(q[0]-y)]
            if not dot and len(horizontal)==len(vertical)==2:
                dsu.union(horizontal[0],horizontal[1]);dsu.union(vertical[0],vertical[1]);crossings.append((x,y));split=True
        if not split:
            for i,_ in arms[1:]:dsu.union(arms[0][0],i)
            if len(arms)>=3:junctions.append((x,y))
    stroke=float(scene.diagnostics.get('wire_stroke_width',1.0));bridge_radius=max(1.5,min(4.,stroke*1.25));bridges=[]
    ends=[(p,a[0][0]) for p,a in at.items() if len(a)==1 and len(adj.get(p,[]))==1]
    nearby=cKDTree(np.asarray([p for p,_ in ends],float)).query_pairs(bridge_radius) if len(ends)>1 else set()
    for j,k in sorted(nearby):
        a,ai=ends[j];b,bi=ends[k]
        if dsu.find(ai)==dsu.find(bi):continue
        delta=np.array(b,float)-np.array(a,float);dist=np.linalg.norm(delta)
        if not 0<dist<=bridge_radius or min(abs(delta[0]),abs(delta[1]))>.35*dist:continue
        va=np.array(a)-np.array(adj[a][0]);vb=np.array(b)-np.array(adj[b][0])
        if np.dot(va,delta)>0 and np.dot(vb,-delta)>0:
            dsu.union(ai,bi);bridges.append(((a[1],a[0]),(b[1],b[0])))
    seglists=[simplify(p) for p in paths];allsegments=[(i,a,b) for i,ss in enumerate(seglists) for a,b in ss]
    pins=[(c,p,f'{c.key}.{p.number}') for c in scene.components for p in c.pins];pdsu=DSU(npaths+len(pins))
    for i in range(npaths):pdsu.union(i,dsu.find(i))
    radius=max(2.5,min(12.,2.2*stroke+2));snaps=[];unattached=[];candidate_count=0;reject={'distance':0,'direction':0,'orientation':0,'pixels':0}
    for pi,(c,p,ref) in enumerate(pins):
        outward=_direction(p,c);best=None
        for wi,a,b in allsegments:
            distance,q=projection(p.tip,a,b)
            if distance>radius:reject['distance']+=1;continue
            candidate_count+=1;delta=np.array(q)-np.array(p.tip);forward=float(np.dot(delta,outward))
            if mode=='directional' and forward < -max(1.,stroke):reject['direction']+=1;continue
            line=np.array(b)-np.array(a);ln=np.linalg.norm(line)
            align=abs(float(np.dot(line/max(ln,1e-6),outward)))
            # A pin may meet a perpendicular bus at a T junction, so line
            # orientation is a score term rather than a hard rejection.
            if mode=='directional' and distance>stroke*1.5 and align<.55:reject['orientation']+=1
            qx,qy=map(int,map(round,q));y0=max(0,qy-1);y1=min(mask.shape[0],qy+2);x0=max(0,qx-1);x1=min(mask.shape[1],qx+2)
            if not mask[y0:y1,x0:x1].any():reject['pixels']+=1;continue
            score=distance+.35*max(0,-forward)+(1-align)*stroke*.5
            if best is None or score<best[0]:best=(score,wi,q,distance,forward,align)
        if best:
            score,wi,q,distance,forward,align=best;pdsu.union(npaths+pi,wi)
            snaps.append({'pin':ref,'path':wi,'distance':round(distance,3),'forward':round(forward,3),'alignment':round(align,3),'radius':round(radius,3),'from':p.tip,'to':q})
        else:unattached.append(ref)
    contacts=[]
    for i,(_,p,_) in enumerate(pins):
        for j in range(i+1,len(pins)):
            if math.dist(p.tip,pins[j][1].tip)<=max(1,stroke*.65):pdsu.union(npaths+i,npaths+j);contacts.append((i,p.tip))
    # Label endpoint association: labels merge logical groups, never create geometry.
    label_hits=[];label_root={};endpoint_records=[(i,(p[1],p[0])) for i,path in enumerate(paths) for p in (path[0],path[-1])]
    for t in scene.texts if merge_labels else []:
        if not _is_label(t,scene):continue
        near=sorted((_bbox_point_dist(pt,t.bbox),i,pt) for i,pt in endpoint_records)
        if not near or near[0][0]>max(5,stroke*4):continue
        dist,i,pt=near[0];label=norm_label(t.text)
        if label in label_root:pdsu.union(i,label_root[label])
        else:label_root[label]=i
        label_hits.append({'label':label,'path':i,'distance':round(dist,3),'point':pt})
    # A malformed/ambiguous pin OCR result can produce the same exported
    # component.pin reference twice.  It is one logical terminal, so force all
    # occurrences into one DSU set before materialising nets.  Keeping this
    # invariant here prevents a label merge from exporting the same pin in two
    # nets while diagnostics still expose the duplicate upstream prediction.
    ref_nodes=defaultdict(list)
    for i,(_,_,ref) in enumerate(pins):ref_nodes[ref].append(npaths+i)
    duplicate_refs={ref:nodes for ref,nodes in ref_nodes.items() if len(nodes)>1}
    for nodes in duplicate_refs.values():
        for node in nodes[1:]:pdsu.union(nodes[0],node)
    groups=defaultdict(lambda:{'pins':set(),'segments':[]})
    for i,(_,_,ref) in enumerate(pins):groups[pdsu.find(npaths+i)]['pins'].add(ref)
    for i,segs in enumerate(seglists):groups[pdsu.find(i)]['segments'].extend(segs)
    for i,tip in contacts:
        g=groups[pdsu.find(npaths+i)]
        if (tip,tip) not in g['segments']:g['segments'].append((tip,tip))
    scene.nets=[Net(sorted(g['pins']),g['segments']) for g in groups.values() if g['pins']]
    scene.diagnostics.update({'topology':'v2','snap_mode':mode,'wire_paths':npaths,'wire_segments':len(allsegments),'wire_pixels':int((mask>0).sum()),
        'wire_groups':len({dsu.find(i) for i in range(npaths)}),'junctions':junctions,'nonconnecting_crossings':crossings,'short_bridges':bridges,
        'snap_radius':round(radius,3),'snap_candidates_evaluated':candidate_count,'snap_rejections':reject,'snaps':snaps,'unattached_pins':unattached,
        'unattached_pin_ratio':len(unattached)/max(1,len(pins)),'signal_label_hits':label_hits,'signal_labels_merged':len(label_root),'flying_label_merge_enabled':merge_labels,
        'nets':len(scene.nets),'singleton_nets':sum(len(n.pins)==1 for n in scene.nets),'net_edges':sum(len(n.segments) for n in scene.nets),
        'wire_groups_without_pins':sum(not g['pins'] for g in groups.values()),
        'duplicate_export_pin_refs':sorted(duplicate_refs)})
    return sk
