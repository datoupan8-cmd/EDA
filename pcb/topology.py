"""Conservative PCB adaptation of CC/snapping ideas; implemented independently."""
from collections import defaultdict
import math
import cv2
import numpy as np
from .schema import DSU,Net
from .wire import trace_paths,simplify,projection

def build_topology(scene,mask,variant='improved',snap_radius=4.0):
    paths,sk,adj=trace_paths(mask)
    dsu=DSU(len(paths));at=defaultdict(list)
    for i,path in enumerate(paths):
        at[path[0]].append((i,path[1]));at[path[-1]].append((i,path[-2]))
    thickness=cv2.distanceTransform(mask,cv2.DIST_L2,3)
    junctions=[];crossings=[]
    for node,arms in at.items():
        if len(arms)<2:continue
        y,x=node
        split=False
        if variant=='improved' and len(arms)==4:
            local=[]
            for idx,_ in arms:
                p=paths[idx] if paths[idx][0]==node else paths[idx][::-1]
                q=p[min(5,len(p)-1)];local.append((idx,q))
            widths=[thickness[q] for _,q in local]
            is_dot=thickness[y,x]>max(1.8,float(np.median(widths))*1.6)
            if not is_dot:
                horizontal=[i for i,q in local if abs(q[1]-x)>abs(q[0]-y)]
                vertical=[i for i,q in local if abs(q[1]-x)<abs(q[0]-y)]
                if len(horizontal)==len(vertical)==2:
                    dsu.union(*horizontal);dsu.union(*vertical);split=True;crossings.append((x,y))
        if not split:
            for i,_ in arms[1:]:dsu.union(arms[0][0],i)
            if len(arms)>=3:junctions.append((x,y))
    # Endpoint bridges only: <=2 px, collinear, opposing directions. No global dilation.
    bridges=[]
    if variant=='improved':
        ends=[(p,arms[0][0]) for p,arms in at.items() if len(arms)==1 and len(adj.get(p,[]))==1]
        for j,(a,ai) in enumerate(ends):
            for b,bi in ends[j+1:]:
                if ai==bi or dsu.find(ai)==dsu.find(bi):continue
                ay,ax=a;by,bx=b;delta=np.array([by-ay,bx-ax],float);dist=np.linalg.norm(delta)
                if not 0<dist<=2.1 or min(abs(by-ay),abs(bx-ax))>0.2:continue
                va=np.array(a)-np.array(adj[a][0]);vb=np.array(b)-np.array(adj[b][0])
                if np.dot(va,delta)>0 and np.dot(vb,-delta)>0:
                    dsu.union(ai,bi);bridges.append(((ax,ay),(bx,by)))
    segments=[simplify(p) for p in paths]
    allsegments=[(i,a,b) for i,ss in enumerate(segments) for a,b in ss]
    pins=[(c,p,f'{c.key}.{p.number}') for c in scene.components for p in c.pins]
    # Pin identities remain separate even if they share a physical tip.
    pdsu=DSU(len(paths)+len(pins));snaps=[];unattached=[]
    for i in range(len(paths)):pdsu.union(i,dsu.find(i))
    radius=0.75 if variant=='baseline' else snap_radius
    for pi,(c,p,ref) in enumerate(pins):
        best=None
        for wi,a,b in allsegments:
            distance,q=projection(p.tip,a,b)
            if distance>radius:continue
            if p.base and np.linalg.norm(np.array(p.tip)-p.base)>1:
                outward=np.array(p.tip)-p.base
                if np.dot(np.array(q)-p.tip,outward)<-np.linalg.norm(outward)*1.5:continue
            candidate=(distance,wi,q)
            if best is None or candidate[:2]<best[:2]:best=candidate
        if best:
            distance,wi,q=best;pdsu.union(len(paths)+pi,wi)
            snaps.append({'pin':ref,'distance':round(distance,3),'from':p.tip,'to':q})
        else:unattached.append(ref)
    contacts=[]
    for i,(_,a,_) in enumerate(pins):
        for j in range(i+1,len(pins)):
            if math.dist(a.tip,pins[j][1].tip)<=1.0:
                pdsu.union(len(paths)+i,len(paths)+j);contacts.append((i,a.tip))
    labels={}
    for i,(c,p,_) in enumerate(pins):
        if c.net_label:
            if c.net_label in labels:pdsu.union(len(paths)+i,labels[c.net_label])
            labels[c.net_label]=len(paths)+i
    groups=defaultdict(lambda:{'pins':set(),'segments':[]})
    for i,(_,_,ref) in enumerate(pins):groups[pdsu.find(len(paths)+i)]['pins'].add(ref)
    for i,segs in enumerate(segments):groups[pdsu.find(i)]['segments'].extend(segs)
    for i,tip in contacts:
        g=groups[pdsu.find(len(paths)+i)]
        if (tip,tip) not in g['segments']:g['segments'].append((tip,tip))
    # Bridge pixels are not present in the image; never fabricate physical edges for them.
    scene.nets=[Net(sorted(g['pins']),g['segments']) for _,g in sorted(groups.items()) if g['pins']]
    scene.diagnostics.update({'variant':variant,'wire_paths':len(paths),'wire_segments':len(allsegments),
        'wire_pixels':int((mask>0).sum()),'junctions':junctions,'nonconnecting_crossings':crossings,
        'short_bridges':bridges,'snaps':snaps,'unattached_pins':unattached,
        'unassigned_wire_groups':sum(not g['pins'] for g in groups.values()),
        'crossing_policy':'four-arm no-dot straight-through heuristic' if variant=='improved' else 'all touching pixels connected'})
    return sk
