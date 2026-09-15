"""Pixel wire extraction and path tracing. Logical tree edges are never exported as wires."""
import cv2
import numpy as np
from skimage.morphology import skeletonize

def extract_wire(image,scene):
    b,g,r=[x.astype(np.int16) for x in cv2.split(image)]
    green=(g>r+25)&(g>b+20)&(g<235)
    colored=int(green.sum())>max(35,image.shape[0]*image.shape[1]*0.00008)
    if colored:
        mask=green.astype(np.uint8)*255
        # KiCad may draw junction dots blue while wires are green. Preserve compact
        # dots supported by wire pixels on at least two sides, not arbitrary blue text.
        blue=((b>r+30)&(b>g+30)).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(blue)
        dots=0
        for i in range(1,count):
            x,y,ww,hh,area=map(int,stats[i])
            if not (3<=area<=350 and max(ww,hh)<=24 and .55<=ww/max(1,hh)<=1.8 and area/(ww*hh)>=.55):continue
            margin=3;xa=max(0,x-margin);xb=min(scene.width,x+ww+margin);ya=max(0,y-margin);yb=min(scene.height,y+hh+margin)
            sides=[green[ya:y,xa:xb].any(),green[y+hh:yb,xa:xb].any(),green[ya:yb,xa:x].any(),green[ya:yb,x+ww:xb].any()]
            if sum(sides)>=2:
                local=mask[y:y+hh,x:x+ww];local[labels[y:y+hh,x:x+ww]==i]=255;dots+=1
        scene.diagnostics['blue_junction_dots']=dots
    else:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # Ignore broad page borders. Keep image geometry at its native resolution.
        for t in scene.texts:
            x1,y1,x2,y2=map(int,t.bbox)
            cv2.rectangle(mask,(max(0,x1),max(0,y1)),(x2,y2),0,-1)
    for c in scene.components:
        x1,y1,x2,y2=map(int,c.body_bbox or c.bbox)
        cv2.rectangle(mask,(x1+1,y1+1),(x2-1,y2-1),0,-1)
    # Mask only the outer image frame, not every long line.
    mask[:2,:]=0;mask[-2:,:]=0;mask[:,:2]=0;mask[:,-2:]=0
    scene.diagnostics['wire_palette']='green' if colored else 'monochrome'
    return mask

def trace_paths(mask):
    sk=skeletonize(mask>0)
    pixels=set(zip(*np.where(sk)))
    def neighbors(p):
        y,x=p;out=[]
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
            q=y+dy,x+dx
            if q not in pixels:continue
            # Avoid triangle shortcuts around corners and junctions.
            if dx and dy and ((y,x+dx) in pixels or (y+dy,x) in pixels):continue
            out.append(q)
        return out
    adj={p:neighbors(p) for p in pixels}
    critical={p for p,n in adj.items() if len(n)!=2}
    visited=set();paths=[]
    def edge(a,b):return (a,b) if a<b else (b,a)
    def walk(start,nxt):
        path=[start];prev,cur=start,nxt;visited.add(edge(prev,cur))
        while True:
            path.append(cur)
            if cur in critical or cur==start:break
            rest=[n for n in adj[cur] if n!=prev]
            if not rest:break
            nxt=rest[0]
            if edge(cur,nxt) in visited:break
            visited.add(edge(cur,nxt));prev,cur=cur,nxt
        return path
    for p in sorted(critical):
        for n in adj[p]:
            if edge(p,n) not in visited:paths.append(walk(p,n))
    # Closed loops have no critical pixel.
    for p in sorted(pixels):
        for n in adj[p]:
            if edge(p,n) not in visited:paths.append(walk(p,n))
    # One-pixel specks are not wires.
    return [p for p in paths if len(p)>=3],sk,adj

def simplify(path,epsilon=1.0):
    points=np.array([(x,y) for y,x in path],np.float32).reshape(-1,1,2)
    approx=cv2.approxPolyDP(points,epsilon,False).reshape(-1,2)
    return [(tuple(map(float,a)),tuple(map(float,b))) for a,b in zip(approx,approx[1:]) if np.linalg.norm(a-b)>0]

def projection(p,a,b):
    p,a,b=np.asarray(p,float),np.asarray(a,float),np.asarray(b,float)
    v=b-a;t=float(np.clip(np.dot(p-a,v)/max(np.dot(v,v),1e-9),0,1));q=a+t*v
    return float(np.linalg.norm(p-q)),tuple(q)
