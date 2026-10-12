"""Synthetic-only cases; no dataset or expected result generation in tests."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import cv2
import numpy as np
from pcb.schema import Scene,Component,Pin,Text


def cases():
    templates=[
        ('straight',[(10,50),(90,50)],[('line',(10,50),(90,50),1)],[]),
        ('t_junction',[(10,50),(90,50),(50,10)],[('line',(10,50),(90,50),1),('line',(50,10),(50,50),1)],[]),
        ('cross_no_dot',[(10,50),(90,50),(50,10),(50,90)],[('line',(10,50),(90,50),1),('line',(50,10),(50,90),1)],[]),
        ('cross_dot',[(10,50),(90,50),(50,10),(50,90)],[('line',(10,50),(90,50),1),('line',(50,10),(50,90),1),('dot',(50,50),4)],[]),
        ('elbow',[(10,50),(50,90)],[('line',(10,50),(50,50),1),('line',(50,50),(50,90),1)],[]),
        ('short_gap',[(10,50),(90,50)],[('line',(10,50),(49,50),1),('line',(51,50),(90,50),1)],[]),
        ('long_gap',[(10,50),(90,50)],[('line',(10,50),(40,50),1),('line',(50,50),(90,50),1)],[]),
        ('closed_loop',[(20,50),(80,50)],[('rect',(20,20),(80,80),1)],[]),
        ('empty_singletons',[(20,30),(80,70)],[],[]),
        ('direct_contact',[(20,30),(20,30)],[],[]),
        ('duplicate_refs',[(10,50),(90,50)],[('line',(10,50),(40,50),1),('line',(60,50),(90,50),1)],[]),
        ('label_merge',[(10,50),(90,50)],[('line',(10,50),(30,50),1),('line',(70,50),(90,50),1)],
         [('VCC',(29,47,36,54)),('VCC',(64,47,71,54))]),
        ('unexportable',[(10,50),(90,50)],[('line',(10,50),(90,50),1)],[]),
        ('directional_behind',[(40,50)],[('line',(42,20),(42,80),1)],[]),
        ('perpendicular_bus',[(47,50)],[('line',(50,10),(50,90),1)],[]),
        ('wire_only',[],[('line',(10,50),(90,50),1)],[]),
    ]
    for name,points,shapes,texts in templates:
        mask=np.zeros((100,100),np.uint8)
        for shape in shapes:
            if shape[0]=='line':cv2.line(mask,shape[1],shape[2],255,shape[3])
            elif shape[0]=='rect':cv2.rectangle(mask,shape[1],shape[2],255,shape[3])
            else:cv2.circle(mask,shape[1],shape[2],255,-1)
        components=[Component(f'U{i+1}','box',(1,1,3,3),pins=[Pin('1','1',p)]) for i,p in enumerate(points)]
        if name=='duplicate_refs':components[1].key=components[0].key
        if name=='unexportable':components[1].pins[0].number='UNK1';components[1].pins[0].exportable=False
        if name=='directional_behind':
            components[0].bbox=(40,40,60,60);components[0].pins[0].base=(50,50);components[0].pins[0].side='left'
        if name=='perpendicular_bus':
            components[0].bbox=(20,40,40,60);components[0].pins[0].base=(40,50);components[0].pins[0].side='right'
        s=Scene(100,100,components,[Text(text,box) for text,box in texts],diagnostics={'wire_stroke_width':1.})
        yield name,s,mask


def plain(value):return json.loads(json.dumps(value,default=lambda v:v.tolist()))


def capture():
    from pcb.topology_stage.v2 import build_topology_v2
    from pcb.submission import export
    path=Path(__file__).parent/'fixtures/network_split_baseline.json'
    if path.exists():raise ValueError('Do not regenerate expected values after refactoring')
    entries=[]
    for name,source,mask in cases():
        for mode in ('fixed','directional'):
            for merge in (False,True):
                s=copy.deepcopy(source);sk=build_topology_v2(s,mask,mode,merge)
                entries.append({'case':name,'mode':mode,'merge_labels':merge,'prediction':export(s),
                                'diagnostics':plain(s.diagnostics),'skeleton_pixels':np.argwhere(sk).tolist()})
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'source_commit':'05415bd3a0cf7a4c2f8ee844d1a2c356291745d6',
                               'synthetic_only':True,'cases':entries},indent=2),encoding='utf8')
    print(f'Captured {len(entries)} synthetic legacy outputs before refactoring.')


if __name__=='__main__':capture()
