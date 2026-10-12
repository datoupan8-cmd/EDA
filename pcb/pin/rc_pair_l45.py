"""Known-two-endpoint estimation with actual predicted boxes; no identity edits."""
from __future__ import annotations
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ..core.interfaces import PinSemanticsOutput


@dataclass(frozen=True)
class PairPolicy:
    height:int=96
    width:int=192
    minimum_long_half:float=32.
    minimum_short_half:float=16.
    box_extent_factor:float=1.5
    epochs:int=40
    batch:int=64
    learning_rate:float=.001
    weight_decay:float=.0001
    seed:int=20261012


POLICY=PairPolicy()


def frame(component, pins, shape, policy=POLICY):
    """Axis comes from frozen pin sides, never target geometry or a style label."""
    if component.type not in ('r','c') or len(pins)!=2 or not all(p.exportable for p in pins):return None
    sides={p.side for p in pins}
    if sides=={'left','right'}:vertical=False
    elif sides=={'top','bottom'}:vertical=True
    else:return None
    a,b,c,d=map(float,component.bbox)
    if not a<c or not b<d:return None
    long=(d-b) if vertical else (c-a)
    short=(c-a) if vertical else (d-b)
    return {'center':[(a+c)/2,(b+d)/2],'vertical':vertical,
            'long_half':max(policy.minimum_long_half,policy.box_extent_factor*long),
            'short_half':max(policy.minimum_short_half,policy.box_extent_factor*short),
            'bbox':[a,b,c,d],'body_bbox':list(component.body_bbox or component.bbox),
            'type_code':float(component.type=='r'),'shape':list(shape[:2])}


def canonical(points, spec):
    pts=np.asarray(points,dtype=np.float32)
    cx,cy=spec['center']
    if spec['vertical']:out=np.stack(((pts[:,1]-cy)/spec['long_half'],(pts[:,0]-cx)/spec['short_half']),1)
    else:out=np.stack(((pts[:,0]-cx)/spec['long_half'],(pts[:,1]-cy)/spec['short_half']),1)
    return out


def global_points(points,spec):
    pts=np.asarray(points,dtype=np.float32).reshape(2,2)
    cx,cy=spec['center']
    if spec['vertical']:return np.stack((cx+pts[:,1]*spec['short_half'],cy+pts[:,0]*spec['long_half']),1)
    return np.stack((cx+pts[:,0]*spec['long_half'],cy+pts[:,1]*spec['short_half']),1)


def ordered_pins(pins):
    return sorted(pins,key=lambda p:p.side in ('right','bottom'))


def rect_canonical(rect,spec):
    a,b,c,d=rect
    pts=canonical([(a,b),(c,d)],spec)
    return [float(pts[:,0].min()),float(pts[:,1].min()),float(pts[:,0].max()),float(pts[:,1].max())]


def sample(image,component,pins,policy=POLICY):
    spec=frame(component,pins,image.shape,policy)
    if spec is None:return None
    yy,xx=np.mgrid[:policy.height,:policy.width].astype(np.float32)
    xx=2*xx/(policy.width-1)-1
    yy=2*yy/(policy.height-1)-1
    cx,cy=spec['center']
    if spec['vertical']:gx=cx+yy*spec['short_half'];gy=cy+xx*spec['long_half']
    else:gx=cx+xx*spec['long_half'];gy=cy+yy*spec['short_half']
    crop=cv2.remap(image,gx,gy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    prior=canonical([p.tip for p in ordered_pins(pins)],spec).reshape(4)
    features=np.asarray([*rect_canonical(spec['bbox'],spec),*rect_canonical(spec['body_bbox'],spec),spec['type_code']],np.float32)
    return crop[:,:,::-1].copy(),prior,features,spec


class PairResidualNet(nn.Module):
    def __init__(self,policy=POLICY):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(8,16,5,2,2),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv2d(16,24,3,2,1),nn.GroupNorm(4,24),nn.SiLU(),
            nn.Conv2d(24,32,3,2,1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.Conv2d(32,48,3,2,1),nn.GroupNorm(4,48),nn.SiLU())
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(48*6*12+4,64),nn.SiLU(),nn.Linear(64,4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,policy.height),torch.linspace(-1,1,policy.width),indexing='ij')
        self.register_buffer('grid_x',xx[None,None]);self.register_buffer('grid_y',yy[None,None])

    def forward(self,rgb,prior,features):
        x=self.grid_x.expand(len(rgb),-1,-1,-1);y=self.grid_y.expand(len(rgb),-1,-1,-1)
        masks=[]
        for offset in (0,4):
            a,b,c,d=(features[:,offset+i,None,None,None] for i in range(4))
            masks.append(((x>=a)&(x<=c)&(y>=b)&(y<=d)).to(rgb.dtype))
        typ=features[:,8,None,None,None].expand_as(x)
        encoded=self.encoder(torch.cat((rgb,*masks,typ,x,y),1)).flatten(1)
        h=torch.cat((encoded,prior),1)
        # Flatten already done; reuse the remaining head layers.
        delta=self.head[1:](h).tanh()
        return prior+delta


def calibration_points(component,pins,alpha):
    spec=frame(component,pins,(1,1))
    if spec is None:return None
    a,b,c,d=spec['bbox'];cx,cy=spec['center']
    short=(c-a) if spec['vertical'] else (d-b)
    lead=float(alpha)*short
    return np.asarray([(cx,b-lead),(cx,d+lead)] if spec['vertical'] else [(a-lead,cy),(c+lead,cy)],np.float32)


def valid_pair(points,spec,image_shape):
    pts=np.asarray(points,dtype=float)
    h,w=image_shape[:2]
    if pts.shape!=(2,2) or not np.isfinite(pts).all():return False
    if not np.all((pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)):return False
    return bool(canonical(pts,spec)[0,0]<canonical(pts,spec)[1,0])


class RCPointSemantics:
    """Run frozen L42 identities first, then propose points for its existing pair."""
    def __init__(self,mode,resources,projection=False,baseline=None):
        from .model_control_l42 import L42ControlSemantics
        self.baseline=baseline or L42ControlSemantics()
        self.mode=mode;self.resources=Path(resources);self.projection=projection;self.model=None
        self.calibration=None;self.last_trace=[]

    def run_image(self,image,component,localization,context):
        original=self.baseline.run_image(image,component,localization,context)
        result=copy.deepcopy(original)
        self.last_trace=[];maps=[]
        if self.mode=='calibration' and self.calibration is None:
            self.calibration=json.loads((self.resources/'calibration.json').read_text('utf-8'))
        if self.mode=='regression' and self.model is None:
            payload=torch.load(self.resources/'best.pt',map_location='cpu',weights_only=True)
            if payload['policy']!=asdict(POLICY):raise ValueError('Frozen pair policy changed')
            self.model=PairResidualNet().to(context.resources['pin']['device'])
            self.model.load_state_dict(payload['state_dict']);self.model.eval()
        for owner in result.components:
            if owner.type not in ('r','c'):continue
            sampled=sample(image,owner,owner.pins)
            if sampled is None:continue
            crop,prior,features,spec=sampled
            if self.mode=='calibration':points=calibration_points(owner,owner.pins,self.calibration['alpha'])
            else:
                device=context.resources['pin']['device']
                with torch.inference_mode():
                    rgb=torch.from_numpy(crop.transpose(2,0,1).copy()).to(device).float()[None]/127.5-1
                    output=self.model(rgb,torch.from_numpy(prior).to(device)[None],torch.from_numpy(features).to(device)[None])
                    points=global_points(output.cpu().numpy()[0],spec)
            accepted=valid_pair(points,spec,image.shape)
            pins=ordered_pins(owner.pins)
            row={'component':owner.key,'type':owner.type,'accepted':accepted,'bbox_source':'actual_predicted_detection_bbox',
                 'old_points':[list(p.tip) for p in pins],'new_points':points.tolist(),
                 'projection_only':self.projection,'mode':self.mode}
            if accepted:
                for pin,point in zip(pins,points):
                    maps.append({'owner':owner.key,'number':pin.number,'point':[float(x) for x in point]})
                    if not self.projection:pin.tip=tuple(float(x) for x in point)
            self.last_trace.append(row)
        result.diagnostics['rc_pair_l45']={'mode':self.mode,'projection_only':self.projection,'trace':self.last_trace,
                                          'export_points':maps,'GT_used_in_inference':False}
        return result


class PinPointProjectionSubmission:
    """Pin-owned private experiment: old anchors reach networks, same exporter sees new points.

    No public schema, Submission implementation or coordinate conversion change.
    """
    def __init__(self,baseline):self.baseline=baseline

    def run(self,scene,context):
        projected=copy.deepcopy(scene)
        lookup={(r['owner'],r['number']):tuple(r['point']) for r in scene.diagnostics.get('rc_pair_l45',{}).get('export_points',[])}
        for owner in projected.components:
            for pin in owner.pins:
                if (owner.key,pin.number) in lookup:pin.tip=lookup[owner.key,pin.number]
        return self.baseline.run(projected,context)
