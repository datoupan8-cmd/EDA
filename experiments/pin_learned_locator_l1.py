"""Image-only, component-conditioned pin heatmap prototype; no data-file reads.

Points are original OpenCV pixels. Canonical strips keep the outward normal as
the row axis; outputs adapt to the existing terminal contract. Only box changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pcb.core.interfaces import PinLocalizationOutput


@dataclass(frozen=True)
class Policy:
    height: int = 128
    width: int = 256
    stride: int = 192
    origin: int = 48
    sigma: float = 1.5
    threshold: float = .35
    nms_radius: int = 3
    base_channels: int = 16
    epochs: int = 40
    batch: int = 16
    learning_rate: float = .001
    seed: int = 20261005


POLICY = Policy()
SIDES = ('left', 'right', 'top', 'bottom')


def image_scale(shape) -> float:
    return max(.55, min(2.2, max(shape[:2]) / 1200.))


def nearest_side(point, box) -> str:
    x, y = point
    a, b, c, d = box
    # Distance to finite segments, not to infinitely extended lines.
    distances = (math.hypot(x-a, max(b-y, 0., y-d)),
                 math.hypot(x-c, max(b-y, 0., y-d)),
                 math.hypot(y-b, max(a-x, 0., x-c)),
                 math.hypot(y-d, max(a-x, 0., x-c)))
    return SIDES[int(np.argmin(distances))]


def strip_specs(box, shape, policy=POLICY) -> list[dict]:
    a, b, c, d = map(float, box)
    if not all(math.isfinite(v) for v in (a, b, c, d)) or a >= c or b >= d:
        raise ValueError('Invalid component box')
    scale = image_scale(shape)
    specs = []
    for side in SIDES:
        low, high = (b, d) if side in ('left', 'right') else (a, c)
        low, high = low-8*scale, high+8*scale
        span = policy.width*scale
        if high-low <= span:
            starts = [(low+high-span)/2]
        else:
            starts = list(np.arange(low, high-span, policy.stride*scale))
            if not starts or abs(starts[-1]-(high-span)) > .01:
                starts.append(high-span)
        specs.extend(dict(side=side, start=float(t), scale=scale, box=[a,b,c,d]) for t in starts)
    return specs


def strip_to_global(col, row, spec, policy=POLICY):
    normal = (np.asarray(row)-policy.origin)*spec['scale']
    tangent = spec['start'] + np.asarray(col)*spec['scale']
    a, b, c, d = spec['box']
    if spec['side'] == 'left': return a-normal, tangent
    if spec['side'] == 'right': return c+normal, tangent
    if spec['side'] == 'top': return tangent, b-normal
    return tangent, d+normal


def global_to_strip(point, spec, policy=POLICY):
    x, y = point
    a, b, c, d = spec['box']
    side = spec['side']
    normal = {'left': a-x, 'right': x-c, 'top': b-y, 'bottom': y-d}[side]
    tangent = y if side in ('left', 'right') else x
    return ((tangent-spec['start'])/spec['scale'], normal/spec['scale']+policy.origin)


def sample_strip(image, spec, policy=POLICY) -> np.ndarray:
    rows, cols = np.mgrid[:policy.height, :policy.width].astype(np.float32)
    xx, yy = strip_to_global(cols, rows, spec, policy)
    crop = cv2.remap(image, xx.astype(np.float32), yy.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    rgb = crop[:, :, ::-1].astype(np.float32)/127.5-1.
    a, b, c, d = spec['box']
    owner = ((xx >= a) & (xx <= c) & (yy >= b) & (yy <= d)).astype(np.float32)
    return np.concatenate((rgb, owner[:, :, None]), axis=2).transpose(2,0,1).copy()


def point_targets(points, spec, policy=POLICY):
    """CenterNet-style Gaussian point targets with exact fractional offsets."""
    heat = np.zeros((1, policy.height, policy.width), np.float32)
    offset = np.zeros((2, policy.height, policy.width), np.float32)
    mask = np.zeros_like(heat)
    rows, cols = np.mgrid[:policy.height, :policy.width]
    used = 0
    for point in points:
        if nearest_side(point, spec['box']) != spec['side']: continue
        col, row = global_to_strip(point, spec, policy)
        x, y = int(round(col)), int(round(row))
        if not (0 <= x < policy.width and 0 <= y < policy.height): continue
        blob = np.exp(-((cols-x)**2+(rows-y)**2)/(2*policy.sigma**2))
        heat[0] = np.maximum(heat[0], blob)
        offset[:, y, x] = (col-x, row-y)
        mask[0, y, x] = 1
        used += 1
    return heat, offset, mask, used


class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming,outgoing,3,padding=1,bias=False),
            nn.GroupNorm(4,outgoing), nn.SiLU(), nn.Conv2d(outgoing,outgoing,3,padding=1,bias=False),
            nn.GroupNorm(4,outgoing), nn.SiLU())

    def forward(self, x): return self.layers(x)


class PinPointNet(nn.Module):
    def __init__(self, policy=POLICY):
        super().__init__()
        n = policy.base_channels
        self.a, self.b, self.c = Block(4,n), Block(n,n*2), Block(n*2,n*4)
        self.d, self.e = Block(n*6,n*2), Block(n*3,n)
        self.head = nn.Conv2d(n,3,1)
        nn.init.constant_(self.head.bias, 0.)
        with torch.no_grad(): self.head.bias[0] = -2.19

    def forward(self, x):
        a = self.a(x); b = self.b(F.avg_pool2d(a,2)); c = self.c(F.avg_pool2d(b,2))
        d = self.d(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        e = self.e(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        return self.head(e)


def point_loss(output, heat, offset, mask):
    # Compute in float32 even when convolutions run in mixed precision.
    p = output[:, :1].float().sigmoid().clamp(1e-5,1-1e-5)
    positives = heat.eq(1.)
    loss = -(torch.log(p)*(1-p).pow(2)*positives
             + torch.log(1-p)*p.pow(2)*(1-heat).pow(4)*(~positives)).sum()
    count = mask.sum().clamp_min(1)
    offset_loss = ((output[:, 1:].float()-offset).abs()*mask).sum()/count
    return loss/count + offset_loss


def decode_map(output, spec, shape, policy=POLICY) -> list[dict]:
    heat = output[0].sigmoid()
    maximum = F.max_pool2d(heat[None,None], 2*policy.nms_radius+1, stride=1,
                           padding=policy.nms_radius)[0,0]
    rows, cols = torch.where((heat >= policy.threshold) & (heat == maximum))
    result = []
    a, b, c, d = spec['box']; h, w = shape[:2]
    for row, col in zip(rows.tolist(), cols.tolist()):
        dx, dy = output[1:,row,col].float().tolist()
        x, y = strip_to_global(col+dx,row+dy,spec,policy)
        x, y = float(x), float(y)
        if not (0 <= x < w and 0 <= y < h): continue
        if nearest_side((x,y), spec['box']) != spec['side']: continue
        side = spec['side']
        base = {'left':(a,y), 'right':(c,y), 'top':(x,b), 'bottom':(x,d)}[side]
        result.append(dict(tip=(x,y),base=base,side=side,method='learned_point_l1',
                           wire_support_score=None,point_confidence=float(heat[row,col])))
    return result


def merge_points(rows, scale, policy=POLICY):
    kept = []
    for row in sorted(rows,key=lambda r:-r['point_confidence']):
        if not any(math.dist(row['tip'],old['tip']) <= policy.nms_radius*scale for old in kept):
            kept.append(row)
    order = {s:i for i,s in enumerate(SIDES)}
    return sorted(kept,key=lambda r:(order[r['side']],r['base'][1] if r['side'] in ('left','right') else r['base'][0]))


@torch.inference_mode()
def locate_boxes(image, components, baseline, model, device, policy=POLICY):
    """Replace box localization, retaining every nonbox terminal verbatim."""
    rows, events = [], []
    model.eval()
    for component, (_, original) in zip(components, baseline.terminals):
        if component.type != 'box':
            rows.append((component,copy.deepcopy(original))); continue
        specs = strip_specs(component.body_bbox or component.bbox,image.shape,policy)
        proposals = []
        for start in range(0,len(specs),policy.batch):
            chunk = specs[start:start+policy.batch]
            x = torch.from_numpy(np.stack([sample_strip(image,s,policy) for s in chunk])).to(device)
            output = model(x).cpu()
            for out, spec in zip(output,chunk): proposals.extend(decode_map(out,spec,image.shape,policy))
        candidates = merge_points(proposals,image_scale(image.shape),policy)
        rows.append((component,candidates))
        events.append(dict(component=component.key,strips=len(specs),candidates=len(candidates)))
    return PinLocalizationOutput(rows,diagnostics={'learned_locator_l1':events})
