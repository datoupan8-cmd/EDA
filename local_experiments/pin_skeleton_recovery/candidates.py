"""Append-only, image-only box recovery. No GT or filesystem inputs.

B uses the earlier SINA-style outer-ring clusters. C separates contacts by
outward skeleton branches, so a line parallel to the box cannot combine many
pins into a single ring centroid. Both use identical local text preservation,
outward-support validation, deduplication and unchanged V3 tip extension.
Reference/adaptation provenance: docs/SINA_LICENSE.txt (MIT, MEDAL 2025).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import numpy as np

from pcb.core.interfaces import PinLocalizationOutput
from pcb.pin.localization.v3 import PinLocalizationStageV3
from pcb.pin.localization.box_skeleton_experiment import (
    BoxCandidateConfig, SIDES, dimensions, preserve_strokes, raw_ink,
    skeleton_terminals, text_mask, tip_from_base,
)


@dataclass(frozen=True)
class RecoveryConfig:
    minimum_run_factor: float = 2.0
    annulus_pixels: float = 3.0
    dedup_scale: float = 7.0
    tangent_band_pixels: int = 1
    support_fraction: float = .65
    minimum_support_pixels: int = 3
    branch_group_gap: int = 1

    def __post_init__(self):
        if (self.minimum_run_factor < 1 or self.annulus_pixels <= 0
                or self.dedup_scale <= 0 or self.tangent_band_pixels < 0
                or not 0 < self.support_fraction <= 1
                or self.minimum_support_pixels < 1 or self.branch_group_gap < 1):
            raise ValueError('Invalid recovery thresholds')


def outward_profile(mask, component, side, tangent, config):
    """Binary normal profile in a narrow tangential band outside a frozen body."""
    (x1, y1, x2, y2), _, reach, _ = dimensions(mask.shape, component)
    boundary = {'left': x1, 'right': x2, 'top': y1, 'bottom': y2}[side]
    normal = boundary + np.arange(1, reach + 1) * (-1 if side in ('left', 'top') else 1)
    center = int(round(tangent))
    band = np.arange(center-config.tangent_band_pixels, center+config.tangent_band_pixels+1)
    yy, xx = (band[:, None], normal[None, :]) if side in ('left', 'right') else (normal[:, None], band[None, :])
    yy, xx = np.broadcast_arrays(yy, xx)
    valid = (yy >= 0) & (yy < mask.shape[0]) & (xx >= 0) & (xx < mask.shape[1])
    sample = np.zeros(yy.shape, bool)
    sample[valid] = mask[yy[valid], xx[valid]].astype(bool)
    return np.any(sample, axis=0 if side in ('left', 'right') else 1)


def supported(profile, scale, config):
    """Require outer evidence beyond the ring, not just a parallel body outline."""
    ring = max(1, int(round(config.annulus_pixels*scale)))
    # Outer support ends at the unchanged V3 tip distance (or image edge).
    end = min(len(profile), max(ring+config.minimum_support_pixels, int(round(max(6., 13.*scale)))))
    outer = profile[ring:end]
    return bool(len(outer) >= config.minimum_support_pixels
                and int(outer.sum()) >= config.minimum_support_pixels
                and float(outer.mean()) >= config.support_fraction)


def split_branches(kept, component, components, config):
    """Split a connected outer-ring contour by radial skeleton evidence.

    No new ink, morphological bridging, body refinement or endpoint-to-tip
    assumption. The skeleton is exactly the B skeleton; only its contacts
    are separated by their outward branch projections.
    """
    previous = BoxCandidateConfig(config.minimum_run_factor, config.annulus_pixels)
    _, detail, sk = skeleton_terminals(kept, component, components, previous)
    if 'crop' not in detail:
        return [], {'reason': 'empty_crop'}
    a, b, c, d = detail['crop']
    (x1, y1, x2, y2), scale, _, margin = dimensions(kept.shape, component)
    # Profile coordinates are local to the skeleton crop.
    # V3 scale depends on the full image, so sample global coordinates within
    # this crop rather than recomputing scale from the crop dimensions.
    proposals, events = [], []
    ring = max(1, int(round(config.annulus_pixels*scale)))
    reach = dimensions(kept.shape, component)[2]
    for side in SIDES:
        low, high = (y1+margin, y2-margin) if side in ('left', 'right') else (x1+margin, x2-margin)
        hits = []
        boundary = {'left': x1, 'right': x2, 'top': y1, 'bottom': y2}[side]
        normal = boundary + np.arange(1, reach+1)*(-1 if side in ('left', 'top') else 1)
        for tangent in range(low, high+1):
            band = np.arange(tangent-config.tangent_band_pixels, tangent+config.tangent_band_pixels+1)
            yy, xx = (band[:, None], normal[None, :]) if side in ('left', 'right') else (normal[:, None], band[None, :])
            yy, xx = np.broadcast_arrays(yy-b, xx-a)
            valid = (yy >= 0) & (yy < sk.shape[0]) & (xx >= 0) & (xx < sk.shape[1])
            pixels = np.zeros(yy.shape, bool)
            pixels[valid] = sk[yy[valid], xx[valid]] > 0
            profile = np.any(pixels, axis=0 if side in ('left', 'right') else 1)
            if profile[:ring+1].any() and supported(profile, scale, config):
                hits.append(tangent)
        groups = []
        for value in hits:
            if not groups or value-groups[-1][-1] > config.branch_group_gap:
                groups.append([value])
            else:
                groups[-1].append(value)
        for group in groups:
            tangent = float(np.median(group))
            base = {'left': (float(x1), tangent), 'right': (float(x2), tangent),
                    'top': (tangent, float(y1)), 'bottom': (tangent, float(y2))}[side]
            base = (min(kept.shape[1]-1., max(0., base[0])), min(kept.shape[0]-1., max(0., base[1])))
            proposals.append({'base': base, 'tip': tip_from_base(base, side, max(6.,13.*scale), kept.shape),
                              'side': side, 'method': 'sina_split_outward_branch',
                              'wire_support_score': len(group), 'branch_span': [group[0], group[-1]]})
        events.append({'side': side, 'supported_tangent_pixels': len(hits), 'branch_count': len(groups)})
    return proposals, {'branch_groups': events, 'ring_cluster_count': len(detail['regions'])}


def append_validated(original, proposals, kept, owner, components, config):
    """Keep the original sequence byte-for-byte; append only supported new contacts."""
    result, rejected = copy.deepcopy(original), []
    _, scale, _, _ = dimensions(kept.shape, owner)
    for proposal in proposals:
        reason = None
        if any(t['side'] == proposal['side'] and math.dist(t['tip'], proposal['tip']) < config.dedup_scale*scale for t in result):
            reason = 'duplicate_baseline_or_extra'
        tangent = proposal['base'][1] if proposal['side'] in ('left','right') else proposal['base'][0]
        if reason is None and not supported(outward_profile(kept, owner, proposal['side'], tangent, config), scale, config):
            reason = 'insufficient_outward_support'
        if reason is None:
            for other in components:
                if other is owner:
                    continue
                x1,y1,x2,y2 = other.body_bbox or other.bbox
                # Sample the proposed stub to exclude paths through another body.
                points = np.linspace(proposal['base'], proposal['tip'], 12)[1:]
                if np.any((points[:,0] > x1) & (points[:,0] < x2) & (points[:,1] > y1) & (points[:,1] < y2)):
                    reason = 'other_component_barrier'
                    break
        if reason:
            rejected.append({'reason': reason, 'proposal': proposal})
        else:
            result.append(copy.deepcopy(proposal))
    if result[:len(original)] != original:
        raise AssertionError('Baseline candidates changed')
    return result, rejected


class RecoveryStage:
    """Process-local strategy: V3 prefix plus box-only recovery."""
    def __init__(self, config: RecoveryConfig, split: bool):
        self.config, self.split = config, split

    def run(self, image, component, context):
        baseline = PinLocalizationStageV3().run(image, component, context)
        ink = raw_ink(image)
        masked = text_mask(ink.shape, component.texts)
        rows, events = [], []
        for owner, original in baseline.terminals:
            if owner.type != 'box':
                rows.append((owner, original))
                continue
            prior = BoxCandidateConfig(self.config.minimum_run_factor, self.config.annulus_pixels)
            kept, restored, _ = preserve_strokes(ink, masked, owner, prior)
            if self.split:
                proposals, detail = split_branches(kept, owner, component.components, self.config)
            else:
                proposals, detail, _ = skeleton_terminals(kept, owner, component.components, prior)
            terminals, rejected = append_validated(original, proposals, kept, owner, component.components, self.config)
            rows.append((owner, terminals))
            events.append({'component': owner.key, 'baseline_count': len(original), 'proposal_count': len(proposals),
                           'added_count': len(terminals)-len(original), 'restored_pixels': int(restored.sum()),
                           'rejected': rejected, 'detail': detail})
        return PinLocalizationOutput(rows, {'local_skeleton_recovery': {'split_branches': self.split, 'rows': events}})
