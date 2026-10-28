"""Pin-private model evidence; no global OCR or component property changes."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from ..schema import Text
from .frozen_l39.library_semantics import library_candidates, model_matches, stage as library_stage, POLICY as LIBRARY_POLICY
from .structural_l40_r1 import apply as structure_apply


@dataclass(frozen=True)
class ModelReadPolicy:
    margin_min: float = 24.
    margin_max: float = 64.
    body_fraction: float = .15
    magnification: float = 3.
    padding: int = 12
    rotations: tuple[int, ...] = (0, 90, 180, 270)


POLICY = ModelReadPolicy()


def owner_roi(component, texts, shape, policy=POLICY):
    h, w = shape[:2]
    x1, y1, x2, y2 = component.body_bbox or component.bbox
    margin = max(policy.margin_min, min(policy.margin_max, policy.body_fraction * min(x2-x1, y2-y1)))
    box = [x1-margin, y1-margin, x2+margin, y2+margin]
    touching = [t.bbox for t in texts if min(box[2], t.bbox[2]) > max(box[0], t.bbox[0])
                and min(box[3], t.bbox[3]) > max(box[1], t.bbox[1])]
    for a, b, c, d in touching:
        box = [min(box[0], a-2), min(box[1], b-2), max(box[2], c+2), max(box[3], d+2)]
    return (max(0, math.floor(box[0])), max(0, math.floor(box[1])),
            min(w, math.ceil(box[2])), min(h, math.ceil(box[3])))


def unrotate_bbox(polygon, padded_width, padded_height, angle, sx, sy, pad, roi):
    from .local_ocr import rotate_points
    points = rotate_points(polygon, padded_width, padded_height, angle, inverse=True)
    return ((float(points[:, 0].min())-pad)/sx+roi[0],
            (float(points[:, 1].min())-pad)/sy+roi[1],
            (float(points[:, 0].max())-pad)/sx+roi[0],
            (float(points[:, 1].max())-pad)/sy+roi[1])


class ModelOCR:
    """Same installed RapidOCR weights, new crop and content-addressed cache."""
    def __init__(self, cache, policy=POLICY):
        self.cache = Path(cache)
        self.policy = policy
        self.backend = None
        self.calls = self.cache_hits = 0

    @property
    def model_signature(self):
        return self.backend.model_signature if self.backend else None

    def read_owner(self, image, component, texts):
        import cv2
        import numpy as np
        from .frozen_l39.terminal_reader import TerminalRowOCR
        if self.backend is None:
            self.backend = TerminalRowOCR(self.cache / 'backend')
        engine = self.backend.ensure_engine()
        roi = owner_roi(component, texts, image.shape, self.policy)
        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        signature = {'version': 'l41-model-evidence-1', 'policy': asdict(self.policy),
                     'models': self.model_signature, 'shape': crop.shape}
        key = hashlib.sha256(crop.tobytes() + json.dumps(signature, sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + '.json')
        if path.is_file():
            payload = json.loads(path.read_text('utf-8'))
            self.cache_hits += 1
        else:
            scaled = cv2.resize(crop, None, fx=self.policy.magnification, fy=self.policy.magnification,
                                interpolation=cv2.INTER_CUBIC)
            sy, sx = scaled.shape[0]/crop.shape[0], scaled.shape[1]/crop.shape[1]
            pad = self.policy.padding
            padded = cv2.copyMakeBorder(scaled, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            ph, pw = padded.shape[:2]
            rows = []
            for angle in self.policy.rotations:
                output, _ = engine(np.ascontiguousarray(np.rot90(padded, angle//90)))
                self.calls += 1
                for polygon, text, score in output or []:
                    box = unrotate_bbox(polygon, pw, ph, angle, sx, sy, pad, roi)
                    # Save even low-confidence/library-missing strings for review.
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    rows.append({'text': str(text), 'score': float(score),
                                 'relative_bbox': [box[i]-roi[i % 2] for i in range(4)], 'angle': angle})
            payload = {'signature': signature, 'rows': rows}
            self.cache.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        h, w = image.shape[:2]
        result = []
        for i, row in enumerate(payload['rows']):
            b = [row['relative_bbox'][j]+roi[j % 2] for j in range(4)]
            b = (max(0., b[0]), max(0., b[1]), min(float(w), b[2]), min(float(h), b[3]))
            if b[2] > b[0] and b[3] > b[1]:
                result.append(Text(row['text'], b, row['score'], f'L41_model:{key}:{i}'))
        return result


def evidence_for_owner(scene, owner, tokens, library, family):
    """Original L37 evidence gates plus agreement with the source crop owner."""
    usable, events = [], []
    for token in tokens:
        matched, trace = model_matches(scene.components, [token], library)
        status = 'accepted' if owner in matched else 'other_owner_or_ambiguous' if matched else 'not_accepted'
        if status == 'accepted':
            usable.append(token)
        events.append({'component': owner, 'family': family, 'token': asdict(token),
                       'status': status, 'library_candidates': library_candidates(token.text, library),
                       'ownership_events': trace})
    return usable, events


def apply(image, scene, reader, library, templates):
    original_models = scene.diagnostics.get('L37_public_library', {}).get('models', {})
    preparations = scene.diagnostics.get('pin_l39_stage_trace', {}).get('joint', {}).get('role_preparation', [])
    local_by_owner = {r['component']: [Text(**t) for t in r.get('canonical_tokens', [])] for r in preparations}
    accepted_tokens, events, owners = [], [], []
    before_calls, before_hits = reader.calls, reader.cache_hits
    for component in scene.components:
        if component.type != 'box' or component.key.startswith('UNRESOLVED_'):
            continue
        if component.key in original_models:
            owners.append({'component': component.key, 'status': 'existing_model_preserved'})
            continue
        tokens, local_events = evidence_for_owner(scene, component.key, local_by_owner.get(component.key, []), library, 'existing_pin_local')
        events.extend(local_events)
        matches, _ = model_matches(scene.components, [*scene.texts, *tokens], library)
        # New reads are skipped only when the old local pool supplies one
        # unambiguous exact model. The same canonical pool is otherwise frozen.
        read_count = 0
        if component.key not in matches:
            readings = reader.read_owner(image, component, scene.texts)
            read_count = len(readings)
            new_tokens, new_events = evidence_for_owner(scene, component.key, readings, library, 'whole_component_OCR')
            tokens += new_tokens
            events.extend(new_events)
        accepted_tokens.extend(tokens)
        owners.append({'component': component.key, 'status': 'model_read_attempted',
                       'roi': owner_roi(component, scene.texts, image.shape), 'OCR_readings': read_count,
                       'eligible_model_tokens': len(tokens)})
    working = copy.deepcopy(scene)
    if accepted_tokens:
        working.texts = [*working.texts, *copy.deepcopy(accepted_tokens)]
        working, library_trace = library_stage(working, library)
        working.texts = copy.deepcopy(scene.texts)
        # Replaying a frozen lookup cannot change old trusted owners merely
        # because this new evidence path exists.
        old_components = {c.key: c for c in scene.components}
        for c in working.components:
            if c.key in original_models and asdict(c) != asdict(old_components[c.key]):
                raise AssertionError('Existing model owner changed before frozen structure stage')
    else:
        library_trace = copy.deepcopy(scene.diagnostics.get('L37_public_library', {}))
    effective = working.diagnostics.get('L37_public_library', {}).get('models', {})
    new_models = {key: value for key, value in effective.items() if key not in original_models}
    candidate = structure_apply(working, templates)
    candidate.diagnostics['L41_model_reading'] = {
        'policy': asdict(POLICY), 'library_policy': asdict(LIBRARY_POLICY),
        'owners': owners, 'events': events, 'new_models': new_models,
        'library_number_changes': library_trace.get('accepted_changes', []),
        'new_OCR_calls': reader.calls-before_calls, 'cache_hits': reader.cache_hits-before_hits,
        'OCR_model_sha256': reader.model_signature, 'global_OCR_changed': False,
        'component_properties_changed': False, 'terminal_geometry_changed': False, 'GT_used': False,
    }
    return candidate
