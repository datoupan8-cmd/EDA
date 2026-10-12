"""Independent public identity assets; frozen L37/L41/L40 algorithms."""
from __future__ import annotations
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex

from .frozen_l39.library_semantics import library_candidates, stage as library_stage
from .model_reader_l41 import apply as model_apply
from .structural_l40 import load_templates

ASSET_ROOT = Path(__file__).resolve().parents[2]/'models/pin/l42_public_identity'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_legacy(text, source):
    """Single-unit KiCad legacy symbols, with original y-up converted once."""
    accepted, rejected = {}, []
    block, start = [], 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.startswith('DEF '):
            block, start = [line], line_number
        elif block:
            block.append(line)
            if line != 'ENDDEF':
                continue
            try:
                rows = [shlex.split(r) for r in block if r and not r.startswith('#')]
                header = rows[0]
                name = header[1]
                if int(header[7]) != 1:
                    raise ValueError('multiunit_not_supported')
                rectangles = [r for r in rows if r[0]=='S']
                if len(rectangles) != 1:
                    raise ValueError('single_rectangle_required')
                r = rectangles[0]
                if int(r[5]) not in (0, 1) or int(r[6]) != 1:
                    raise ValueError('unsupported_body_unit_or_conversion')
                x1, y1, x2, y2 = map(float, r[1:5])
                body = [min(x1,x2), min(-y1,-y2), max(x1,x2), max(-y1,-y2)]
                if body[2] <= body[0] or body[3] <= body[1]:
                    raise ValueError('invalid_body')
                pins, visible = [], []
                for r in rows:
                    if r[0] != 'X':
                        continue
                    if int(r[9]) not in (0,1) or int(r[10]) != 1:
                        raise ValueError('unsupported_pin_unit_or_conversion')
                    pin_name, number = r[1], r[2]
                    if pin_name in ('', '~') or not number:
                        raise ValueError('unnamed_pin')
                    x,y,length = map(float, r[3:6])
                    if r[6] not in ('L','R','U','D') or length < 0:
                        raise ValueError('invalid_pin_geometry')
                    pin = {'name': pin_name, 'number': number}
                    pins.append(pin)
                    hidden = any('N' in flag for flag in r[12:])
                    if not hidden:
                        visible.append({**pin, 'point':[x,-y],
                                        'uv':[(x-body[0])/(body[2]-body[0]),(-y-body[1])/(body[3]-body[1])]})
                if len(pins)<2 or len({p['number'] for p in pins}) != len(pins):
                    raise ValueError('duplicate_number_or_incomplete_symbol')
                aliases = [name] + [alias for r in rows if r[0]=='ALIAS' for alias in r[1:]]
                footprint = next((r[1] for r in rows if r[0]=='F2'), '')
                base = {'model':name, 'pins':pins, 'source':source['url'], 'source_sha256':source['sha256'],
                        'source_symbol':name, 'line_start':start, 'footprint':footprint,
                        'template':{'model':name,'source':source['url'],'source_sha256':source['sha256'],
                                    'body':body,'pins':visible} if len(visible)>=2 else None}
                entries = {}
                for alias in aliases:
                    if any(s in alias.lower() for s in ('xxx','xx','?','*')):
                        continue  # Placeholder strings are not exact MPNs.
                    if alias in accepted or alias in entries:
                        raise ValueError('duplicate_model_alias')
                    entries[alias] = {**copy.deepcopy(base),'model':alias}
                    if entries[alias]['template']:
                        entries[alias]['template']['model'] = alias
                accepted.update(entries)
            except (ValueError, IndexError) as error:
                rejected.append({'symbol':block[0].split()[1], 'line_start':start,'reason':str(error)})
            block = []
    if block:
        raise ValueError('truncated_legacy_library')
    return accepted, rejected


def load_assets(original_library, expanded=True, directory=ASSET_ROOT):
    library = copy.deepcopy(original_library)
    templates, previous = load_templates(library=original_library)
    if not expanded:
        return library, templates, {'expanded':False,'old_assets':previous}
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text('utf-8'))
    for record in manifest['files']:
        path = (directory/record['file']).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != record['sha256']:
            raise ValueError('L42 asset path/hash mismatch')
    catalog = json.loads((directory/'catalog.json').read_text('utf-8'))
    added, skipped = [], []
    for model, entry in catalog['models'].items():
        if library_candidates(model, original_library):
            skipped.append(model)
            continue  # Never replace/shadow the existing public knowledge.
        library[model] = {'pins':copy.deepcopy(entry['pins']), 'source':entry['source'],
                          'source_sha256':entry['source_sha256']}
        if entry.get('template'):
            templates[model] = copy.deepcopy(entry['template'])
        added.append(model)
    return library, templates, {'expanded':True,'added_models':added,'existing_models_not_shadowed':skipped,
                               'manifest_sha256':digest(directory/'manifest.json'),'old_assets':previous}


def apply(image, scene, reader, library, templates):
    """Same asset entry point for A/B, before the frozen image model adapter."""
    original_models = copy.deepcopy(scene.diagnostics.get('L37_public_library',{}).get('models',{}))
    working, trace = library_stage(scene, library)
    originals = {c.key:c for c in scene.components}
    # Existing trusted identities remain frozen even if new asset model names
    # occur nearby. Only previously unsupported owners may use new knowledge.
    for c in working.components:
        if c.key in original_models:
            c.pins = copy.deepcopy(originals[c.key].pins)
    trace['models'].update(original_models)
    trace['accepted_changes'] = [r for r in trace['accepted_changes'] if r['component'] not in original_models]
    working.diagnostics['L37_public_library'] = trace
    result = model_apply(image, working, reader, library, templates)
    result.diagnostics['L42_assets'] = {
        'global_model_owners_added':sorted(set(trace['models'])-set(original_models)),
        'public_lookup_changes':trace['accepted_changes'],
        'old_model_owners_preserved':sorted(original_models),
        'GT_used':False,'global_OCR_modified':False,'terminal_geometry_modified':False,
    }
    return result
