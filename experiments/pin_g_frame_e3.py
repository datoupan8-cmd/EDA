"""Frozen-G E3: frame barrier versus stub restoration, independent Wire factors.

Every Pin/Component field is fixed. Four variants run unchanged Topology; targets
are opened only afterwards for evaluation. No production registry/config edits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from pin_g_wire_e1 import (
    verify_formal_state, AccessGuard, load_cases, read_json, write_json, digest,
    PipelineConfig, PipelineContext, signed_frontend, reconstruct_scene,
    saved_paths, pair_changes, normalized_pairs, build_default_registry, git_state,
)
from pin_skeleton_followup import SETS, strict_ids
from wire_stub_preservation import restore_supported_stubs
from wire_frame_guard import suppress_box_frames, POLICY
from pcb.io import read_image, write_image
from pcb.submission import validate_strict
from pcb.wire import trace_paths
from pin_tip_repair import shortest_path
from evaluate_v2 import evaluate_case, aggregate, source_from_name
import cv2
import numpy as np

OUT = ROOT / "reports/pin_g_frame_e3"
VARIANTS = ("B", "P", "F", "PF")
LABELS = {"B": "G + original Wire", "P": "G + E1 stub preservation",
          "F": "G + frame guard only", "PF": "G + E1 stubs + final frame guard"}


def freeze_policy():
    record = {"frame_policy": asdict(POLICY),
              "source_sha256": {f: digest(ROOT / f) for f in
                ("experiments/wire_frame_guard.py", "experiments/wire_stub_preservation.py", "experiments/pin_g_frame_e3.py")},
              "GT_used_for_policy": False, "no_threshold_grid_search": True}
    path = OUT / "frozen_policy.json"
    if path.exists():
        if read_json(path) != record:
            raise AssertionError("Policy/source changed after freeze; do not retune on Check")
    else:
        write_json(path, record)
    return record


def local_overlay(image, wires, scene, decisions, path):
    """Pick the largest removed-frame example from image evidence, never GT."""
    removed = Counter()
    for d in decisions:
        removed[d["component"]] += d.get("removed_pixel_proposals", 0)
    if not removed or max(removed.values()) == 0:
        return
    owner = removed.most_common(1)[0][0]
    c = next(c for c in scene.components if c.key == owner)
    x1, y1, x2, y2 = c.body_bbox or c.bbox
    a, b = max(0, round(x1 - 100)), max(0, round(y1 - 20))
    d, e = min(scene.width, round(x2 + 100)), min(scene.height, round(y2 + 20))
    panels = [image[b:e, a:d].copy()]
    panels += [cv2.cvtColor(wires[v].mask[b:e, a:d], cv2.COLOR_GRAY2BGR) for v in VARIANTS]
    overlay = image[b:e, a:d].copy()
    overlay[(wires['B'].mask[b:e, a:d] > 0) & (wires['F'].mask[b:e, a:d] == 0)] = (0, 0, 255)
    panels.append(overlay)
    labels = (owner + " image", "B original", "P stubs", "F frame", "PF stubs/frame", "Red: frame removals")
    for i, panel in enumerate(panels):
        panel = cv2.copyMakeBorder(panel, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(panel, labels[i], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
        panels[i] = panel
    write_image(path, np.concatenate([np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)], axis=0))


def boundary_route(scene, wire, diagnostics):
    """Previously discovered 0014 pair, diagnostic only; no inference exception."""
    refs = ("U1.18", "U1.24")
    snaps = {s['pin']: s for s in diagnostics.get('snaps', [])}
    c = next((c for c in scene.components if c.key == 'U1'), None)
    if c is None or not all(r in snaps for r in refs):
        return {"route_exists": False, "both_snapped": all(r in snaps for r in refs)}
    paths, _, adj = trace_paths(wire.mask)
    ends = [min(paths[snaps[r]['path']], key=lambda p: (p[1]-snaps[r]['to'][0])**2 + (p[0]-snaps[r]['to'][1])**2) for r in refs]
    route = shortest_path(adj, *ends)
    boundary = round((c.body_bbox or c.bbox)[0])
    return {"route_exists": bool(route), "both_snapped": True, "route_pixels": len(route),
            "route_pixels_near_left_body_2px": int(sum(abs(p[1]-boundary) <= 2 for p in route))}


def run(subset):
    started = time.perf_counter()
    formal = verify_formal_state()
    before_git = git_state()
    frozen = freeze_policy()
    if subset == 'check':
        design = read_json(OUT / 'design/latest.json')
        if not design['guards']['all_new_variants_pass']:
            raise AssertionError('Design failed; no Check/150 run allowed')
        if design['metadata']['frozen_policy'] != frozen:
            raise AssertionError('Policy changed after Design')
    guard = AccessGuard()
    cases = load_cases()
    config = PipelineConfig.load(ROOT / 'configs/current.json')
    model_sha = digest(ROOT / 'models/component_yolo11n_continue_v2_best.pt')
    expected = {r['case_id']: r['target_sha256'] for r in read_json(ROOT/'reports/pin_g_wire_e1/e0/latest.json')['cases']}
    keys = ('0011', '0014') if subset == 'smoke' else SETS[subset]
    folder = OUT / subset / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    registry = build_default_registry()
    records, inputs = [], {'configs/current.json': digest(ROOT/'configs/current.json')}
    evaluations = {v: [] for v in VARIANTS}
    total_changes = {v: Counter() for v in ('F_vs_B', 'PF_vs_B', 'PF_vs_P', 'PF_vs_F')}
    for key in keys:
        case = cases[key]
        image = read_image(case.image_path)
        h, w = image.shape[:2]
        source_subset = 'design' if key in SETS['design'] else 'check'
        source = saved_paths(source_subset, key)['G']
        raw = read_json(source/'raw_terminals.json')
        reference = read_json(source/'result.json')
        for filename in ('result.json', 'diagnostics.json', 'raw_terminals.json'):
            inputs[str(source/filename)] = digest(source/filename)
        front = signed_frontend(case, config, model_sha)
        context = PipelineContext(case.image_path.name, w, h, config, None)
        scene, _ = reconstruct_scene(front, raw, image, context)
        structures = copy.deepcopy(scene.components)
        scenes = {'B': scene}
        wires = {'B': registry.create('wire', 'v3').run(image, scene, context)}
        scenes['P'] = copy.deepcopy(scene)
        wires['P'] = restore_supported_stubs(image, scenes['P'], wires['B'])
        scenes['F'] = copy.deepcopy(scene)
        wires['F'] = suppress_box_frames(image, scenes['F'], wires['B'])
        scenes['PF'] = copy.deepcopy(scenes['P'])
        wires['PF'] = suppress_box_frames(image, scenes['PF'], wires['P'])
        predictions, diagnostics = {}, {}
        for v in VARIANTS:
            sc, wire = scenes[v], wires[v]
            if sc.components != structures:
                raise AssertionError('Wire strategy modified G Component/Pin domain objects')
            if not np.array_equal(wire.color_debug, wires['B'].color_debug):
                raise AssertionError('Palette changed')
            if v in ('F', 'PF'):
                parent = 'B' if v == 'F' else 'P'
                if ((wire.mask > 0) & (wires[parent].mask == 0)).any():
                    raise AssertionError('Frame policy added pixels')
            same = next((u for u in predictions if np.array_equal(wire.mask, wires[u].mask)), None)
            if same:
                predictions[v] = copy.deepcopy(predictions[same])
                diagnostics[v] = copy.deepcopy(diagnostics[same])
                diagnostics[v].update({k:d for k,d in sc.diagnostics.items() if k.startswith('wire_frame_guard') or k.startswith('wire_stub_preservation')})
            else:
                registry.create('topology', 'v3').run(sc, wire, context)
                predictions[v] = registry.create('submission', 'official').run(sc, context).data
                diagnostics[v] = sc.diagnostics
            validate_strict(predictions[v], (w, h))
            if predictions[v]['components'] != reference['components'] or predictions[v]['pins'] != reference['pins']:
                raise AssertionError('Frozen G fields changed in submission')
            if v != 'B':
                write_json(folder/'predictions'/v/key/'result.json', predictions[v])
                write_json(folder/'predictions'/v/key/'diagnostics.json', diagnostics[v])
        if predictions['B'] != reference:
            raise AssertionError('Original G full JSON replay differs')
        # All four masks/predictions exist before evaluator-only target access.
        with guard.evaluator():
            if digest(case.target_path) != expected[key]:
                raise AssertionError('Target version drift from frozen E0')
            target = read_json(case.target_path)
            truth_ids = strict_ids(reference, target)
            evs = {v: evaluate_case(predictions[v], target, diagnostics[v]) for v in VARIANTS}
            for v in VARIANTS:
                if strict_ids(predictions[v], target) != truth_ids:
                    raise AssertionError('Strict Pin TP set changed')
                evaluations[v].append({'case_id': key, 'source': source_from_name(case.image_path.name), 'status':'ok', 'evaluation':evs[v]})
            changes = {name: pair_changes(predictions[a], predictions[b], target) for name,a,b in
                (('F_vs_B','B','F'), ('PF_vs_B','B','PF'), ('PF_vs_P','P','PF'), ('PF_vs_F','F','PF'))}
            if key == '0011':
                recovery = {v: ('U2.4','U2.9') in normalized_pairs(predictions[v], target) for v in VARIANTS}
            else:
                recovery = None
        for name, data in changes.items():
            total_changes[name].update({k:len(v) for k,v in data.items()})
        record = {'case_id':key, 'source':source_from_name(case.image_path.name), 'evaluation':evs,
                  'pair_changes':changes, 'known_0011_pair_present':recovery,
                  'frame':{v:scenes[v].diagnostics['wire_frame_guard'] for v in ('F','PF')},
                  'G_full_JSON_replay':True, 'G_component_pin_identical':True,
                  'frame_deleted_only':True}
        if key == '0014':
            record['known_boundary_route'] = {v:boundary_route(scenes[v],wires[v],diagnostics[v]) for v in VARIANTS}
        records.append(record)
        if subset == 'smoke':
            local_overlay(image, wires, scene, record['frame']['F']['decisions'], folder/f'{key}_frame_overlay.png')
        print(json.dumps({'E3_case':key, 'frame_pixels_removed':record['frame']['F']['removed_pixels'],
             'Pin_TP':evs['F']['metrics']['Pin']['tp'],
             'pair_changes':{name:{k:len(v) for k,v in d.items()} for name,d in changes.items()}}), flush=True)
    guards = {}
    for name in ('F_vs_B','PF_vs_B','PF_vs_F'):
        guards[name] = {'no_correct_pair_loss':total_changes[name]['correct_lost']==0,
                        'no_new_incorrect_pairs':total_changes[name]['incorrect_added']==0}
    guards['all_new_variants_pass'] = all(all(values.values()) for values in guards.values())
    for filename, sha in inputs.items():
        path = Path(filename)
        if digest(path if path.is_absolute() else ROOT/path) != sha:
            raise AssertionError('Frozen input changed')
    payload = {'metadata':{'OFFICIAL_SCORE':False, 'subset':subset, 'case_ids':list(keys),
                 'case_count':len(keys), 'run_directory':str(folder), 'frozen_policy':frozen,
                 'formal_state':verify_formal_state(), 'input_sha256':inputs, 'git_before':before_git,
                 'dataset_root':str(cases[keys[0]].image_path.parents[1]), 'variant_labels':LABELS,
                 'Pin_frozen_at_G':True, 'Topology_frozen':True, 'models_or_OCR_executed':False,
                 'target_in_inference_attempts':guard.target_attempts, 'sealed_attempts':guard.sealed_attempts,
                 'sealed_holdout_used':False, 'full_150_executed':False,
                 'check_is_previously_used_development_regression_set':True, 'seconds':time.perf_counter()-started},
               'end_to_end':{v:aggregate(evaluations[v]) for v in VARIANTS},
               'pair_change_counts':{k:dict(v) for k,v in total_changes.items()}, 'guards':guards,
               'strict_schema_pass_count':len(keys)*4, 'cases':records}
    write_json(folder/'report.json', payload)
    write_json(OUT/subset/'latest.json', payload)
    print(json.dumps({'E3_finished':subset, 'guards':guards, 'report':str(folder/'report.json')}), flush=True)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('smoke','design','check'), required=True)
    run(parser.parse_args().task)
