"""Finite image-only preparation and separate L24-policy evaluation of L25."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
from tools.evaluate_pin_qa_diagnostic import (DESIGN, METHODS, POLICIES, ReadGuard,
    aggregate_scenarios, digest, evaluate_pins, git_state, read)
from pcb.io import write_json
from pin_nonbox_export_l25 import POLICY, admit, old_network_projection

OUT=ROOT/'reports/pin_nonbox_export_l25'
REFERENCE=ROOT/'reports/pin_visible_semantics_l22'
SOURCES=('experiments/pin_nonbox_export_l25.py','experiments/pin_nonbox_export_l25_run.py',
         'tests/pin/test_pin_nonbox_export_l25.py','reports/pin_nonbox_export_l25/experiment_plan.md')


def verify(initial):
    for p,h in {**initial['protected_hashes'],**initial['experiment_sources']}.items():
        if digest(Path(p)) != h:
            raise AssertionError(f'Frozen source/input changed: {p}')
    state=git_state()
    if any(state[k] != initial['git'][k] for k in ('head','branch','tracked_diff_sha256')):
        raise AssertionError('Tracked Git state changed')


def initial_state():
    path=OUT/'initial_state.json'
    if path.exists():
        result=read(path);verify(result);return result
    previous=read(ROOT/'reports/pin_qa_diagnostic_l24/input_seal.json')
    hashes=copy.deepcopy(previous['protected_hashes'])
    for p in (REFERENCE/'prepared.json',ROOT/'experiments/pin_pair_score_l11_run.py',
              ROOT/'experiments/wire_frame_guard.py',ROOT/'tools/evaluate_pin_qa_diagnostic.py'):
        hashes[str(p)]=digest(p)
    for p,h in read(REFERENCE/'prepared.json')['prediction_sha256'].items():
        # Only A image-only source enters inference. Manual B_ORACLE is excluded.
        if Path(p).resolve().is_relative_to((REFERENCE/'predictions/A').resolve()):
            hashes[p]=h
    sources={str(ROOT/p):digest(ROOT/p) for p in SOURCES}
    result={'protected_hashes':hashes,'experiment_sources':sources,
            'git':git_state(),'policy':POLICY,'OFFICIAL_SCORE':False}
    verify(result);write_json(path,result);return result


def hydrate(meta, inputs):
    """Use signed frontend and complete archived image-only Pins, not oracle B."""
    from pcb.schema import Component, Pin, Scene, Text
    from pcb.io import read_image
    from pcb.core.context import PipelineContext
    from pcb.core.config import PipelineConfig
    case=meta['case_id']
    if case not in DESIGN:raise ValueError('Fixed Design only')
    front_path=ROOT.parent/f'tmp/pin_sina_frontend/{case}.json'
    raw_path=ROOT/f'reports/pin_terminal_reader_l2/design/raw_terminals/{case}.json'
    events_path=ROOT/f'reports/pin_terminal_reader_l2/design/predictions/L1/{case}/diagnostics.json'
    internal_path=REFERENCE/f'predictions/A/{case}/internal_pins.json'
    diag_path=REFERENCE/f'predictions/A/{case}/diagnostics.json'
    pred_path=REFERENCE/f'predictions/A/{case}/result.json'
    for p in (front_path,raw_path,events_path,internal_path,diag_path,pred_path):inputs[str(p)]=digest(p)
    front,raw,events,internal=(read(p) for p in (front_path,raw_path,events_path,internal_path))
    for p,h in front['signature']['frontend_source_sha256'].items():
        if digest(ROOT/p.replace('\\','/')) != h:raise AssertionError('Signed frontend source changed')
    image_path=Path(meta['target_path']).parent/meta['image_name']
    inputs[str(image_path)]=digest(image_path)
    if inputs[str(image_path)] != meta['image_sha256'] or front['signature']['image_sha256'] != meta['image_sha256']:
        raise AssertionError('Original image version changed')
    image=read_image(image_path)
    if image.shape[:2] != (meta['height'],meta['width']):raise AssertionError('Image dimensions changed')
    components=[]
    for record in front['components']:
        c=Component(**{**record,'bbox':tuple(record['bbox']),
                       'body_bbox':tuple(record['body_bbox']) if record.get('body_bbox') else None})
        c.pins=[Pin(**{**p,'tip':tuple(p['tip']),'base':tuple(p['base']) if p['base'] is not None else None})
                for p in internal[c.key]]
        rr=next(r for r in raw if r['component']==c.key)
        if rr['type'] != c.type or len(c.pins) != len(rr['terminals']):
            raise AssertionError('Archived localization / Pin count mismatch')
        for p,t in zip(c.pins,rr['terminals']):
            if p.side != t['side'] or math.dist(p.tip,t['tip'])>1e-9 or math.dist(p.base,t['base'])>1e-9:
                raise AssertionError('Original raw candidate geometry changed')
        components.append(c)
    scene=Scene(meta['width'],meta['height'],components,
                [Text(**{**t,'bbox':tuple(t['bbox'])}) for t in front['texts']],
                diagnostics=copy.deepcopy(read(diag_path)))
    context=PipelineContext(meta['image_name'],meta['width'],meta['height'],
                            PipelineConfig.load(ROOT/'configs/current.json'),None)
    return scene,events['pin_events'],raw,image,context,read(pred_path)


def generate(meta, initial):
    from pin_pair_score_l11_run import predict, json_hash
    from pcb.core.registry import build_default_registry
    from pcb.submission import validate_strict
    inputs={};scene,events,raw,image,context,saved=hydrate(meta,inputs)
    before=json_hash([asdict(scene),events,raw])
    a_scene=copy.deepcopy(scene);b_scene,trace=admit(scene,events)
    if json_hash([asdict(scene),events,raw]) != before:raise AssertionError('Input mutation')
    registry=build_default_registry()
    a,a_geo=predict(a_scene,image,context,registry)
    if a != saved:raise AssertionError(f"A replay parity failed: {meta['case_id']}")
    if a_geo != meta['geometry']:raise AssertionError('A saved geometry parity failed')
    b,b_geo=predict(b_scene,image,context,registry)
    if a_geo != b_geo:raise AssertionError('Downstream geometry changed')
    if a['components'] != b['components']:raise AssertionError('Component output changed')
    for ck,pp in a['pins'].items():
        if any(b['pins'][ck].get(pk) != p for pk,p in pp.items()):raise AssertionError('Old output Pin changed/lost')
        if a['components'][ck]['type']=='box' and pp != b['pins'][ck]:raise AssertionError('Box output changed')
    refs={f'{ck}.{pk[4:]}' for ck,pins in a['pins'].items() for pk in pins}
    if old_network_projection(a,refs) != old_network_projection(b,refs):
        raise AssertionError('Old-ref net membership or edges changed')
    files={}
    for name,pred,current in [('A',a,a_scene),('B',b,b_scene)]:
        dest=OUT/f'predictions/{name}/{meta["case_id"]}'
        validate_strict(pred,(meta['width'],meta['height']))
        for filename,value in [('result.json',pred),('diagnostics.json',current.diagnostics),
                               ('internal_pins.json',{c.key:[asdict(p) for p in c.pins] for c in current.components})]:
            p=dest/filename;write_json(p,value);files[str(p)]=digest(p)
    record={'case_id':meta['case_id'],'meta':meta,'trace':trace,'input_sha256':inputs,
            'prediction_sha256':files,'A_full_prediction_equal':True,'candidate_geometry_invariant':True,
            'downstream_geometry_equal':True,'old_ref_subgraph_equal':True,'geometry':a_geo,
            'admitted':sum(r['admitted'] for r in trace),
            'rejected_reasons':dict(Counter(r['reason'] for r in trace if not r['admitted']))}
    verify(initial)
    for p,h in inputs.items():
        if digest(Path(p)) != h:raise AssertionError('Frozen image/cache changed during prepare')
    write_json(OUT/f'prepared_cases/{meta["case_id"]}.json',record)
    print(json.dumps({k:record[k] for k in ('case_id','admitted','rejected_reasons','A_full_prediction_equal','downstream_geometry_equal')},ensure_ascii=False),flush=True)
    return record


def prepare(smoke=False):
    started=time.perf_counter();guard=ReadGuard();initial=initial_state()
    if (OUT/'prepared.json').exists():raise RuntimeError('Already sealed; no repeated Design search')
    metadata=read(REFERENCE/'prepared.json')['cases']
    if tuple(m['case_id'] for m in metadata) != DESIGN:raise AssertionError('Design manifest changed')
    records=[]
    for meta in metadata[:1] if smoke else metadata:
        path=OUT/f'prepared_cases/{meta["case_id"]}.json'
        if path.exists():
            row=read(path)
            for p,h in {**row['input_sha256'],**row['prediction_sha256']}.items():
                if digest(Path(p)) != h:raise AssertionError('Saved smoke/prepare changed')
            if row['meta'] != meta:raise AssertionError('Manifest changed')
        else:row=generate(meta,initial)
        records.append(row)
    if guard.target_opens or guard.blocked:raise AssertionError('Unexpected target/forbidden read')
    payload={'cases':records,'all_predictions_saved_before_target_reads':True,
             'target_reads':0,'new_OCR_YOLO_VLM_solver_localization_calls':0,
             'seconds':time.perf_counter()-started,'OFFICIAL_SCORE':False}
    write_json(OUT/('smoke.json' if smoke else 'prepared.json'),payload)
    print(json.dumps({'phase':'smoke' if smoke else 'prepare','cases':len(records),
                      'target_reads':0,'seconds':payload['seconds']},ensure_ascii=False),flush=True)


def predicted_refs(pred):
    return {f'{ck}.{pk[4:]}' for ck,pp in pred['pins'].items() for pk in pp}


def mapped_pair_audit(a,b,target,qa_a,qa_b):
    """Evaluator-only identity assumption; never change predictions or infer graph."""
    from evaluate_v2 import net_members, pair_set
    def mapping(row):
        return {f"{m['pred'][0]}.{m['pred'][1][4:]}":f"{m['gt'][0]}.{m['gt'][1][4:]}" for m in row['matches']}
    def pairs(pred,identities):
        return pair_set([[identities[r] for r in net_members(n) if r in identities] for n in pred['nets'].values()])
    am,bm=mapping(qa_a),mapping(qa_b)
    ap,bp=pairs(a,am),pairs(b,bm)
    truth=pair_set([net_members(n) for n in target['nets'].values()])
    added=predicted_refs(b)-predicted_refs(a)
    unassessed=set()
    for n in b['nets'].values():
        for pair in itertools.combinations(sorted(net_members(n)),2):
            if added.intersection(pair) and not all(r in bm for r in pair):unassessed.add(pair)
    return {'official_network_score':False,'mapping_used_only_in_evaluator':True,
            'correct_gained':sorted((bp&truth)-(ap&truth)),
            'correct_lost':sorted((ap&truth)-(bp&truth)),
            'incorrect_added':sorted((bp-truth)-(ap-truth)),
            'incorrect_removed':sorted((ap-truth)-(bp-truth)),
            'new_ref_pairs_unassessed':sorted(unassessed),
            'new_refs_with_legal_pin_match':sorted(added.intersection(bm))}


def evaluate():
    from pin_g_wire_e1 import pair_changes
    from pcb.coordinates import opencv_to_target
    from pcb.submission import validate_strict
    started=time.perf_counter();guard=ReadGuard();initial=read(OUT/'initial_state.json');verify(initial)
    if (OUT/'complete.json').exists():raise RuntimeError('One finite Design evaluation only')
    prepared=read(OUT/'prepared.json');preds={'A':{},'B':{}}
    if tuple(r['case_id'] for r in prepared['cases']) != DESIGN:raise AssertionError('Five cases not sealed')
    for row in prepared['cases']:
        for p,h in {**row['prediction_sha256'],**row['input_sha256']}.items():
            if digest(Path(p)) != h:raise AssertionError('Sealed prediction/input changed')
        for v in preds:
            preds[v][row['case_id']]=read(OUT/f'predictions/{v}/{row["case_id"]}/result.json')
            validate_strict(preds[v][row['case_id']],(row['meta']['width'],row['meta']['height']))
    if guard.target_opens:raise AssertionError('Predictions must load before targets')
    targets={}
    for row in prepared['cases']:
        p=Path(row['meta']['target_path'])
        with guard.evaluating(p):raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['meta']['target_sha256']:
            raise AssertionError('Official target version drift; stop for version diff')
        targets[row['case_id']]=json.loads(raw.decode('utf-8-sig'))
    old=read(ROOT/'reports/pin_qa_diagnostic_l24/complete.json')
    scenarios={};details=[];admitted_rows=[];legacy_pairs={};network_audits={}
    for policy in POLICIES:
        for method in METHODS if policy!='legacy_strict' else METHODS[:1]:
            label=policy if method==METHODS[0] else policy+'_min_distance_sensitivity'
            grouped={v:[] for v in preds};network_audits[label]=[]
            for row in prepared['cases']:
                case=row['case_id'];a,b=(preds[v][case] for v in ('A','B'));target=targets[case]
                qa={v:evaluate_pins(preds[v][case],target,policy,method) for v in preds}
                for v,d in qa.items():d['case_id']=case;grouped[v].append(d)
                for scope in ('overall','box','nonbox'):
                    if qa['A']['box'] != qa['B']['box']:raise AssertionError('Box metric changed')
                old_gt={tuple(m['gt']) for m in qa['A']['matches']}
                new_gt={tuple(m['gt']) for m in qa['B']['matches']}
                if old_gt-new_gt:raise AssertionError('Old correct Pin correspondence lost')
                if policy!='legacy_strict':
                    network_audits[label].append({'case_id':case,**mapped_pair_audit(a,b,target,qa['A'],qa['B'])})
                if label=='qa_fields_only':
                    by_pred={tuple(m['pred']):m for m in qa['B']['matches']}
                    for item in row['trace']:
                        if not item['admitted']:continue
                        owner=item['owner'];pin=item['new_pin'];gk=qa['B']['component_key_map'].get(owner)
                        xy=opencv_to_target(*pin['tip'],row['meta']['height'])
                        candidates=target['pins'].get(gk,{}) if gk else {}
                        nearest=min(((math.dist(xy,(p['point']['x'],p['point']['y'])),pk,p) for pk,p in candidates.items()),default=None,key=lambda x:x[0])
                        matched=by_pred.get((owner,'pin_'+pin['number']))
                        reason=('OWNER_NOT_MAPPED' if not gk else 'NO_GT_PIN_ON_OWNER' if not nearest else
                                'CORRECT_5PX_ONE_TO_ONE' if matched else
                                'NEAR_5PX_BUT_NOT_ASSIGNED' if nearest[0]<=5 else
                                'OFFSET_5_TO_10PX' if nearest[0]<=10 else
                                'OFFSET_10_TO_20PX' if nearest[0]<=20 else 'FARTHER_THAN_20PX')
                        base_xy=opencv_to_target(*pin['base'],row['meta']['height']) if pin['base'] else None
                        base_error=math.dist(base_xy,(nearest[2]['point']['x'],nearest[2]['point']['y'])) if nearest and base_xy else None
                        admitted_rows.append({'case_id':case,'owner':owner,'type':item['type'],'terminal':item['terminal'],
                            'local_number':pin['number'],'point':xy,'GT_owner':gk,'bucket':reason,
                            'nearest_distance_unconstrained':nearest[0] if nearest else None,
                            'nearest_GT_pin':nearest[1] if nearest else None,'GT_base_distance_diagnostic_only':base_error,
                            'match':matched})
                if label=='legacy_strict':legacy_pairs[case]=pair_changes(a,b,target)
            summary={v:aggregate_scenarios(r) for v,r in grouped.items()}
            for scope in ('overall','box','nonbox'):
                if summary['A'][scope] != old['aggregate']['L13_B'][label][scope]:
                    raise AssertionError('Historical L24 baseline metric mismatch')
            scenarios[label]=summary;details.append({'scenario':label,'variants':grouped})
    principal=scenarios['qa_fields_only'];a,b=(principal[v]['nonbox'] for v in ('A','B'))
    primary_details=next(r for r in details if r['scenario']=='qa_fields_only')['variants']
    improved=[x['case_id'] for x,y in zip(primary_details['A'],primary_details['B']) if y['nonbox']['f1']>x['nonbox']['f1']]
    audits=[row for rows in network_audits.values() for row in rows]
    guards={'nonbox_TP_and_micro_F1_net_gain':b['tp']>a['tp'] and b['f1']>a['f1'],
            'nonbox_relative_gain_20pct':b['tp']>=1.2*a['tp'] or b['f1']>=1.2*a['f1'],
            'at_least_3_nonbox_cases_improved':len(improved)>=3,
            'nonbox_macro_not_lower':b['macro_f1']>=a['macro_f1'],
            'box_and_old_Pin_correspondence_no_loss':True,
            'old_ref_subgraph_and_downstream_geometry_equal':True,
            'no_new_verifiable_wrong_Pair_under_assumptions':not any(r['incorrect_added'] for r in audits),
            'no_verifiable_correct_Pair_loss_under_assumptions':not any(r['correct_lost'] for r in audits)}
    if guard.blocked:raise AssertionError('Forbidden read attempted')
    verify(initial)
    payload={'stage':'L25 nonbox export admission','aggregate':scenarios,'details':details,
        'admitted_pin_attribution':admitted_rows,'admitted_buckets':dict(Counter(r['bucket'] for r in admitted_rows)),
        'legacy_raw_reference_Pair_changes':legacy_pairs,'evaluator_only_mapped_Pair_audits':network_audits,
        'guards':guards,'mechanism_and_safety_passed':all(guards.values()),
        'enter_check':False,'adopt_current':False,'official_reply_pending':True,
        'improved_nonbox_cases':improved,'OFFICIAL_SCORE':False,'FinalScore':'NOT_COMPUTED',
        'SEALED_HOLDOUT_USED_FOR_DEVELOPMENT':False,'Check_run':False,'full150_run':False,
        'new_OCR_YOLO_VLM_solver_localization_calls':0,'inference_target_reads':0,
        'evaluation_target_reads':len(guard.target_opens),'blocked_access_attempts':guard.blocked,
        'candidate_geometry_and_box_outputs_unchanged':True,'seconds':time.perf_counter()-started,
        'source_and_input_hashes_unchanged':True}
    write_json(OUT/'complete.json',payload)
    print(json.dumps({'L25_complete':True,'nonbox_A':a,'nonbox_B':b,
        'overall_A':principal['A']['overall'],'overall_B':principal['B']['overall'],
        'admitted_buckets':payload['admitted_buckets'],'guards':guards,'target_reads':len(guard.target_opens),
        'seconds':payload['seconds']},ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['smoke','prepare','evaluate'])
    args=p.parse_args()
    if args.phase=='evaluate':evaluate()
    else:prepare(smoke=args.phase=='smoke')


if __name__=='__main__':main()
