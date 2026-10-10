"""Agent-run follow-up: saved-result conversion audit and frozen pin ablations.

No model/OCR execution; only signed image-only frontend snapshots are used.
All dataset access is limited to fixed Design/Check subsets of 0001..0150.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from evaluate_v2 import component_key_map, pin_counts, evaluate_case, aggregate
from pcb.coordinates import target_to_opencv
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.interfaces import ComponentStageOutput, PinLocalizationOutput, PinSemanticsOutput
from pcb.core.registry import build_default_registry
from pcb.io import read_image, write_json
from pcb.schema import Component, Text
from pcb.submission import validate_strict
from pcb.text_detection import TokenRole
from pcb.pin.localization.tip_placement_v4 import place_boundary_tip
from tools.run_box_skeleton_experiment import AccessGuard, digest, full_inference, empty_counts, add_counts, metrics
from tools.run_box_skeleton_experiment import component_payload
from tools.evaluate_box_text_mask_counterfactual import _strict_box_pairs, _gt_pins
from local_experiments.pin_skeleton_recovery.runner import (
    SETS, read_json, selected_cases, constrained_matches, verify_formal_state, localization,
)
from local_experiments.pin_skeleton_recovery.candidates import RecoveryConfig, RecoveryStage

OUT = ROOT/'reports/pin_skeleton_followup'
MANUAL = ROOT/'local_experiments/pin_skeleton_recovery'
NORMALS = {'left':(-1,0),'right':(1,0),'top':(0,-1),'bottom':(0,1)}


def strict_ids(pred,gt):
    """Same key/name/5px tests as pin_counts; verify count rather than assume parity."""
    mapping=component_key_map(pred['components'],gt['components'])
    found=set()
    for owner,pins in pred['pins'].items():
        gowner=mapping.get(owner)
        for key,pin in pins.items():
            ref=gt['pins'].get(gowner,{}).get(key)
            if ref and pin['pinname']==ref['pinname'] and math.dist(
                    (pin['point']['x'],pin['point']['y']),(ref['point']['x'],ref['point']['y']))<=5:
                found.add((gowner,key))
    expected=pin_counts(pred['pins'],gt['pins'],mapping)[0]
    if len(found)!=expected:
        raise AssertionError('TP set/count discrepancy')
    return found


def event_lookup(raw,diagnostics):
    groups=defaultdict(list)
    for event in diagnostics.get('pin_events',[]):
        groups[event['component']].append(event)
    for block in raw:
        if len(groups[block['component']]) != len(block['terminals']):
            raise AssertionError(f"Raw terminal/event count mismatch: {block['component']}")
        for terminal,event in zip(block['terminals'],groups[block['component']]):
            if math.dist(terminal['tip'],event['tip'])>1e-7 or terminal['side']!=event['side']:
                raise AssertionError('Event/terminal geometry mismatch')
    return groups


def candidate_reason(terminal,event,gt_id,gt,owner,gowner,pred,height,distance):
    """One primary bucket per appended candidate; field failures also retained."""
    result={'matched_gt_pin':gt_id,'distance':distance,'number':event.get('number'),
            'pinname':event.get('pinname'),'exportable':event.get('exportable',True)}
    if gt_id is None:
        return {**result,'bucket':'NO_OWNER_SIDE_MATCH_20','blockers':[]}
    key=gt_id[len(gowner)+1:]
    reference=gt['pins'][gowner][key]
    expected_key='pin_'+str(event.get('number',''))
    exported=pred['pins'][owner].get(expected_key) if result['exportable'] else None
    key_match=key==expected_key
    name_match=reference['pinname']==event.get('pinname','')
    point_error=math.dist(
        tuple(target_to_opencv(exported['point']['x'],exported['point']['y'],height)),
        tuple(target_to_opencv(reference['point']['x'],reference['point']['y'],height))) if exported else None
    blockers=[]
    if distance>5: blockers.append('RAW_POINT_GT_5PX')
    if not exported: blockers.append('EXPORT_FILTERED')
    if not key_match: blockers.append('PIN_KEY_MISMATCH')
    if not name_match: blockers.append('PINNAME_MISMATCH')
    if point_error is not None and point_error>5: blockers.append('FINAL_POINT_GT_5PX')
    bucket=('NEAR_5_TO_20' if distance>5 else
            'EXACT5_EXPORT_FILTERED' if not exported else
            'EXACT5_KEY_MISMATCH' if not key_match else
            'EXACT5_PINNAME_MISMATCH' if not name_match else
            'EXACT5_FINAL_POINT_MISMATCH' if point_error>5 else 'EXACT5_STRICT_TP')
    dx=terminal['tip'][0]-target_to_opencv(reference['point']['x'],reference['point']['y'],height)[0]
    dy=terminal['tip'][1]-target_to_opencv(reference['point']['x'],reference['point']['y'],height)[1]
    nx,ny=NORMALS[terminal['side']]
    return {**result,'bucket':bucket,'blockers':blockers,'gt_key':key,'gt_pinname':reference['pinname'],
            'key_match':key_match,'pinname_match':name_match,'final_point_error':point_error,
            'normal_error':dx*nx+dy*ny,'tangent_error':dy if nx else dx,
            'observability':'REVIEW_REQUIRED'}


def audit():
    state=verify_formal_state()
    guard=AccessGuard()
    saved=read_json(MANUAL/'outputs/design/latest.json')
    if saved['metadata']['case_ids'] != SETS['design']:
        raise AssertionError('Audit requires the complete fixed Design set')
    run=Path(saved['metadata']['run_directory'])
    dataset=Path(saved['metadata']['dataset_root'])
    totals={v:Counter() for v in ('B','C')}
    rows=[];changes={v:{'gained':[],'lost':[]} for v in ('B','C')}
    all_ids={v:set() for v in ('A','B','C')}
    per_case=[]
    for case in selected_cases(dataset,'design'):
        key=f'{case.case_id:04d}'
        old=next(r for r in saved['cases'] if r['case_id']==key)
        if digest(case.image_path)!=old['image_sha256']:
            raise AssertionError('Dataset image changed since Design')
        height=read_image(case.image_path).shape[0]
        preds={v:read_json(run/'predictions'/v/key/'result.json') for v in ('A','B','C')}
        raw={v:read_json(run/'predictions'/v/key/'raw_terminals.json') for v in ('A','B','C')}
        diags={v:read_json(run/'predictions'/v/key/'diagnostics.json') for v in ('A','B','C')}
        events={v:event_lookup(raw[v],diags[v]) for v in ('A','B','C')}
        with guard.evaluator():
            if digest(case.target_path)!=old['target_sha256']:
                raise AssertionError('Target changed since Design')
            gt=read_json(case.target_path)
            ids={v:strict_ids(preds[v],gt) for v in ('A','B','C')}
            for v in ids: all_ids[v].update((key,*p) for p in ids[v])
            for v in ('B','C'):
                gained=[(key,*p) for p in sorted(ids[v]-ids['A'])]
                lost=[(key,*p) for p in sorted(ids['A']-ids[v])]
                changes[v]['gained'].extend(gained);changes[v]['lost'].extend(lost)
                pairs=dict(_strict_box_pairs(preds[v]['components'],gt['components']))
                for original,block in zip(raw['A'],raw[v]):
                    if original['component']!=block['component'] or block['terminals'][:len(original['terminals'])]!=original['terminals']:
                        raise AssertionError('Saved V3 prefix not retained')
                    if block['type']!='box': continue
                    owner=block['component'];gowner=pairs.get(owner)
                    gt_points=_gt_pins(gt,gowner,height) if gowner else []
                    # Matching is geometry-only; semantics cannot help choose GT.
                    m5=constrained_matches(gt_points,block['terminals'],5)
                    m20=constrained_matches(gt_points,block['terminals'],20)
                    by_index5={index:(pinid,dist) for pinid,(index,dist) in m5.items()}
                    by_index20={index:(pinid,dist) for pinid,(index,dist) in m20.items()}
                    for index in range(len(original['terminals']),len(block['terminals'])):
                        terminal=block['terminals'][index];event=events[v][owner][index]
                        matched=by_index5.get(index,by_index20.get(index,(None,None)))
                        result=(candidate_reason(terminal,event,*matched[:1],gt,owner,gowner,preds[v],height,matched[1])
                                if gowner else {'bucket':'COMPONENT_NOT_STRICT_MATCHED','blockers':[]})
                        record={'case_id':key,'variant':v,'component':owner,'candidate_index':index,
                                'tip':terminal['tip'],'base':terminal['base'],'side':terminal['side'],
                                'method':terminal['method'],**result}
                        rows.append(record);totals[v][record['bucket']]+=1
            per_case.append({'case_id':key,'strict_tp':{v:len(i) for v,i in ids.items()},
                             'gained':{v:len(ids[v]-ids['A']) for v in ('B','C')},
                             'lost':{v:len(ids['A']-ids[v]) for v in ('B','C')}})
    for v in ('B','C'):
        if sum(totals[v].values())!=saved['metadata']['added_candidates'][v]:
            raise AssertionError('Candidate bucket counts do not sum to additions')
    offsets={}
    for v in ('B','C'):
        near=[r for r in rows if r['variant']==v and r.get('distance') is not None]
        offsets[v]={'near_count':len(near),
                    'normal_mae':statistics.mean(abs(r['normal_error']) for r in near) if near else None,
                    'tangent_mae':statistics.mean(abs(r['tangent_error']) for r in near) if near else None}
    payload={'metadata':{'OFFICIAL_SCORE':False,'case_ids':SETS['design'],'sealed_holdout_used':False,
                         'inference_rerun':False,'formal_state':verify_formal_state(),'saved_run':str(run)},
             'strict_tp':{v:len(s) for v,s in all_ids.items()},'changes':changes,
             'candidate_buckets':{v:dict(c) for v,c in totals.items()},'offsets':offsets,
             'cases':per_case,'candidate_records':rows}
    folder=OUT/'conversion_audit';write_json(folder/'audit.json',payload)
    lines=['# 新增骨架候选得分转化审计','','固定6例Design，读取已保存结果；未重跑推理。全部为本地非官方诊断。','',
           '|组|严格Pin TP|相对A新增TP|相对A丢失TP|','|---|---:|---:|---:|']
    lines.append(f"|A|{len(all_ids['A'])}|0|0|")
    for v in ('B','C'):
        lines.append(f"|{v}|{len(all_ids[v])}|{len(changes[v]['gained'])}|{len(changes[v]['lost'])}|")
    lines+=['','|追加候选主桶|B|C|','|---|---:|---:|']
    for bucket in sorted(set(totals['B'])|set(totals['C'])):
        lines.append(f"|{bucket}|{totals['B'][bucket]}|{totals['C'][bucket]}|")
    lines+=['',f"20px内追加候选误差：{json.dumps(offsets,ensure_ascii=False)}",
            '主桶以owner/side一对一匹配判定；未匹配不等于图像假阳性，也可能是重复、几何偏移或GT差异。',
            'EXPORT_FILTERED不能自动等同于OCR漏识别；逐记录保留多个字段阻塞项。',
            '没有做逐裁剪人工可观测性确认，不凭内部样式key宣布不可观测。',
            '正式源码/配置/模型209个保护文件未变；未读取封存集。']
    (folder/'audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'strict_tp':payload['strict_tp'],'buckets':payload['candidate_buckets'],
                      'gained_lost':{v:{k:len(items) for k,items in c.items()} for v,c in changes.items()},
                      'offsets':offsets,'report':str(folder/'audit.md')},ensure_ascii=False),flush=True)
    return payload


class BoxOrderedSemantics:
    """Reuse existing V6 only on box; every other type stays on formal V3."""
    def run(self,component,localization,context):
        from pcb.pin.semantics.v3 import PinSemanticsStageV3
        from pcb.pin_semantics_v6 import assign_pin_semantics_v6
        output=PinSemanticsStageV3().run(component,localization,context)
        old=event_lookup([{'component':c.key,'terminals':terms} for c,terms in localization.terminals],output.diagnostics)
        rows=[]
        for item,terminals in localization.terminals:
            if item.type=='box':
                pins,events=assign_pin_semantics_v6(item,terminals,component.roles)
                item.pins=pins
                rows.extend({'component':item.key,**e} for e in events)
            else:
                rows.extend(old[item.key])
        output.diagnostics['pin_events']=rows
        output.diagnostics['pin_semantics']='box_only_existing_v6_other_types_v3'
        output.diagnostics['exportable_pins']=sum(p.exportable for c in component.components for p in c.pins)
        output.diagnostics['pin_v3_frozen']=False
        return output


def interval_distance(value,low,high):
    return max(float(low)-float(value),0.,float(value)-float(high))


def number_bbox_score(role,terminal,component):
    """Experimental number distance uses the text extent, with frozen V5 gates.

    Numbers printed above a wire are not centered on the wire. Keep the
    outside/role/regex tests and all limits; change only how distance is measured.
    The number center must still be outside the correct component side.
    """
    from pcb.pin_semantics import PIN_NUMBER
    from pcb.pin_semantics_v5 import _number_normal_distance,_tangent_limit
    if not PIN_NUMBER.fullmatch(role.normalized): return 0.
    if role.role=='PIN_NUMBER': role_bonus=.25
    elif component.type in {'box','block'} and role.role=='VALUE' and role.normalized.isdigit(): role_bonus=0.
    else: return 0.
    box=component.body_bbox or component.bbox
    if _number_normal_distance(role,terminal,box) is None: return 0.
    x1,y1,x2,y2=box;a,b,c,d=role.token.bbox
    side=terminal['side'];vertical=side in ('left','right')
    normal={'left':max(0.,x1-c),'right':max(0.,a-x2),
            'top':max(0.,y1-d),'bottom':max(0.,b-y2)}[side]
    tangent=interval_distance(terminal['base'][1] if vertical else terminal['base'][0],
                              b if vertical else a,d if vertical else c)
    span=x2-x1 if vertical else y2-y1
    normal_limit=max(30.,min(120.,.25*span))
    tangent_limit=_tangent_limit(role,terminal,box)
    if normal>normal_limit or tangent>tangent_limit: return 0.
    return (1.+2.*(1.-tangent/tangent_limit)+.5*(1.-normal/normal_limit)
            +float(role.confidence)+role_bonus)


class BoxBBoxSemantics:
    """Process-local scorer adapter, for serial experiments only (not production).

    Reuses V6 ordering/pairing/dedup with number_bbox_score. The imported scoring
    reference is restored in finally; no source or registry files are modified.
    """
    def run(self,component,localization,context):
        import pcb.pin_semantics_v6 as ordered
        old_score=ordered._number_score
        try:
            ordered._number_score=number_bbox_score
            output=BoxOrderedSemantics().run(component,localization,context)
        finally:
            ordered._number_score=old_score
        output.diagnostics['pin_semantics']='box_only_ordered_bbox_number_distance_other_types_v3'
        return output


def corrected_box_tips(output,shape):
    """Existing P1 formula; freeze candidate count/order/base/side/method/tangent."""
    result=copy.deepcopy(output);changes=[]
    for (old_owner,old_rows),(owner,rows) in zip(output.terminals,result.terminals):
        if owner.type!='box':
            if rows!=old_rows: raise AssertionError('Non-box geometry changed')
            continue
        for index,(old,row) in enumerate(zip(old_rows,rows)):
            row['tip'],detail=place_boundary_tip(row,shape)
            for field in ('base','side','method'):
                if row.get(field)!=old.get(field): raise AssertionError('Candidate invariant failed')
            axis=1 if row['side'] in ('left','right') else 0
            if row['tip'][axis]!=old['tip'][axis]: raise AssertionError('Tangent changed')
            changes.append({'component':owner.key,'candidate':index,**detail})
    result.diagnostics['box_tip_followup']={'formula':'max(6,6*V3 scale)','changes':changes}
    return result


def signed_frontend(case,config,model_sha):
    """Require pre-existing validated snapshots: no OCR/model fallback."""
    path=ROOT.parent/'tmp/pin_sina_frontend'/f'{case.case_id:04d}.json'
    value=read_json(path)
    files=['pcb/component_detection_v4.py','pcb/component/v4.py','pcb/text_detection.py',
           'pcb/component_text_assignment.py','pcb/ocr_backends.py','pcb/text/stage.py']
    expected={'image_sha256':digest(case.image_path),'config':config.as_dict(),'model_sha256':model_sha,
              'frontend_source_sha256':{name:digest(ROOT/name) for name in files}}
    signature=copy.deepcopy(value.get('signature') or {})
    signature['frontend_source_sha256']={key.replace('\\','/'):sha for key,sha in signature.get('frontend_source_sha256',{}).items()}
    if signature!=expected: raise AssertionError(f'Untrusted frontend snapshot: {path}')
    components=[Component(**{**r,'bbox':tuple(r['bbox']),'body_bbox':tuple(r['body_bbox']) if r.get('body_bbox') else None}) for r in value['components']]
    texts=[Text(**{**r,'bbox':tuple(r['bbox'])}) for r in value['texts']]
    roles=[TokenRole(**{**r,'token':Text(**{**r['token'],'bbox':tuple(r['token']['bbox'])})}) for r in value['roles']]
    return ComponentStageOutput(components,texts,roles,value['diagnostics'])


def followup_localization(front,outputs,target,height):
    """Same frozen strict-box rule; geometry-only matching for all factor variants."""
    pairs=_strict_box_pairs(component_payload(front,height),target['components'])
    counts={v:empty_counts() for v in outputs}
    lookup={v:{c.key:r for c,r in output.terminals} for v,output in outputs.items()}
    for owner,gowner in pairs:
        points=_gt_pins(target,gowner,height)
        for variant in outputs:
            candidates=lookup[variant][owner]
            block=empty_counts()
            block.update(components=1,gt=len(points),pred=len(candidates),
                         absolute_count_error=abs(len(points)-len(candidates)),
                         exact_count=int(len(points)==len(candidates)))
            for radius in (5,10,20):
                block[f'tp_{radius}']=len(constrained_matches(points,candidates,radius))
            add_counts(counts[variant],block)
    return counts


# A/B/C are the previous candidate variants, with formal V3 text semantics.
# S changes semantics only; T changes box tips only. Suffixes show both factors.
FOLLOWUPS={
    'A':('A',False,False), 'B':('B',False,False), 'C':('C',False,False),
    'S':('A',False,True),
    'T':('A',True,False), 'TB':('B',True,False), 'TC':('C',True,False),
    'TS':('A',True,True), 'TBS':('B',True,True), 'TCS':('C',True,True),
}

TEXT_FACTORS={'A':('A',False,False),'S':('A',False,True),
              'TS':('A',True,True),'N':('A',False,'bbox'),'TN':('A',True,'bbox')}


def bridge(task,dataset,resume=None):
    verify_formal_state()
    config=PipelineConfig.load(ROOT/'configs/current.json')
    if config!=PipelineConfig.current(): raise AssertionError('Formal config changed')
    guard=AccessGuard();started=time.perf_counter()
    fixed=RecoveryConfig(**read_json(MANUAL/'config.json'))
    candidates={'A':None,'B':RecoveryStage(fixed,False),'C':RecoveryStage(fixed,True)}
    registry=build_default_registry()
    registry.register('pin_semantics','followup_box_ordered',BoxOrderedSemantics)
    registry.register('pin_semantics','followup_box_bbox',BoxBBoxSemantics)
    model_sha=digest(ROOT/'models/component_yolo11n_continue_v2_best.pt')
    folder=resume.resolve() if resume else OUT/task/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if resume and folder.parent!=(OUT/task).resolve():
        raise ValueError('Resume directory must be a run of this task')
    if not resume: folder.mkdir(parents=True,exist_ok=False)
    restored_predictions=0
    # Design showed no additional strict TP from skeleton recovery. Check only
    # the predeclared minimal 2x2 tip/semantics factors; parameters stay frozen.
    factors=(TEXT_FACTORS if task.startswith('text_') else
             {v:FOLLOWUPS[v] for v in ('A','S','T','TS')} if task=='check' else FOLLOWUPS)
    totals={v:empty_counts() for v in factors}
    evals={v:[] for v in factors};tp_sets={v:set() for v in factors}
    cases=selected_cases(dataset,'design' if task in ('design','text_design') else 'check')
    if task=='smoke': cases=[c for c in selected_cases(dataset,'design') if c.case_id in (17,87,130)]
    all_cases=[];baseline_parity=0
    from pcb.pin.localization.v3 import PinLocalizationStageV3
    for case in cases:
        key=f'{case.case_id:04d}';image=read_image(case.image_path);height,width=image.shape[:2]
        front=signed_frontend(case,config,model_sha)
        context=PipelineContext(case.image_path.name,width,height,config,None)
        original=PinLocalizationStageV3().run(image,copy.deepcopy(front),context)
        outputs={'A':original}
        for v in sorted({base for base,tip,sem in factors.values()}-{'A'}):
            outputs[v]=candidates[v].run(image,copy.deepcopy(front),context)
            for (oc,old),(nc,new) in zip(original.terminals,outputs[v].terminals):
                if (oc.key,oc.bbox,oc.body_bbox)!=(nc.key,nc.bbox,nc.body_bbox) or new[:len(old)]!=old:
                    raise AssertionError('Component or original candidate changed')
                if oc.type!='box' and old!=new: raise AssertionError('Non-box changed')
        prepared={v:corrected_box_tips(outputs[base],image.shape) if tip else copy.deepcopy(outputs[base])
                  for v,(base,tip,sem) in factors.items()}
        results={};diagnostics={}
        for v,(base,tip,sem) in factors.items():
            local=copy.deepcopy(front);loc=copy.deepcopy(prepared[v])
            loc.terminals=[(c,loc.terminals[i][1]) for i,c in enumerate(local.components)]
            cfg=replace(config,pin_semantics='followup_box_bbox' if sem=='bbox' else 'followup_box_ordered' if sem else 'v3')
            start=time.perf_counter()
            saved=folder/'predictions'/v/key
            raw_expected=[{'component':c.key,'type':c.type,'terminals':r} for c,r in prepared[v].terminals]
            if resume:
                if read_json(saved/'raw_terminals.json')!=json.loads(json.dumps(raw_expected)):
                    raise AssertionError('Saved geometry differs from frozen variant')
                results[v]=read_json(saved/'result.json')
                diagnostics[v]=read_json(saved/'diagnostics.json')
                event_lookup(raw_expected,diagnostics[v])
                restored_predictions+=1
            else:
                results[v],diagnostics[v]=full_inference(image,local,loc,replace(context,config=cfg),registry)
                diagnostics[v]['seconds']=time.perf_counter()-start
            validate_strict(results[v],(width,height))
            if results[v]['components']!=results['A']['components']:
                raise AssertionError('Component JSON changed')
            if v!='A':
                for c in front.components:
                    if c.type!='box' and results[v]['pins'][c.key]!=results['A']['pins'][c.key]:
                        raise AssertionError('Non-box final pins changed')
            if not resume:
                write_json(saved/'result.json',results[v])
                write_json(saved/'diagnostics.json',diagnostics[v])
                write_json(saved/'raw_terminals.json',raw_expected)
        reference=ROOT.parent/'tmp/pin_p6_5_formal_predictions'/key/'result.json'
        if read_json(reference)!=results['A']: raise AssertionError(f'{key}: baseline JSON parity failed')
        baseline_parity+=1
        with guard.evaluator():
            target=read_json(case.target_path)
            loc=followup_localization(front,prepared,target,height)
            case_tp={}
            for v in factors:
                add_counts(totals[v],loc[v])
                evaluation=evaluate_case(results[v],target,diagnostics[v])
                ids=strict_ids(results[v],target);tp_sets[v].update((key,*pin) for pin in ids)
                if len(ids)!=evaluation['metrics']['Pin']['tp']: raise AssertionError('Evaluator TP mismatch')
                case_tp[v]=len(ids)
                evals[v].append({'case_id':key,'source':case.image_path.name,'status':'ok','evaluation':evaluation})
        all_cases.append({'case_id':key,'localization':loc,'strict_tp':case_tp})
        print(json.dumps({'case':key,'PinTP':case_tp,'box_TP5':{v:loc[v]['tp_5'] for v in factors}},ensure_ascii=False),flush=True)
    # Source attribution is the original evaluator helper, no filename-based algorithm rules.
    from evaluate_v2 import source_from_name
    for rows in evals.values():
        for row in rows: row['source']=source_from_name(row['source'])
    result={'metadata':{'OFFICIAL_SCORE':False,'case_ids':[f'{c.case_id:04d}' for c in cases],
                        'sealed_holdout_used':False,'task':task,'baseline_parity_count':baseline_parity,
                        'formal_state':verify_formal_state(),'parameters':{'recovery':fixed.__dict__,'tip_formula':'max(6,6*V3 scale)','semantics':'V6 box-only; N/TN change numeric distance to bbox extent, all limits frozen'},
                        'stage_factors':factors,'runtime_seconds':time.perf_counter()-started,
                        'restored_prediction_count':restored_predictions,
                        'check_selection':'A/S/T/TS fixed after Design; no recovery candidates or parameter retuning' if task=='check' else None,
                        'target_in_inference_attempts':guard.target_attempts,'sealed_attempts':guard.sealed_attempts,
                        'run_directory':str(folder)},
            'localization':{v:metrics(t) for v,t in totals.items()},
            'end_to_end':{v:aggregate(e) for v,e in evals.items()},
            'tp_changes':{v:{'gained':sorted(tp_sets[v]-tp_sets['A']),'lost':sorted(tp_sets['A']-tp_sets[v])} for v in factors},
            'cases':all_cases}
    write_json(folder/'report.json',result);write_json(OUT/task/'latest.json',result)
    lines=[f'# 骨架候选桥接实验：{task}','','所有结果为本地非官方诊断。无模型/OCR重跑；只改box实验插槽。','',
           '|组|box定位TP5|box定位Recall20|PinTP|Pin宏F1|新增/丢失TP|HyperF1|LineF1|PairF1|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in factors:
        m=result['end_to_end'][v]['overall']['metrics'];loc=result['localization'][v];ch=result['tp_changes'][v]
        lines.append(f"|{v}|{loc['tp_5']}|{loc['at_20']['recall']:.4f}|{m['Pin']['tp']}|{m['Pin']['macro_f1']:.6f}|{len(ch['gained'])}/{len(ch['lost'])}|{m['NetHypergraph']['macro_f1']:.6f}|{m['NetLine']['macro_f1']:.6f}|{m['PinPair']['macro_f1']:.6f}|")
    lines+=['','A/B/C：原定位/骨架补充/分支补充；S：只换box保序语义；T：只换box Tip；TB/TC：补充+Tip；TS/TBS/TCS再加box保序语义。',
            '原candidate生成顺序/base/side/method/数量在Tip前后保持；仅tip法线坐标变化。',
            '非box Pin JSON、Component JSON完全一致；正式209个保护文件未变。',
            'Check是固定开发诊断子集，已有历史使用，不是封存集或真正未见测试集。']
    if task.startswith('text_'):
        lines+=['N：只换脚号距离度量；TN：N加已有Tip公式。阈值/角色/正则/顺序配对/脚名分数都不改。',
                '号码文本中心仍需位于正确外侧；没有直接放宽数值阈值，没有增加OCR或terminal。',
                '临时scorer仅供串行实验，finally恢复模块引用，不接入生产配置。']
    (OUT/task/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"REPORT: {OUT/task/'summary.md'}",flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=('audit','smoke','design','check','text_design','text_check'),default='audit')
    parser.add_argument('--dataset-root',type=Path)
    parser.add_argument('--resume-directory',type=Path)
    args=parser.parse_args()
    if args.task=='audit': audit()
    else:
        if args.dataset_root is None:
            args.dataset_root=Path(read_json(MANUAL/'outputs/design/latest.json')['metadata']['dataset_root'])
        bridge(args.task,args.dataset_root,args.resume_directory)
