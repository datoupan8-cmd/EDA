"""One held-out-by-Pin-training A/B, with unchanged frontend and semantics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT)]

import torch
from pin_learned_locator_l1 import POLICY, PinPointNet, locate_boxes
from pin_learned_locator_l1_train import OUT, DATASET, Guard, digest, write, freeze
from pin_g_wire_e1 import verify_formal_state, git_state, pair_changes, signed_frontend
from pin_tip_regression import reconstruct_scene
from pin_tip_evidence import refine_scene_tips
from pin_contextual_roles_e6 import refine_scene as roles_e6
from pin_g_frame_e3 import suppress_box_frames
from pin_g_recovery_e4 import dump_raw
from pin_skeleton_followup import strict_ids
from tools.run_box_skeleton_experiment import load_frontend
from tools.analyze_pin_localization import strict_component_pairs, coordinate_assignment
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.registry import build_default_registry
from pcb.data_policy import discover_allowed_cases
from pcb.coordinates import target_to_opencv
from pcb.io import read_image
from pcb.pin.localization.v3 import PinLocalizationStageV3
from pcb.submission import export, validate_strict
from evaluate_v2 import evaluate_case, aggregate, source_from_name, pin_counts, prf


def box_counts(pred,target,raw,height):
    pairs=[(p,g) for p,g in strict_component_pairs(pred['components'],target['components'])
           if pred['components'][p]['type']=='box']
    lookup={r['component']:r['terminals'] for r in raw}; total=Counter()
    for p,g in pairs:
        gt=[target_to_opencv(v['point']['x'],v['point']['y'],height)
            for v in target['pins'].get(g,{}).values()]
        terms=lookup.get(p,[]); assignments=coordinate_assignment(terms,gt)
        total.update(components=1,gt=len(gt),pred=len(terms),count_abs=abs(len(gt)-len(terms)))
        for radius in (5,10,20): total[f'tp_{radius}']+=sum(m['distance']<=radius for m in assignments)
    counts=pin_counts({p:pred['pins'].get(p,{}) for p,g in pairs},
                     {g:target['pins'].get(g,{}) for p,g in pairs},dict(pairs),False)
    return {'geometry':dict(total),'strict':dict(zip(('tp','pred','gt'),counts))}


def scene_from(front,raw,image,context,baseline=False):
    scene,loc=reconstruct_scene(front,raw,image,context)
    if baseline:
        scene,loc,_=refine_scene_tips(scene,loc,front.roles,image,'joint')
        raw=dump_raw(loc)
    scene,decisions=roles_e6(scene,raw,front.roles)
    scene.diagnostics['L1_frozen_roles']=decisions
    return scene,loc,raw


def run():
    started=time.perf_counter(); frozen=freeze(); state=verify_formal_state(); git_before=git_state()
    if (OUT/'validation_complete.json').exists():
        raise RuntimeError('Validation already completed; no repeated selection on this split')
    guard=Guard(); config=PipelineConfig.load(ROOT/'configs/current.json')
    manifest=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'))
    complete=json.loads((OUT/'training_complete.json').read_text(encoding='utf-8'))
    checkpoint=torch.load(OUT/'best.pt',map_location='cpu',weights_only=True)
    if checkpoint['frozen_policy']!=frozen or complete['weight_sha256']!=digest(OUT/'best.pt'):
        raise AssertionError('Checkpoint protocol/hash mismatch')
    if set(checkpoint['train_ids']) & set(manifest['validation_ids']): raise AssertionError('Case split leak')
    if checkpoint['manifest_sha256']!=digest(OUT/'training_manifest.json'): raise AssertionError('Manifest changed')
    device='cuda'; model=PinPointNet().to(device); model.load_state_dict(checkpoint['state_dict']); model.eval()
    cases={c.case_id:c for c in discover_allowed_cases(DATASET)}
    registry=build_default_registry()
    names=('GF_ROLE','L1'); evaluations={v:[] for v in names}; totals={v:Counter() for v in names}
    box_total={v:Counter() for v in names}; cond_total={v:Counter() for v in names}
    paired=Counter(); gained=[]; lost=[]; records=[]; inputs={}; frontend_cached=0
    source_files=['pcb/component_detection_v4.py','pcb/component/v4.py','pcb/text_detection.py',
                  'pcb/component_text_assignment.py','pcb/ocr_backends.py','pcb/text/stage.py']
    runtime={'model_sha256':digest(ROOT/'models/component_yolo11n_continue_v2_best.pt'),
             'frontend_source_sha256':{f:digest(ROOT/f) for f in source_files}}
    cache=ROOT.parent/'tmp/pin_sina_frontend'; output=OUT/'validation/predictions'
    for number in manifest['validation_ids']:
        case=cases[number]; key=f'{number:04d}'; image=read_image(case.image_path); h,w=image.shape[:2]
        if digest(case.image_path)!=manifest['image_sha256'][str(number)]: raise AssertionError('Image changed')
        context=PipelineContext(case.image_path.name,w,h,config,None)
        if (cache/(key+'.json')).exists():
            front=signed_frontend(case,config,runtime['model_sha256']); cached=True
        else:
            front,cached=load_frontend(case,image,context,cache,runtime)
        frontend_cached+=int(cached); inputs[key]=digest(cache/(key+'.json'))
        original=PinLocalizationStageV3().run(image,copy.deepcopy(front),context)
        baseline,bloc,braw=scene_from(front,dump_raw(original),image,context,True)
        learned=locate_boxes(image,front.components,bloc,model,device)
        candidate,lloc,lraw=scene_from(front,dump_raw(learned),image,context)
        scenes={'GF_ROLE':baseline,'L1':candidate}; raws={'GF_ROLE':braw,'L1':lraw}; predictions={}
        for variant in names:
            scene=scenes[variant]
            wire=registry.create('wire','v3').run(image,scene,context)
            wire=suppress_box_frames(image,scene,wire)
            registry.create('topology','v3').run(scene,wire,context)
            pred=registry.create('submission','official').run(scene,context).data
            validate_strict(pred,(w,h)); predictions[variant]=pred
            write(output/variant/key/'result.json',pred)
            write(output/variant/key/'raw_terminals.json',raws[variant])
            write(output/variant/key/'diagnostics.json',scene.diagnostics)
        if predictions['GF_ROLE']['components']!=predictions['L1']['components']:
            raise AssertionError('Component changed')
        for c in front.components:
            if c.type!='box' and predictions['GF_ROLE']['pins'].get(c.key)!=predictions['L1']['pins'].get(c.key):
                raise AssertionError('Nonbox exported Pin changed')
        # Only now may this validation target enter the evaluator.
        target,target_sha=guard.target(case,training=False)
        ids={v:strict_ids(predictions[v],target) for v in names}
        change_gained=sorted(ids['L1']-ids['GF_ROLE']); change_lost=sorted(ids['GF_ROLE']-ids['L1'])
        gained.extend({'case_id':key,'pin':p} for p in change_gained)
        lost.extend({'case_id':key,'pin':p} for p in change_lost)
        row={'case_id':key,'source':source_from_name(case.image_path.name),'target_sha256':target_sha,
             'gained':len(change_gained),'lost':len(change_lost),'metrics':{},'matched_box':{}}
        for variant in names:
            evaluation=evaluate_case(predictions[variant],target,scenes[variant].diagnostics)
            evaluations[variant].append({'case_id':key,'source':row['source'],
                                          'status':'ok','evaluation':evaluation})
            row['metrics'][variant]=evaluation['metrics']['Pin']
            row['matched_box'][variant]=box_counts(predictions[variant],target,raws[variant],h)
            box_total[variant].update(row['matched_box'][variant]['geometry'])
            cond_total[variant].update(row['matched_box'][variant]['strict'])
            totals[variant].update({k:evaluation['metrics']['Pin'][k] for k in ('tp','pred','gt')})
            if evaluation['metrics']['Pin']['tp']!=len(ids[variant]): raise AssertionError('TP definitions disagree')
        changes=pair_changes(predictions['GF_ROLE'],predictions['L1'],target)
        row['pair_changes']={k:len(v) for k,v in changes.items()}; paired.update(row['pair_changes'])
        records.append(row); write(OUT/'validation/progress.json',{'cases':records,'gained':gained,'lost':lost})
        print(json.dumps({'validation_case':key,'pin':{v:row['metrics'][v] for v in names}},ensure_ascii=True),flush=True)
    summary={v:aggregate(evaluations[v]) for v in names}
    a,b=(summary[v]['overall']['metrics']['Pin'] for v in names)
    improved=[r['case_id'] for r in records if r['metrics']['L1']['f1']>r['metrics']['GF_ROLE']['f1']]
    guards={'TP_relative_gain_at_least_20pct':b['tp']>=1.2*a['tp'] and b['tp']>a['tp'],
        'micro_F1_relative_gain_at_least_20pct':b['f1']>=1.2*a['f1'] and b['f1']>a['f1'],
        'macro_F1_not_lower':b['macro_f1']>=a['macro_f1'],'at_least_3_improved_cases':len(improved)>=3,
        'no_old_strict_TP_loss':not lost}
    guards['Pin_passed']=all(guards.values())
    guards['system_passed']=guards['Pin_passed'] and not paired['incorrect_added'] and not paired['correct_lost']
    boxes={v:{'counts':dict(box_total[v]),'thresholds':{str(r):prf(box_total[v][f'tp_{r}'],
            box_total[v]['pred'],box_total[v]['gt']) for r in (5,10,20)},
            'strict_Pin':prf(*(cond_total[v][k] for k in ('tp','pred','gt')))} for v in names}
    payload={'metadata':{'OFFICIAL_SCORE':False,'sealed_holdout_used':False,'case_count':len(records),
        'validation_ids':manifest['validation_ids'],'train_ids':manifest['train_ids'],
        'validation_previously_used_for_development':True,'frontend_cached_count':frontend_cached,
        'frozen_policy':frozen,'model_epoch_selected_by_training_loss':checkpoint['epoch'],
        'weight_sha256':digest(OUT/'best.pt'),'frontend_snapshot_sha256':inputs,
        'GT_read_before_prediction':False,'blocked_access_attempts':guard.blocked,
        'formal_state_before':state,'formal_state_after':verify_formal_state(),
        'git_state_before':git_before,'git_state_after':git_state(),'seconds':time.perf_counter()-started},
        'end_to_end':summary,'matched_box':boxes,'guards':guards,'pair_changes':dict(paired),
        'gained':gained,'lost':lost,'cases':records,'improved_cases':improved}
    write(OUT/'validation_complete.json',payload)
    with (OUT/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['case_id','source','baseline_tp','L1_tp','baseline_pred','L1_pred','baseline_f1','L1_f1','gained','lost'])
        for r in records:
            av,bv=(r['metrics'][v] for v in names)
            writer.writerow([r['case_id'],r['source'],av['tp'],bv['tp'],av['pred'],bv['pred'],av['f1'],bv['f1'],r['gained'],r['lost']])
    lines=['# L1学习式Pin定位：一次30例开发验证','','本地非官方诊断；只训练允许开发集的训练case。30例未用于本Pin模型训练，但已用于历史开发，不是封存测试。','',
        '|指标|GF_ROLE|L1|','|---|---:|---:|']
    for k in ('tp','pred','gt','precision','recall','f1','macro_f1'):
        lines.append(f'|{k}|{a[k]:.6f}|{b[k]:.6f}|')
    lines+=['',f'预设价值门槛：{json.dumps(guards,ensure_ascii=False)}',f'改善case：{improved}',
        f'旧严格TP损失：{len(lost)}；新增：{len(gained)}；Pair变化：{dict(paired)}',
        '',f'固定阈值matched-box定位：{json.dumps(boxes,ensure_ascii=False)}','',
        '本轮没有改OCR、Component、脚号脚名算法、非box定位、F、Wire/Topology、输出或评分器。',
        '通过才能讨论接入；未通过不调验证阈值、不覆盖正式配置。模型、源码和结果保留。',
        '未完整推理150例；未读取封存集/QuickTest；未commit/push。']
    (OUT/'decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':{'GF_ROLE':a,'L1':b},'guards':guards,'report':str(OUT/'decision.md')},ensure_ascii=True),flush=True)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.parse_args(); run()
