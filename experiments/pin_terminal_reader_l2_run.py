"""L2 Design then one fixed 30-case regression; no localization re-selection."""
from __future__ import annotations

import argparse
from collections import Counter
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

from pin_terminal_reader_l2 import POLICY, TerminalRowOCR, refine_scene
from pin_learned_locator_l1_train import OUT as L1, DATASET, Guard, digest, freeze as freeze_l1
from pin_learned_locator_l1_eval import scene_from, box_counts
from pin_g_wire_e1 import verify_formal_state, git_state, signed_frontend, pair_changes
from pin_g_recovery_e4 import dump_raw
from pin_skeleton_followup import strict_ids
from pin_g_frame_e3 import suppress_box_frames
from pcb.core.config import PipelineConfig
from pcb.core.context import PipelineContext
from pcb.core.registry import build_default_registry
from pcb.data_policy import discover_allowed_cases
from pcb.io import read_image, write_json, write_image
from pcb.submission import validate_strict
from evaluate_v2 import evaluate_case,aggregate,source_from_name,prf

OUT=ROOT/'reports/pin_terminal_reader_l2'
DESIGN=(14,17,18,87,103)


def freeze():
    previous=freeze_l1()
    paths=('experiments/pin_terminal_reader_l2.py','experiments/pin_terminal_reader_l2_run.py',
           'tests/pin/test_pin_terminal_reader_l2.py','reports/pin_terminal_reader_l2/experiment_plan.md')
    value={'policy':asdict(POLICY),'single_factor':'box terminal-conditioned text reading and joint ownership',
           'source_sha256':{p:digest(ROOT/p) for p in paths},'design_ids':list(DESIGN),
           'L1_weights_sha256':digest(L1/'best.pt'),'L1_protocol':previous,
           'L1_validation_sha256':digest(L1/'validation_complete.json'),
           'OFFICIAL_SCORE':False,'sealed_holdout_used':False}
    path=OUT/'frozen_policy.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=value:
        raise AssertionError('Frozen L2 changed; no validation threshold search')
    if not path.exists(): write_json(path,value)
    return value


def predict(scene,image,context,registry):
    wire=registry.create('wire','v3').run(image,scene,context)
    wire=suppress_box_frames(image,scene,wire)
    registry.create('topology','v3').run(scene,wire,context)
    pred=registry.create('submission','official').run(scene,context).data
    validate_strict(pred,(context.width,context.height))
    return pred


def debug(image,raw,trace,folder):
    """A small image-only contact sheet, no cherry-picked GT improvements."""
    import cv2
    import numpy as np
    lookup={b['component']:b['terminals'] for b in raw}
    cells=[]
    for row in trace[:12]:
        a,b,c,d=row['window']['roi']; crop=image[b:d,a:c].copy()
        if not crop.size: continue
        crop=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        canvas=np.full((100,max(400,crop.shape[1]),3),255,np.uint8)
        canvas[:min(68,crop.shape[0]),:crop.shape[1]]=crop[:68]
        selected=row['selected']
        text=f"{row['component']}[{row['candidate']}] {row['window']['side']} -> "
        text+=f"{selected['number']} / {selected['name']}" if selected else 'ABSTAIN / KEEP OLD'
        cv2.putText(canvas,text,(4,88),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
        cells.append(canvas)
    if cells:
        width=max(c.shape[1] for c in cells)
        sheet=np.full((len(cells)*100,width,3),255,np.uint8)
        for i,cell in enumerate(cells):sheet[i*100:(i+1)*100,:cell.shape[1]]=cell
        write_image(folder/'first_rows.png',sheet)


def run(subset):
    started=time.perf_counter(); frozen=freeze(); state=verify_formal_state(); git_before=git_state()
    dest=OUT/subset
    if (dest/'complete.json').exists(): raise RuntimeError('This L2 run already completed; no repeat model selection')
    guard=Guard(); config=PipelineConfig.load(ROOT/'configs/current.json')
    manifest=json.loads((L1/'training_manifest.json').read_text(encoding='utf-8'))
    validation=json.loads((L1/'validation_complete.json').read_text(encoding='utf-8'))
    if subset=='check':
        design=json.loads((OUT/'design/complete.json').read_text(encoding='utf-8'))
        if not design['guards']['Design_progress']:
            raise RuntimeError('Design failed; Check prohibited under the experiment plan')
    numbers=list(DESIGN) if subset=='design' else manifest['validation_ids']
    if set(DESIGN)&set(manifest['validation_ids']): raise AssertionError('Design/Check overlap')
    cases={c.case_id:c for c in discover_allowed_cases(DATASET)}
    registry=build_default_registry(); reader=TerminalRowOCR(OUT/'ocr_cache')
    model=None
    if subset=='design':
        import torch
        from pin_learned_locator_l1 import PinPointNet
        torch.set_num_threads(4)
        checkpoint=torch.load(L1/'best.pt',map_location='cpu',weights_only=True)
        model=PinPointNet().to('cuda'); model.load_state_dict(checkpoint['state_dict']); model.eval()
    names=('L1','L2'); records=[]; evaluations={v:[] for v in names}
    box_total={v:Counter() for v in names}; cond_total={v:Counter() for v in names}
    gained=[];lost=[];pairs=Counter();stats=Counter();inputs={}; contracts=0
    for number in numbers:
        key=f'{number:04d}';case=cases[number];image=read_image(case.image_path);h,w=image.shape[:2]
        if digest(case.image_path)!=manifest['image_sha256'][str(number)]: raise AssertionError('Image changed')
        context=PipelineContext(case.image_path.name,w,h,config,None)
        front=signed_frontend(case,config,digest(ROOT/'models/component_yolo11n_continue_v2_best.pt'))
        if subset=='design':
            from pcb.pin.localization.v3 import PinLocalizationStageV3
            from pin_learned_locator_l1 import locate_boxes
            original=PinLocalizationStageV3().run(image,copy.deepcopy(front),context)
            _,bloc,_=scene_from(front,dump_raw(original),image,context,True)
            raw=dump_raw(locate_boxes(image,front.components,bloc,model,'cuda'))
        else:
            source=L1/'validation/predictions/L1'/key
            raw=json.loads((source/'raw_terminals.json').read_text(encoding='utf-8'))
            inputs[key]={'raw_sha256':digest(source/'raw_terminals.json'),
                         'prediction_sha256':digest(source/'result.json')}
        baseline,_,_=scene_from(front,raw,image,context)
        candidate,trace,statistics=refine_scene(image,baseline,raw,reader)
        stats.update(statistics)
        scenes={'L1':baseline,'L2':candidate}; predictions={}
        for variant in names:
            scene=scenes[variant]; predictions[variant]=predict(scene,image,context,registry)
            saved=dest/'predictions'/variant/key
            write_json(saved/'result.json',predictions[variant]);contracts+=1
            write_json(saved/'diagnostics.json',scene.diagnostics)
        if subset=='check':
            reference=json.loads((source/'result.json').read_text(encoding='utf-8'))
            if predictions['L1']!=reference: raise AssertionError('Reconstructed L1 full-JSON parity failure')
        write_json(dest/'raw_terminals'/f'{key}.json',raw)
        write_json(dest/'text_trace'/f'{key}.json',trace)
        if subset=='design' and number in DESIGN[:2]:debug(image,raw,trace,dest/'debug'/key)
        if predictions['L1']['components']!=predictions['L2']['components']: raise AssertionError('Component changed')
        for c in front.components:
            if c.type!='box' and predictions['L1']['pins'][c.key]!=predictions['L2']['pins'][c.key]:
                raise AssertionError('Nonbox changed')
        # Both full image-only predictions are complete before target access.
        target,target_sha=guard.target(case,training=False)
        if subset=='check':
            old=next(r for r in validation['cases'] if r['case_id']==key)
            if old['target_sha256']!=target_sha: raise AssertionError('L1/L2 GT version changed')
        ids={v:strict_ids(predictions[v],target) for v in names}
        new=sorted(ids['L2']-ids['L1']); missing=sorted(ids['L1']-ids['L2'])
        gained.extend({'case_id':key,'pin':p} for p in new)
        lost.extend({'case_id':key,'pin':p} for p in missing)
        row={'case_id':key,'source':source_from_name(case.image_path.name),'target_sha256':target_sha,
             'gained':len(new),'lost':len(missing),'metrics':{},'matched_box':{},'reader':statistics}
        for v in names:
            evaluation=evaluate_case(predictions[v],target,scenes[v].diagnostics)
            evaluations[v].append({'case_id':key,'source':row['source'],'status':'ok','evaluation':evaluation})
            row['metrics'][v]=evaluation['metrics']['Pin']
            row['matched_box'][v]=box_counts(predictions[v],target,raw,h)
            box_total[v].update(row['matched_box'][v]['geometry'])
            cond_total[v].update(row['matched_box'][v]['strict'])
            if row['metrics'][v]['tp']!=len(ids[v]):raise AssertionError('TP set disagreement')
        if row['matched_box']['L1']['geometry']!=row['matched_box']['L2']['geometry']:
            raise AssertionError('Frozen geometry statistics changed')
        changes=pair_changes(predictions['L1'],predictions['L2'],target)
        row['pair_changes']={k:len(v) for k,v in changes.items()};pairs.update(row['pair_changes'])
        records.append(row)
        write_json(dest/'progress.json',{'cases':records,'gained':gained,'lost':lost})
        print(json.dumps({'case':key,'L1':row['metrics']['L1'],'L2':row['metrics']['L2'],
                          'local_accepted':statistics['accepted']},ensure_ascii=True),flush=True)
    summary={v:aggregate(evaluations[v]) for v in names}
    a,b=(summary[v]['overall']['metrics']['Pin'] for v in names)
    improved=[r['case_id'] for r in records if r['metrics']['L2']['f1']>r['metrics']['L1']['f1']]
    regressed=[r['case_id'] for r in records if r['metrics']['L2']['f1']<r['metrics']['L1']['f1']]
    guards={'TP_relative_gain_at_least_20pct':b['tp']>=1.2*a['tp'] and b['tp']>a['tp'],
            'micro_F1_relative_gain_at_least_20pct':b['f1']>=1.2*a['f1'] and b['f1']>a['f1'],
            'macro_F1_not_lower':b['macro_f1']>=a['macro_f1'],
            'at_least_3_improved_cases':len(improved)>=3,'no_old_strict_TP_loss':not lost}
    guards['Pin_passed']=all(guards.values())
    guards['Design_progress']=b['tp']>a['tp'] and b['f1']>a['f1'] and not lost
    guards['system_passed']=guards['Pin_passed'] and not pairs['incorrect_added'] and not pairs['correct_lost']
    boxes={v:{'counts':dict(box_total[v]),'strict_Pin':prf(*(cond_total[v][k] for k in ('tp','pred','gt')))} for v in names}
    payload={'metadata':{'OFFICIAL_SCORE':False,'sealed_holdout_used':False,'subset':subset,
        'case_ids':numbers,'frozen_policy':frozen,'L1_input_snapshots':inputs,
        'inference_GT_reads':False,'blocked_access_attempts':guard.blocked,
        'formal_before':state,'formal_after':verify_formal_state(),'git_before':git_before,
        'git_after':git_state(),'contracts_passed':contracts,'model_SHA':reader.model_signature,
        'ocr_detector_calls':reader.calls,'ocr_direct_recognizer_crops':reader.direct_calls,
        'ocr_cache_hits':reader.cache_hits,'seconds':time.perf_counter()-started},
        'end_to_end':summary,'matched_box':boxes,'reader_statistics':dict(stats),
        'guards':guards,'pair_changes':dict(pairs),'gained':gained,'lost':lost,'cases':records,
        'improved_cases':improved,'regressed_cases':regressed,
        'unchanged_cases':[r['case_id'] for r in records if r['case_id'] not in improved+regressed]}
    write_json(dest/'complete.json',payload)
    with (dest/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case_id','source','L1_TP','L2_TP','L1_pred','L2_pred','L1_F1','L2_F1','gained','lost'])
        for r in records:
            x,y=(r['metrics'][v] for v in names)
            writer.writerow([r['case_id'],r['source'],x['tp'],y['tp'],x['pred'],y['pred'],x['f1'],y['f1'],r['gained'],r['lost']])
    lines=[f'# L2 {subset}：terminal条件局部读取与联合归属','','本地非官方诊断；本实验没有修改定位或正式current。','',
           '|严格Pin指标|L1|L2|','|---|---:|---:|']
    for k in ('tp','pred','gt','precision','recall','f1','macro_f1'):lines.append(f'|{k}|{a[k]:.6f}|{b[k]:.6f}|')
    lines+=['',f'预设护栏：{json.dumps(guards,ensure_ascii=False)}',f'改善：{improved}；回归：{regressed}',
            f'新增TP：{len(gained)}；旧TP损失：{len(lost)}；Pair：{dict(pairs)}',
            f'相同matched-box条件严格Pin：{json.dumps(boxes,ensure_ascii=False)}',
            f'局部读取统计：{dict(stats)}；耗时{payload["metadata"]["seconds"]:.2f}秒。',
            '未推理全部150例、未用封存集、未commit/push。Check只做固定开发回归，不声称未见泛化。']
    (dest/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    freeze()
    print(json.dumps({'subset':subset,'L1':a,'L2':b,'guards':guards,'report':str(dest/'summary.md')},ensure_ascii=True),flush=True)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--subset',choices=('design','check'),required=True)
    run(parser.parse_args().subset)
