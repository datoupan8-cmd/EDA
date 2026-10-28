"""Frozen L42 comparisons for calibration, image pair regression and point projection."""
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
import argparse
import importlib.util
import json
import math
import shutil
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('l45_train_protocol',ROOT/'docs/pin/prepare_train_l45_20261012.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
spec=importlib.util.spec_from_file_location('retained_l42_protocol',ROOT/'docs/pin/run_l42_validation_20261012.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
STUDY,REPORT=prep.STUDY,prep.REPORT
read,write,sha,event=base.read,base.write,base.sha,base.event
COHORTS={'design':[14,17,18,87,103],'check':list(range(5,151,5)),'full150':list(range(1,151))}
METHODS=('calibration','regression','regression_projection')
CACHE=Path('D:/Codex_Experiments/Pin_20261012/work/pin_l44_text_stroke_20261012/cache')

def signature():
    initial=read(STUDY/'study_initial.json')
    if prep.protect()!=initial['source_files']:raise ValueError('Study source changed')
    trained=read(STUDY/'training_complete.json')
    for name,key in (('best.pt','checkpoint_sha256'),('calibration.json','calibration_sha256')):
        if sha(STUDY/name)!=trained[key]:raise ValueError('Frozen learned resource changed')
    return {'base':base.signature(),'study_files':initial['source_files'],
            'training_checkpoint_sha256':trained['checkpoint_sha256'],
            'calibration_sha256':trained['calibration_sha256'],'actual_control':'L42ControlSemantics',
            'original_312_files_verified':True,'default_changed':False,'L43_used':False,'OFFICIAL_SCORE':False}

def progress(args,records,phase):
    write(args.output/'progress.json',{'records':records,'expected':len(COHORTS[args.cohort]),'phase':phase,'GT_reads':0})
    prep.status(phase,method=args.method,cohort=args.cohort,completed=len(records),expected=len(COHORTS[args.cohort]))

def preflight(args):
    from tools.run_team_pipeline import select_inputs
    if args.output.exists():raise ValueError('Fresh output required')
    if args.cohort!='design':
        previous='design' if args.cohort=='check' else 'check'
        result=read(STUDY/args.method/previous/'evaluation.json')
        if not result['advance']:raise ValueError('Predeclared actual positive prior cohort required')
        if result['network_cost_requires_review']:raise ValueError('Resolve network cost using fixed point projection before expanding')
    if args.method=='regression_projection':
        if not (STUDY/'regression'/args.cohort/'inference_complete.json').exists():
            raise ValueError('Same-model ordinary regression predictions needed for projection parity')
    current=signature()
    rows=select_inputs(None,base.DATASET,','.join(str(n) for n in COHORTS[args.cohort]))
    items=[{'case_id':cid,'path':str(image),'sha256':sha(image),
            'reference':str(ROOT/'runs/pin_l42_full150_20261012'/cid/'B/result.json'),
            'reference_sha256':sha(ROOT/'runs/pin_l42_full150_20261012'/cid/'B/result.json')}
           for cid,image,folder in rows]
    write(args.output/'initial.json',{'signature':current,'method':args.method,'cohort':args.cohort,'inputs':items,
          'cache':str(CACHE),'projection_only':args.method.endswith('projection'),'GT_free_inference':True})
    progress(args,[],'PREFLIGHT_PASSED');event('preflight_pass',method=args.method,cohort=args.cohort,count=len(items))

def verify_saved(output,records):
    for row in records:
        for arm in ('A','B'):
            for filename,digest in row[arm]['files'].items():
                if sha(output/row['case_id']/arm/filename)!=digest:raise ValueError('Saved evidence changed')

def infer(args):
    from tools.team_access_guard import Guard
    guard=Guard();network=[]
    def offline(name,values):
        if name=='socket.connect':network.append(str(values[1:]));raise PermissionError('Inference must be offline')
    sys.addaudithook(offline)
    initial=read(args.output/'initial.json');current=signature()
    if current!=initial['signature']:raise ValueError('Frozen experiment changed')
    records=read(args.output/'progress.json')['records'];verify_saved(args.output,records)
    import torch
    from pcb.core.team_config import load_team_config
    from pcb.core.context import PipelineContext
    from pcb.io import read_image
    from pcb.submission import validate_strict
    from pcb.pin.model_control_l42 import L42ControlSemantics
    from pcb.pin.rc_pair_l45 import RCPointSemantics,PinPointProjectionSubmission
    from tools.run_team_pipeline import build_runtime
    config=load_team_config(ROOT/'configs/team.json')
    pipeline,ocr,detector,device=build_runtime(config,'cuda','cuda',CACHE)
    mode='calibration' if args.method=='calibration' else 'regression'
    projection=args.method.endswith('projection')
    stages={'A':L42ControlSemantics(),'B':RCPointSemantics(mode,STUDY,projection)}
    original_submission=pipeline.submission
    projected_submission=PinPointProjectionSubmission(original_submission)
    completed={r['case_id'] for r in records};start=time.perf_counter()
    for item in initial['inputs']:
        cid=item['case_id']
        if cid in completed:continue
        if sha(item['path'])!=item['sha256']:raise ValueError('PNG changed')
        row={'case_id':cid,'image':item['path']};snapshots={};results={}
        torch.cuda.reset_peak_memory_stats()
        for arm in ('A','B'):
            folder=args.output/cid/arm
            if folder.exists():raise ValueError('Partial output retained')
            pipeline.pin_semantics=stages[arm]
            pipeline.submission=projected_submission if arm=='B' and projection else original_submission
            sem=stages[arm] if arm=='A' else stages[arm].baseline
            calls=sem.reader.calls if sem.reader else 0;hits=sem.reader.cache_hits if sem.reader else 0
            t0=time.perf_counter()
            with guard.inferring():
                image=read_image(Path(item['path']));h,w=image.shape[:2]
                context=PipelineContext(Path(item['path']).name,w,h,config,ocr,detector,
                    resources={'pin':{'device':device,'words_cache':CACHE/'pin_words','model_cache':CACHE/'pin_model_control_l42'}})
                artifacts=pipeline.run(image,context);validate_strict(artifacts.result,(w,h))
                scene=asdict(artifacts.scene)
                write(folder/'result.json',artifacts.result);write(folder/'scene.json',scene)
                write(folder/'diagnostics.json',artifacts.scene.diagnostics)
                results[arm]=artifacts.result
                snapshots[arm]={'texts':scene['texts'],'owners':[{k:v for k,v in c.items() if k!='pins'} for c in scene['components']],
                                'pins':{c['key']:c['pins'] for c in scene['components']}}
            if arm=='A':
                if sha(item['reference'])!=item['reference_sha256'] or read(folder/'result.json')!=read(item['reference']):
                    raise AssertionError('Frozen L42 control mismatch')
            trace=stages['B'].last_trace if arm=='B' else []
            row[arm]={'seconds':time.perf_counter()-t0,'files':{n:sha(folder/n) for n in ('result.json','scene.json','diagnostics.json')},
                      'model_OCR_calls':sem.reader.calls-calls,'model_cache_hits':sem.reader.cache_hits-hits,
                      'RC_pairs_proposed':len(trace),'RC_pairs_accepted':sum(r['accepted'] for r in trace)}
            event('arm_complete',method=args.method,case=cid,arm=arm,seconds=round(row[arm]['seconds'],2))
        if results['A']['components']!=results['B']['components'] or snapshots['A']['texts']!=snapshots['B']['texts'] or snapshots['A']['owners']!=snapshots['B']['owners']:
            raise AssertionError('Non-Pin upstream changed')
        for owner,component in results['A']['components'].items():
            if component['type'] not in ('r','c'):
                if results['A']['pins'][owner]!=results['B']['pins'][owner]:raise AssertionError('Other type Pin changed')
            else:
                a,b=results['A']['pins'][owner],results['B']['pins'][owner]
                if set(a)!=set(b) or any(a[k]['pinname']!=b[k]['pinname'] for k in a):raise AssertionError('RC identity/count changed')
            ap,bp=snapshots['A']['pins'][owner],snapshots['B']['pins'][owner]
            if [{k:v for k,v in p.items() if k!='tip'} for p in ap]!=[{k:v for k,v in p.items() if k!='tip'} for p in bp]:
                raise AssertionError('Pin fields other than tip changed')
        if projection:
            if results['A']['nets']!=results['B']['nets']:raise AssertionError('Projection must preserve complete network JSON')
            ordinary=read(STUDY/'regression'/args.cohort/cid/'B/result.json')
            if results['B']['pins']!=ordinary['pins']:raise AssertionError('Same-model projected Pin output differs')
        row.update({'A_reference_equal':True,'upstream_equal':True,'only_RC_points_changed':True,
                    'network_JSON_equal':results['A']['nets']==results['B']['nets'],
                    'projection_Pin_parity':True if projection else None,
                    'GPU_peak_allocated':torch.cuda.max_memory_allocated(),'GPU_peak_reserved':torch.cuda.max_memory_reserved()})
        records.append(row);progress(args,records,'INFERENCE_RUNNING');event('case_complete',completed=len(records),expected=len(initial['inputs']))
    verify_saved(args.output,records)
    if signature()!=current or guard.target_opens or network or len(records)!=len(initial['inputs']):raise AssertionError('Completion gate failed')
    complete={'count':len(records),'records':records,'GT_reads':guard.target_opens,'blocked_accesses':guard.blocked,
              'network_attempts':network,'all_A_reference_equal':True,'all_upstream_equal':True,
              'seconds':time.perf_counter()-start,'cold_start_timing':False,'OFFICIAL_SCORE':False}
    write(args.output/'inference_complete.json',complete);progress(args,records,'PREDICTIONS_COMPLETE_EVALUATION_PENDING')
    event('inference_complete',method=args.method,cohort=args.cohort,count=len(records),GT_reads=guard.target_opens)

def geometry_counts(pred,gt):
    from evaluate_v2 import component_key_map,point,prf
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    mapping=component_key_map(pred['components'],gt['components'])
    pcount=sum(len(pins) for owner,pins in pred['pins'].items() if pred['components'][owner]['type'] in ('r','c'))
    gcount=sum(len(pins) for owner,pins in gt['pins'].items() if gt['components'].get(owner,{}).get('type') in ('r','c'))
    tp=0
    for owner,pins in pred['pins'].items():
        mapped=mapping.get(owner);typ=pred['components'][owner]['type']
        if typ not in ('r','c') or gt['components'].get(mapped,{}).get('type')!=typ:continue
        a=list(pins.values());b=list(gt['pins'].get(mapped,{}).values())
        if not a or not b:continue
        matrix=np.asarray([[math.dist(point(p['point']),point(g['point']))<=5 for g in b] for p in a],np.int8)
        rr,cc=linear_sum_assignment(-matrix);tp+=int(matrix[rr,cc].sum())
    return prf(tp,pcount,gcount)

def evaluate(args):
    from tools.team_access_guard import Guard
    from evaluate_v2 import evaluate_case,aggregate,source_from_name,component_key_map,point,prf
    initial=read(args.output/'initial.json');complete=read(args.output/'inference_complete.json')
    if complete['count']!=len(COHORTS[args.cohort]) or complete['GT_reads'] or (args.output/'evaluation.json').exists():raise ValueError('All predictions and fresh evaluation required')
    if signature()!=initial['signature']:raise ValueError('Frozen resources changed')
    verify_saved(args.output,complete['records']);guard=Guard();cases={'A':[],'B':[]};geometry={'A':[],'B':[]};diff=[];targets=[]
    for item in complete['records']:
        cid=item['case_id'];saved=[args.output/cid/arm/'result.json' for arm in ('A','B')]
        predictions={arm:read(path) for arm,path in zip(('A','B'),saved)}
        paths=list(Path(item['image']).parent.glob('*_target.json'))
        if len(paths)!=1 or int(cid) not in COHORTS[args.cohort]:raise ValueError('Unsafe evaluation case')
        with guard.evaluating(paths[0],saved):gt=read(paths[0]);targets.append({'case':cid,'sha256':sha(paths[0])})
        true={};source=source_from_name(item['image'])
        for arm,pred in predictions.items():
            result=evaluate_case(pred,gt);cases[arm].append({'case_id':cid,'status':'ok','source':source,'evaluation':result})
            geom=geometry_counts(pred,gt);geometry[arm].append({'case_id':cid,'source':source,**geom})
            mapping=component_key_map(pred['components'],gt['components']);true[arm]=set()
            for owner,pins in pred['pins'].items():
                for key,pin in pins.items():
                    g=gt['pins'].get(mapping.get(owner),{}).get(key)
                    if g and pin['pinname']==g['pinname'] and math.dist(point(pin['point']),point(g['point']))<=5:true[arm].add((mapping[owner],key))
        ea,eb=cases['A'][-1]['evaluation'],cases['B'][-1]['evaluation']
        diff.append({'case_id':cid,'source':source,'gained':sorted(true['B']-true['A']),'lost':sorted(true['A']-true['B']),
                     'strict_net_TP':len(true['B'])-len(true['A']),
                     'geometry_net_TP':geometry['B'][-1]['tp']-geometry['A'][-1]['tp'],
                     'Pin_F1_delta':eb['metrics']['Pin']['f1']-ea['metrics']['Pin']['f1'],
                     'local_score_delta':eb['FinalScore']-ea['FinalScore']})
    summary={arm:aggregate(rows) for arm,rows in cases.items()};a,b=(summary[arm]['overall'] for arm in ('A','B'))
    delta={'Pin_TP':b['metrics']['Pin']['tp']-a['metrics']['Pin']['tp'],
           'Pin_macro_F1':b['metrics']['Pin']['macro_f1']-a['metrics']['Pin']['macro_f1'],
           'Pin_micro_F1':b['metrics']['Pin']['f1']-a['metrics']['Pin']['f1'],
           'local_score':b['FinalScore']-a['FinalScore']}
    geo={arm:{**prf(sum(r['tp'] for r in rows),sum(r['pred'] for r in rows),sum(r['gt'] for r in rows)),
              'macro_f1':sum(r['f1'] for r in rows)/len(rows)} for arm,rows in geometry.items()}
    geo_delta={'TP':geo['B']['tp']-geo['A']['tp'],'macro_F1':geo['B']['macro_f1']-geo['A']['macro_f1']}
    strict_positive=delta['Pin_TP']>0 and delta['Pin_macro_F1']>0
    geometry_positive=geo_delta['TP']>0 and geo_delta['macro_F1']>0
    positive=strict_positive or geometry_positive
    key='strict_net_TP' if strict_positive else 'geometry_net_TP'
    largest=max(diff,key=lambda r:(r[key],-int(r['case_id'])))['case_id']
    excluded=sum(r[key] for r in diff if r['case_id']!=largest)
    multi=sum(r[key]>0 for r in diff)>1
    advance=positive if args.cohort=='design' else positive and multi and excluded>0
    costs={family:b['metrics'][family]['macro_f1']-a['metrics'][family]['macro_f1'] for family in ('NetHypergraph','NetLine')}
    network_review=any(v<0 for v in costs.values()) and not args.method.endswith('projection')
    decision=('ADVANCE_TO_FIXED_CHECK' if args.cohort=='design' else 'ADVANCE_TO_FULL150' if args.cohort=='check' else 'FULL150_POSITIVE_REVIEW_OFFICIAL_COST') if advance else 'STOP_FINITE_REVIEW'
    result={'OFFICIAL_SCORE':False,'method':args.method,'cohort':args.cohort,'summary':summary,'cases':cases,
            'delta':delta,'differences':diff,'RC_geometry_diagnostic':geo,'RC_geometry_delta':geo_delta,
            'diagnostic_is_not_official_score':True,'geometry_cases':geometry,'decision':decision,'advance':advance,
            'network_cost_requires_review':network_review,'network_macro_deltas':costs,
            'excluded_largest':{'case':largest,'metric':key,'net_TP_without_largest':excluded},
            'target_hashes':targets,'GT_reads_in_inference':0,'evaluation_separate_process':True,
            'prediction_count':2*len(targets),'all_A_reference_equal':True,'all_upstream_equal':True,
            'all_other_Pin_fields_preserved':True,'model_resources_frozen':True,'runtime_costs':complete}
    write(args.output/'evaluation.json',result);write(REPORT/f'{args.method}_{args.cohort}_评价.json',result)
    write(args.output/'final_verification.json',{'status':'PASSED','decision':decision,'GT_reads_in_inference':0,
          'runtime_signature_verified':signature()==initial['signature'],'A_reference_parity':True})
    note=f'''# {args.method} / {args.cohort}验证

严格Pin TP：{a['metrics']['Pin']['tp']} → {b['metrics']['Pin']['tp']}（{delta['Pin_TP']:+d}）。
Pin macro F1差值：{delta['Pin_macro_F1']:+.8f}；本地综合差值：{delta['local_score']:+.8f}。
R/C位置5px诊断TP：{geo['A']['tp']} → {geo['B']['tp']}（{geo_delta['TP']:+d}），macro差值{geo_delta['macro_F1']:+.8f}；该诊断忽略非box身份字段，不是官方评分器。
网络macro变化：{costs}。需要处理网络成本：{network_review}。
剔除最大图{largest}后的{key}净TP：{excluded}。

决定：{decision}。GT-free预测保存后独立评价，训练模型和参数固定；A复现既有L42。原始输出：`{args.output}`。
'''
    (REPORT/f'{args.method}_{args.cohort}_报告.md').write_text(note,'utf-8')
    prep.status('COHORT_COMPLETE',method=args.method,cohort=args.cohort,decision=decision,strict_delta=delta,geometry_delta=geo_delta)
    event('evaluation_complete',method=args.method,cohort=args.cohort,delta=delta,geometry_delta=geo_delta,decision=decision,network_cost_requires_review=network_review)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('preflight','infer','evaluate'),required=True)
    parser.add_argument('--method',choices=METHODS,required=True);parser.add_argument('--cohort',choices=tuple(COHORTS),required=True)
    args=parser.parse_args();args.output=STUDY/args.method/args.cohort
    if not STUDY.parent.is_dir() or not (prep.TASK/'outputs').resolve().is_relative_to(Path('D:/Codex_Experiments/Pin_20261012/outputs').resolve()):raise RuntimeError('D output route unavailable')
    {'preflight':preflight,'infer':infer,'evaluate':evaluate}[args.phase](args)
