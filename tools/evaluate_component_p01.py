"""Standalone P01 vs current Component control, fixed 0001..0150 only."""
import argparse,hashlib,json,shutil,statistics,sys,time
from dataclasses import asdict,replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for relative in ('.runtime/ort_gpu_debug','.runtime/alternative_frontend','.runtime/parseq_runtime'):sys.path.insert(0,str(ROOT/relative))
from tools.evaluate_component_local_joint import select_cases,write,export_components


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ocr-cache',type=Path,help='Optional reusable image-keyed shared OCR cache')
    p.add_argument('--shared-ocr',type=Path,help='Optional sealed image-only shared OCR tokens; hash checked for every image')
    p.add_argument('--device',default='auto');p.add_argument('--ocr-device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--reference',type=Path,help='Optional previous image-only comparison root; never target annotations')
    args=p.parse_args()
    from tools.team_access_guard import Guard
    guard=Guard();rows=select_cases(args.dataset);ids=[c for c,_,_ in rows]
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use fresh output')
    args.output.mkdir(parents=True,exist_ok=True)
    import torch,cv2
    torch.set_num_threads(4);cv2.setNumThreads(4)
    from pcb.component.detector_yolo import YoloComponentDetector
    from pcb.component.selective_filter import PROFILE,ComponentStageSelectiveP01
    from pcb.component.ocr_ablation import PairedPPOcrV5Backend
    from pcb.component.registry import register_stages
    from pcb.core.registry import StageRegistry
    from pcb.core.team_config import load_team_config
    from pcb.core.context import PipelineContext
    from pcb.core.interfaces import TextStageOutput
    from pcb.io import read_image
    from pcb.schema import Text
    current_config=load_team_config(ROOT/'configs/team.json')
    config=replace(current_config,component='ordered_yolo11m_local_parseq_v1')
    selected_config=replace(config,component=PROFILE)
    registry=StageRegistry();register_stages(registry);candidate=registry.create('component',PROFILE);control=registry.create('component',config.component)
    if not isinstance(candidate,ComponentStageSelectiveP01):raise AssertionError('Wrong registry binding')
    detector=YoloComponentDetector(ROOT/'models/component/yolo11m_finetune120/best.pt',args.device,cache_dir=args.output/'runtime_cache/yolo')
    import onnxruntime
    ocr_cuda=(torch.cuda.is_available() and 'CUDAExecutionProvider' in onnxruntime.get_available_providers()) if args.ocr_device=='auto' else args.ocr_device=='cuda'
    shared_manifest=None
    if args.shared_ocr is not None:
        shared_manifest=json.loads((args.shared_ocr/'manifest.json').read_text(encoding='utf-8'))
        if shared_manifest['ACTUAL_CASE_IDS']!=ids or shared_manifest['GT_opens'] or not shared_manifest['image_only']:raise ValueError('Unsealed shared OCR')
        reader=None;ocr_sha=shared_manifest['OCR_source_sha256']
    else:
        if ocr_cuda:
            import onnxruntime
            if hasattr(onnxruntime,'preload_dlls'):onnxruntime.preload_dlls()
        reader=PairedPPOcrV5Backend(args.ocr_cache or args.output/'runtime_cache/shared_ocr',normalization='legacy',use_cuda=ocr_cuda)
        ocr_sha=sha(ROOT/'pcb/component/ocr_ablation.py')
    source_names=['tools/evaluate_component_p01.py','pcb/component/selective_filter.py','pcb/component/registry.py','pcb/component/local_combination.py','pcb/component/parseq_local.py','pcb/component/parseq_adapted.py','pcb/component/ocr_ablation.py','configs/component/current.json','configs/component/p01.json','configs/pin/current.json','configs/wire_topology/current.json','evaluate_v2.py']
    sources={}
    for name in source_names:
        sources[name]=sha(ROOT/name);dest=args.output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    manifest={'ACTUAL_CASE_IDS':ids,'scope':'Component_only','OFFICIAL_SCORE':False,'reserved_used':False,'variants':['baseline','P01'],'source_sha256':sources,
        'P01_parameters_sha256':candidate.parameter_sha256,'OCR_source_sha256':ocr_sha,'fullpage_OCR':'sealed image-only shared tokens' if shared_manifest else 'same image-only PP-OCRv5 beam output for both candidates; new inference if no cache',
        'shared_OCR_manifest_sha256':sha(args.shared_ocr/'manifest.json') if shared_manifest else None,
        'other_candidates_loaded':False,'threshold_changed':False,'evaluation_case_exclusions':[],'GT_used_in_inference':False}
    write(args.output/'manifest.json',manifest);print(json.dumps({'ACTUAL_CASE_IDS':ids,'scope':'P01_only_against_control','GT_opens':0}),flush=True)
    hashes={};timings={};start=time.perf_counter();ocr_cache_hits=0
    for index,(cid,image_path,_) in enumerate(rows,1):
        with guard.inferring():
            image=read_image(image_path);h,w=image.shape[:2]
            if shared_manifest:
                text_path=args.shared_ocr/(cid+'.json')
                if sha(text_path)!=shared_manifest['sha256'][cid]:raise ValueError('Shared OCR changed')
                data=json.loads(text_path.read_text(encoding='utf-8'))
                if data['image_sha256']!=hashlib.sha256(image.tobytes()).hexdigest():raise ValueError('Shared OCR image mismatch')
                texts=TextStageOutput([Text(**row) for row in data['texts']]);ocr_cache_hits+=1
            else:
                texts=TextStageOutput(reader.recognize_pair(image)['beam']);ocr_cache_hits+=bool(reader.last_diagnostics['cache_hit'])
            context=PipelineContext(image_path.stem,w,h,selected_config,None,detector=detector,resources={'component':{'cache_dir':args.output/'runtime_cache'}})
            before=[asdict(t) for t in texts.texts];t0=time.perf_counter();p01=candidate.run(image,texts,context);p01_seconds=time.perf_counter()-t0
            baseline=control.run(image,texts,replace(context,config=config))
            if [asdict(t) for t in texts.texts]!=before or p01.texts is not texts.texts:raise AssertionError('Shared OCR changed')
            for name,result in [('baseline',baseline),('P01',p01)]:
                path=args.output/cid/name/'components.json';write(path,export_components(result,h));hashes[str(path.relative_to(args.output))]=sha(path)
        # These are frozen image-only outputs, never target annotations.
        if args.reference is not None:
            for name in ('baseline','P01'):
                reference=args.reference/cid/name/'components.json'
                if json.loads((args.output/cid/name/'components.json').read_text(encoding='utf-8'))!=json.loads(reference.read_text(encoding='utf-8')):raise AssertionError('Standalone parity '+cid+'/'+name)
        timings[cid]={'P01_component_seconds':p01_seconds,'YOLO_cache_hit':p01.diagnostics['yolo']['cache_hit'],'PARSeq_cache_hit':p01.diagnostics['component_local_recognition']['cache_hit']}
        write(args.output/'progress.json',{'completed':index,'case':cid,'GT_opens':guard.target_opens})
        if index%10==0:print(json.dumps({'inferred':index,'GT_opens':guard.target_opens}),flush=True)
    if guard.target_opens or guard.blocked:raise AssertionError('Inference accessed labels')
    write(args.output/'predictions_sealed_before_GT.json',{'ACTUAL_CASE_IDS':ids,'GT_opens':0,'sha256':hashes});write(args.output/'timings.json',timings)
    from evaluate_v2 import component_key_map,component_counts,prf
    cases=[]
    for cid,_,folder in rows:
        paths=[args.output/cid/name/'components.json' for name in ('baseline','P01')];targets=list(folder.glob('*_target*.json'))
        if len(targets)!=1 or targets[0].resolve().parent!=folder:raise ValueError('Unsafe target')
        with guard.evaluating(targets[0],paths):gt=json.loads(targets[0].read_text(encoding='utf-8'))['components']
        scores={}
        for name,path in zip(('baseline','P01'),paths):
            if sha(path)!=hashes[str(path.relative_to(args.output))]:raise AssertionError('Prediction changed')
            pred=json.loads(path.read_text(encoding='utf-8'));scores[name]=prf(*component_counts(pred,gt,component_key_map(pred,gt)))
        cases.append({'case_id':cid,'scores':scores})
    aggregate={}
    for name in ('baseline','P01'):
        aggregate[name]={}
        for split,group in [('all150',cases),('train120',[r for r in cases if int(r['case_id'])%5]),('check30',[r for r in cases if not int(r['case_id'])%5])]:
            values=[r['scores'][name] for r in group];aggregate[name][split]={'macro_ComponentF1':statistics.mean(v['f1'] for v in values),'micro_Component':prf(*(sum(v[k] for v in values) for k in ('tp','pred','gt')))}
    delta={s:aggregate['P01'][s]['macro_ComponentF1']-aggregate['baseline'][s]['macro_ComponentF1'] for s in ('all150','train120','check30')}
    per_page=[r['scores']['P01']['f1']-r['scores']['baseline']['f1'] for r in cases]
    if any(sha(ROOT/name)!=value for name,value in sources.items()):raise AssertionError('Source/config changed during test')
    report={**manifest,'aggregate':aggregate,'delta':delta,'cases':cases,'GT_opens':guard.target_opens,'guard_blocked':guard.blocked,'exact_parity_with_fixed_P01':True if args.reference is not None else None,
        'improved':sum(d>1e-12 for d in per_page),'worse':sum(d < -1e-12 for d in per_page),'same':sum(abs(d)<=1e-12 for d in per_page),
        'local_improvement_confirmed':all(d>0 for d in delta.values()),'P01_component_mean_seconds':statistics.mean(r['P01_component_seconds'] for r in timings.values()),
        'elapsed_seconds':time.perf_counter()-start,'workspace_default_unchanged':True,'submission_profile':PROFILE,'shared_OCR_cache_hits':ocr_cache_hits}
    write(args.output/'report.json',report);print(json.dumps({'aggregate':aggregate,'delta':delta,'local_improvement_confirmed':report['local_improvement_confirmed']}),flush=True)


if __name__=='__main__':main()
