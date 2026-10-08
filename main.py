"""PNG-only official-schema baseline. This entry point never reads target JSON."""
from pathlib import Path
import sys
project_root=Path(__file__).resolve().parent
from pcb.runtime_paths import prepend_compatible_vendor_paths
prepend_compatible_vendor_paths(project_root)
local_runtime=project_root.parent/'work/runtime'
# The user's YOLO environment already contains binary vision packages. Only
# add the bundled fallback when OpenCV is genuinely unavailable; placing a
# different-Python binary runtime first would shadow the working environment.
try:
    import cv2  # noqa: F401
except ImportError:
    if local_runtime.exists():sys.path.insert(0,str(local_runtime))

import argparse,hashlib,json,time,traceback
from dataclasses import asdict
import cv2
from pcb.io import read_image,write_json,write_image
from pcb.vision import OCR,detect_scene
from pcb.submission import export,validate_strict

HOLDOUT_START=151

def overlay(image,scene):
    out=image.copy()
    for c in scene.components:
        x1,y1,x2,y2=map(int,c.bbox);cv2.rectangle(out,(x1,y1),(x2,y2),(255,90,0),1)
        cv2.putText(out,c.key,(x1,max(10,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.35,(160,0,160),1)
        for p in c.pins:cv2.circle(out,tuple(round(v) for v in p.tip),2,(0,0,255),-1)
    for i,n in enumerate(scene.nets):
        color=((i*67+80)%220,(i*127+70)%220,(i*41+40)%220)
        for a,b in n.segments:cv2.line(out,tuple(round(v) for v in a),tuple(round(v) for v in b),color,1)
    return out

def _debug_v3(image,scene,debug,mask,sk,suppressed,corridor,case_dir):
    import numpy as np
    base=image.copy()
    candidates=base.copy();final=base.copy();text_boxes=base.copy();links=base.copy();pins=base.copy();semantics=base.copy()
    for row in scene.diagnostics.get('large_box_candidates',[]):
        x1,y1,x2,y2=map(int,row);cv2.rectangle(candidates,(x1,y1),(x2,y2),(0,170,255),1)
    for row in scene.diagnostics.get('capacitor_geometry_candidates',[]):
        cv2.circle(candidates,tuple(map(int,row['center'])),5,(255,0,255),1)
    for c in scene.components:
        x1,y1,x2,y2=map(int,c.bbox);cv2.rectangle(final,(x1,y1),(x2,y2),(255,80,0),1);cv2.putText(final,c.key,(x1,max(10,y1-2)),0,.35,(180,0,160),1)
        for p in c.pins:
            cv2.circle(pins,tuple(map(int,map(round,p.tip))),3,(0,0,255),-1)
            cv2.putText(semantics,f'{p.number}:{p.name or ""}',tuple(map(int,map(round,p.tip))),0,.3,(0,0,180),1)
    for row in scene.diagnostics.get('text_roles',[]):
        x1,y1,x2,y2=map(int,row['bbox']);cv2.rectangle(text_boxes,(x1,y1),(x2,y2),(0,140,0),1);cv2.putText(text_boxes,row['role'][:4],(x1,max(9,y1-1)),0,.28,(0,100,0),1)
    centers={c.key:((c.bbox[0]+c.bbox[2])/2,(c.bbox[1]+c.bbox[3])/2) for c in scene.components}
    for row in scene.diagnostics.get('text_component_associations',[]):
        c=centers.get(row['component']); token=next((t for t in scene.texts if t.text==row['token']),None)
        if c and token:cv2.line(links,tuple(map(int,token.center)),tuple(map(int,c)),(0,120,255),1)
    write_image(case_dir/'01_component_candidates.png',candidates);write_image(case_dir/'02_component_final.png',final)
    write_image(case_dir/'03_text_boxes.png',text_boxes);write_image(case_dir/'04_text_component_links.png',links)
    write_image(case_dir/'05_pin_candidates.png',pins);write_image(case_dir/'06_pin_semantics.png',semantics)
    write_image(case_dir/'07_component_text_suppressed.png',suppressed if suppressed is not None else debug.get('text_mask',np.zeros(mask.shape,np.uint8)))
    write_image(case_dir/'08_wire_mask.png',mask)
    n,lab=cv2.connectedComponents((mask>0).astype('uint8'));cc=np.zeros_like(image)
    for i in range(1,n):cc[lab==i]=((i*71)%255,(i*131)%255,(i*43)%255)
    write_image(case_dir/'09_wire_cc.png',cc);write_image(case_dir/'10_topology_graph.png',overlay(image,scene));write_image(case_dir/'11_final_overlay.png',overlay(image,scene))
    if corridor is not None:write_image(case_dir/'terminal_preservation_corridors.png',corridor)

def _debug_v4(image,scene,debug,case_dir):
    def proposal_image(rows,color):
        out=image.copy()
        for row in rows:
            box=row.bbox if hasattr(row,'bbox') else row['bbox']
            typ=row.type if hasattr(row,'type') else row['type']
            conf=row.confidence if hasattr(row,'confidence') else row['confidence']
            x1,y1,x2,y2=map(int,box);cv2.rectangle(out,(x1,y1),(x2,y2),color,1)
            cv2.putText(out,f'{typ} {conf:.2f}',(x1,max(10,y1-2)),0,.32,color,1)
        return out
    write_image(case_dir/'01_yolo_proposals.png',proposal_image(debug.get('yolo_proposals',[]),(20,80,230)))
    write_image(case_dir/'02_geometry_proposals.png',proposal_image(debug.get('geometry_proposals',[]),(230,80,20)))
    write_image(case_dir/'03_fused_proposals.png',proposal_image(debug.get('fused_proposals',[]),(20,180,180)))
    tokens=image.copy()
    for role in scene.diagnostics.get('text_roles',[]):
        x1,y1,x2,y2=map(int,role['bbox']);cv2.rectangle(tokens,(x1,y1),(x2,y2),(30,150,30),1)
        cv2.putText(tokens,role['role'][:5],(x1,max(9,y1-1)),0,.28,(20,100,20),1)
    write_image(case_dir/'04_ocr_tokens.png',tokens)
    centers={i:((p.bbox[0]+p.bbox[2])/2,(p.bbox[1]+p.bbox[3])/2) for i,p in enumerate(debug.get('fused_proposals',[]))}
    def matching_image(block):
        out=tokens.copy()
        for event in block.get('accepted',[]):
            comp=event.get('component');proposal_index=event.get('proposal_index')
            c=centers.get(proposal_index)
            token_text=event.get('token')
            token=next((t for t in scene.texts if t.text==token_text),None)
            if c is None and comp:
                obj=next((x for x in scene.components if x.key==comp),None)
                if obj:c=((obj.bbox[0]+obj.bbox[2])/2,(obj.bbox[1]+obj.bbox[3])/2)
            if c and token:cv2.line(out,tuple(map(int,token.center)),tuple(map(int,c)),(0,80,255),1)
        return out
    write_image(case_dir/'05_designator_matching.png',matching_image(scene.diagnostics.get('designator_assignment',{})))
    write_image(case_dir/'06_name_matching.png',matching_image(scene.diagnostics.get('name_assignment',{})))
    write_image(case_dir/'07_value_matching.png',matching_image(scene.diagnostics.get('value_assignment',{})))
    final=image.copy()
    provenance={row['component']:row for row in scene.diagnostics.get('component_provenance',[])}
    for c in scene.components:
        x1,y1,x2,y2=map(int,c.bbox);cv2.rectangle(final,(x1,y1),(x2,y2),(255,80,0),1)
        src=provenance.get(c.key,{}).get('proposal_source','?')
        label=f'{c.key}|{c.type}|{c.name or ""}|{c.value or ""}|{src}'
        cv2.putText(final,label,(x1,max(10,y1-2)),0,.28,(170,0,150),1)
    write_image(case_dir/'08_final_components.png',final)

def run_one(image_path,case_dir,ocr,variant='baseline',pipeline='v2',enable_flying_labels=False,save_debug_images=False,detector=None,component_stage='F',text_rules='v4_1'):
    start=time.perf_counter();image=read_image(image_path)
    debug={};corridor=None
    if pipeline=='v2':
        import copy
        from pcb.vision_v2 import refine_v1_scene
        scene=refine_v1_scene(image,ocr,copy.deepcopy(detect_scene(image,ocr)),safe_fused_only=True,sanitize_pins=True,global_roles=False)
    elif pipeline=='v4_component':
        from pcb.vision_v4 import detect_scene_v4
        scene,debug=detect_scene_v4(image,ocr,detector,Path(image_path).name,component_stage,text_rules=text_rules)
    elif pipeline.startswith('v3'):
        from pcb.vision_v3 import detect_scene_v3
        scene,debug=detect_scene_v3(image,ocr,Path(image_path).name,pin_v3=pipeline!='v3_component')
    else:scene=detect_scene(image,ocr)
    color_debug=suppressed=None
    if pipeline=='v2' or pipeline in ('v3_component','v3_pin'):
        from pcb.wire_v2 import extract_wire_v2
        from pcb.topology_v2 import build_topology_v2
        mask,color_debug,suppressed=extract_wire_v2(image,scene)
        sk=build_topology_v2(scene,mask,'directional' if variant!='baseline' else 'fixed',merge_labels=enable_flying_labels)
    elif pipeline in ('v3_wire','v3','v4_component'):
        from pcb.wire_v3 import extract_wire_v3
        mask,color_debug,suppressed,corridor=extract_wire_v3(image,scene)
        if pipeline in ('v3','v4_component'):
            from pcb.topology_v3 import build_topology_v3
            sk=build_topology_v3(scene,mask,merge_labels=enable_flying_labels)
        else:
            from pcb.topology_v2 import build_topology_v2
            sk=build_topology_v2(scene,mask,'directional',merge_labels=enable_flying_labels)
    else:
        from pcb.wire import extract_wire
        from pcb.topology import build_topology
        mask=extract_wire(image,scene);sk=build_topology(scene,mask,variant)
    if getattr(ocr,'last_diagnostics',None):scene.diagnostics['ocr']=dict(ocr.last_diagnostics)
    result=export(scene)
    validate_strict(result,(scene.width,scene.height))
    case_dir=Path(case_dir);write_json(case_dir/'result.json',result)
    elapsed=time.perf_counter()-start
    config_hash=hashlib.sha256(json.dumps({'pipeline':pipeline,'component_stage':component_stage,'text_rules':text_rules,'variant':variant,'flying_labels':enable_flying_labels,'ocr_backend':getattr(ocr,'name','rapidocr'),'mode':'image-only'},sort_keys=True).encode()).hexdigest()[:12]
    scene.diagnostics.update({'input_image':str(image_path),'mode':'image-only','pipeline':pipeline,'variant':variant,'config_hash':config_hash,'seconds':round(elapsed,3),
        'components':len(result['components']),'pins':sum(len(x) for x in result['pins'].values()),'terminal_candidates':sum(len(c.pins) for c in scene.components),
        'unexported_terminals':sum(not p.exportable for c in scene.components for p in c.pins),'scene_nets_before_export_filter':len(scene.nets),
        'scene_edges_before_export_filter':sum(len(n.segments) for n in scene.nets),'nets':len(result['nets']),
        'net_edges':sum(len(n['edges']) for n in result['nets'].values()),'strict_contract_valid':True,'token_count':0})
    write_json(case_dir/'diagnostics.json',scene.diagnostics)
    if save_debug_images:
        write_json(case_dir/'scene.json',asdict(scene));write_image(case_dir/'overlay.png',overlay(image,scene));write_image(case_dir/'wire_mask.png',mask);write_image(case_dir/'skeleton.png',sk.astype('uint8')*255)
        if color_debug is not None:write_image(case_dir/'wire_color_debug.png',color_debug)
        if suppressed is not None:write_image(case_dir/'suppressed_regions.png',suppressed)
        if pipeline.startswith('v3'):_debug_v3(image,scene,debug,mask,sk,suppressed,corridor,case_dir)
        elif pipeline=='v4_component':_debug_v4(image,scene,debug,case_dir)
    return scene.diagnostics

def run_one_modular(image_path,case_dir,ocr,config,save_debug_images=False,detector=None):
    """Run the registered Stage pipeline while preserving legacy case behavior."""
    from pcb.core import ModularPipeline, PipelineContext, build_default_registry
    start=time.perf_counter();image=read_image(image_path);height,width=image.shape[:2]
    context=PipelineContext(Path(image_path).name,width,height,config,ocr,detector)
    artifacts=ModularPipeline(config,build_default_registry()).run(image,context)
    scene=artifacts.scene;result=artifacts.result
    case_dir=Path(case_dir);write_json(case_dir/'result.json',result)
    elapsed=time.perf_counter()-start
    config_hash=hashlib.sha256(json.dumps({'pipeline':config.legacy_pipeline_name,'component_stage':config.component_stage,'text_rules':config.text_rules,'variant':config.variant,'flying_labels':config.enable_flying_labels,'ocr_backend':getattr(ocr,'name','rapidocr'),'mode':'image-only'},sort_keys=True).encode()).hexdigest()[:12]
    scene.diagnostics.update({'input_image':str(image_path),'mode':'image-only','pipeline':config.legacy_pipeline_name,'variant':config.variant,'config_hash':config_hash,'seconds':round(elapsed,3),
        'components':len(result['components']),'pins':sum(len(x) for x in result['pins'].values()),'terminal_candidates':sum(len(c.pins) for c in scene.components),
        'unexported_terminals':sum(not p.exportable for c in scene.components for p in c.pins),'scene_nets_before_export_filter':len(scene.nets),
        'scene_edges_before_export_filter':sum(len(n.segments) for n in scene.nets),'nets':len(result['nets']),
        'net_edges':sum(len(n['edges']) for n in result['nets'].values()),'strict_contract_valid':True,'token_count':0})
    write_json(case_dir/'diagnostics.json',scene.diagnostics)
    if save_debug_images:
        write_json(case_dir/'scene.json',asdict(scene));write_image(case_dir/'overlay.png',overlay(image,scene));write_image(case_dir/'wire_mask.png',artifacts.wire.mask);write_image(case_dir/'skeleton.png',artifacts.topology.skeleton.astype('uint8')*255)
        if artifacts.wire.color_debug is not None:write_image(case_dir/'wire_color_debug.png',artifacts.wire.color_debug)
        if artifacts.wire.suppressed is not None:write_image(case_dir/'suppressed_regions.png',artifacts.wire.suppressed)
        if config.component=='v4':_debug_v4(image,scene,artifacts.component_debug,case_dir)
        elif config.component=='v3':_debug_v3(image,scene,artifacts.component_debug,artifacts.wire.mask,artifacts.topology.skeleton,artifacts.wire.suppressed,artifacts.wire.corridor,case_dir)
    return scene.diagnostics

def _official_root(path):
    root=Path(path)
    if root.name=='200_train_cases':return root
    child=root/'200_train_cases'
    if child.is_dir():return child
    raise ValueError('--input_dir must be 200_train_cases or its direct parent')

def official_images(root,start,end):
    if not (1<=start<=end<HOLDOUT_START):raise ValueError('This benchmark build is hard-guarded to cases 0001..0150; holdout access refused')
    base=_official_root(root);rows=[]
    for number in range(start,end+1):
        case=f'{number:04d}';folder=base/case
        images=sorted(folder.glob('*.png'))
        if len(images)!=1:raise FileNotFoundError(f'{case}: expected exactly one PNG, got {len(images)}')
        rows.append((case,images[0]))
    return rows

def parse_args():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--image',type=Path);g.add_argument('--input_dir',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--output_dir',type=Path,default=Path('runs/v2_baseline'))
    p.add_argument('--case_start',type=int,default=1);p.add_argument('--case_end',type=int,default=150)
    p.add_argument('--case_ids',help='Optional comma-separated allowed cases, e.g. 0002,0050,0087')
    p.add_argument('--variant',choices=['baseline','improved'],default='baseline');p.add_argument('--pipeline',choices=['v1','v2','v3_component','v3_pin','v3_wire','v3','v4_component'],default='v4_component')
    p.add_argument('--component_stage',choices=['B','C','D','E','F','BEST'],default='BEST')
    p.add_argument('--text_rules',choices=['v4','v4_1'],default='v4_1')
    p.add_argument('--weights',type=Path,default=Path('models/component_yolo11n_continue_v2_best.pt'))
    p.add_argument('--device',default='auto',help='YOLO device: auto, cpu, or CUDA index such as 0')
    p.add_argument('--yolo_cache_dir',type=Path,default=Path('runs/yolo_cache'))
    p.add_argument('--ocr_backend',choices=['easyocr_tiled','rapidocr','hybrid','selective_local'],default='hybrid')
    p.add_argument('--ocr_device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--ocr_model_dir',type=Path,default=Path('models/easyocr'))
    p.add_argument('--allow_ocr_download',action='store_true',help='Allow EasyOCR to download missing model files')
    p.add_argument('--enable_flying_labels',action='store_true');p.add_argument('--save_debug_images',action='store_true')
    p.add_argument('--cache_dir',type=Path,default=Path('runs/ocr_cache_v4_1'));p.add_argument('--time_limit',type=float,default=7000)
    p.add_argument('--orchestrator',choices=['legacy','modular'],default='legacy',help='legacy preserves the original entry; modular uses registered Stage wrappers')
    p.add_argument('--pipeline_config',type=Path,help='JSON Stage selection used only with --orchestrator modular')
    return p.parse_args()

def main():
    args=parse_args()
    if args.ocr_backend in {'easyocr_tiled','hybrid','selective_local'}:
        from pcb.ocr_backends import HybridOCR,SelectiveLocalOCR,TiledEasyOCR
        tiled=TiledEasyOCR(args.cache_dir,args.ocr_model_dir,device=args.ocr_device,allow_download=args.allow_ocr_download)
        if args.ocr_backend=='hybrid':ocr=HybridOCR(tiled,OCR(args.cache_dir))
        elif args.ocr_backend=='selective_local':ocr=SelectiveLocalOCR(tiled,OCR(args.cache_dir))
        else:ocr=tiled
    else:ocr=OCR(args.cache_dir)
    modular_config=None
    if args.orchestrator=='modular':
        from pcb.core.config import PipelineConfig
        if args.pipeline_config:
            modular_config=PipelineConfig.load(args.pipeline_config)
        else:
            if args.pipeline!='v4_component':raise ValueError('Without --pipeline_config, modular orchestration preserves the current v4_component baseline only')
            modular_config=PipelineConfig.current(component_stage=args.component_stage,text_rules=args.text_rules,variant=args.variant,enable_flying_labels=args.enable_flying_labels)
    start=time.perf_counter();records=[];detector=None
    if args.pipeline=='v4_component' or (modular_config is not None and modular_config.component=='v4'):
        from pcb.component_detector_yolo import YoloComponentDetector
        detector=YoloComponentDetector(args.weights,args.device,cache_dir=args.yolo_cache_dir)
    if args.image:
        rows=[(args.image.stem,args.image)]
    elif args.case_ids:
        numbers=[int(value.strip()) for value in args.case_ids.split(',') if value.strip()]
        if not numbers or any(number<1 or number>=HOLDOUT_START for number in numbers):raise ValueError('--case_ids accepts only 0001..0150')
        rows=[]
        for number in numbers:rows.extend(official_images(args.input_dir,number,number))
    else:
        rows=official_images(args.input_dir,args.case_start,args.case_end)
    for index,(case,image_path) in enumerate(rows):
        if time.perf_counter()-start>args.time_limit:
            records.extend({'case_id':c,'image':str(p),'status':'not_started_time_budget','stage':'scheduler'} for c,p in rows[index:]);break
        case_dir=(args.output.parent if args.image and args.output else args.output_dir/case)
        try:
            if args.orchestrator=='modular':diag=run_one_modular(image_path,case_dir,ocr,modular_config,args.save_debug_images,detector)
            else:diag=run_one(image_path,case_dir,ocr,args.variant,args.pipeline,args.enable_flying_labels,args.save_debug_images,detector,args.component_stage,args.text_rules)
            if args.image and args.output and args.output.name!='result.json':
                # Copy the already-validated payload to the explicit filename.
                data=json.loads((case_dir/'result.json').read_text(encoding='utf-8'));write_json(args.output,data)
            record={'case_id':case,'image':str(image_path),'status':'ok',**diag};records.append(record)
            print(json.dumps({k:record[k] for k in ('case_id','components','pins','nets','seconds','status')},ensure_ascii=False),flush=True)
        except Exception as exc:
            record={'case_id':case,'image':str(image_path),'status':'failed','stage':'inference_or_validation','error':str(exc),'traceback':traceback.format_exc()};records.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
    summary={'mode':'image-only','ocr_backend':args.ocr_backend,'case_start':args.case_start if args.input_dir else None,'case_end':args.case_end if args.input_dir else None,
        'holdout_guard':'151..200 and 10_GTcase are not enumerated','seconds':round(time.perf_counter()-start,3),'success_count':sum(r['status']=='ok' for r in records),
        'failure_count':sum(r['status']!='ok' for r in records),'cases':records,'official_score':None}
    if args.orchestrator=='modular':summary['orchestrator']='modular'
    write_json(args.output_dir/'batch_summary.json',summary)
    return int(summary['failure_count']>0)

if __name__=='__main__':raise SystemExit(main())
