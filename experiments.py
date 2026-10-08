"""Reproducible ablation on supplied public cases. Diagnostic scores are NOT official."""
from pathlib import Path
import sys
runtime=Path(__file__).resolve().parent.parent/'work/runtime'
if runtime.exists():sys.path.insert(0,str(runtime))
import argparse,json,time,copy,traceback
from dataclasses import asdict
from collections import Counter
from pcb.io import read_image,write_image,write_json,discover_images,annotation_for
from pcb.annotation import load_annotation,reference_nets
from pcb.vision import OCR,detect_scene
from pcb.wire import extract_wire
from pcb.topology import build_topology
from pcb.submission import export
from pcb.evaluate import evaluate
from main import overlay

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_dir',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,default=Path('runs/experiment'))
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--audit_only',action='store_true',help='Inspect all file/annotation structures without model inference')
    a=p.parse_args();out=a.output_dir;images=discover_images(a.input_dir)
    audit=[];categories=Counter()
    for image in images:
        try:
            d=json.loads(annotation_for(image).read_text(encoding='utf8'))
            categories.update(c['category'] for c in d['components'].values())
            audit.append({'case':image.parent.name,'image':str(image),'components':len(d['components']),
                          'pins':sum(len(c.get('pins',{})) for c in d['components'].values()),'canvas':d['canvas']})
        except Exception as e:audit.append({'case':image.parent.name,'error':str(e)})
    write_json(out/'dataset_inventory.json',{'cases':audit,'categories':categories,'image_count':len(images)})
    if a.audit_only:
        print(json.dumps({'image_count':len(images),'errors':sum('error' in r for r in audit)}));return 0
    records=[];ocr=OCR(out/'ocr_cache')
    for image in images[:a.limit or None]:
        case=image.parent.name;start=time.perf_counter()
        try:
            im=read_image(image)
            # Run the complete visual front end before loading any sidecar annotation.
            front_start=time.perf_counter();visual=detect_scene(im,ocr);front_seconds=time.perf_counter()-front_start
            ref_scene,data=load_annotation(annotation_for(image));ref_scene.nets=reference_nets(ref_scene,data)
            ref=export(ref_scene);write_json(out/case/'derived_reference_NOT_OFFICIAL.json',ref)
            record={'case':case,'image':str(image),'image_size':list(im.shape[:2]),'frontend_seconds':front_seconds,
                    'reference':'derived from noisy CVJsonStd, not official ground truth','variants':{}}
            for mode,scene0 in [('image',visual),('annotation-assisted',ref_scene)]:
                for variant in ['baseline','improved']:
                    ts=time.perf_counter();scene=copy.deepcopy(scene0);mask=extract_wire(im,scene)
                    sk=build_topology(scene,mask,variant);pred=export(scene)
                    name=mode+'_'+variant;folder=out/case/name
                    elapsed=time.perf_counter()-ts
                    write_json(folder/'result.json',pred)
                    write_json(folder/'diagnostics.json',scene.diagnostics)
                    metrics=evaluate(pred,ref);write_json(folder/'diagnostic_metrics.json',metrics)
                    record['variants'][name]={'components':len(scene.components),'pins':sum(len(c.pins) for c in scene.components),
                        'nets':len(scene.nets),'unattached_pins':len(scene.diagnostics['unattached_pins']),
                        'topology_and_export_seconds':elapsed,'metrics':metrics}
                    if mode=='image' and variant=='improved':
                        write_image(folder/'overlay.png',overlay(im,scene));write_image(folder/'wire_mask.png',mask)
                        write_json(folder/'scene.json',asdict(scene))
            record['seconds']=time.perf_counter()-start;record['status']='ok';records.append(record)
            print(json.dumps({'case':case,'seconds':record['seconds'],'image':record['variants']['image_improved']},ensure_ascii=False),flush=True)
        except Exception as e:
            records.append({'case':case,'status':'failed','error':str(e),'traceback':traceback.format_exc()})
            print(json.dumps(records[-1]),flush=True)
        write_json(out/'summary.json',{'official':False,'records':records})
    return 1 if any(r['status']!='ok' for r in records) else 0

if __name__=='__main__':raise SystemExit(main())
