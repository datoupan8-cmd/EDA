"""Build reproducible train/development manifests without changing Dataset300."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT.parent/'work/runtime'
if RUNTIME.exists():sys.path.insert(0,str(RUNTIME))
sys.path.insert(0,str(ROOT))
import numpy as np
from pcb.io import discover_images,read_image,annotation_for,write_json

DEFAULT_DEV=['0001','0002','0003','0005','0006','0007','0010','0015','0020','0030','0040','0050',
             '0075','0100','0125','0150','0175','0200','0201','0202','0210','0225','0250','0275','0300']

def describe(p):
    im=read_image(p);d=json.loads(annotation_for(p).read_text(encoding='utf8'))
    delta=im.max(axis=2).astype(np.int16)-im.min(axis=2).astype(np.int16)
    name=p.name.lower()
    source='datasheet' if 'datasheet' in name else 'kicad' if 'kicad' in name else 'altium' if 'altium' in name else 'other'
    comps=d.get('components',{})
    cats={str(c.get('category','unknown')).lower() for c in comps.values()}
    return {'case_id':p.parent.name,'image':str(p.resolve()),'annotation':str(annotation_for(p).resolve()),
            'source':source,'width':im.shape[1],'height':im.shape[0],'color_fraction':round(float((delta>25).mean()),4),
            'components':len(comps),'pins':sum(len(c.get('pins',{})) for c in comps.values()),
            'large_ic':any(str(c.get('category','')).lower()=='box' and len(c.get('pins',{}))>=24 for c in comps.values()),
            'connector':any(x in cats for x in ('pin','net_input','net_output','net_bidirection')),
            'flying_net':any(bool(e.get('isFlywire')) or bool(e.get('netlabel')) for e in d.get('edges',{}).values()),
            'dense_pins':sum(len(c.get('pins',{})) for c in comps.values())>=100,'categories':sorted(cats)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--output_dir',type=Path,default=ROOT/'splits')
    a=ap.parse_args();records=[describe(p) for p in discover_images(a.dataset)];by={r['case_id']:r for r in records}
    missing=[x for x in DEFAULT_DEV if x not in by]
    if missing:raise ValueError(f'Missing fixed cases: {missing}')
    dev=[by[x] for x in DEFAULT_DEV];train=[r for r in records if r['case_id'] not in set(DEFAULT_DEV)]
    write_json(a.output_dir/'v2_dev.json',{'frozen':True,'case_count':len(dev),'cases':dev})
    write_json(a.output_dir/'v2_train.json',{'frozen':True,'case_count':len(train),'cases':train})
    summary={'all':len(records),'train':len(train),'dev':len(dev),'dev_case_ids':DEFAULT_DEV,
             'dev_sources':{s:sum(r['source']==s for r in dev) for s in ('kicad','altium','datasheet','other')},
             'dev_color':sum(r['color_fraction']>.01 for r in dev),'dev_monochrome':sum(r['color_fraction']<=.01 for r in dev),
             'dev_large_ic':sum(r['large_ic'] for r in dev),'dev_connector':sum(r['connector'] for r in dev),
             'dev_flying_net':sum(r['flying_net'] for r in dev),'dev_dense_pins':sum(r['dense_pins'] for r in dev)}
    write_json(a.output_dir/'v2_split_summary.json',summary);print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
