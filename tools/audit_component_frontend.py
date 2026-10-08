"""Summarize component-front-end quality from a completed frozen stage matrix."""
from pathlib import Path
import argparse,json,statistics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--details',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    rows=[r for r in json.loads(a.details.read_text(encoding='utf8'))['cases'] if 'v1_baseline' in r]
    keys=['ComponentF1','component_key_precision','component_key_recall','component_bbox_precision','component_bbox_recall']
    out={'OFFICIAL':False,'case_count':len(rows),'frontend':'OpenCV symbol proposals + RapidOCR designator anchors','metrics':{}}
    for k in keys:
        vals=[]
        for r in rows:
            m=r['v1_baseline']['metrics'];vals.append(m[k] if k in m else m['diagnostic_layers'][k])
        out['metrics'][k]=statistics.mean(vals) if vals else None
    out['trained_detector_comparison']={'status':'not_run','reason':'torch and ultralytics are not installed; no pretrained PCB-specific weight is present','dataset_tool':'tools/prepare_component_dataset.py','training_tool':'tools/train_component_detector.py'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()
