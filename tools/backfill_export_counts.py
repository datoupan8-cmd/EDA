"""Backfill exported JSON counts into an already completed stage matrix."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v2_experiments import aggregate
from pcb.io import write_json
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run_dir',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    detail=a.run_dir/'stage_details.json';data=json.loads(detail.read_text(encoding='utf8'));names=[]
    for row in data['cases']:
        for name,record in row.items():
            if name in ('case_id','error','traceback') or not isinstance(record,dict):continue
            if name not in names:names.append(name)
            pred=json.loads((a.run_dir/row['case_id']/name/'result.json').read_text(encoding='utf8'))
            record.update({'exported_pins':sum(len(x) for x in pred['pins'].values()),'exported_nets':len(pred['nets']),'exported_edges':sum(len(x['edges']) for x in pred['nets'].values())})
    write_json(detail,data);report=json.loads(a.report.read_text(encoding='utf8'));report['stages']=aggregate(data['cases'],names);write_json(a.report,report)
if __name__=='__main__':main()
