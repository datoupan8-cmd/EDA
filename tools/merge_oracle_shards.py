"""Merge disjoint Oracle shard outputs using the experiment's aggregation."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v2_experiments import aggregate,ORACLES
from pcb.io import write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('shards',nargs='+',type=Path);ap.add_argument('--output_dir',type=Path,required=True);a=ap.parse_args()
    rows=[]
    for shard in a.shards:
        rows.extend(json.loads((shard/'oracle_details.json').read_text(encoding='utf8'))['cases'])
    rows=sorted(rows,key=lambda x:x['case_id']);a.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(a.output_dir/'oracle_details.json',{'official':False,'cases':rows})
    report={'OFFICIAL':False,'reference':'CVJsonStd-derived diagnostic reference','split_manifest':'splits/v2_dev.json','case_count':len(rows),'oracles':aggregate(rows,ORACLES)}
    write_json(ROOT/'reports/v2_oracle_ablation.json',report);print(json.dumps({'cases':len(rows),'oracles':list(report['oracles'])}))

if __name__=='__main__':main()
