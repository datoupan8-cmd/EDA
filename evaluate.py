from pathlib import Path
import sys,argparse,json
runtime=Path(__file__).resolve().parent.parent/'work/runtime'
try:
    import scipy  # noqa: F401
except ImportError:
    if runtime.exists():sys.path.insert(0,str(runtime))
from pcb.evaluate import evaluate
from pcb.submission import validate
from pcb.io import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description='NON-OFFICIAL diagnostic evaluator. Official evaluator has not been supplied.')
    p.add_argument('--prediction',required=True,type=Path);p.add_argument('--reference',required=True,type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();pred=json.loads(a.prediction.read_text(encoding='utf8'));ref=json.loads(a.reference.read_text(encoding='utf8'))
    validate(pred);validate(ref);r=evaluate(pred,ref);print(json.dumps(r,indent=2,ensure_ascii=False))
    if a.output:write_json(a.output,r)
