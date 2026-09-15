"""Build the curated V2 delivery archive and verify every ZIP member."""
from __future__ import annotations
import hashlib,json,os,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT.parent/'PCB_Competition_Solution_V2.zip'
PREFIX=ROOT.name

ROOT_FILES={'.dockerignore','.gitignore','Dockerfile','README.md','THIRD_PARTY_NOTICES.md',
            'main.py','evaluate_v2.py','oracle_experiments.py','requirements.txt'}
REPORTS={
 'v2_baseline_metrics.json','v2_oracle_metrics.json','v2_error_budget.md',
 'v2_schema_migration.md','v2_evaluator_notes.md','v2_next_stage.md',
 'v2_baseline_summary.md','v2_contract_validation.json',
 'v2_all_prediction_contract_validation.json','v2_test_results.txt',
 'v2_final_validation.json','contract_oracle_gt_component.json',
 'contract_oracle_gt_component_pin_tip.json','contract_oracle_gt_components_pins.json',
 'contract_oracle_gt_edges_graph.json'
}

def selected(path:Path)->bool:
    rel=path.relative_to(ROOT);parts=rel.parts
    if path.name.endswith('.pyc') or '__pycache__' in parts:return False
    if len(parts)==1:return path.name in ROOT_FILES
    if parts[0]=='pcb':return path.suffix=='.py'
    if parts[0]=='tests':return path.suffix=='.py'
    if parts[0]=='tools':return path.name in {'validate_predictions.py','package_official_v2.py'}
    if parts[0]=='reports':return path.name in REPORTS or path.name=='v2_delivery_manifest.json'
    if parts[:2]==('runs','v2_baseline'):return path.is_file()
    if parts[:2]==('experiments','oracle'):return path.is_file()
    if parts[:2]==('examples','0001'):return path.is_file()
    return False

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    files=sorted((p for p in ROOT.rglob('*') if p.is_file() and selected(p) and p.name!='v2_delivery_manifest.json'),key=lambda p:p.as_posix())
    manifest={'delivery':PREFIX,'file_count_without_manifest':len(files),'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
    manifest_path=ROOT/'reports'/'v2_delivery_manifest.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    files.append(manifest_path);files.sort(key=lambda p:p.as_posix())
    temp=ARCHIVE.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,f'{PREFIX}/{p.relative_to(ROOT).as_posix()}')
    with zipfile.ZipFile(temp) as z:
        bad=z.testzip();members=z.namelist()
    if bad is not None:raise RuntimeError(f'corrupt member: {bad}')
    if len(members)!=len(files):raise RuntimeError('member count mismatch')
    os.replace(temp,ARCHIVE)
    result={'archive':str(ARCHIVE),'members':len(members),'bytes':ARCHIVE.stat().st_size,'sha256':sha(ARCHIVE),'testzip':None}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
