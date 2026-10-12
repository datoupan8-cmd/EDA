"""Check owner file boundaries across branch commits, index and working tree.

This is a collaboration guard, not a security sandbox or an algorithm test.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PREFIXES={
    'wire':('pcb/wire_stage/','tests/wire/','models/wire/','docs/network/wire/'),
    'net_hypergraph':('pcb/net_hypergraph/','configs/net_hypergraph/','tests/net_hypergraph/',
                      'models/net_hypergraph/','docs/network/hypergraph/'),
    'net_lines':('pcb/net_lines/','configs/net_lines/','tests/net_lines/',
                 'models/net_lines/','docs/network/lines/'),
}
INTEGRATION_PREFIXES=('pcb/network_shared/','pcb/topology_stage/','tests/topology/','docs/network/')
INTEGRATION_FILES={'README.md','docs/TEAM_MODULE_CONTRACTS.md','docs/TEAM_RUN_GUIDE.md',
                   '.github/workflows/team-modules.yml','.github/pull_request_template.md',
                   'tests/parity/test_full_pipeline_parity.py',
                   'tools/run_team_tests.py','tools/run_team_pipeline.py','tools/network_replay_parity.py',
                   'tools/check_network_change_scope.py','tools/show_network_versions.py'}


def allowed(path,owner):
    path=path.replace('\\','/')
    owners=tuple(PREFIXES) if owner in ('network','integration') else (owner,)
    if any(path.startswith(prefix) for kind in owners for prefix in PREFIXES[kind]):return True
    if path=='configs/wire_topology/current.json':return owner in ('wire','network','integration')
    return owner=='integration' and (path.startswith(INTEGRATION_PREFIXES) or path in INTEGRATION_FILES)


def wire_fields_allowed(old,new):
    return {k for k in set(old)|set(new) if old.get(k)!=new.get(k)} <= {'wire'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner',choices=(*PREFIXES,'network','integration'),required=True)
    p.add_argument('--base',default='origin/feature/network-metrics-split')
    p.add_argument('--git',default=shutil.which('git'))
    args=p.parse_args()
    if not args.git:raise SystemExit('Git not found on PATH; provide --git PATH_TO_GIT.')
    def git(*parts):
        return subprocess.check_output([args.git,*parts],cwd=ROOT,text=True,encoding='utf8').strip()
    base=git('merge-base',args.base,'HEAD')
    paths=set(git('diff','--name-only','--no-renames',base,'HEAD').splitlines())
    paths.update(git('diff','--name-only','--no-renames','HEAD').splitlines())
    paths.update(git('ls-files','--others','--exclude-standard').splitlines())
    bad=sorted(path for path in paths if not allowed(path,args.owner))
    wire_path='configs/wire_topology/current.json'
    if wire_path in paths and args.owner!='integration':
        old=json.loads(git('show',base+':'+wire_path))
        staged=json.loads(git('show',':'+wire_path))
        committed=json.loads(git('show','HEAD:'+wire_path))
        working=json.loads((ROOT/wire_path).read_text(encoding='utf8'))
        if not all(wire_fields_allowed(old,value) for value in (committed,staged,working)):
            bad.append(wire_path+' (only wire selection may change)')
    print(json.dumps({'owner':args.owner,'base':base,'changed_file_count':len(paths),'violations':bad,
                      'status':'FAILED' if bad else 'PASSED'},ensure_ascii=False,indent=2))
    return 1 if bad else 0


if __name__=='__main__':sys.exit(main())
