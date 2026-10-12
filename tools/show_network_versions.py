"""Show frozen team selections and separately owned network selectors."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pcb.core.team_config import load_team_config
from pcb.network_shared.selection import selected_versions

if __name__=='__main__':
    config=load_team_config(ROOT/'configs/team.json')
    print(json.dumps({'component':config.component,'pin_localization':config.pin_localization,
                      'pin_semantics':config.pin_semantics,'wire':config.wire,'topology_wrapper':config.topology,
                      **selected_versions(),'topology_v1_uses_legacy_unsplit_implementation':True},indent=2))
