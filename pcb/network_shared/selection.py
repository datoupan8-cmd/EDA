"""Read two owner-specific selectors without changing core/team configuration."""
import json
from pathlib import Path
from ..core.registry import StageRegistry

ROOT=Path(__file__).resolve().parents[2]


def selected_versions(root=ROOT):
    result={}
    for kind in ('net_hypergraph','net_lines'):
        data=json.loads((Path(root)/'configs'/kind/'current.json').read_text(encoding='utf8'))
        if set(data)!={'version'} or not isinstance(data['version'],str) or not data['version']:
            raise ValueError(f'{kind} selector may contain only a nonempty version')
        result[kind]=data['version']
    return result


def backends(versions=None):
    from ..net_hypergraph.registry import register_stages as logical
    from ..net_lines.registry import register_stages as lines
    registry=StageRegistry();logical(registry);lines(registry)
    versions=selected_versions() if versions is None else versions
    if set(versions)!={'net_hypergraph','net_lines'}:raise ValueError('Two explicit network versions required')
    return registry.create('net_hypergraph',versions['net_hypergraph']),registry.create('net_lines',versions['net_lines'])
