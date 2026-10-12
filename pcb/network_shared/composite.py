"""Validate and assemble owner outputs into the unchanged Scene.nets schema."""
import math
from numbers import Real,Integral
from ..schema import Net
from .contracts import NetworkPlan,LogicalResult,GeometryResult,NetGroup,GroupLines,snapshot
from .graph import build_graph
from .selection import backends


def validate_plan(plan,scene,graph):
    if not isinstance(plan,NetworkPlan):raise TypeError('Expected NetworkPlan')
    if not isinstance(plan.groups,tuple):raise TypeError('NetworkPlan groups must be immutable tuples')
    valid_refs={f'{c.key}.{p.number}' for c in scene.components for p in c.pins}
    groups=set();pins=set();paths=set()
    for g in plan.groups:
        if not isinstance(g,NetGroup) or isinstance(g.group_id,bool) or not isinstance(g.group_id,Integral):
            raise TypeError('NetGroup with an integer ID required')
        if not all(isinstance(v,tuple) for v in (g.pins,g.path_ids,g.contact_points)):
            raise TypeError('Membership fields must be immutable tuples')
        if any(not isinstance(ref,str) for ref in g.pins):raise TypeError('Pin references must be strings')
        for p in g.contact_points:
            if not isinstance(p,tuple) or len(p)!=2 or any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) for v in p):
                raise ValueError('Contact points must be immutable finite x/y pairs')
        if g.group_id in groups:raise ValueError('Duplicate group ID')
        groups.add(g.group_id)
        if len(set(g.pins))!=len(g.pins) or set(g.pins)-valid_refs or pins.intersection(g.pins):
            raise ValueError('Unknown or multiply assigned input Pin reference')
        pins.update(g.pins)
        if len(set(g.path_ids))!=len(g.path_ids) or paths.intersection(g.path_ids):
            raise ValueError('Path assigned more than once')
        if any(isinstance(p,bool) or not isinstance(p,int) or not 0<=p<len(graph.paths) for p in g.path_ids):
            raise ValueError('Invalid path ID')
        paths.update(g.path_ids)


def assemble(plan,geometry):
    if not isinstance(geometry,GeometryResult):raise TypeError('Expected GeometryResult')
    if not isinstance(geometry.groups,tuple):raise TypeError('Geometry groups must be immutable tuples')
    lines={}
    for group in geometry.groups:
        if not isinstance(group,GroupLines) or not isinstance(group.segments,tuple):
            raise TypeError('GroupLines must return immutable segment tuples')
        if group.group_id in lines:raise ValueError('Duplicate geometry group ID')
        for segment in group.segments:
            if not isinstance(segment,tuple) or len(segment)!=2:raise ValueError('Exactly two immutable segment endpoints required')
            for p in segment:
                if not isinstance(p,tuple) or len(p)!=2 or any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) for v in p):
                    raise ValueError('Finite original-image x/y coordinates required')
        lines[group.group_id]=group.segments
    if set(lines)!={g.group_id for g in plan.groups}:raise ValueError('Geometry must use exactly the logical group IDs')
    return [Net(list(g.pins),list(lines[g.group_id])) for g in plan.groups if g.pins]


def build_networks(scene,mask,mode='directional',merge_labels=True,traced=None,
                   hypergraph=None,lines=None,versions=None):
    view=snapshot(scene);graph=build_graph(mask,traced)
    if hypergraph is None or lines is None:
        default_hypergraph,default_lines=backends(versions)
        hypergraph=default_hypergraph if hypergraph is None else hypergraph
        lines=default_lines if lines is None else lines
    logical=hypergraph.run(view,graph,mode=mode,merge_labels=merge_labels)
    if not isinstance(logical,LogicalResult):raise TypeError('Expected LogicalResult')
    validate_plan(logical.plan,view,graph)
    geometry=lines.run(graph,logical.plan)
    nets=assemble(logical.plan,geometry)
    diagnostics=dict(logical.diagnostics)
    diagnostics.update(nets=len(nets),singleton_nets=sum(len(n.pins)==1 for n in nets),
                       net_edges=sum(len(n.segments) for n in nets))
    # Backends never receive the mutable public Scene. Commit only the two
    # fields owned by topology after all outputs have passed validation.
    scene.nets=nets;scene.diagnostics.update(diagnostics)
    return graph.skeleton.copy()
