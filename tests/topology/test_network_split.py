import copy
from dataclasses import asdict,FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from pcb.schema import Net
from pcb.submission import export
from pcb.topology_stage.v2 import build_topology_v2
from pcb.network_shared.composite import build_networks
from pcb.network_shared.contracts import GeometryResult,GroupLines,NetGroup,NetworkPlan,LogicalResult,snapshot
from pcb.network_shared.graph import build_graph
from pcb.network_shared.selection import backends,selected_versions
from pcb.net_hypergraph.legacy_v2 import LegacyHypergraph
from pcb.net_lines.legacy_v2 import LegacyLines
from tests.topology.split_cases import cases,plain

LEGACY={'net_hypergraph':'legacy_v2','net_lines':'legacy_v2'}


class SplitEachLine:
    def run(self,graph,plan):
        legacy=LegacyLines().run(graph,plan);rows=[]
        for group in legacy.groups:
            segments=[]
            for a,b in group.segments:
                middle=((a[0]+b[0])/2,(a[1]+b[1])/2)
                segments.extend(((a,middle),(middle,b)))
            rows.append(GroupLines(group.group_id,tuple(segments)))
        return GeometryResult(tuple(rows))


class MergeGroups:
    def run(self,scene,graph,**options):
        old=LegacyHypergraph().run(scene,graph,**options)
        groups=old.plan.groups
        group=NetGroup(0,tuple(sorted({p for g in groups for p in g.pins})),
                       tuple(sorted(p for g in groups for p in g.path_ids)),
                       tuple(p for g in groups for p in g.contact_points))
        return LogicalResult(NetworkPlan((group,)),old.diagnostics)


class NetworkSplit(unittest.TestCase):
    def test_frozen_64_legacy_predictions_diagnostics_skeletons(self):
        path=Path(__file__).parent/'fixtures/network_split_baseline.json'
        baseline=json.loads(path.read_text(encoding='utf8'))
        self.assertTrue(baseline['synthetic_only']);self.assertEqual(len(baseline['cases']),64)
        sources={name:(scene,mask) for name,scene,mask in cases()}
        for expected in baseline['cases']:
            with self.subTest(case=expected['case'],mode=expected['mode'],labels=expected['merge_labels']):
                source,mask=sources[expected['case']];s=copy.deepcopy(source)
                immutable=asdict(s)['components'];texts=asdict(s)['texts']
                sk=build_topology_v2(s,mask,expected['mode'],expected['merge_labels'])
                self.assertEqual(export(s),expected['prediction'])
                self.assertEqual(plain(s.diagnostics),expected['diagnostics'])
                self.assertEqual(np.argwhere(sk).tolist(),expected['skeleton_pixels'])
                self.assertEqual(asdict(s)['components'],immutable);self.assertEqual(asdict(s)['texts'],texts)

    def test_line_strategy_changes_edges_only(self):
        _,s,mask=next(c for c in cases() if c[0]=='straight');other=copy.deepcopy(s)
        build_networks(s,mask,versions=LEGACY);build_networks(other,mask,lines=SplitEachLine(),versions=LEGACY)
        a,b=export(s),export(other)
        self.assertEqual(a['components'],b['components']);self.assertEqual(a['pins'],b['pins'])
        self.assertEqual([n['hyperGraph'] for n in a['nets'].values()],[n['hyperGraph'] for n in b['nets'].values()])
        self.assertNotEqual([n['edges'] for n in a['nets'].values()],[n['edges'] for n in b['nets'].values()])

    def test_logical_strategy_changes_groups_not_physical_lines(self):
        _,s,mask=next(c for c in cases() if c[0]=='long_gap');other=copy.deepcopy(s)
        build_networks(s,mask,versions=LEGACY);build_networks(other,mask,hypergraph=MergeGroups(),versions=LEGACY)
        self.assertEqual(len(s.nets),2);self.assertEqual(len(other.nets),1)
        self.assertEqual(sorted(edge for n in s.nets for edge in n.segments),sorted(edge for n in other.nets for edge in n.segments))
        self.assertEqual(asdict(s)['components'],asdict(other)['components'])

    def test_scene_views_and_graph_are_read_only(self):
        _,s,mask=next(cases());view=snapshot(s);graph=build_graph(mask)
        with self.assertRaises(FrozenInstanceError):view.components[0].pins[0].tip=(0,0)
        with self.assertRaises(TypeError):view.diagnostics['wire_stroke_width']=999
        with self.assertRaises(ValueError):graph.mask[0,0]=255
        with self.assertRaises(TypeError):graph.adjacency[(0,0)]=()
        self.assertFalse(np.array_equal(graph.mask,np.zeros_like(mask)))

    def test_bad_geometry_rejected_before_scene_commit(self):
        class BadLines:
            def run(self,graph,plan):return GeometryResult((GroupLines(-999,()),))
        _,s,mask=next(cases());s.nets=[Net(['previous.ref'],[])];before=copy.deepcopy(asdict(s))
        with self.assertRaises(ValueError):build_networks(s,mask,lines=BadLines(),versions=LEGACY)
        self.assertEqual(asdict(s),before)

    def test_unknown_pin_reference_rejected(self):
        class BadHypergraph:
            def run(self,scene,graph,**options):return LogicalResult(NetworkPlan((NetGroup(0,('not_a_pin',),()),)),{})
        _,s,mask=next(cases())
        with self.assertRaises(ValueError):build_networks(s,mask,hypergraph=BadHypergraph(),versions=LEGACY)

    def test_mutable_membership_cannot_leak_into_line_module(self):
        class MutableHypergraph:
            def run(self,scene,graph,**options):
                return LogicalResult(NetworkPlan((NetGroup(0,['U1.1'],(),()),)),{})
        _,s,mask=next(cases());before=asdict(s)
        with self.assertRaises(TypeError):build_networks(s,mask,hypergraph=MutableHypergraph(),versions=LEGACY)
        self.assertEqual(asdict(s),before)

    def test_independent_registry_versions_and_invalid_selection(self):
        logical,lines=backends({'net_hypergraph':'legacy_v2','net_lines':'legacy_v2'})
        self.assertIsInstance(logical,LegacyHypergraph);self.assertIsInstance(lines,LegacyLines)
        with self.assertRaises(ValueError):backends({'net_hypergraph':'missing','net_lines':'legacy_v2'})

    def test_owner_selector_rejects_cross_fields(self):
        temp_root=Path(__file__).resolve().parents[2]/'runs/test_tmp'
        temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            root=Path(folder)
            self.assertTrue(root.resolve().is_relative_to(temp_root.resolve()))
            for kind in ('net_hypergraph','net_lines'):
                p=root/'configs'/kind/'current.json';p.parent.mkdir(parents=True)
                p.write_text('{"version":"legacy_v2"}',encoding='utf8')
            self.assertEqual(selected_versions(root),{'net_hypergraph':'legacy_v2','net_lines':'legacy_v2'})
            (root/'configs/net_lines/current.json').write_text('{"version":"legacy_v2","pin_semantics":"other"}',encoding='utf8')
            with self.assertRaises(ValueError):selected_versions(root)

    def test_public_skeleton_is_still_writable(self):
        _,s,mask=next(cases());sk=build_topology_v2(s,mask)
        self.assertTrue(sk.flags.writeable)

    def test_legacy_api_is_fixed_but_team_stage_reads_owner_selection(self):
        from pcb.topology_stage.versions import TopologyStageV3
        _,s,mask=next(cases())
        invalid={'net_hypergraph':'missing','net_lines':'legacy_v2'}
        with patch('pcb.network_shared.selection.selected_versions',return_value=invalid):
            build_topology_v2(s,mask)
            with self.assertRaises(ValueError):TopologyStageV3()

    def test_stage_freezes_both_selections_at_construction(self):
        from pcb.topology_stage.versions import TopologyStageV3
        from pcb.core.interfaces import WireStageOutput
        from pcb.core.context import PipelineContext
        from pcb.core.config import PipelineConfig
        choices={'net_hypergraph':'legacy_v2','net_lines':'legacy_v2'}
        stage=TopologyStageV3(network_versions=choices)
        choices['net_lines']='missing'
        _,s,mask=next(cases())
        context=PipelineContext('synthetic.png',100,100,PipelineConfig(),None)
        stage.run(s,WireStageOutput(mask),context)
        self.assertEqual(len(s.nets),1)
        self.assertEqual(stage.network_versions['net_lines'],'legacy_v2')
        with self.assertRaises(ValueError):TopologyStageV3(network_versions=choices)


if __name__=='__main__':unittest.main()
