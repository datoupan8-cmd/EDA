"""Compatibility API; membership and segment geometry now have separate owners."""
from ..network_shared.composite import build_networks
# Keep historical helpers importable. New code should use backend interfaces.
from ..net_hypergraph.legacy_v2 import SIGNAL,VALUE,norm_label,_direction,_is_label,_bbox_point_dist


def build_topology_v2(scene,mask,mode='directional',merge_labels=True,traced=None):
    # Historical callers and parity fixtures keep their original algorithm.
    # Team Topology stages select the independently maintained backends.
    return build_networks(scene,mask,mode,merge_labels,traced,
                          versions={'net_hypergraph':'legacy_v2','net_lines':'legacy_v2'})
