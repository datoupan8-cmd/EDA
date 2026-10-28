"""Integration-owned graph construction; stable inputs for both backends."""
from types import MappingProxyType
import cv2
import numpy as np
from ..wire_stage.v1 import trace_paths,simplify_with_indices
from .contracts import WireGraph


def build_graph(mask,traced=None):
    paths,skeleton,adj=traced if traced is not None else trace_paths(mask)
    original=np.asarray(mask).copy()
    thickness=cv2.distanceTransform(original,cv2.DIST_L2,3)
    skeleton=np.asarray(skeleton).copy()
    indexed=tuple(simplify_with_indices(p) for p in paths)
    reference=tuple(item.segments for item in indexed)
    reference_ranges=tuple(item.segment_edge_ranges for item in indexed)
    paths=tuple(tuple(tuple(p) for p in path) for path in paths)
    adjacency=MappingProxyType({p:tuple(neighbors) for p,neighbors in adj.items()})
    for array in (original,thickness,skeleton):array.setflags(write=False)
    height,width=original.shape
    return WireGraph(width,height,paths,adjacency,reference,original,thickness,skeleton,
                     reference_ranges)
