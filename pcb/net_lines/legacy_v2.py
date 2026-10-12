"""Unchanged legacy segment materialisation, separately replaceable."""
from ..network_shared.contracts import GeometryResult,GroupLines


class LegacyLines:
    def run(self,graph,plan):
        result=[]
        for group in plan.groups:
            segments=[]
            for path_id in group.path_ids:
                segments.extend(graph.reference_segments[path_id])
            for point in group.contact_points:
                edge=(point,point)
                if edge not in segments:segments.append(edge)
            result.append(GroupLines(group.group_id,tuple(segments)))
        return GeometryResult(tuple(result))
