"""Test helpers for comparing exported network geometry semantically.

The project output assigns edges to a network and lists that network's pin
members, but does not assign individual edges to individual pins.  This helper
therefore preserves the network member set while normalizing only edge order,
direction, and exact collinear degree-two subdivision vertices.
"""

from collections import Counter, defaultdict


def _point(value):
    return (float(value["x"]), float(value["y"]))


def _edge(a, b):
    return (a, b) if a <= b else (b, a)


def _suppress_collinear_degree_two(edges):
    """Coalesce exact collinear subdivisions without removing turns/junctions."""
    counts = Counter(edges)
    while True:
        adjacent = defaultdict(list)
        for (a, b), count in counts.items():
            if count <= 0 or a == b:
                continue
            adjacent[a].extend([b] * count)
            adjacent[b].extend([a] * count)

        replacement = None
        for middle in sorted(adjacent):
            neighbors = adjacent[middle]
            if len(neighbors) != 2 or neighbors[0] == neighbors[1]:
                continue
            left, right = neighbors
            left_edge, right_edge = _edge(left, middle), _edge(middle, right)
            if counts[left_edge] != 1 or counts[right_edge] != 1:
                continue
            v1 = (left[0] - middle[0], left[1] - middle[1])
            v2 = (right[0] - middle[0], right[1] - middle[1])
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            if cross == 0 and dot < 0:
                replacement = (left_edge, right_edge, _edge(left, right))
                break
        if replacement is None:
            return counts
        left_edge, right_edge, combined = replacement
        del counts[left_edge]
        del counts[right_edge]
        counts[combined] += 1


def normalize_submission(result):
    """Return network semantics keyed by pin membership, not generated IDs.

    Strict equality is used for collinearity; no tolerance or coordinate
    rounding is introduced. Zero-length contact edges remain explicit.
    """
    normalized = {}
    for _net_id, net in result["nets"].items():
        hypergraph = net["hyperGraph"]
        if not (isinstance(hypergraph, str) and hypergraph.startswith("(") and hypergraph.endswith(")")):
            raise ValueError(f"unexpected hyperGraph format: {hypergraph!r}")
        body = hypergraph[1:-1].strip()
        members = frozenset(item.strip() for item in body.split(",") if item.strip())
        if not members:
            raise ValueError("network has no pin members")
        if members in normalized:
            raise ValueError(f"duplicate network membership: {sorted(members)}")

        nonzero = []
        zero = Counter()
        for _edge_id, endpoints in net["edges"].items():
            if len(endpoints) != 2:
                raise ValueError("edge must contain exactly two endpoints")
            a, b = _point(endpoints[0]), _point(endpoints[1])
            if a == b:
                zero[a] += 1
            else:
                nonzero.append(_edge(a, b))
        merged = _suppress_collinear_degree_two(nonzero)
        # Keep multiplicity so duplicate exported edges cannot disappear.
        edge_records = tuple(sorted(
            [edge for edge, count in merged.items() for _ in range(count)]
            + [(point, point) for point, count in zero.items() for _ in range(count)]
        ))
        normalized[members] = edge_records
    return normalized


def edge_graph_summary(edges):
    """Summarize connected components/degrees in normalized edge geometry."""
    adjacency = defaultdict(set)
    for a, b in edges:
        if a == b:
            adjacency.setdefault(a, set())
        else:
            adjacency[a].add(b)
            adjacency[b].add(a)
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node] & remaining:
                remaining.remove(neighbor)
                stack.append(neighbor)
    degrees = [len(neighbors) for neighbors in adjacency.values()]
    return {
        "component_count": components,
        "vertex_count": len(adjacency),
        "degree_1_count": sum(degree == 1 for degree in degrees),
        "degree_2_count": sum(degree == 2 for degree in degrees),
        "degree_3_plus_count": sum(degree >= 3 for degree in degrees),
    }
