"""Small dependency-free assignment helpers used by Component V4."""
from __future__ import annotations

import math
from collections.abc import Sequence


def maximum_weight_assignment(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Return a maximum-weight rectangular one-to-one assignment in O(n^3)."""
    if not weights or not weights[0]:
        return []
    rows, columns = len(weights), len(weights[0])
    if any(len(row) != columns for row in weights):
        raise ValueError("weight matrix must be rectangular")
    transposed = rows > columns
    matrix = (
        [[float(weights[row][column]) for row in range(rows)] for column in range(columns)]
        if transposed else [[float(value) for value in row] for row in weights]
    )
    n, m = len(matrix), len(matrix[0])
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        minimum, used = [math.inf] * (m + 1), [False] * (m + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0, delta, column1 = p[column0], math.inf, 0
            for column in range(1, m + 1):
                if used[column]:
                    continue
                reduced = -matrix[row0 - 1][column - 1] - u[row0] - v[column]
                if reduced < minimum[column]:
                    minimum[column], way[column] = reduced, column0
                if minimum[column] < delta:
                    delta, column1 = minimum[column], column
            for column in range(m + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    pairs = [(p[column] - 1, column - 1) for column in range(1, m + 1) if p[column]]
    return [(column, row) for row, column in pairs] if transposed else pairs

