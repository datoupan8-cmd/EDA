"""Original deterministic maximum-weight assignment; no scoring or data access."""
import math
from typing import Sequence

def maximum_weight_assignment(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Return a maximum-weight rectangular assignment in O(n^3).

    The implementation is dependency-free so the evaluator remains usable in the
    competition runtime even when SciPy is unavailable.
    """

    if not weights or not weights[0]:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if any(len(row) != column_count for row in weights):
        raise ValueError("weight matrix must be rectangular")
    transposed = row_count > column_count
    matrix = (
        [[float(weights[row][column]) for row in range(row_count)] for column in range(column_count)]
        if transposed
        else [[float(value) for value in row] for row in weights]
    )
    n = len(matrix)
    m = len(matrix[0])

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        min_value = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, m + 1):
                if used[column]:
                    continue
                reduced_cost = -matrix[row0 - 1][column - 1] - u[row0] - v[column]
                if reduced_cost < min_value[column]:
                    min_value[column] = reduced_cost
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(m + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    pairs = [(p[column] - 1, column - 1) for column in range(1, m + 1) if p[column] != 0]
    if transposed:
        return [(column, row) for row, column in pairs]
    return pairs
