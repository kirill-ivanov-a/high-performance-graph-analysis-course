from typing import List, Tuple

import pygraphblas as gb

__all__ = ["sssp", "mssp"]


def mssp(
    adj_matrix: gb.Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    if not adj_matrix.square:
        raise ValueError("adj_matrix must be square")

    if any(start >= adj_matrix.nrows or start < 0 for start in start_vertices):
        raise ValueError(
            f"start_vertex[i] must be between 0 and {adj_matrix.nrows - 1}"
        )

    dist = gb.Matrix.sparse(
        adj_matrix.type, nrows=len(start_vertices), ncols=adj_matrix.ncols
    )

    for i, j in enumerate(start_vertices):
        dist.assign_scalar(0, i, j)

    for _ in range(adj_matrix.nrows):
        dist.mxm(
            adj_matrix,
            semiring=adj_matrix.type.min_plus,
            out=dist,
            accum=adj_matrix.type.min,
        )

    def __create_result(res_size, vertices, distances):
        result = [-1] * res_size
        for i, vertex in enumerate(vertices):
            result[vertex] = distances[i]
        return result

    return [
        (vertex, __create_result(adj_matrix.nrows, *dist[i].to_lists()))
        for i, vertex in enumerate(start_vertices)
    ]


def sssp(adj_matrix: gb.Matrix, start_vertex: int) -> List[int]:
    return mssp(adj_matrix, [start_vertex])[0][1]
