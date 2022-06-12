from typing import List, Tuple

import pygraphblas as gb

from project.exceptions import (
    MatrixTypeException,
    MatrixDimensionException,
    NegativeCycleException,
)

__all__ = ["sssp", "mssp"]


def mssp(
    adj_matrix: gb.Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    if not adj_matrix.square:
        raise MatrixDimensionException("adj_matrix must be square")

    if any(start >= adj_matrix.nrows or start < 0 for start in start_vertices):
        raise ValueError(
            f"start_vertex[i] must be between 0 and {adj_matrix.nrows - 1}"
        )

    if not hasattr(adj_matrix.type, "min_plus"):
        raise MatrixTypeException(
            f"Unsupported type of matrix elements: {adj_matrix.type}"
        )

    dist = gb.Matrix.sparse(
        adj_matrix.type, nrows=len(start_vertices), ncols=adj_matrix.ncols
    )

    for i, j in enumerate(start_vertices):
        dist.assign_scalar(0, i, j)

    for i in range(adj_matrix.nrows):
        prev_dist = dist.dup()
        dist.mxm(
            adj_matrix,
            semiring=adj_matrix.type.min_plus,
            out=dist,
            accum=adj_matrix.type.min,
        )
        if (dist == prev_dist).reduce_bool(gb.types.BOOL.land_monoid):
            break
        if i == adj_matrix.nrows - 1:
            raise NegativeCycleException("A negative cycle detected in the graph")

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
