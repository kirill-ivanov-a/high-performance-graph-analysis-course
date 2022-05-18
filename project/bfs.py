from typing import List, Tuple
from itertools import count

import pygraphblas as gb

__all__ = ["bfs", "multiple_bfs"]


def bfs(adj_matrix: gb.Matrix, start_vertex: int) -> List[int]:
    if not adj_matrix.square:
        raise ValueError("adj_matrix must be square")

    if start_vertex >= adj_matrix.nrows or start_vertex < 0:
        raise ValueError(f"start_vertex must be between 0 and {adj_matrix.nrows - 1}")

    if adj_matrix.type != gb.types.BOOL:
        raise ValueError(
            f"Unsupported adj_matrix type: {adj_matrix.type}. Expected: BOOL"
        )

    layer = gb.Vector.sparse(gb.types.BOOL, adj_matrix.nrows)
    visited = gb.Vector.sparse(gb.types.BOOL, adj_matrix.nrows)
    result = gb.Vector.dense(gb.types.INT64, adj_matrix.nrows, fill=-1)

    result[start_vertex] = 0
    visited[start_vertex] = True
    layer[start_vertex] = True

    for num_layer in count(1):
        prev_nnz = visited.nvals
        layer.vxm(adj_matrix, mask=visited, out=layer, desc=gb.descriptor.RC)
        visited.eadd(layer, layer.type.lxor_monoid, out=visited, desc=gb.descriptor.R)
        result.assign_scalar(num_layer, mask=layer)
        if visited.nvals == prev_nnz:
            break

    return list(result.vals)


def multiple_bfs(
    adj_matrix: gb.Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    if not adj_matrix.square:
        raise ValueError("adj_matrix must be square")

    if any(start >= adj_matrix.nrows or start < 0 for start in start_vertices):
        raise ValueError(
            f"start_vertex[i] must be between 0 and {adj_matrix.nrows - 1}"
        )

    if adj_matrix.type != gb.types.BOOL:
        raise ValueError(
            f"Unsupported adj_matrix type: {adj_matrix.type}. Expected: BOOL"
        )

    layers = gb.Matrix.sparse(
        gb.types.BOOL, nrows=len(start_vertices), ncols=adj_matrix.ncols
    )
    visited = gb.Matrix.sparse(
        gb.types.BOOL, nrows=len(start_vertices), ncols=adj_matrix.ncols
    )
    results = gb.Matrix.dense(
        gb.types.INT64, nrows=len(start_vertices), ncols=adj_matrix.ncols, fill=-1
    )

    for i, j in enumerate(start_vertices):
        layers.assign_scalar(True, i, j)
        visited.assign_scalar(True, i, j)
        results.assign_scalar(0, i, j)

    for num_layer in count(1):
        prev_nnz = visited.nvals
        layers.mxm(adj_matrix, mask=visited, out=layers, desc=gb.descriptor.RC)
        visited.eadd(layers, layers.type.lxor_monoid, out=visited, desc=gb.descriptor.R)
        results.assign_scalar(num_layer, mask=layers)
        if visited.nvals == prev_nnz:
            break

    return [(vertex, list(results[i].vals)) for i, vertex in enumerate(start_vertices)]
