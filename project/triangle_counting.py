from typing import List

import pygraphblas as gb

from project.exceptions import MatrixTypeException, MatrixDimensionException

__all__ = ["count_triangles"]


def count_triangles(adj_matrix: gb.Matrix) -> List[int]:
    if not adj_matrix.square:
        raise MatrixDimensionException("adj_matrix must be square")

    if adj_matrix.type != gb.types.BOOL:
        raise MatrixTypeException(
            f"Unsupported adj_matrix type: {adj_matrix.type}. Expected: BOOL"
        )

    adj_matrix = adj_matrix + adj_matrix.transpose()

    nonzero_vertices, num_triangles = (
        adj_matrix.mxm(
            adj_matrix, cast=gb.types.INT64, accum=gb.types.INT64.PLUS, mask=adj_matrix
        ).reduce_vector()
        / 2
    ).to_lists()

    result = [0] * adj_matrix.nrows
    for i, vertex in enumerate(nonzero_vertices):
        result[vertex] = num_triangles[i]
    return result
