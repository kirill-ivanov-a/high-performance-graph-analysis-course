from typing import List

import pygraphblas as gb


def count_triangles(adj_matrix: gb.Matrix) -> List[int]:
    if not adj_matrix.square:
        raise ValueError("adj_matrix must be square")

    if adj_matrix.type != gb.types.BOOL:
        raise ValueError(
            f"Unsupported adj_matrix type: {adj_matrix.type}. Expected: BOOL"
        )

    res = adj_matrix

    for i in range(2):
        res = adj_matrix.mxm(res, cast=gb.types.INT64, accum=gb.types.INT64.PLUS)

    def __create_result(res_size, vertices, triangle_nums):
        result = [0] * res_size
        for i, vertex in enumerate(vertices):
            result[vertex] = triangle_nums[i]
        return result

    res = res.diag().reduce_vector()
    res /= 2

    return __create_result(adj_matrix.nrows, *res.to_lists())
