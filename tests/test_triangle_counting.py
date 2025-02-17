import pytest
import pygraphblas as gb
from project import count_triangles, MatrixDimensionException


def test_non_square_count_triangles():
    adj_matrix = gb.Matrix.dense(gb.INT64, nrows=1, ncols=2)
    with pytest.raises(MatrixDimensionException):
        count_triangles(adj_matrix)


@pytest.mark.parametrize(
    "I, J, V, size, expected_ans",
    [
        (
            [1, 0, 4, 4, 4, 3, 5, 2, 5, 1, 5, 2],
            [0, 5, 0, 5, 3, 5, 3, 3, 2, 5, 1, 1],
            [True] * 12,
            6,
            [2, 2, 2, 2, 2, 5],
        ),
        (
            [0, 1, 0, 5, 0, 4, 4, 5, 4, 3, 3, 5, 3, 2, 5, 2, 1, 5, 1, 2],
            [1, 0, 5, 0, 4, 0, 5, 4, 3, 4, 5, 3, 2, 3, 2, 5, 5, 1, 2, 1],
            [True] * 20,
            6,
            [2, 2, 2, 2, 2, 5],
        ),
        (
            [0, 1, 1, 2, 2, 0, 3, 2],
            [1, 0, 2, 1, 0, 2, 2, 3],
            [True] * 8,
            4,
            [1, 1, 1, 0],
        ),
        ([0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2], [True] * 6, 3, [1, 1, 1]),
        ([1], [1], [False], 5, [0] * 5),
    ],
)
def test_count_triangles(I, J, V, size, expected_ans):
    adj_matrix = gb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert count_triangles(adj_matrix) == expected_ans
