import pytest
import pygraphblas as gb
from project import sssp, mssp


@pytest.mark.parametrize(
    "I, J, V, size, start_vertex, expected_ans",
    [
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0.0, 3.0, 1.0, 5000.0, 5000.0, 5000.0, 5000.0, 1.0, 5000.0],
            3,
            0,
            [0.0, 2.0, 1.0],
        ),
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 3, 1, 5000, 5000, 5000, 5000, 1, 5000],
            4,
            0,
            [0.0, 2.0, 1.0, -1],
        ),
        ([0], [1], [5], 5, 0, [0, 5, -1, -1, -1]),
    ],
)
def test_sssp(I, J, V, size, start_vertex, expected_ans):
    adj_matrix = gb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert sssp(adj_matrix, start_vertex) == expected_ans


@pytest.mark.parametrize(
    "I, J, V, size, start_vertices, expected_ans",
    [
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0.0, 3.0, 1.0, 5000.0, 5000.0, 5000.0, 5000.0, 1.0, 5000.0],
            3,
            [0, 1, 2],
            [(0, [0.0, 2.0, 1.0]), (1, [5000.0, 0.0, 5000.0]), (2, [5000.0, 1.0, 0.0])],
        ),
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 3, 1, 5000, 5000, 5000, 5000, 1, 5000],
            4,
            [0, 1, 2],
            [
                (0, [0.0, 2.0, 1.0, -1]),
                (1, [5000.0, 0.0, 5000.0, -1]),
                (2, [5000.0, 1.0, 0.0, -1]),
            ],
        ),
        (
            [0],
            [1],
            [5],
            5,
            [0, 1, 2],
            [
                (0, [0, 5, -1, -1, -1]),
                (1, [-1, 0, -1, -1, -1]),
                (2, [-1, -1, 0, -1, -1]),
            ],
        ),
    ],
)
def test_mssp(I, J, V, size, start_vertices, expected_ans):
    adj_matrix = gb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert mssp(adj_matrix, start_vertices) == expected_ans
