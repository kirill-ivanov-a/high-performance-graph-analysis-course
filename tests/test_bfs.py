import pytest
import pygraphblas as gb
from project import bfs, multiple_bfs


@pytest.mark.parametrize(
    "I, J, V, size, start_vertex, expected_ans",
    [
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            5,
            0,
            [0, 1, 1, -1, -1],
        ),
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [False, True, True, True, True],
            5,
            0,
            [0, 2, 1, -1, -1],
        ),
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            5,
            3,
            [-1, -1, -1, 0, 1],
        ),
        (
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [True, True, True, True],
            5,
            0,
            [0, 1, 2, 3, 4],
        ),
        ([0], [1], [False], 5, 0, [0, -1, -1, -1, -1]),
    ],
)
def test_bfs(I, J, V, size, start_vertex, expected_ans):
    adj_matrix = gb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert bfs(adj_matrix, start_vertex) == expected_ans


@pytest.mark.parametrize(
    "I, J, V, size, start_vertices, expected_ans",
    [
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            5,
            [0, 0],
            [(0, [0, 1, 1, -1, -1]), (0, [0, 1, 1, -1, -1])],
        ),
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            5,
            [0, 3],
            [(0, [0, 1, 1, -1, -1]), (3, [-1, -1, -1, 0, 1])],
        ),
        (
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [True, True, True, True],
            5,
            [0, 1, 2, 3, 4],
            [
                (0, [0, 1, 2, 3, 4]),
                (1, [-1, 0, 1, 2, 3]),
                (2, [-1, -1, 0, 1, 2]),
                (3, [-1, -1, -1, 0, 1]),
                (4, [-1, -1, -1, -1, 0]),
            ],
        ),
        (
            [0],
            [1],
            [False],
            5,
            [0, 2],
            [(0, [0, -1, -1, -1, -1]), (2, [-1, -1, 0, -1, -1])],
        ),
    ],
)
def test_multiple_bfs(I, J, V, size, start_vertices, expected_ans):
    adj_matrix = gb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert multiple_bfs(adj_matrix, start_vertices) == expected_ans


def test_non_square_bfs():
    adj_matrix = gb.Matrix.dense(gb.BOOL, nrows=1, ncols=2)
    with pytest.raises(ValueError):
        bfs(adj_matrix, 0)


def test_non_square_multiple_bfs():
    adj_matrix = gb.Matrix.dense(gb.BOOL, nrows=1, ncols=2)
    with pytest.raises(ValueError):
        multiple_bfs(adj_matrix, [0])


def test_invalid_matrix_type_bfs():
    adj_matrix = gb.Matrix.dense(gb.INT64, nrows=2, ncols=2)
    with pytest.raises(ValueError):
        bfs(adj_matrix, 0)


def test_invalid_matrix_type_multiple_bfs():
    adj_matrix = gb.Matrix.dense(gb.INT64, nrows=2, ncols=2)
    with pytest.raises(ValueError):
        multiple_bfs(adj_matrix, [0])


def test_invalid_start_vertex_bfs():
    adj_matrix = gb.Matrix.dense(gb.BOOL, nrows=2, ncols=2)
    with pytest.raises(ValueError):
        bfs(adj_matrix, 2)


def test_invalid_start_vertex_multiple_bfs():
    adj_matrix = gb.Matrix.dense(gb.BOOL, nrows=2, ncols=2)
    with pytest.raises(ValueError):
        multiple_bfs(adj_matrix, [2])
