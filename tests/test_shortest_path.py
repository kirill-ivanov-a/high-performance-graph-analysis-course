import pytest
import pygraphblas as gb
from project import (
    sssp,
    mssp,
    MatrixTypeException,
    MatrixDimensionException,
    NegativeCycleException,
)


@pytest.fixture(
    params=[
        dict(I=[0, 1], J=[1, 0], V=[5, -6]),
        dict(I=[0, 1, 0, 2], J=[1, 2, 2, 1], V=[1.0, -1.0, 2.0, 0.5]),
    ]
)
def negative_cycle(request):
    return gb.Matrix.from_lists(**request.param)


@pytest.fixture(params=[gb.FC32, gb.FC64, gb.BOOL])
def unsupported_type(request):
    return request.param


def test_negative_cycle_sssp(negative_cycle):
    with pytest.raises(NegativeCycleException):
        sssp(negative_cycle, 0)


def test_negative_cycle_mssp(negative_cycle):
    with pytest.raises(NegativeCycleException):
        mssp(negative_cycle, [0, 1])


def test_non_square_sssp():
    adj_matrix = gb.Matrix.dense(gb.INT64, nrows=1, ncols=2)
    with pytest.raises(MatrixDimensionException):
        sssp(adj_matrix, 0)


def test_non_square_mssp():
    adj_matrix = gb.Matrix.dense(gb.INT64, nrows=1, ncols=2)
    with pytest.raises(MatrixDimensionException):
        mssp(adj_matrix, [0])


def test_invalid_start_vertex_sssp():
    adj_matrix = gb.Matrix.dense(gb.INT64, nrows=2, ncols=2)
    with pytest.raises(ValueError):
        sssp(adj_matrix, 2)


def test_invalid_start_vertex_mssp():
    adj_matrix = gb.Matrix.dense(gb.INT64, nrows=2, ncols=2)
    with pytest.raises(ValueError):
        mssp(adj_matrix, [2])


def test_invalid_matrix_type_sssp(unsupported_type):
    adj_matrix = gb.Matrix.dense(unsupported_type, nrows=2, ncols=2)
    with pytest.raises(MatrixTypeException):
        sssp(adj_matrix, 0)


def test_invalid_matrix_type_mssp(unsupported_type):
    adj_matrix = gb.Matrix.dense(unsupported_type, nrows=2, ncols=2)
    with pytest.raises(MatrixTypeException):
        mssp(adj_matrix, [0])


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
