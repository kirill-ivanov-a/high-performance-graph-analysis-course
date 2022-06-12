__all__ = ["MatrixTypeException", "MatrixDimensionException", "NegativeCycleException"]


class MatrixDimensionException(Exception):
    pass


class MatrixTypeException(Exception):
    pass


class NegativeCycleException(Exception):
    pass
