from enum import Enum, auto
from numpy import ndarray
from scipy.sparse import spmatrix
from typing import TypeAlias

Matrix: TypeAlias = ndarray | spmatrix


class Pair:
    def __init__(self, i: int, j: int, gap: float = 1.0, sample_weight: float = 1.0):
        self.i = i
        self.j = j
        self.gap = gap
        self.sample_weight = sample_weight


class LossType(Enum):
    Hinge = auto()
    SquaredHinge = auto()


class PenalisationType(Enum):
    L1 = auto()
    L2 = auto()


class TuningMetric(Enum):
    KendallTau = auto()
    # See documents/theory.pdf for what the LpMainObjective is
    LpMainObjective = auto()
