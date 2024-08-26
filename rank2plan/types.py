from enum import Enum, auto


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
