from rank2plan import Model, Pair
from rank2plan.lp_models import PrimalLpModel
from rank2plan.lp_models.constraint_generation import ConstraintModel
from pulp import LpSolver
from typing import List
from numpy import ndarray
import numpy as np


class LpModel(Model):
    def __init__(
        self,
        solver: LpSolver,
        use_column_generation=False,
        use_constraint_generation=False,
        C=1.0,
        tol=1e-4,
        verbose=False,
    ) -> None:
        if C <= 0:
            raise ValueError(f"C ({C}) must be positive")
        if not use_column_generation and not use_constraint_generation:
            self._underlying = PrimalLpModel(solver, C=C, verbose=verbose)
        elif use_constraint_generation and not use_column_generation:
            self._underlying = ConstraintModel(solver, C=C, verbose=verbose, tol=tol)
        else:
            raise NotImplementedError("Column generation not implemented yet")

    def fit(self, X, pairs):
        pairs = _filter_pairs(X, pairs)
        self._underlying.fit(X, pairs)

    def predict(self, X):
        return self._underlying.predict(X)

    def weights(self):
        return self._underlying.weights()


def _filter_pairs(X: ndarray, pairs: List[Pair]) -> List[Pair]:
    """Remove pairs where the feature vectors are identical, as they do not
    contribute to the model.

    Args:
        X (ndarray): The feature matrix
        pairs (List[Pair]): The pairs

    Returns:
        List[Pair]: pairs with distinct feature vectors
    """
    return [pair for pair in pairs if not np.array_equal(X[pair.i], X[pair.j])]
