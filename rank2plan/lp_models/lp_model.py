from rank2plan import Model, Pair
from rank2plan.lp_models import PrimalLpModel
from rank2plan.lp_models.constraint_column_generation import (
    ConstraintColumnModel,
)
from pulp import LpSolver
from typing import List
from numpy import ndarray
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


class LpModel(Model):
    def __init__(
        self,
        solver: LpSolver,
        use_column_generation=False,
        use_constraint_generation=False,
        C=1.0,
        tol=1e-4,
    ) -> None:
        """Create a new LP model.

        Args:
            solver (LpSolver): The solver to use

            use_column_generation (bool, optional): Whether to use column
            generation. Defaults to False.

            use_constraint_generation (bool, optional): Whether to use
            constraint generation. Defaults to False.

            C (float, optional): The SVM regularisation parameter. Defaults to
            1.0.

            tol (float, optional): Tolerence value used in constraint or column
            generation for adding constraints and columns. Defaults to 1e-4.

        Raises:
            ValueError: If C is not positive NotImplementedError: For now, using
            only column generation without constraint generation is not
            implemented
        """
        if C <= 0:
            raise ValueError(f"C ({C}) must be positive")
        if not use_column_generation and not use_constraint_generation:
            self._underlying = PrimalLpModel(solver, C=C)
        elif use_constraint_generation:
            self._underlying = ConstraintColumnModel(
                solver, C, tol, no_feature_sampling=(not use_column_generation)
            )
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
