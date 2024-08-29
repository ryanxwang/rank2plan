from rank2plan import Model, Pair
from rank2plan.lp_models import PrimalLpModel
from rank2plan.lp_models.constraint_column_generation import (
    ConstraintModel,
    ConstraintColumnModel,
)
from pulp import LpSolver
from typing import List, Optional
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
        dynamic_regularisation_target: Optional[float] = None,
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

            dynamic_regularisation_target (Optional[float], optional): If
            provided, will use dynamic regularisation by starting with the
            provided C value and adjusting till the ratio of the average slack
            value and the L1 norm of weights is close to this value . Defaults
            to None.

        Raises:
            ValueError: If C is not positive
            NotImplementedError: For now, using only column generation without
            constraint generation is not implemented
        """
        if C <= 0:
            raise ValueError(f"C ({C}) must be positive")
        if not use_column_generation and not use_constraint_generation:
            if dynamic_regularisation_target is not None:
                raise ValueError(
                    "Dynamic regularisation is only available when using column or constraint generation"
                )
            self._underlying = PrimalLpModel(solver, C=C)
        elif use_constraint_generation and not use_column_generation:
            self._underlying = ConstraintModel(
                solver,
                C,
                tol,
                dynamic_regularisation_target,
            )
        elif use_constraint_generation and use_column_generation:
            self._underlying = ConstraintColumnModel(
                solver,
                C=C,
                tol=tol,
                dynamic_regularisation_target=dynamic_regularisation_target,
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
