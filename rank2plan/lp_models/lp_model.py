from rank2plan import Model, Pair
from rank2plan.lp_models import PrimalLpModel
from rank2plan.lp_models.constraint_column_generation import (
    ConstraintColumnModel,
)
from rank2plan.lp_models.objective_values import compute_main_objective
from rank2plan.metrics import kendall_tau
from rank2plan.lp_models.constraint_column_generation.utils import compute_X_tilde
from pulp import LpSolver
from typing import List
from numpy import ndarray
import numpy as np
import logging
from bayes_opt import BayesianOptimization

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

    def tune(
        self,
        X_train: ndarray,
        pairs_train: List[Pair],
        X_val: ndarray,
        pairs_val: List[Pair],
        C_range=(0.001, 100),
        tuning_rounds=25,
    ) -> float:
        if tuning_rounds < 5:
            raise ValueError("tuning_rounds should be at least 5")

        pairs_train = _filter_pairs(X_train, pairs_train)
        pairs_val = _filter_pairs(X_val, pairs_val)

        # X_tilde_val = compute_X_tilde(X, pairs_val)

        def validation_score(weights: ndarray) -> float:
            # negative because we are minimising
            # return -compute_main_objective(X_tilde_val, pairs_val, weights)
            scores = self._underlying.predict(X_val)
            return kendall_tau(pairs_val, scores)

        def f(C: float):
            self._underlying.refit_with_C_value(C, save_state=True)
            val_score = validation_score(self.weights())  # type: ignore
            LOGGER.info(f"Validation score for C={C}: {val_score}")
            return val_score

        optimiser = BayesianOptimization(
            f=f,
            pbounds={"C": C_range},
            verbose=0,
            random_state=1,
        )

        self._underlying.fit(X_train, pairs_train, save_state=True)
        initial_score = validation_score(self.weights())  # type: ignore
        optimiser.register(params={"C": self._underlying.C}, target=initial_score)
        optimiser.maximize(init_points=5, n_iter=tuning_rounds - 5)

        best_C = optimiser.max["params"]["C"]  # type: ignore
        best_score = -optimiser.max["target"]  # type: ignore
        LOGGER.info(
            f"Refitting on complete data set with found best C: {best_C}, which gave the best score: {best_score}",
        )
        self._underlying.state = None
        self._underlying.C = best_C
        return best_C

    def fit(self, X: ndarray, pairs: List[Pair]) -> None:
        """Fit the model.

        Args:
            X (ndarray): The feature matrix

            pairs (List[Pair]): The pairs
        """
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
