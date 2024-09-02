from typing import List, Optional
from rank2plan.model import Model
from rank2plan.types import Pair
import numpy as np
from numpy import ndarray
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpDot,
    lpSum,
    LpAffineExpression,
    LpSolver,
)
import logging

LOGGER = logging.getLogger(__name__)


class PrimalLpModel(Model):
    """The primal LP model for RankSVM. Based on equation (5) of (Dedieu et al,
    2022). This should mainly be used for debugging and testing purposes.
    """

    def __init__(
        self,
        solver: LpSolver,
        C=1.0,
    ) -> None:
        """Initialise the primal LP model.

        Args:
            solver (LpSolver): The solver to use.
            C (float, optional): Regularisation parameter. Defaults to 1.0.
        """
        self.solver = solver
        self.C = C
        self._weights = None

    def fit(self, X: ndarray, pairs: List[Pair]) -> None:
        N = X.shape[0]
        P = X.shape[1]

        prob = LpProblem("PrimalLp", LpMinimize)
        beta_plus: List[LpVariable] = []
        beta_minus: List[LpVariable] = []

        for p in range(P):
            beta_plus.append(LpVariable(f"beta_plus_{p}", lowBound=0))
            beta_minus.append(LpVariable(f"beta_minus_{p}", lowBound=0))

        LOGGER.info("Created variables, constructing h values")
        LOGGER.info(f"Constructing h values, building problem")

        xi: List[LpVariable] = []
        sample_weights: List[float] = []
        for pair_id, pair in enumerate(pairs):
            h_value_i = lpDot(beta_plus, X[pair.i]) - lpDot(beta_minus, X[pair.i])
            h_value_j = lpDot(beta_plus, X[pair.j]) - lpDot(beta_minus, X[pair.j])
            xi.append(LpVariable(f"xi_{pair_id}_{pair.i}_{pair.j}", lowBound=0))
            # this order of j and i makes lower scores better
            prob += h_value_j - h_value_i >= pair.gap - xi[-1]
            sample_weights.append(pair.sample_weight)

        main_objective: LpAffineExpression = lpDot(xi, sample_weights)  # type: ignore
        regularisation_objective: LpAffineExpression = lpSum(beta_plus) + lpSum(
            beta_minus
        )
        prob += main_objective * self.C + regularisation_objective
        LOGGER.info("Problem build, solving")

        prob.solve(self.solver)

        LOGGER.info("Finished solving")
        LOGGER.info(f"Overall objective: {prob.objective.value()}")  # type: ignore
        LOGGER.info(f"Main objective: {main_objective.value()}")
        LOGGER.info(f"Regularisation objective: {regularisation_objective.value()}")

        self._weights = np.array(
            [beta_plus[i].varValue - beta_minus[i].varValue for i in range(P)]  # type: ignore
        )

    def predict(self, X: ndarray) -> ndarray:
        return X @ self._weights

    def weights(self) -> Optional[ndarray]:
        return self._weights
