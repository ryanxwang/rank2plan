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
    PULP_CBC_CMD,
    LpAffineExpression,
)


class PrimalLpModel(Model):
    """The primal LP model for RankSVM. Based on equation (5) of (Dedieu et al,
    2022).
    """

    def __init__(
        self,
        C=1.0,
        msgs=False,
        solver_time_limit: Optional[float] = None,
        seed=0,
    ) -> None:
        """Initialize the model.

        Args:
            C (float, optional): The regularisation parameter of the SVM.
            Defaults to 1.0.
            msgs (bool, optional): Whether to print messages, including those of
            the LP solver. Defaults to False.
            solver_time_limit (Optional[float], optional): The time limit for
            the LP solver in seconds. This does not limit the time used to
            create the problem, which can take some time for large problems.
            Defaults to None
            seed (int, optional): The seed for the LP solver. Defaults to 0.
        """
        self.C = C
        self.msgs = msgs
        self.solver_time_limit = solver_time_limit
        self.seed = seed
        self._weights = None

    def fit(self, X: ndarray, pairs: List[Pair]) -> None:
        N = X.shape[0]
        P = X.shape[1]

        prob = LpProblem("SimpleLP", LpMinimize)
        beta_plus: List[LpVariable] = []
        beta_minus: List[LpVariable] = []

        for p in range(P):
            beta_plus.append(LpVariable(f"beta_plus_{p}", lowBound=0))
            beta_minus.append(LpVariable(f"beta_minus_{p}", lowBound=0))

        h_values: List[LpAffineExpression] = [
            lpDot(beta_plus, X[i]) - lpDot(beta_minus, X[i]) for i in range(N)
        ]

        xi: List[LpVariable] = []
        sample_weights: List[float] = []
        for pair_id, pair in enumerate(pairs):
            if np.array_equal(X[pair.i], X[pair.j]):
                continue

            xi.append(LpVariable(f"xi_{pair_id}_{pair.i}_{pair.j}", lowBound=0))
            # this order of j and i makes lower scores better
            prob += h_values[pair.j] - h_values[pair.i] >= pair.gap - xi[-1]
            sample_weights.append(pair.sample_weight)

        main_objective: LpAffineExpression = self.C * lpDot(xi, sample_weights)  # type: ignore
        regularisation_objective: LpAffineExpression = lpSum(beta_plus) + lpSum(
            beta_minus
        )
        prob += main_objective + regularisation_objective

        # TODO-someday: make solver configurable
        solver = PULP_CBC_CMD(
            msg=self.msgs,
            options=[f"RandomS {self.seed}"],
            timeLimit=self.solver_time_limit,
            mip=False,
        )
        prob.solve(solver)

        if self.msgs:
            print(f"Overall objective: {prob.objective.value()}")  # type: ignore
            print(f"Main objective: {main_objective.value()}")
            print(f"Regularisation object: {regularisation_objective.value()}")

        self._weights = np.array(
            [beta_plus[i].varValue - beta_minus[i].varValue for i in range(P)]  # type: ignore
        )

    def predict(self, X: ndarray) -> ndarray:
        return X @ self._weights

    def weights(self) -> Optional[ndarray]:
        return self._weights
