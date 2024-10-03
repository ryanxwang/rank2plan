from typing import List, Optional
from rank2plan.types import Pair, Matrix
from rank2plan.lp_models.lp_underlying import LpUnderlying
from rank2plan.lp_models.utils import sparseLpDot
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
from dataclasses import dataclass
from scipy.sparse import issparse, spmatrix
import logging

LOGGER = logging.getLogger(__name__)


@dataclass
class FitState:
    """A very primitive state, instead of saving the built problem, we just
    rebuild each time, since PrimalLpModel is mainly for debugging and testing
    anyway.
    """

    pairs: List[Pair]


class PrimalLpModel(LpUnderlying):
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
        self._C = C
        self.state: Optional[FitState] = None
        self._weights = None

    def fit(self, X_tilde: Matrix, pairs: List[Pair], save_state=False) -> None:
        assert X_tilde.shape is not None
        _, P = X_tilde.shape

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
            xi.append(LpVariable(f"xi_{pair_id}_{pair.i}_{pair.j}", lowBound=0))
            # this order of j and i makes lower scores better
            if issparse(X_tilde):
                assert isinstance(X_tilde, spmatrix)
                prob += (
                    sparseLpDot(beta_plus, X_tilde.getrow(pair_id))
                    - sparseLpDot(beta_minus, X_tilde.getrow(pair_id))
                    >= pair.gap - xi[-1]
                )
            else:
                assert isinstance(X_tilde, np.ndarray)
                prob += (
                    lpDot(beta_plus, X_tilde[pair_id])
                    - lpDot(beta_minus, X_tilde[pair_id])
                    >= pair.gap - xi[-1]
                )
            sample_weights.append(pair.sample_weight)

        main_objective: LpAffineExpression = lpDot(xi, sample_weights)  # type: ignore
        regularisation_objective: LpAffineExpression = lpSum(beta_plus) + lpSum(
            beta_minus
        )
        prob += main_objective * self._C + regularisation_objective
        LOGGER.info("Problem build, solving")

        prob.solve(self.solver)

        LOGGER.info("Finished solving")
        LOGGER.info(f"Overall objective: {prob.objective.value()}")  # type: ignore
        LOGGER.info(f"Main objective: {main_objective.value()}")
        LOGGER.info(f"Regularisation objective: {regularisation_objective.value()}")

        self._weights = np.array(
            [beta_plus[i].varValue - beta_minus[i].varValue for i in range(P)]  # type: ignore
        )
        if save_state:
            self.state = FitState(pairs)

    def refit_with_C_value(self, X_tilde: Matrix, C: float, save_state=False) -> None:
        assert self.state is not None
        LOGGER.info(f"Refitting with C value changed from {self._C} to {C}")
        self._C = C
        self.fit(X_tilde, self.state.pairs, save_state=save_state)

    def predict(self, X: Matrix) -> ndarray:
        assert isinstance(self._weights, np.ndarray)
        return X @ self._weights

    def weights(self) -> Optional[ndarray]:
        return self._weights

    def clear_state(self) -> None:
        self.state = None

    @property
    def C(self) -> float:
        return self._C

    @C.setter
    def C(self, value: float) -> None:
        self._C = value
