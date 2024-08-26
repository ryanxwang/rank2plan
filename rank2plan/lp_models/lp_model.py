from rank2plan.model import Model
from rank2plan.lp_models import PrimalLpModel
from pulp import LpSolver


class LpModel(Model):
    def __init__(
        self,
        solver: LpSolver,
        use_column_generation=False,
        use_constraint_generation=False,
        C=1.0,
        verbose=False,
    ) -> None:
        if C <= 0:
            raise ValueError(f"C ({C}) must be positive")
        if not use_column_generation and not use_constraint_generation:
            self._underlying = PrimalLpModel(solver, C=C, verbose=verbose)

    def fit(self, X, pairs):
        self._underlying.fit(X, pairs)

    def predict(self, X):
        return self._underlying.predict(X)

    def weights(self):
        return self._underlying.weights()
