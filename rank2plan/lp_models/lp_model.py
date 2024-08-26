from rank2plan.model import Model
from rank2plan.lp_models import PrimalLpModel


class LpModel(Model):
    def __init__(
        self,
        use_column_generation=False,
        use_constraint_generation=False,
        C=1.0,
        msgs=False,
        solver_time_limit=None,
        seed=0,
    ) -> None:
        if not use_column_generation and not use_constraint_generation:
            self._underlying = PrimalLpModel(
                C=C, msgs=msgs, solver_time_limit=solver_time_limit, seed=seed
            )

    def fit(self, X, pairs):
        self._underlying.fit(X, pairs)

    def predict(self, X):
        return self._underlying.predict(X)

    def weights(self):
        return self._underlying.weights()
