from rank2plan import LpModel
from rank2plan.metrics import kendall_tau
import numpy as np


def test_primal_lp(small_ranking_dataset, pulp_cbc):
    model = LpModel(
        pulp_cbc, use_column_generation=False, use_constraint_generation=False
    )
    X_train, pairs = small_ranking_dataset
    model.fit(X_train, pairs)

    train_scores = model.predict(X_train)
    assert train_scores.shape == (7,)
    assert (
        kendall_tau(pairs, train_scores) > 0.5
    )  # might not get perfect score due to regularisation

    X_test = np.array([[1.1, 1.1], [2.3, 2.3], [1.0, 1.0]])

    test_scores = model.predict(X_test)

    assert test_scores.shape == (3,)
    assert test_scores[1] < test_scores[0]
    assert test_scores[1] < test_scores[2]
