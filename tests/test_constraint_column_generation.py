from rank2plan import LpModel
from rank2plan.metrics import kendall_tau
import numpy as np
import random


def test_constraint_column_generation_simple(small_ranking_dataset, pulp_cbc):
    random.seed(0)
    model = LpModel(
        pulp_cbc,
        use_column_generation=True,
        use_constraint_generation=True,
    )
    X, pairs = small_ranking_dataset
    model.fit(X, pairs)

    train_scores = model.predict(X)
    assert train_scores.shape == (7,)
    assert kendall_tau(pairs, train_scores) > 0.5

    X_test = np.array([[1.1, 1.1], [2.3, 2.3], [1.0, 1.0]])
    test_scores = model.predict(X_test)

    assert test_scores.shape == (3,)
    assert test_scores[1] < test_scores[0]
    assert test_scores[1] < test_scores[2]


def test_constraint_column_generation_miconic(miconic_mock_dataset, pulp_cbc):
    random.seed(0)
    model = LpModel(
        pulp_cbc,
        use_column_generation=True,
        use_constraint_generation=True,
    )
    X, pairs = miconic_mock_dataset
    model.fit(X, pairs)
    train_scores = model.predict(X)
    assert kendall_tau(pairs, train_scores) > 0.5
