from rank2plan import LpModel
from rank2plan.metrics import kendall_tau
import numpy as np
import random


def test_primal_lp_simple(small_ranking_dataset, pulp_cbc):
    random.seed(0)
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


def test_primal_lp_simple_sparse(small_sparse_ranking_dataset, pulp_cbc):
    random.seed(0)
    model = LpModel(
        pulp_cbc, use_column_generation=False, use_constraint_generation=False
    )
    X_train, pairs = small_sparse_ranking_dataset
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


def test_primal_lp_miconic(miconic_mock_dataset, pulp_cbc):
    random.seed(0)
    model = LpModel(
        pulp_cbc,
        use_column_generation=False,
        use_constraint_generation=False,
    )
    X, pairs = miconic_mock_dataset
    model.fit(X, pairs)
    train_scores = model.predict(X)
    assert kendall_tau(pairs, train_scores) > 0.4


def test_primal_lp_miconic_sparse(miconic_mock_sparse_dataset, pulp_cbc):
    random.seed(0)
    model = LpModel(
        pulp_cbc,
        use_column_generation=False,
        use_constraint_generation=False,
    )
    X, pairs = miconic_mock_sparse_dataset
    model.fit(X, pairs)
    train_scores = model.predict(X)
    assert kendall_tau(pairs, train_scores) > 0.4


# We don't test tune_then_fit here to avoid tests taking too long
