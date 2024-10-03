from rank2plan import LpModel
from rank2plan.metrics import kendall_tau
import numpy as np
import random


def test_constraint_column_generation_simple(small_ranking_dataset, pulp_cbc):
    random.seed(0)
    model = LpModel(
        pulp_cbc, use_column_generation=True, use_constraint_generation=True
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


def test_constraint_column_generation_simple_sparse(
    small_sparse_ranking_dataset, pulp_cbc
):
    random.seed(0)
    model = LpModel(
        pulp_cbc, use_column_generation=True, use_constraint_generation=True
    )
    X, pairs = small_sparse_ranking_dataset
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
    assert kendall_tau(pairs, train_scores) > 0.4


def test_constraint_column_generation_miconic_sparse(
    miconic_mock_sparse_dataset, pulp_cbc
):
    random.seed(0)
    model = LpModel(
        pulp_cbc,
        use_column_generation=True,
        use_constraint_generation=True,
    )
    X, pairs = miconic_mock_sparse_dataset
    model.fit(X, pairs)
    train_scores = model.predict(X)
    assert kendall_tau(pairs, train_scores) > 0.4


def test_constraint_column_generation_tune_then_fit_miconic(
    miconic_mock_dataset, pulp_cbc
):
    random.seed(0)
    model = LpModel(
        pulp_cbc,
        use_column_generation=True,
        use_constraint_generation=True,
    )
    X, pairs = miconic_mock_dataset

    split_index = int(len(pairs) * 0.8)
    pairs_train, pairs_val = pairs[:split_index], pairs[split_index:]
    X_train = X
    X_val = X

    model.tune(X_train, pairs_train, X_val, pairs_val)
    train_scores = model.predict(X)
    assert kendall_tau(pairs, train_scores) > 0.4


def test_constraint_column_generation_tune_then_fit_miconic_sparse(
    miconic_mock_sparse_dataset, pulp_cbc
):
    random.seed(0)
    model = LpModel(
        pulp_cbc,
        use_column_generation=True,
        use_constraint_generation=True,
    )
    X, pairs = miconic_mock_sparse_dataset

    split_index = int(len(pairs) * 0.8)
    pairs_train, pairs_val = pairs[:split_index], pairs[split_index:]
    X_train = X
    X_val = X

    model.tune(X_train, pairs_train, X_val, pairs_val)
    train_scores = model.predict(X)
    assert kendall_tau(pairs, train_scores) > 0.4
