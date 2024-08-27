from rank2plan import LpModel
import numpy as np


def test_constraint_generation_simple(small_ranking_dataset, pulp_cbc):
    model = LpModel(
        pulp_cbc,
        use_column_generation=False,
        use_constraint_generation=True,
        verbose=True,
    )
    X, pairs = small_ranking_dataset
    model.fit(X, pairs)

    X_test = np.array([[1.1, 1.1], [2.3, 2.3], [1.0, 1.0]])
    scores = model.predict(X_test)

    assert scores.shape == (3,)
    assert scores[1] < scores[0]
    assert scores[1] < scores[2]


def test_constraint_generation_miconic(miconic_mock_dataset, pulp_cbc):
    model = LpModel(
        pulp_cbc,
        use_column_generation=False,
        use_constraint_generation=True,
        verbose=True,
    )
    X, pairs = miconic_mock_dataset
    model.fit(X, pairs)

    scores = model.predict(X)
