from rank2plan import LpModel
import numpy as np


def test_primal_lp(small_ranking_dataset):
    model = LpModel(use_column_generation=False, use_constraint_generation=False)
    X_train, pairs = small_ranking_dataset
    model.fit(X_train, pairs)

    X_test = np.array([[1.1, 1.1], [2.3, 2.3], [1.0, 1.0]])
    scores = model.predict(X_test)

    assert scores.shape == (3,)
    assert scores[1] < scores[0]
    assert scores[1] < scores[2]
