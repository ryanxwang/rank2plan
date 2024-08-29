from numpy import ndarray
import numpy as np
from typing import List
from rank2plan import Pair


def compute_main_objective(X: ndarray, pairs: List[Pair], beta: ndarray) -> float:
    """Compute the main objective of the model, not considering C value.

    Args:
        X (ndarray): The feature matrix
        pairs (List[Pair]): The pairs
        beta (ndarray): The weights

    Returns:
        float: The main objective
    """
    res = 0
    for pair in pairs:
        res += max(
            0, pair.sample_weight * (pair.gap - (X[pair.j] - X[pair.i]).T.dot(beta))
        )
    return res


def compute_regularisation_objective(beta: ndarray) -> float:
    return np.sum(np.abs(beta))


def compute_overall_objective(
    X: ndarray, pairs: List[Pair], beta: ndarray, C: float
) -> float:
    return compute_main_objective(
        X, pairs, beta
    ) * C + compute_regularisation_objective(beta)
