from numpy import ndarray
import numpy as np
from typing import List
from rank2plan import Pair


def compute_X_tilde(X: ndarray, pairs: List[Pair]) -> ndarray:
    """Compute the X_tilde matrix.

    Args:
        X (ndarray): The original feature matrix, shape (n_samples, n_features).
        pairs (List[Pair]): The list of pairs, shape (n_pairs,).

    Returns:
        ndarray: The X_tilde matrix, shape (n_pairs, n_features).
    """
    X_tilde = []
    for pair in pairs:
        X_tilde.append(pair.sample_weight * (X[pair.j] - X[pair.i]))
    return np.array(X_tilde)
