from numpy import ndarray
import numpy as np
from typing import List
from rank2plan import Pair, Matrix
from scipy.sparse import issparse, spmatrix, csr_matrix


def compute_X_tilde(X: Matrix, pairs: List[Pair]) -> Matrix:
    """Compute the X_tilde matrix.

    Args:
        X (Matrix): The original feature matrix, shape (n_samples, n_features).
        pairs (List[Pair]): The list of pairs, shape (n_pairs,). If X is sparse,
        it should be optimised for row access.

    Returns:
        Matrix: The X_tilde matrix, shape (n_pairs, n_features). Will be sparse
        if and only if X is sparse. If sparse, will be a CSR matrix.
    """
    if issparse(X):
        assert isinstance(X, spmatrix)
        data = []
        row_indices = []
        col_indices = []
        for row, pair in enumerate(pairs):
            diff = pair.sample_weight * (X.getrow(pair.j) - X.getrow(pair.i))
            data.extend(diff.data)
            row_indices.extend([row] * len(diff.data))
            col_indices.extend(diff.indices)
        return csr_matrix(
            (data, (row_indices, col_indices)), shape=(len(pairs), X.shape[1])
        )
    else:
        assert isinstance(X, np.ndarray)
        X_tilde = []
        for pair in pairs:
            X_tilde.append(pair.sample_weight * (X[pair.j] - X[pair.i]))
        return np.array(X_tilde)
