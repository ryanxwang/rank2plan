from pulp import lpDot, lpSum
from scipy.sparse import issparse


def sparseLpDot(a, b):
    if not issparse(a) and not issparse(b):
        return lpDot(a, b)
    if issparse(a) and not issparse(b):
        a, b = b, a

    if issparse(a):
        raise ValueError(
            "Both a and b are sparse, this is just a dot product, not an lpDot"
        )

    result = []
    for i, val in zip(b.indices, b.data):
        result.append(a[i] * val)
    return lpSum(result)
