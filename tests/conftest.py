import pytest
import numpy as np
from numpy import ndarray
from typing import List, Tuple
from rank2plan.types import Pair
from pulp import LpSolver, PULP_CBC_CMD
import json
from scipy.sparse import csr_matrix, spmatrix


@pytest.fixture
def small_ranking_dataset() -> Tuple[ndarray, List[Pair]]:
    X = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [1.2, 1.2],
            [1.3, 1.3],
            [0.9, 0.9],
            [2.2, 2.2],
            [1.3, 1.3],
        ]
    )
    pairs = [Pair(1, 0), Pair(1, 2), Pair(1, 3), Pair(1, 4), Pair(5, 6)]

    return (X, pairs)


@pytest.fixture
def small_sparse_ranking_dataset(small_ranking_dataset) -> Tuple[spmatrix, List[Pair]]:
    X, pairs = small_ranking_dataset
    X = csr_matrix(X)

    return (X, pairs)


@pytest.fixture
def miconic_mock_dataset() -> Tuple[ndarray, List[Pair]]:
    raw = json.load(open("tests/data/miconic_mock_data.json"))
    X = np.array(raw["features"])
    pairs = [
        Pair(
            pair["i"],
            pair["j"],
            sample_weight=pair["importance"],
            gap=1.0 if pair["relation"] == "Better" else 0.0,
        )
        for pair in raw["pairs"]
    ]

    # sample only some pairs for speed
    pairs = pairs[:200]
    return (X, pairs)


@pytest.fixture
def miconic_mock_sparse_dataset(miconic_mock_dataset) -> Tuple[spmatrix, List[Pair]]:
    X, pairs = miconic_mock_dataset
    X = csr_matrix(X)

    return (X, pairs)


@pytest.fixture
def pulp_cbc() -> LpSolver:
    return PULP_CBC_CMD(mip=False, msg=False, timeLimit=10, options=["RandomS 0"])
