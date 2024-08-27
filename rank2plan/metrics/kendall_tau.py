from rank2plan import Pair
from typing import List
from numpy import ndarray


def kendall_tau(pairs: List[Pair], scores: ndarray) -> float:
    """Compute the Kendall's tau correlation coefficient between the predicted
    scores and the ranking pairs.

    Args:
        pairs (List[Pair]): The list of ranking pairs, shape (n_pairs,).
        scores (ndarray): The predicted scores, shape (n_samples,).

    Returns:
        float: The Kendall's tau correlation coefficient.
    """
    total = 0
    concordant = 0
    discordant = 0
    for pair in pairs:
        i, j = pair.i, pair.j
        # ignore gaps in kendall tau
        if scores[i] <= scores[j]:
            concordant += pair.sample_weight
        else:
            discordant += pair.sample_weight
        total += pair.sample_weight

    return (concordant - discordant) / total
