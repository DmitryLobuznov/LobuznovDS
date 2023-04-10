"""
Calculations of CummulativeGain for k objects.
"""
from typing import List
import numpy as np


def cumulative_gain(relevance: List[float], k: int) -> float:
    """Score is cumulative gain at k (CG@k)

    Parameters
    ----------
    relevance:  `List[float]`
        Relevance labels (Ranks)
    k : `int`
        Number of elements to be counted

    Returns
    -------
    score : float
    """
    score = np.sum(np.sort(relevance[:k]))
    return score


# if __name__ == "__main__":
#     relevance = [0.99, 0.94, 0.88, 0.74, 0.71, 0.68]
#     k = 5
#     print(cumulative_gain(relevance, k))        #  4.26...
