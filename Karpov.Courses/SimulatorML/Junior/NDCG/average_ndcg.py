"""
Calculations of Average Normalized Discounted Cummulative Gain for top-k objects with 2 methods
    and for list of relevances.

Methods:
* standard
    DCG@k   = sum_{i=1}^{k} {rel_i}/{log2(i+1)}
* industry
    DCG@k = sum_{i=1}^{k} {2^{rel_i} - 1}/{log2(i+1)}
nDCG@k  = DCG@k / IDCG@k
Average nDCG@k = sum_{i=1}^{n} {nDCG@k(qi)} / {n}, qi - query
"""
from typing import List
import numpy as np


def compute_gain(y_value: float, gain_scheme: str) -> float:
    """Calculate gain for y_value by selected gain_scheme.

    Parameters
    ----------
    y_value: `float`
        Value to compute gain
    gain_scheme: `str`
        Gain calculation scheme.
        `const` - y_value
        `exp2`  - 2^{y_value} - 1

    Returns
    -------
    gain: `float`
        Gain value
    """
    if gain_scheme == "exp2":
        gain = 2 ** y_value - 1
    elif gain_scheme == "const":
        gain = y_value
    else:
        raise ValueError(f"{gain_scheme} method not supported, only `exp2` and `const`.")
    return float(gain)


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if method == 'standard':
        gain_scheme = 'const'
    elif method == 'industry':
        gain_scheme = 'exp2'
    else:
        raise ValueError()
    score = 0
    for idx, cur_r in enumerate(relevance[:k], 1):
        gain = compute_gain(cur_r, gain_scheme=gain_scheme)
        score += gain / np.log2(idx+1)
    return float(score)


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    dcg  = discounted_cumulative_gain(relevance, k, method)
    idcg = discounted_cumulative_gain(sorted(relevance, reverse=True), k, method)
    score = float(dcg / idcg)
    return score


def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = 'standard') -> float:
    """avarage nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    score = np.mean([normalized_dcg(relevance, k, method) for relevance in list_relevances])
    return score


# if __name__=="__main__":
#     list_relevances = [
#         [0.99, 0.94, 0.88, 0.89, 0.72, 0.65],
#         [0.99, 0.92, 0.93, 0.74, 0.61, 0.68],
#         [0.99, 0.96, 0.81, 0.73, 0.76, 0.69]
#     ]
#     k = 5
#     method = 'standard'
#     print(avg_ndcg(list_relevances, k, method))     # 0.99613...
