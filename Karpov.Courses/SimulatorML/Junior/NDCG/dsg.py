"""
Calculations of Discounted Cummulative Gain for k objects with 2 methods.

Methods:
* standard
    DCG@k = sum_{i=1}^{k} {rel_i}/{log2(i+1)}
* industry
    DCG@k = sum_{i=1}^{k} {2^{rel_i} - 1}/{log2(i+1)}
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
    # relevance_sorted = sorted(relevance, reverse=True)[:k]
    score = 0
    for idx, cur_r in enumerate(relevance[:k], 1):
        gain = compute_gain(cur_r, gain_scheme=gain_scheme)
        score += gain / np.log2(idx+1)
    return float(score)


# if __name__=='__main__':
#     relevance = [0.99, 0.74, 0.71, 0.94, 0.88, 0.68]
#     k = 5
#     method = 'standard'
#     print(discounted_cumulative_gain(relevance, k, method))     # 2.6164...

#     relevance1 = [0.99, 0.94, 0.88]
#     relevance2 = [0.99, 0.83, 0.89]
    # print(discounted_cumulative_gain(relevance1, k, method),
    #       discounted_cumulative_gain(relevance2, k, method))
