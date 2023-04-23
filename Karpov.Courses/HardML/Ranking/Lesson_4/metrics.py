"""Ranking metrics realizations."""
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
from math import log2
from pprint import pprint


def num_swapped_pairs(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> int:
    """Функция для подсчёта неправильно упорядоченных пар,
        правильно <=> от наибольшего к наименьшему в ys_true.
        Что эквивалентно числу перестановок пар.

    Parameters
    ----------
    ys_true : `torch.Tensor`
        Correct labels rank.
    ys_pred : `torch.Tensor`
        Predicted labels rank.

    Returns
    -------
    swapped_cnt : `int`
        #_of_swapped_pairs.
    """
    ys_pred_sorted, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    
    num_objects = ys_true_sorted.shape[0]
    swapped_cnt = 0
    for cur_obj in range(num_objects - 1):
        for next_obj in range(cur_obj + 1, num_objects):
            if ys_true_sorted[cur_obj] < ys_true_sorted[next_obj]:
                if ys_pred_sorted[cur_obj] > ys_pred_sorted[next_obj]:
                    swapped_cnt += 1
            elif ys_true_sorted[cur_obj] > ys_true_sorted[next_obj]:
                if ys_pred_sorted[cur_obj] < ys_pred_sorted[next_obj]:
                    swapped_cnt += 1
    return swapped_cnt


def precission_at_k(ys_true: torch.Tensor, ys_pred: torch.Tensor, k: int) -> float:
    """Metric:
        Calculates the precission at top-k predictions.

    Parameters
    ----------
    ys_true : `torch.Tensor`
        Correct labels rank.
    ys_pred : `torch.Tensor`
        Predicted labels rank.

    Returns
    -------
    p_at_k : `float`
    """
    if ys_true.sum() == 0:
        return -1
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    hits = ys_true_sorted[:k].sum()
    p_at_k = hits / min(ys_true.sum(), k)
    return float(p_at_k)


def reciprocal_rank(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    """Metric:
        Calculates the reciprocal rank.

    Parameters
    ----------
    ys_true : `torch.Tensor`
        Correct labels rank.
    ys_pred : `torch.Tensor`
        Predicted labels rank.

    Returns
    -------
    reciprocal_rank : `float`
    """
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    
    for idx, cur_y in enumerate(ys_true_sorted, 1):
        if cur_y == 1:
            return 1 / idx
    return 0


def average_precision(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    """Metric:
        Calculates the Average Precision (AP) = \sum_K {#_relevant_documents} / k.

    Parameters
    ----------
    ys_true : `torch.Tensor`
        Correct labels rank.
    ys_pred : `torch.Tensor`
        Predicted labels rank.

    Returns
    -------
    avg_precision : `float`
    """
    if ys_true.sum() == 0:
        return -1
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    rolling_sum = 0
    num_correct_ans = 0
    
    for idx, cur_y in enumerate(ys_true_sorted, start=1):
        if cur_y == 1:
            num_correct_ans += 1
            rolling_sum += num_correct_ans / idx
    if num_correct_ans == 0:
        return 0
    else:
        return rolling_sum / num_correct_ans


def compute_gain(y_value: float, gain_scheme: str) -> float:
    """Compute the Gain for a given y_value and gain_scheme.
        Gain schemes:
        * const : gain = rank;
        * exp2  : gain = 2^rank - 1.

    Parameters
    ----------
    y_value : float
        Rank label value.
    gain_scheme : str
        Gain scheme.

    Returns
    -------
    gain : float
    """
    if gain_scheme == "exp2":
        gain = 2 ** y_value - 1
    elif gain_scheme == "const":
        gain = y_value
    else:
        raise ValueError(f"{gain_scheme} method not supported, only `exp2` and `const`.")
    return float(gain)


def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str, top_k: int=None) -> float:
    """Metric:
        Calculates the Discounted Cumulative Gain (DCG)

    Parameters
    ----------
    ys_true : `torch.Tensor`
        Correct labels rank.
    ys_pred : `torch.Tensor`
        Predicted labels rank.
    gain_scheme : `str`
        Gain scheme. Allowed values = ['const', 'exp2']
            * const : gain = rank;
            * exp2  : gain = 2^rank - 1.
    top_k : `Optional[int]`
        Top k most relevant objects.

    Returns
    -------
    dcg : `float`
    """
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    if top_k:
        argsort = argsort[:top_k]
    ret = 0
    for idx, cur_y in enumerate(ys_true_sorted, 1):
        gain = compute_gain(cur_y, gain_scheme)
        ret += gain / log2(idx + 1)
    return ret


def ndcg(ys_true: torch.Tensor,
         ys_pred: torch.Tensor,
         gain_scheme: str = 'const',
         return_ideal: bool = False) -> Union[float, Tuple[float, float]]:
    """Metric:
        Calculates the Normalized Discounted Cumulative Gain (NDCG)

    Parameters
    ----------
    ys_true : `torch.Tensor`
        Correct labels rank.
    ys_pred : `torch.Tensor`
        Predicted labels rank.
    gain_scheme : `str`
        Gain scheme. Allowed values = ['const', 'exp2']
            * const : gain = rank;
            * exp2  : gain = 2^rank - 1.
    return_ideal : `bool`
        Return ideal_ndcg also or not.
        Default=False.
        If True, return (ndcg, ideal_ndcg)

    Returns
    -------
    * ndcg_value : `float`
    * (ndcg_value, ideal_dcg) : `Tuple[float, float]`
    """
    pred_dcg = dcg(ys_true, ys_pred, gain_scheme)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
    
    ndcg_value = pred_dcg / ideal_dcg
    return ndcg_value if not return_ideal else (ndcg_value, ideal_dcg)


def ndcg_k(y_true: torch.Tensor, y_pred: torch.Tensor, gain_scheme: str, top_k: int) -> float:
    """Metric:
        Calculates the Normalized Discounted Cumulative Gain (NDCG) by k most relevant.

    Parameters
    ----------
    ys_true : `torch.Tensor`
        Correct labels rank.
    ys_pred : `torch.Tensor`
        Predicted labels rank.
    gain_scheme : `str`
        Gain scheme. Allowed values = ['const', 'exp2']
            * const : gain = rank;
            * exp2  : gain = 2^rank - 1.
    top_k : `int`
        Top k most relevant objects.

    Returns
    -------
    * ndcg_value : `float`
    * (ndcg_value, ideal_dcg) : `Tuple[float, float]`
    """
    def dcg(y_true: torch.Tensor, y_pred: torch.Tensor)-> float:
        # Sort ys
        _, argsort = torch.sort(y_pred, dim=0, descending=True)
        argsort = argsort[:top_k]
        y_true_sorted = y_true[argsort]
        # Metric value
        dcg_value = 0
        for i, l in enumerate(y_true_sorted, 1):
            dcg_value += (2 ** l - 1) / log2(i + 1)
        return float(dcg_value)

    empirical_dcg = dcg(y_true, y_pred)
    ideal_dcg = dcg(y_true, y_true)

    return empirical_dcg / ideal_dcg


def p_found(ys_true: torch.Tensor, ys_pred: torch.Tensor, p_break: float = 0.15 ) -> float:
    """Metric:
        Calculates the P-Found = \sum_{i=1}^n pLook(i) \cdot pRel(i),
            where calculates recursively pLook[i] = pLook[i-1]*(1 - pRel[i-1])*(1 - pBreak).
            pBreak - probability to break tries to find a match.

    Parameters
    ----------
    ys_true : `torch.Tensor`
        Correct labels rank.
    ys_pred : `torch.Tensor`
        Predicted labels rank.
    p_break : `float`
        Probability to break tries to find a match. Allowed values are between 0 and 1.
        Default = 0.15

    Returns
    -------
    p_found : `float`
    """
    p_look = 1
    p_found = 0
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]

    for cur_y in ys_true_sorted:
        p_found += p_look * float(cur_y)
        p_look = p_look * (1 - float(cur_y)) * (1 - p_break)
    
    return p_found