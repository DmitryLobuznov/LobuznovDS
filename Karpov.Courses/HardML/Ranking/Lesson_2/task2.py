from math import log2
from torch import Tensor, rand, sort, randn, randint


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    # Number of objects
    N_objects = ys_true.shape[0]
    # Sort ys
    ys_pred_sorted, indices = sort(ys_pred, dim=0, descending=True)
    ys_true_sorted = ys_true[indices]
    print("true:", ys_true_sorted)
    print("pred:", ys_pred_sorted)
    # Counter
    cnt_swap = 0
    # Loop over objects
    for i in range(N_objects - 1):
        for j in range(i+1, N_objects): # j = i+1
            if (ys_true_sorted[i] > ys_true_sorted[j]) and (ys_pred_sorted[i] < ys_pred_sorted[j]):
                cnt_swap += 1
            elif (ys_pred_sorted[i] > ys_pred_sorted[j]) and (ys_true_sorted[i] < ys_true_sorted[j]):
                cnt_swap += 1
    return cnt_swap



def compute_gain(y_value: float, gain_scheme: str) -> float:
    assert gain_scheme in ["const", "exp2"]
    if gain_scheme == "const":
        gain = float(y_value)
    elif gain_scheme == "exp2":
        gain = float(2 ** y_value - 1)
    return gain


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    # Number of objects
    N_objects = ys_true.shape[0]
    # Sort ys
    ys_pred_sorted, indices = sort(ys_pred, dim=0, descending=True)
    ys_true_sorted = ys_true[indices]
    # Metric value
    dcg_value = 0 
    # Loop over documents
    
    dcg_value = sum([compute_gain(y_true, gain_scheme) / log2(i + 1) for i, y_true in enumerate(ys_true_sorted, start=1)])
    return float(dcg_value)


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    empirical_dcg = dcg(ys_true, ys_pred, gain_scheme=gain_scheme)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme=gain_scheme)
    ndcg = empirical_dcg / ideal_dcg
    return ndcg


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    if ys_true.sum() == 0:
        return -1
    # Sort ys
    ys_pred_sorted, indices = sort(ys_pred, dim=0, descending=True)
    ys_true_sorted = ys_true[indices]
    # Select first k documents
    rel_sum = ys_true_sorted[:k].sum()
    precision = rel_sum / min(ys_true.sum(), k)
    return float(precision)


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    # Sort ys
    ys_pred_sorted, indices = sort(ys_pred, dim=0, descending=True)
    ys_true_sorted = ys_true[indices]

    for i, y_true in enumerate(ys_true_sorted, start=1):
        if y_true == 1:
            return 1 / i
    return 0


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    pLook, pFound = 1, 0
    # Sort ys
    ys_pred_sorted, indices = sort(ys_pred, dim=0, descending=True)
    ys_true_sorted = ys_true[indices]
    # Loop over documents
    for y_true in ys_true_sorted:
        float_y_true = float(y_true)
        pFound += pLook * float_y_true
        pLook *= (1 - float_y_true) * (1 - p_break)
    return float(pFound)


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    if ys_true.sum() == 0:
        return -1
    # Correct prediction
    correct_cnt = 0
    # Average sum over K
    sum_K = 0
    # Sort ys
    ys_pred_sorted, indices = sort(ys_pred, dim=0, descending=True)
    ys_true_sorted = ys_true[indices]
    # Loop over documents
    for i, y_true in enumerate(ys_true_sorted, start=1):
        if y_true == 1:
            correct_cnt += 1
            sum_K += correct_cnt / i
    return sum_K / correct_cnt if correct_cnt != 0 else 0


# if __name__ == '__main__':
    # true, pred = randint(1, 10, size=(5,)), randint(1, 10, size=(5,))
    # print(f"{num_swapped_pairs(true, pred) = }")
    # print(f"{dcg(true, pred, gain_scheme='const') = }")
    # print(f"{dcg(true, pred, gain_scheme='exp2') = }")
    # print(f"{ndcg(true, pred) = }")
    # print(f"{precission_at_k(true, pred, k=3) = }")
    # print(f"{precission_at_k(true, pred, k=10) = }")
    # print(f"{reciprocal_rank(true, pred) = }")
    # print(f"{p_found(true, pred) = }")
    # print(f"{average_precision(true, pred) = }")