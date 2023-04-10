"""
Реализовать оценку доверительного интервала для ROC-AUC c помощью бутстрепа.

На вход подаётся обученный классификатор (модели из scikit-learn), тестовая выборка,
    а также размер доверительного интервала и количество бутстрепнутых выборок.

"""
from typing import Tuple, List
from scipy import stats
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score


def get_normal_ci(bootstrap_stats: List[float],
                  pe: float,
                  alpha: float) -> Tuple[float, float]:
    """Calculates normal confidence interval.

    Parameters
    ----------
    bootstrap_stats: `np.ndarray`
        Bootstrap statistic values.
    pe : `float`
        Point estimation.
    alpha: `float`
        Confidence level.

    Returns
    -------
    (lcb, ucb): `Tuple[float, float]
        Lower and Upper bound of confidence interval.
    """
    delta = stats.norm.ppf(1 - alpha / 2) * np.std(bootstrap_stats)
    lcb, ucb = pe - delta, pe + delta
    return (lcb, ucb)


def get_pivotal_ci(bootstrap_stats: np.ndarray,
                   pe: float,
                   alpha: float) -> Tuple[float, float]:
    """Строит центральный доверительный интервал."""
    left, right= 2 * pe - np.quantile(bootstrap_stats, [1 - alpha / 2,  alpha / 2])
    if left > right:
        left, right = right, left
    return left, right


def get_roc_auc_ci(bootstrap_stats: List[float],
                   alpha: float) -> Tuple[float, float]:
    """Calculates confidence interval.

    Parameters
    ----------
    bootstrap_stats: `np.ndarray`
        Bootstrap statistic values.
    alpha: `float`
        Confidence level.

    Returns
    -------
    (lcb, ucb): `Tuple[float, float]
        Lower and Upper bound of confidence interval.
    """
    n_scores = len(bootstrap_stats)
    sorted_scores = np.array(bootstrap_stats)
    sorted_scores.sort()
    left_idx, right_idx = int(alpha * n_scores), int((1 - alpha) * n_scores)
    lcb, ucb = sorted_scores[left_idx], sorted_scores[right_idx]
    return (lcb, ucb)


def roc_auc_ci(classifier: ClassifierMixin,
               X: np.ndarray,
               y: np.ndarray,
               conf: float = 0.95,
               n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC metric.

    Parameters
    ----------
    classifier : `sklearn.base.ClassifierMixin`
        Fitted classifier from scikit-learn.
    X : `np.ndarray`
        Features from test sample.
    y : `np.ndarray`
        Labels from test sample.
    conf : `float`
        Confidence level.
    n_bootstraps : `int`
        Number of bootstraps steps.

    Returns
    -------
    (lcb, ucb): `Tuple[float, float]`
        Lower and Upper confidence bounds of confidence interval.
    """
    y_true = y
    # Prediction
    y_pred = classifier.predict(X)
    # Constants
    len_test, alpha = len(X), 1 - conf
    # Point estimation
    point_est = roc_auc_score(y, classifier.predict(X))
    # List for roc-auc scores
    bootstrap_roc_auc = []
    # Generate bootstrap samples
    for _ in range(n_bootstraps):
        indices = np.random.randint(low=0, high=len_test, size=len_test)
        # Can't calculate ROC-AUC. We need at least one negative, one positive sample.
        if len(np.unique(y[indices])) < 2:
            continue
        # Calculate score for bootstrapped sample
        bootstrap_roc_auc.append(roc_auc_score(y_true[indices], y_pred[indices]))
    # lcb, ucb = get_roc_auc_ci(bootstrap_roc_auc, alpha)
    # lcb, ucb = get_normal_ci(bootstrap_roc_auc, point_est, alpha)
    lcb, ucb = get_pivotal_ci(bootstrap_roc_auc, point_est, alpha)
    return (lcb, ucb)
