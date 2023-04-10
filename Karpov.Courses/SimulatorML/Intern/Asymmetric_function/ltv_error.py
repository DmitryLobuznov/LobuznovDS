"""
...
"""
import numpy as np


def ltv_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Функция для вычисление ошибки спроса.
    Хотим больше штрафовать за недопрогноз, чем за перепрогноз.
    Параметры:
    ----------
    y_true: np.ndarray
        Значения целевой переменной.
    y_pred: np.ndarray
        Прогноз значений целевой переменной.

    Возвращаемое значение:
    ----------------------
    error: float
        Значение метрики.
    """
    # Case 1: QuantileLoss(gamma): gamma < 0.5 =>
    gamma = 0.4
    abs_error = np.abs(y_true - y_pred)
    left, right = gamma * abs_error, (1 - gamma) * abs_error
    error = np.where(y_true >= y_pred, left, right)
    return np.mean(error)
