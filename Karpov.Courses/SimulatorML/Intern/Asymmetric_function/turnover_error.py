"""
Мы работаем в компании X6. Наша задача – прогноз товарооборота бытовой техники: нужно понять,
    сколько каждого товара привезти на ту или иную точку для продажи.
Допустим, поставки осуществляются раз в 3 месяца.
Если привезем слишком мало, то весь товар скупят и новые клиенты не смогут купить нужную им технику.
Из-за чего мы потеряем потенциальную прибыль (missed profit), так как не смогли покрыть спрос.
Если привозим слишком много (а зачастую бытовая техника – крупногабаритный товар),
    то это ведёт к переполнению склада, что затрудняет ввоз более новых моделей техники.
Что в данной ситуации предпочтительнее: привезти на склад больше чем нужно или меньше?
Вам необходимо придумать функцию потерь, которая была бы адекватная для данной бизнес-задачи:
"""
import numpy as np


def turnover_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
    # Case 1: RMSLE => 100/100. Correct!
    error = np.mean((np.log(y_true+1) - np.log(y_pred+1)) ** 2)
    # Case 2: SMAPE => 100/100. Correct!
    error = np.mean(
            np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred))/2)
        )*100
    return error
