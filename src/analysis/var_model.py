import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from typing import Tuple, List, Optional, Dict


def select_optimal_lag(data: pd.DataFrame, max_lags: int, criteria: List[str] = ['aic', 'bic']) -> Dict[str, int]:
    """
    Выбирает оптимальный порядок лагов для VAR модели на основе информационных критериев.

    Args:
        data: DataFrame, содержащий временные ряды (предполагается стационарным).
        max_lags: Максимальное количество лагов для проверки.
        criteria: Список информационных критериев для использования ('aic', 'bic', 'hqic', 'fpe').

    Returns:
        Словарь, сопоставляющий каждый критерий с выбранным оптимальным порядком лага.
    """
    print(
        f"Выбор оптимального порядка лагов (max_lags={max_lags}) по критериям: {criteria}")
    if data.isnull().values.any():

        print("Предупреждение: Данные содержат NaN значения. Это может привести к некорректным результатам.")
        
    try:
        model = VAR(data)
        # Note: statsmodels select_order might print results directly.
        # We capture the results programmatically.
        lag_selection_results = model.select_order(maxlags=max_lags)
        print("\nLag Selection Results Summary:")
        print("Результаты выбора порядка лагов:")
        print(lag_selection_results.summary())

        optimal_lags = {}
        for criterion in criteria:
            if criterion in lag_selection_results.selected_orders:
                optimal_lags[criterion] = lag_selection_results.selected_orders[criterion]
            else:
                print(
                    f"Предупреждение: Критерий '{criterion}' не найден в результатах выбора.")
                optimal_lags[criterion] = -1

        print("\nВыбранные оптимальные лаги:")
        for crit, lag in optimal_lags.items():
            print(f"  {crit.upper()}: {lag}")

        return {
            **optimal_lags, # Unpack the optimal lags (e.g., 'aic': 1, 'bic': 1)
            'summary': lag_selection_results # Add the summary object
        }

    except Exception as e:
        print(f"Ошибка при выборе лагов: {e}")
        return {crit: -1 for crit in criteria}


def fit_var_model(data: pd.DataFrame, lag_order: int) -> Optional[VARResults]:
    """
    Подгоняет VAR модель к данным с указанным порядком лага.

    Args:
        data: DataFrame, содержащий временные ряды (предполагается стационарным).
        lag_order: Количество лагов для включения в модель.

    Returns:
        Подобранный объект VARResults или None, если подгонка не удалась.
    """
    print(f"Подгонка VAR модели с порядком лагов: {lag_order}")
    if data.isnull().values.any():
        print("Предупреждение: Данные содержат NaN значения. Подгонка VAR может завершиться ошибкой.")

    if lag_order < 0:
        print("Ошибка: Некорректный порядок лагов (должен быть >= 0).")
        return None
    if lag_order == 0:
        print("Ошибка: Порядок лагов 0 не подходит для анализа причинности по Грейнджеру.")
        return None

    try:
        model = VAR(data)
        results = model.fit(lag_order)
        print("\nРезультаты подгонки VAR модели:")
        print(results.summary())
        return results
    except Exception as e:
        print(f"Ошибка при подгонке VAR модели: {e}")
        return None


def check_model_stability(results: VARResults) -> bool:
    """
    Проверяет, является ли подобранная VAR модель стабильной.
    (Все корни характеристического полинома лежат за пределами единичного круга).

    Args:
        results: Подобранный объект VARResults.

    Returns:
        True, если модель стабильна, False в противном случае.
    """
    print("Проверка стабильности VAR модели...")
    try:
        roots = results.roots
        is_stable = np.all(np.abs(roots) < 1)
        is_stable_sm = results.is_stable(verbose=True)

        print(
f"\nПроверка стабильности модели: {'Стабильна' if is_stable_sm else 'Нестабильна'}")
        return is_stable_sm
    except Exception as e:
        print(f"Ошибка при проверке стабильности модели: {e}")
        return False


if __name__ == '__main__':
    # Example usage (for testing purposes)
    print("\nТестирование функций VAR модели...")

    # Создание примера стационарных данных
    idx = pd.period_range(start='2020-01', periods=100, freq='M')
    data1 = np.random.randn(100)
    # Создание второго ряда, связанного с первым с лагом
    data2 = 0.5 * pd.Series(data1).shift(1).fillna(0) + \
        np.random.randn(100) * 0.5
    df_var_test = pd.DataFrame({'Var1': data1, 'Var2': data2}, index=idx)

    print("\nПример DataFrame для VAR:")
    print(df_var_test.head())

    # Выбор оптимального лага
    max_lags_test = 5
    criteria_test = ['aic', 'bic']
    optimal_lags_result = select_optimal_lag(
        df_var_test, max_lags=max_lags_test, criteria=criteria_test)

    # Подгонка VAR модели (используя BIC лаг, например)
    chosen_lag = optimal_lags_result.get('bic', -1)
    if chosen_lag > 0:  # Убедимся, что найден корректный лаг
        var_results = fit_var_model(df_var_test, lag_order=chosen_lag)

        # Проверка стабильности
        if var_results:
            is_stable = check_model_stability(var_results)
        else:
            print("Ошибка подгонки VAR модели, пропуск проверки стабильности.")
    elif chosen_lag == 0:
        print("Оптимальный лаг 0, VAR модель неприменима в стандартном смысле.")
    else:
        print("Ошибка выбора оптимального лага или получен некорректный лаг.")

    print("\nТестирование VAR модели завершено.")
