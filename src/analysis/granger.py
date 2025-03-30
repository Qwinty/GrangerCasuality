# src/analysis/granger.py
# Функции для выполнения тестов причинности по Грейнджеру на результатах VAR.

import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VARResults
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, Any, Optional, Tuple
import numpy as np


def perform_granger_causality_test(results: VARResults, max_lag: int, significance_level: float = 0.05) -> Optional[Dict[Tuple[str, str], Dict[str, Any]]]:
    """
    Выполняет тесты причинности по Грейнджеру для всех пар переменных в подогнанной VAR модели.

    Args:
        results: Объект VARResults с подогнанной моделью из statsmodels.
        max_lag: Максимальный порядок лага для тестирования причинности (обычно должен быть
                 порядком лага подогнанной VAR модели).
        significance_level: Пороговое значение p-value для определения значимости.

    Returns:
        Словарь, где ключи - кортежи (зависимая_переменная, причинная_переменная),
        а значения - словари, содержащие результаты теста (p-value, F-статистика,
        степени свободы и значимость). Возвращает None в случае ошибки.
    """
    print(
        f"\nВыполнение тестов причинности Грейнджера (max_lag={max_lag}, alpha={significance_level})...")
    if max_lag <= 0:
        print("Ошибка: max_lag должен быть положительным целым числом.")
        return None

    variables = results.names
 
    data = pd.DataFrame(results.model.y, columns=variables) 

    test_results = {}

    for caused_var in variables:
        for causing_var in variables:
            if caused_var == causing_var:
                continue  # Пропустить тестирование переменной на самой себе

            print(f"  Тест: {causing_var} вызывает {caused_var} по Грейнджеру?")

            # Выбор соответствующих столбцов для бивариантного теста
            test_data = data[[caused_var, causing_var]]

            try:
                # statsmodels grangercausalitytests ожидает DataFrame/array
                # Он выполняет тесты для лагов от 1 до max_lag
                # Нас интересуют результаты для указанного max_lag
                gc_results = grangercausalitytests(
                    test_data, [max_lag], verbose=False)

                # Извлечение результатов для указанного max_lag
                # Словарь результатов индексируется по номеру лага
                # [0] обращается к словарю результатов теста
                lag_result = gc_results[max_lag][0]

                f_test_stat = lag_result['ssr_ftest'][0]  # F-статистика
                p_value = lag_result['ssr_ftest'][1]     # p-значение
                # Числитель степеней свободы (лаг)
                df_num = lag_result['ssr_ftest'][2]
                # Знаменатель степеней свободы
                df_den = lag_result['ssr_ftest'][3]

                # Также рассматриваем 'params_ftest', который проверяет совместную значимость коэффициентов лагированной причинной переменной
                f_params_stat = lag_result['params_ftest'][0]
                p_params_value = lag_result['params_ftest'][1]

                significant = p_value < significance_level
                significant_params = p_params_value < significance_level

                test_results[(caused_var, causing_var)] = {
                    'ssr_F': f_test_stat,
                    'ssr_p_value': p_value,
                    'ssr_significant': significant,
                    'params_F': f_params_stat,
                    'params_p_value': p_params_value,
                    'params_significant': significant_params,
                    'lag': max_lag,
                    'df_num': df_num,
                    'df_den': df_den
                }
                print(
                    f"    ssr_ftest: p-значение={p_value:.4f} ({'Значимо' if significant else 'Не значимо'})")
                print(
                    f"    params_ftest: p-значение={p_params_value:.4f} ({'Значимо' if significant_params else 'Не значимо'})")

            except Exception as e:
                print(f"    Ошибка при тестировании {causing_var} -> {caused_var}: {e}")
                test_results[(caused_var, causing_var)] = {'error': str(e)}

    return test_results
 


def summarize_granger_results(results_dict: Dict[Tuple[str, str], Dict[str, Any]]) -> pd.DataFrame:
    """Форматирует результаты теста причинности Грейнджера в удобочитаемый DataFrame."""
    summary_list = []
    for (caused, causing), results in results_dict.items():
        if 'error' in results:
            summary_list.append({
                'Effect': f"{causing} -> {caused}",
                'Lag': results.get('lag', 'N/A'),
                'SSR_p_value': 'Error',
                'SSR_Significant': 'Error',
                'Params_p_value': 'Error',
                'Params_Significant': 'Error',
                'Details': results['error']
            })
        else:
            summary_list.append({
                'Effect': f"{causing} -> {caused}",
                'Lag': results['lag'],
                'SSR_p_value': f"{results['ssr_p_value']:.4f}",
                'SSR_Significant': results['ssr_significant'],
                'Params_p_value': f"{results['params_p_value']:.4f}",
                'Params_Significant': results['params_significant'],
                'Details': f"F={results['ssr_F']:.2f}, df=({results['df_num']:.0f}, {results['df_den']:.0f})"
            })
    return pd.DataFrame(summary_list)


if __name__ == '__main__':
    # Example usage (requires a fitted VAR model)
    print("\nТестирование функций причинности Грейнджера...")

    # --- Mocking VARResults ---
    # Эта часть обычно использует фактические результаты из fit_var_model
    from statsmodels.tsa.vector_ar.var_model import VAR
    idx = pd.period_range(start='2020-01', periods=100, freq='M')
    data1 = np.random.randn(100).cumsum()  # Нестационарные данные
    data2 = 0.5 * pd.Series(data1).shift(1).fillna(0) + \
        np.random.randn(100) * 0.5
    # Делаем данные стационарными для VAR/Granger
    df_granger_test = pd.DataFrame(
        {'Var1': np.diff(data1), 'Var2': np.diff(data2)}, index=idx[1:])

    print("\nПример DataFrame для теста Грейнджера (дифференцированный):")
    print(df_granger_test.head(5))

    var_lag = 2  # Assume optimal lag is 2 for this example
    try:
        model = VAR(df_granger_test)
        mock_results = model.fit(var_lag)
        print("\nСводка по подобранной VAR модели:")
        print(mock_results.summary())

        # Выполнение теста Грейнджера
        granger_test_results = perform_granger_causality_test(
            mock_results, max_lag=var_lag, significance_level=0.05)

        if granger_test_results:
            print("\nСырые результаты теста причинности Грейнджера:")
            summary_df = summarize_granger_results(granger_test_results)
            print("\nСводная таблица теста причинности Грейнджера:")
            print(summary_df)
        else:
            print("Тестирование причинности Грейнджера завершилось ошибкой.")
 
    except Exception as e:
        print(f"\nОшибка при подгонке VAR или тестировании Грейнджера в примере: {e}")
 
    print("\nТестирование причинности Грейнджера завершено.")
