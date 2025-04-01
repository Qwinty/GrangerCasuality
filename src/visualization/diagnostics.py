# src/visualization/diagnostics.py
# Функции для построения графиков диагностики VAR-моделей, таких как IRF и FEVD.

import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VARResults
from typing import Optional, List

# Применить стиль графика из конфига (необязательно)
# import matplotlib as mpl
# try:
#     import config
#     mpl.style.use(config.PLOT_STYLE)
# except (ImportError, FileNotFoundError, KeyError):
#     print("Config file not found or PLOT_STYLE not set, using default style.")


def plot_impulse_response(results: VARResults, impulse: Optional[str] = None, response: Optional[str] = None, periods: int = 10, figsize: tuple = (12, 8), save_path: Optional[str] = None):
    """
    Строит графики функций импульсной характеристики (IRF) для подогнанной VAR-модели.

    Args:
        results: Объект Fitted VARResults.
        impulse: Имя переменной, задающей импульс. Если None, строятся графики всех импульсов.
        response: Имя переменной, получающей отклик. Если None, строятся графики всех откликов.
        periods: Количество периодов для построения графика отклика.
        figsize: Размер фигуры.
        save_path: Необязательный путь для сохранения графика.
    """
    print(
        f"Построение графиков функций импульсной характеристики (периоды: {periods})...")
    try:
        # Метод plot обрабатывает выбор конкретного импульса/отклика или построение графиков для всех
        irf = results.irf(periods=periods)  # Calculate IRFs

        # Создать график. Метод `plot` объекта irf является гибким.
        # Он может строить графики ортогонализованных IRF по умолчанию.
        plot_kwargs = {'figsize': figsize}
        if impulse and response:
            plot_kwargs['impulse'] = impulse
            plot_kwargs['response'] = response
            print(f"  Specific IRF: Impulse={impulse}, Response={response}")
        elif impulse:
            plot_kwargs['impulse'] = impulse
            print(f"  Impulses from: {impulse}")
        elif response:
            plot_kwargs['response'] = response
            print(f"  Responses of: {response}")
        else:
            print("  Построение графиков всех IRF.")

        # Функция plot возвращает объект matplotlib Figure
        fig = irf.plot(**plot_kwargs)
        fig.suptitle('Функции импульсной характеристики', fontsize=16)
        # Настройка макета для предотвращения перекрытия заголовка
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            print(f"Сохранение графика IRF в: {save_path}")
            fig.savefig(save_path)
            plt.close(fig)  # Закрыть график, если сохраняется
        else:
            plt.show()

    except Exception as e:
        print(
            f"Ошибка при построении графиков функций импульсной характеристики: {e}")


def plot_fevd(results: VARResults, periods: Optional[int] = None, figsize: tuple = (12, 8), save_path: Optional[str] = None):
    """
    Строит графики разложения дисперсии ошибки прогноза (FEVD).

    Args:
        results: Объект Fitted VARResults.
        periods: Количество шагов вперед для FEVD. Если None, используется порядок запаздывания модели.
        figsize: Размер фигуры.
        save_path: Необязательный путь для сохранения графика.
    """
    print("Построение графика разложения дисперсии ошибки прогноза...")
    try:
        fevd = results.fevd(periods=periods) # Вычислить FEVD

        # Метод plot создает сводный график FEVD
        # Он возвращает объект matplotlib Figure
        fig = fevd.plot(figsize=figsize)
        fig.suptitle('Разложение дисперсии ошибки прогноза', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Настройка макета

        if save_path:
            print(f"Сохранение графика FEVD в: {save_path}")
            fig.savefig(save_path)
            plt.close(fig) # Закрыть график, если сохраняется
        else:
            plt.show()

    except Exception as e:
        print(f"Ошибка при построении графика FEVD: {e}")


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.api import VAR

    # Пример использования (требуется подогнанная VAR-модель)
    print("\nТестирование функций визуализации диагностики...")

    # --- Reusing VAR fitting example ---
    idx = pd.period_range(start='2020-01', periods=100, freq='M')
    data1 = np.random.randn(100).cumsum()  # Нестационарный
    data2 = 0.5 * pd.Series(data1).shift(1).fillna(0) + \
        np.random.randn(100) * 0.5
    df_diag_test = pd.DataFrame({'Var1': np.diff(data1), 'Var2': np.diff(
        data2)}, index=idx[1:])  # Использовать дифференцированные данные

    print("\nПример DataFrame для диагностики:")
    print(df_diag_test.head())

    var_lag = 2  # Предположим, что оптимальный лаг равен 2
    try:
        model = VAR(df_diag_test)
        var_results_diag = model.fit(var_lag)
        print("\nПодогнанная макетная VAR-модель для диагностики:")
        # print(var_results_diag.summary()) # Необязательно: вывести сводку

        # Тест графика IRF (все)
        plot_impulse_response(var_results_diag, periods=15)
        # plot_impulse_response(var_results_diag, periods=15, save_path="sample_irf_all.png")

        # Тест графика IRF (конкретный)
        # plot_impulse_response(var_results_diag, impulse='Var1', response='Var2', periods=15)
        # plot_impulse_response(var_results_diag, impulse='Var1', response='Var2', periods=15, save_path="sample_irf_v1_to_v2.png")

        # Тест графика FEVD
        plot_fevd(var_results_diag, periods=15)
        # plot_fevd(var_results_diag, periods=15, save_path="sample_fevd.png")

    except Exception as e:
        print(
            f"\nОшибка во время подгонки VAR или построения диагностического графика в примере: {e}")

    print("\nТест визуализации диагностики завершен.")
