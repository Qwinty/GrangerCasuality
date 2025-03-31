# src/visualization/time_series.py
# Функции для визуализации данных временных рядов и результатов анализа.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.vector_ar.var_model import VARResults
from typing import List, Optional, Union

# Apply plot style from config (optional, can be set globally in main script)
# import matplotlib as mpl
# try:
#     import config
#     mpl.style.use(config.PLOT_STYLE)
# except (ImportError, FileNotFoundError, KeyError):
#     print("Config file not found or PLOT_STYLE not set, using default style.")


def plot_time_series(df: pd.DataFrame, columns: Optional[List[str]] = None, title: str = "График временного ряда", xlabel: str = "Время", ylabel: str = "Значение", save_path: Optional[str] = None):
    """Отображает один или несколько временных рядов из DataFrame."""
    print(f"Отображение временного ряда: {title}")
    if columns is None:
        columns = df.columns.tolist() # Отображать все столбцы, если не указаны

    plt.figure(figsize=(12, 6))
    for col in columns:
        if col in df.columns:
            # Убедитесь, что индекс можно отобразить (преобразовать PeriodIndex в Timestamps)
            plot_index = df.index.to_timestamp() if isinstance(df.index, pd.PeriodIndex) else df.index
            plt.plot(plot_index, df[col], label=col)
        else:
            print(f"Предупреждение: Столбец '{col}' не найден в DataFrame.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        print(f"Сохранение графика в: {save_path}")
        plt.savefig(save_path)
        plt.close() # Закрыть график при сохранении, чтобы избежать его отображения
    else:
        plt.show()


def plot_acf_pacf(series: pd.Series, lags: Optional[int] = None, title_suffix: str = "", save_path_prefix: Optional[str] = None):
    """Отображает Автокорреляционную функцию (ACF) и Частичную автокорреляционную функцию (PACF)."""
    print(f"Отображение ACF/PACF для: {series.name}")
    if lags is None:
        # Лаги по умолчанию: min(10*log10(N), N//2 - 1) для ACF/PACF
        n_obs = len(series.dropna())
        lags = min(int(10 * np.log10(n_obs)), n_obs // 2 - 1) if n_obs > 4 else 0


    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # График ACF
    plot_acf(series.dropna(), lags=lags, ax=axes[0], title=f'ACF - {series.name} {title_suffix}')
    axes[0].grid(True)

    # График PACF
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method='ywm', title=f'PACF - {series.name} {title_suffix}') # 'ywm' часто предпочтительнее
    axes[1].grid(True)

    plt.tight_layout()

    if save_path_prefix:
        acf_path = f"{save_path_prefix}_acf.png"
        pacf_path = f"{save_path_prefix}_pacf.png" # Или сохранить объединенный график
        print(f"Сохранение графиков ACF/PACF с префиксом: {save_path_prefix}")
        # Save the whole figure
        fig.savefig(f"{save_path_prefix}_acf_pacf.png")
        plt.close(fig) # Закрыть график при сохранении
    else:
        plt.show()


def plot_cross_correlation(series1: pd.Series, series2: pd.Series, lags: Optional[int] = None, title: str = "График кросс-корреляции", save_path: Optional[str] = None):
    """Отображает кросс-корреляцию между двумя временными рядами."""
    print(f"Отображение кросс-корреляции между {series1.name} и {series2.name}")
    if lags is None:
        n_obs = min(len(series1.dropna()), len(series2.dropna()))
        lags = min(int(10 * np.log10(n_obs)), n_obs // 2 - 1) if n_obs > 4 else 0

    # Выровнять серии (важно, если индексы не совпадают идеально)
    aligned_s1, aligned_s2 = series1.align(series2, join='inner')
    if aligned_s1.empty:
        print("Ошибка: Серии не имеют перекрывающихся периодов времени для кросс-корреляции.")
        return

    plt.figure(figsize=(10, 5))
    # Use plt.xcorr which calculates and plots
    plt.xcorr(aligned_s1.dropna(), aligned_s2.dropna(), usevlines=True, maxlags=lags, normed=True, lw=2)
    plt.grid(True)
    plt.axhline(0, color='black', lw=1) # Добавить горизонтальную линию на 0
    # Добавить линии значимости (приблизительно)
    n = len(aligned_s1.dropna())
    conf_level = 1.96 / np.sqrt(n) # 95% доверительные интервалы
    plt.axhline(conf_level, color='red', linestyle='--', lw=1, label='95% Confidence Interval')
    plt.axhline(-conf_level, color='red', linestyle='--', lw=1)

    plt.title(f"{title}\n({series1.name} vs {series2.name})")
    plt.xlabel("Lag")
    plt.ylabel("Cross-Correlation")
    plt.legend()
    plt.tight_layout()

    if save_path:
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# Заполнитель для интерактивных графиков с использованием Plotly
def plot_time_series_interactive(df: pd.DataFrame, title: str = "Интерактивный график временного ряда", save_path: Optional[str] = None):
    """Отображает временные ряды в интерактивном режиме, используя Plotly."""
    print("Запрошено интерактивное построение графиков (требуется Plotly).")
    try:
        import plotly.express as px
        # Убедитесь, что индекс можно отобразить (преобразовать PeriodIndex)
        plot_df = df.copy()
        if isinstance(plot_df.index, pd.PeriodIndex):
            plot_df.index = plot_df.index.to_timestamp()
        plot_df = plot_df.reset_index()
        date_col = plot_df.columns[0] # Предполагается, что первый столбец является индексом после сброса

        fig = px.line(plot_df, x=date_col, y=plot_df.columns[1:], title=title,
                      labels={'value': 'Value', date_col: 'Time'})
        fig.update_layout(legend_title_text='Variables')

        if save_path:
            print(f"Сохранение интерактивного графика в: {save_path}")
            fig.write_html(save_path)
        else:
            fig.show()

    except ImportError:
        print("Ошибка: Библиотека Plotly не установлена. Невозможно создать интерактивный график.")
    except Exception as e:
        print(f"Ошибка при создании интерактивного графика: {e}")


if __name__ == '__main__':
    import numpy as np # Убедитесь, что numpy импортирован для примера использования

    # Пример использования (в целях тестирования)
    print("\nТестирование функций визуализации...")

    # Создать образец данных
    idx = pd.period_range(start='2020-01', periods=100, freq='M')
    data1 = np.random.randn(100).cumsum()
    data2 = 0.5 * pd.Series(data1).shift(1).fillna(0) + np.random.randn(100) * 0.5
    df_vis_test = pd.DataFrame({'SeriesA': data1, 'SeriesB': data2}, index=idx)

    print("\nПример DataFrame для визуализации:")
    print(df_vis_test.head())

    # Тест графика временного ряда
    plot_time_series(df_vis_test, title="Sample Time Series")
    # plot_time_series(df_vis_test, title="Sample Time Series Saved", save_path="sample_time_series.png") #Сохраненный график временного ряда

    # Тест графика ACF/PACF (на потенциально стационарном ряду - дифференцированный A)
    series_a_diff = df_vis_test['SeriesA'].diff().dropna()
    series_a_diff.name = "Differenced Series A" # Дайте ему имя
    plot_acf_pacf(series_a_diff, lags=20)
    # plot_acf_pacf(series_a_diff, lags=20, save_path_prefix="sample_diff_a") #Сохраненный график ACF/PACF

    # Тест графика кросс-корреляции
    plot_cross_correlation(df_vis_test['SeriesA'], df_vis_test['SeriesB'], lags=20)
    # plot_cross_correlation(df_vis_test['SeriesA'], df_vis_test['SeriesB'], lags=20, save_path="sample_xcorr.png") #Сохраненный график кросс-корреляции

    # Тест интерактивного графика (если установлен plotly)
    plot_time_series_interactive(df_vis_test, title="Interactive Sample Plot")
    # plot_time_series_interactive(df_vis_test, title="Interactive Sample Plot Saved", save_path="interactive_sample.html") #Сохраненный интерактивный график


    print("\nТест визуализации завершен.")