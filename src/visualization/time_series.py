# src/visualization/time_series.py
# Functions for visualizing time series data and analysis results.

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


def plot_time_series(df: pd.DataFrame, columns: Optional[List[str]] = None, title: str = "Time Series Plot", xlabel: str = "Time", ylabel: str = "Value", save_path: Optional[str] = None):
    """Plots one or more time series from a DataFrame."""
    print(f"Plotting time series: {title}")
    if columns is None:
        columns = df.columns.tolist() # Plot all columns if None specified

    plt.figure(figsize=(12, 6))
    for col in columns:
        if col in df.columns:
            # Ensure index is plottable (convert PeriodIndex to Timestamps)
            plot_index = df.index.to_timestamp() if isinstance(df.index, pd.PeriodIndex) else df.index
            plt.plot(plot_index, df[col], label=col)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path)
        plt.close() # Close plot if saving to avoid displaying it
    else:
        plt.show()


def plot_acf_pacf(series: pd.Series, lags: Optional[int] = None, title_suffix: str = "", save_path_prefix: Optional[str] = None):
    """Plots the AutoCorrelation Function (ACF) and Partial AutoCorrelation Function (PACF)."""
    print(f"Plotting ACF/PACF for: {series.name}")
    if lags is None:
        # Default lags: min(10*log10(N), N//2 - 1) for ACF/PACF
        n_obs = len(series.dropna())
        lags = min(int(10 * np.log10(n_obs)), n_obs // 2 - 1) if n_obs > 4 else 0


    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ACF Plot
    plot_acf(series.dropna(), lags=lags, ax=axes[0], title=f'ACF - {series.name} {title_suffix}')
    axes[0].grid(True)

    # PACF Plot
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method='ywm', title=f'PACF - {series.name} {title_suffix}') # 'ywm' is often preferred
    axes[1].grid(True)

    plt.tight_layout()

    if save_path_prefix:
        acf_path = f"{save_path_prefix}_acf.png"
        pacf_path = f"{save_path_prefix}_pacf.png" # Or save combined plot
        print(f"Saving ACF/PACF plots with prefix: {save_path_prefix}")
        # Save the whole figure
        fig.savefig(f"{save_path_prefix}_acf_pacf.png")
        plt.close(fig) # Close plot if saving
    else:
        plt.show()


def plot_cross_correlation(series1: pd.Series, series2: pd.Series, lags: Optional[int] = None, title: str = "Cross-Correlation Plot", save_path: Optional[str] = None):
    """Plots the cross-correlation between two time series."""
    print(f"Plotting cross-correlation between {series1.name} and {series2.name}")
    if lags is None:
        n_obs = min(len(series1.dropna()), len(series2.dropna()))
        lags = min(int(10 * np.log10(n_obs)), n_obs // 2 - 1) if n_obs > 4 else 0

    # Align series (important if indices don't perfectly match)
    aligned_s1, aligned_s2 = series1.align(series2, join='inner')
    if aligned_s1.empty:
        print("Error: Series have no overlapping time periods for cross-correlation.")
        return

    plt.figure(figsize=(10, 5))
    # Use plt.xcorr which calculates and plots
    plt.xcorr(aligned_s1.dropna(), aligned_s2.dropna(), usevlines=True, maxlags=lags, normed=True, lw=2)
    plt.grid(True)
    plt.axhline(0, color='black', lw=1) # Add horizontal line at 0
    # Add significance lines (approximate)
    n = len(aligned_s1.dropna())
    conf_level = 1.96 / np.sqrt(n) # 95% confidence intervals
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


# Placeholder for interactive plots using Plotly
def plot_time_series_interactive(df: pd.DataFrame, title: str = "Interactive Time Series Plot", save_path: Optional[str] = None):
    """Plots time series interactively using Plotly."""
    print("Interactive plotting requested (requires Plotly).")
    try:
        import plotly.express as px
        # Ensure index is plottable (convert PeriodIndex)
        plot_df = df.copy()
        if isinstance(plot_df.index, pd.PeriodIndex):
            plot_df.index = plot_df.index.to_timestamp()
        plot_df = plot_df.reset_index()
        date_col = plot_df.columns[0] # Assumes first column is the index after reset

        fig = px.line(plot_df, x=date_col, y=plot_df.columns[1:], title=title,
                      labels={'value': 'Value', date_col: 'Time'})
        fig.update_layout(legend_title_text='Variables')

        if save_path:
            print(f"Saving interactive plot to: {save_path}")
            fig.write_html(save_path)
        else:
            fig.show()

    except ImportError:
        print("Error: Plotly library not installed. Cannot create interactive plot.")
    except Exception as e:
        print(f"Error creating interactive plot: {e}")


if __name__ == '__main__':
    import numpy as np # Ensure numpy is imported for example usage

    # Example usage (for testing purposes)
    print("\nTesting visualization functions...")

    # Create sample data
    idx = pd.period_range(start='2020-01', periods=100, freq='M')
    data1 = np.random.randn(100).cumsum()
    data2 = 0.5 * pd.Series(data1).shift(1).fillna(0) + np.random.randn(100) * 0.5
    df_vis_test = pd.DataFrame({'SeriesA': data1, 'SeriesB': data2}, index=idx)

    print("\nSample DataFrame for Visualization:")
    print(df_vis_test.head())

    # Test time series plot
    plot_time_series(df_vis_test, title="Sample Time Series")
    # plot_time_series(df_vis_test, title="Sample Time Series Saved", save_path="sample_time_series.png")

    # Test ACF/PACF plot (on a potentially stationary series - differenced A)
    series_a_diff = df_vis_test['SeriesA'].diff().dropna()
    series_a_diff.name = "Differenced Series A" # Give it a name
    plot_acf_pacf(series_a_diff, lags=20)
    # plot_acf_pacf(series_a_diff, lags=20, save_path_prefix="sample_diff_a")

    # Test cross-correlation plot
    plot_cross_correlation(df_vis_test['SeriesA'], df_vis_test['SeriesB'], lags=20)
    # plot_cross_correlation(df_vis_test['SeriesA'], df_vis_test['SeriesB'], lags=20, save_path="sample_xcorr.png")

    # Test interactive plot (if plotly is installed)
    plot_time_series_interactive(df_vis_test, title="Interactive Sample Plot")
    # plot_time_series_interactive(df_vis_test, title="Interactive Sample Plot Saved", save_path="interactive_sample.html")


    print("\nVisualization test finished.")