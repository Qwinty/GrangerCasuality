# src/visualization/diagnostics.py
# Functions for plotting VAR model diagnostics like IRF and FEVD.

import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VARResults
from typing import Optional, List

# Apply plot style from config (optional)
# import matplotlib as mpl
# try:
#     import config
#     mpl.style.use(config.PLOT_STYLE)
# except (ImportError, FileNotFoundError, KeyError):
#     print("Config file not found or PLOT_STYLE not set, using default style.")


def plot_impulse_response(results: VARResults, impulse: Optional[str] = None, response: Optional[str] = None, periods: int = 10, figsize: tuple = (12, 8), save_path: Optional[str] = None):
    """
    Plots the Impulse Response Functions (IRF) for a fitted VAR model.

    Args:
        results: Fitted VARResults object.
        impulse: Name of the variable giving the impulse. If None, plots all impulses.
        response: Name of the variable receiving the response. If None, plots all responses.
        periods: Number of periods to plot the response for.
        figsize: Size of the figure.
        save_path: Optional path to save the plot.
    """
    print(f"Plotting Impulse Response Functions (Periods: {periods})...")
    try:
        # The plot method handles selecting specific impulse/response or plotting all
        irf = results.irf(periods=periods) # Calculate IRFs

        # Create the plot. The `plot` method of irf object is flexible.
        # It can plot orthogonalized IRFs by default.
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
             print("  Plotting all IRFs.")

        # The plot function returns a matplotlib Figure object
        fig = irf.plot(**plot_kwargs)
        fig.suptitle('Impulse Response Functions', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

        if save_path:
            print(f"Saving IRF plot to: {save_path}")
            fig.savefig(save_path)
            plt.close(fig) # Close plot if saving
        else:
            plt.show()

    except Exception as e:
        print(f"Error plotting Impulse Response Functions: {e}")


def plot_fevd(results: VARResults, periods: Optional[int] = None, figsize: tuple = (12, 8), save_path: Optional[str] = None):
    """
    Plots the Forecast Error Variance Decomposition (FEVD).

    Args:
        results: Fitted VARResults object.
        periods: Number of steps ahead for FEVD. If None, uses model lag order.
        figsize: Size of the figure.
        save_path: Optional path to save the plot.
    """
    print("Plotting Forecast Error Variance Decomposition...")
    try:
        fevd = results.fevd(periods=periods) # Calculate FEVD

        # The plot method generates the FEVD summary plot
        # It returns a matplotlib Figure object
        fig = fevd.plot(figsize=figsize)
        fig.suptitle('Forecast Error Variance Decomposition', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        if save_path:
            print(f"Saving FEVD plot to: {save_path}")
            fig.savefig(save_path)
            plt.close(fig) # Close plot if saving
        else:
            plt.show()

    except Exception as e:
        print(f"Error plotting FEVD: {e}")


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.api import VAR

    # Example usage (requires a fitted VAR model)
    print("\nTesting diagnostic visualization functions...")

    # --- Reusing VAR fitting example ---
    idx = pd.period_range(start='2020-01', periods=100, freq='M')
    data1 = np.random.randn(100).cumsum() # Non-stationary
    data2 = 0.5 * pd.Series(data1).shift(1).fillna(0) + np.random.randn(100) * 0.5
    df_diag_test = pd.DataFrame({'Var1': np.diff(data1), 'Var2': np.diff(data2)}, index=idx[1:]) # Use differenced data

    print("\nSample DataFrame for Diagnostics:")
    print(df_diag_test.head())

    var_lag = 2 # Assume optimal lag is 2
    try:
        model = VAR(df_diag_test)
        var_results_diag = model.fit(var_lag)
        print("\nFitted Mock VAR Model for Diagnostics:")
        # print(var_results_diag.summary()) # Optional: print summary

        # Test IRF plot (all)
        plot_impulse_response(var_results_diag, periods=15)
        # plot_impulse_response(var_results_diag, periods=15, save_path="sample_irf_all.png")

        # Test IRF plot (specific)
        # plot_impulse_response(var_results_diag, impulse='Var1', response='Var2', periods=15)
        # plot_impulse_response(var_results_diag, impulse='Var1', response='Var2', periods=15, save_path="sample_irf_v1_to_v2.png")


        # Test FEVD plot
        plot_fevd(var_results_diag, periods=15)
        # plot_fevd(var_results_diag, periods=15, save_path="sample_fevd.png")


    except Exception as e:
        print(f"\nError during VAR fitting or diagnostic plotting in example: {e}")

    print("\nDiagnostic visualization test finished.")