# src/config.py
# Configuration settings for the Granger causality analysis project.

# Data paths
TEMP_DATA_PATH = "../data/Moscow_Temp (2010-2024).csv"
# Choose one of the following for the second dataset:
# SECONDARY_DATA_PATH = "data/Moscow/moscow_dtp_transformed.csv"
SECONDARY_DATA_PATH = "../data/Moscow/moscow_mortality.csv"
SECONDARY_DATA_NAME = "Mortality" # Or "DTP"

# Preprocessing settings
DATE_FORMAT = "%Y-%m"
NORMALIZATION_METHOD_TEMP = "z-score" # Or None
NORMALIZATION_METHOD_SECONDARY = "log" # Or None, "z-score"
AGGREGATION_TEMP = "mean"
AGGREGATION_SECONDARY = "sum"

# Stationarity settings
ADF_SIGNIFICANCE_LEVEL = 0.05
KPSS_SIGNIFICANCE_LEVEL = 0.05
MAX_DIFFERENCING_ORDER = 2

# VAR model settings
MAX_LAG_ORDER = 30
LAG_SELECTION_CRITERIA = ["aic", "bic"] # Options: 'aic', 'bic', 'hqic', 'fpe'

# Granger causality settings
GRANGER_SIGNIFICANCE_LEVEL = 0.05

# Visualization settings
PLOT_STYLE = "seaborn-v0_8-darkgrid"
INTERACTIVE_PLOTS = True # Set to True for Plotly

# Validation settings
BOOTSTRAP_ITERATIONS = 1000
CROSS_VALIDATION_WINDOW = 24 # months
CROSS_VALIDATION_STEPS = 3 # forecast steps

# Logging settings
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "analysis.log"