# src/config.py
# Настройки конфигурации для проекта анализа причинности Грейнджера.

# Пути к данным
TEMP_DATA_PATH = "../data/Moscow_Temp (2010-2024).csv"
# Выберите один из следующих вариантов для второго набора данных:
# SECONDARY_DATA_PATH = "data/Moscow/moscow_dtp_transformed.csv"
SECONDARY_DATA_PATH = "../data/Moscow/moscow_mortality.csv"
SECONDARY_DATA_NAME = "Mortality" # Или "DTP"

# Настройки предварительной обработки
DATE_FORMAT = "%Y-%m"
NORMALIZATION_METHOD_TEMP = "z-score" # Или None
NORMALIZATION_METHOD_SECONDARY = "log" # Или None, "z-score"
AGGREGATION_TEMP = "mean"
AGGREGATION_SECONDARY = "sum"

# Настройки стационарности
ADF_SIGNIFICANCE_LEVEL = 0.05
KPSS_SIGNIFICANCE_LEVEL = 0.05
MAX_DIFFERENCING_ORDER = 2

# Настройки VAR модели
MAX_LAG_ORDER = 30
LAG_SELECTION_CRITERIA = ["aic", "bic"] # Варианты: 'aic', 'bic', 'hqic', 'fpe'

# Настройки причинности Грейнджера
GRANGER_SIGNIFICANCE_LEVEL = 0.05

# Настройки визуализации
PLOT_STYLE = "seaborn-v0_8-darkgrid"
INTERACTIVE_PLOTS = True # Установите значение True для Plotly

# Настройки валидации
BOOTSTRAP_ITERATIONS = 1000
CROSS_VALIDATION_WINDOW = 24 # months
CROSS_VALIDATION_STEPS = 3 # шаги прогноза

# Настройки логгера
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "analysis.log"