# src/utils/logger.py
# Конфигурация логирования для проекта.

import logging
import sys
from typing import Optional

# --- Configuration ---
# Это потенциально может быть перемещено или переопределено в src/config.py
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = "analysis.log"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# --- End Configuration ---

# Хранилище настроенных логгеров, чтобы избежать дублирования обработчиков
_configured_loggers = {}


def setup_logger(name: str = 'granger_analysis',
                 log_level: Optional[str] = None,
                 log_file: Optional[str] = None,
                 use_console: bool = True,
                 use_file: bool = True) -> logging.Logger:
    """
    Настраивает и возвращает экземпляр логгера.

    Аргументы:
        name: Имя логгера.
        log_level: Уровень логирования (например, 'DEBUG', 'INFO'). Переопределяет значение по умолчанию/из конфигурации.
        log_file: Путь к файлу логов. Переопределяет значение по умолчанию/из конфигурации.
        use_console: Следует ли выполнять логирование в консоль.
        use_file: Следует ли выполнять логирование в файл.

    Возвращает:
        Настроенный экземпляр логгера.
    """
    global _configured_loggers

    if name in _configured_loggers:
        return _configured_loggers[name]

    # Определяем окончательный уровень логирования и путь к файлу
    final_log_level_str = log_level or DEFAULT_LOG_LEVEL
    final_log_file = log_file or DEFAULT_LOG_FILE

    # Получаем числовой уровень логирования
    numeric_level = getattr(logging, final_log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        print(
            f"Предупреждение: Недопустимый уровень логирования '{final_log_level_str}'. Используется значение по умолчанию INFO.")
        numeric_level = logging.INFO
        final_log_level_str = "INFO"  # Update string representation

    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Создаем форматер
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Удаляем существующие обработчики, чтобы предотвратить дублирование, если функция вызывается снова
    # (Хотя проверка _configured_loggers должна это предотвратить)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Обработчик консоли
    if use_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Обработчик файла
    if use_file and final_log_file:
        try:
            file_handler = logging.FileHandler(
                final_log_file, mode='a')  # Append mode
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Ошибка при настройке файлового обработчика для {final_log_file}: {e}")
            logger.error(f"Could not attach file handler to {final_log_file}")

    # Предотвращаем распространение в корневой логгер, если добавлены обработчики
    logger.propagate = False

    logger.info(
        f"Логгер '{name}' сконфигурирован. Уровень: {final_log_level_str}. Файл: {final_log_file if use_file else 'None'}. Консоль: {use_console}.")

    _configured_loggers[name] = logger
    return logger


if __name__ == '__main__':
    # Пример использования
    print("Тестирование настройки логгера...")

    # Получаем логгер по умолчанию
    logger1 = setup_logger()
    logger1.debug("Это отладочное сообщение (не должно отображаться по умолчанию).")
    logger1.info("Это информационное сообщение.")
    logger1.warning("Это сообщение-предупреждение.")
    logger1.error("Это сообщение об ошибке.")
    logger1.critical("Это критическое сообщение.")

    print(f"\nПроверьте файл логов: {DEFAULT_LOG_FILE}")

    # Получаем другой логгер с другими настройками
    logger2 = setup_logger(
        name='data_loader', log_level='DEBUG', log_file='data_loading.log')
    logger2.debug("Отладочное сообщение от data_loader.")
    logger2.info("Информационное сообщение от data_loader.")

    print(f"Проверьте файл логов: data_loading.log")

    # Снова получаем первый логгер (должен вернуться тот же экземпляр)
    logger1_again = setup_logger()
    print(
        f"\nЯвляется ли logger1 тем же самым, что и logger1_again? {logger1 is logger1_again}")
    logger1_again.info("Еще одно информационное сообщение от логгера по умолчанию.")

    print("\nТестирование логгера завершено.")
