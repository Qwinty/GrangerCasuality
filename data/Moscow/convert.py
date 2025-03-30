import pandas as pd
import io  # Нужен для чтения данных из строки

# Ваши данные в виде строки (как в примере)
# В реальной ситуации вы бы читали из файла: pd.read_csv('your_file.csv')
df = pd.read_csv('data/Moscow/moscow_mortality.csv', encoding='utf-8')

# Создаем словарь для преобразования названий месяцев в номера (с ведущим нулем)
month_map = {
    'January': '01', 'February': '02', 'March': '03', 'April': '04',
    'May': '05', 'June': '06', 'July': '07', 'August': '08',
    'September': '09', 'October': '10', 'November': '11', 'December': '12'
}

# 1. Преобразуем столбец 'Month' в номера месяцев (строки '01', '02', ...)
df['MonthNumber'] = df['Month'].map(month_map)

# 2. Создаем новый столбец 'Date_MM.YYYY', объединяя номер месяца и год
#    Убедимся, что год тоже строка перед объединением
df['Date_MM.YYYY'] = df['MonthNumber'] + '.' + df['Year'].astype(str)

# Выводим результат (например, первые 5 строк с нужными столбцами)
print(df[['Year', 'Month', 'Date_MM.YYYY']].head())

# Можно вывести весь столбец
print("\nНовый столбец 'Date_MM.YYYY':")
print(df['Date_MM.YYYY'])

# Если нужно сохранить результат в новый CSV:
df.to_csv('moscow_mortality_transformed.csv', index=False) # index=False чтобы не записывать индекс pandas в файл