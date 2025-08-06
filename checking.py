import pandas as pd
import numpy as np
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

def download_and_analyze_hft_data():
    """
    Скачиваем файл
    """
    
    print("Загрузка данных...")
    
    # Скачиваем файл
    url = "https://github.com/timthecreatorr/project-eqi/raw/main/test.feather"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Читаем файл
        df = pd.read_feather(BytesIO(response.content))
        
        print(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
        print(f"Колонки: {list(df.columns)}")
        print(f"Размер данных: {df.shape}")
        print()
        
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return
    
    # Анализ ошибок
    errors_found = []
    
    print("АНАЛИЗ ОШИБОК В HFT ДАННЫХ \n")
    
    # 1. Проверка временных меток
    if 'timestamp' in df.columns or any('time' in col.lower() for col in df.columns):
        time_col = 'timestamp' if 'timestamp' in df.columns else [col for col in df.columns if 'time' in col.lower()][0]
        
        # Немонотонные временные метки
        if not df[time_col].is_monotonic_increasing:
            errors_found.append("Немонотонные временные метки - нарушение последовательности времени")
        
        # Дублирующиеся временные метки
        if df[time_col].duplicated().any():
            errors_found.append("Дублирующиеся временные метки")
        
        # Пропуски во времени (большие разрывы)
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            time_diffs = df[time_col].diff().dropna()
            if (time_diffs > pd.Timedelta(seconds=60)).any():
                errors_found.append("Большие временные разрывы в данных (>60 сек)")
    
    # 2. Проверка цен
    price_cols = [col for col in df.columns if any(word in col.lower() for word in ['px_', 'trade_px'])]
    
    for col in price_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # Нулевые или отрицательные цены
            if (df[col] <= 0).any():
                errors_found.append(f"Нулевые или отрицательные цены в колонке {col}")
            
            # Экстремальные значения цен
            if df[col].std() > 0 and df[col].count() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                if (z_scores > 10).any():
                    errors_found.append(f"Экстремальные ценовые выбросы в колонке {col}")
    
    # 3. Проверка bid/ask спреда
    buy_prices = [col for col in df.columns if col.startswith('px_buy_')]
    sell_prices = [col for col in df.columns if col.startswith('px_sell_')]
    
    if buy_prices and sell_prices:
        # Проверим лучшие цены (уровень 1)
        best_bid = 'px_buy_1'
        best_ask = 'px_sell_1'
        
        if best_bid in df.columns and best_ask in df.columns:
            # Создаем маску для валидных данных
            valid_mask = pd.notna(df[best_bid]) & pd.notna(df[best_ask]) & \
                        pd.to_numeric(df[best_bid], errors='coerce').notna() & \
                        pd.to_numeric(df[best_ask], errors='coerce').notna()
            
            if valid_mask.any():
                bid_numeric = pd.to_numeric(df.loc[valid_mask, best_bid], errors='coerce')
                ask_numeric = pd.to_numeric(df.loc[valid_mask, best_ask], errors='coerce')
                
                # Crossed book (bid > ask)
                if (bid_numeric > ask_numeric).any():
                    errors_found.append("Пересеченная книга ордеров (лучший bid > лучший ask)")
                
                # Отрицательный спред
                spread = ask_numeric - bid_numeric
                if (spread < 0).any():
                    errors_found.append("Отрицательный спред")
                
                # Нулевой спред
                if (spread == 0).any():
                    errors_found.append("Нулевой спред")
    
    # 4. Проверка объемов/количеств
    amount_cols = [col for col in df.columns if col.startswith('amt_') or col == 'trade_amt']
    
    for col in amount_cols:
        if col in df.columns:
            # Преобразуем в числовой тип
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            valid_data = numeric_col.dropna()
            
            if len(valid_data) > 0:
                # Нулевые или отрицательные объемы
                if (valid_data <= 0).any():
                    errors_found.append(f"Нулевые или отрицательные объемы в колонке {col}")
                
                # Экстремально большие объемы
                if valid_data.quantile(0.99) > 0:
                    extreme_threshold = valid_data.quantile(0.99) * 100
                    if (valid_data > extreme_threshold).any():
                        errors_found.append(f"Экстремально большие объемы в колонке {col}")
    
    # 5. Проверка на пропущенные значения
    missing_data = df.isnull().sum()
    if missing_data.any():
        for col, count in missing_data[missing_data > 0].items():
            errors_found.append(f"Пропущенные значения в колонке {col} ({count} записей)")
    
    # 6. Проверка дублирующихся строк
    if df.duplicated().any():
        errors_found.append("Полностью дублирующиеся строки в данных")
    
    # 7. Проверка последовательности номеров сообщений
    if 'msgSeqNum' in df.columns:
        seq_col = df['msgSeqNum']
        if pd.api.types.is_numeric_dtype(seq_col):
            # Проверка монотонности
            if not seq_col.is_monotonic_increasing:
                errors_found.append("Нарушение последовательности номеров сообщений")
            
            # Проверка пропусков в последовательности
            seq_diffs = seq_col.diff().dropna()
            if (seq_diffs > 1).any():
                errors_found.append("Пропуски в последовательности номеров сообщений")
            
            # Проверка дублирующихся номеров
            if seq_col.duplicated().any():
                errors_found.append("Дублирующиеся номера сообщений")
    
    # 8. Проверка типов данных
    for col in df.columns:
        if df[col].dtype == 'object':
            # Проверка на смешанные типы в строковых колонках
            try:
                pd.to_numeric(df[col], errors='raise')
                errors_found.append(f"Колонка {col} содержит числовые данные в строковом формате")
            except:
                pass
    
    # 9. Проверка сделок относительно спреда
    if 'trade_px' in df.columns and 'px_buy_1' in df.columns and 'px_sell_1' in df.columns:
        # Только для строк где есть сделки
        trade_mask = pd.notna(df['trade_px']) & (df['trade_px'] != 0)
        
        if trade_mask.any():
            trade_data = df[trade_mask].copy()
            
            # Преобразуем в числовые типы
            trade_px = pd.to_numeric(trade_data['trade_px'], errors='coerce')
            best_bid = pd.to_numeric(trade_data['px_buy_1'], errors='coerce')
            best_ask = pd.to_numeric(trade_data['px_sell_1'], errors='coerce')
            
            # Маска для валидных данных
            valid_trade_mask = pd.notna(trade_px) & pd.notna(best_bid) & pd.notna(best_ask)
            
            if valid_trade_mask.any():
                # Сделки вне спреда
                outside_spread = (trade_px[valid_trade_mask] < best_bid[valid_trade_mask]) | \
                               (trade_px[valid_trade_mask] > best_ask[valid_trade_mask])
                if outside_spread.any():
                    errors_found.append("Сделки по ценам вне текущего спреда")
    
    # 10. Проверка на константные значения
    for col in df.columns:
        if df[col].nunique() == 1:
            errors_found.append(f"Константные значения в колонке {col}")
    
    # 11. Проверка корреляций (подозрительно высокие)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        # Исключаем диагональ и ищем подозрительно высокие корреляции
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr = corr_matrix.where(mask) > 0.99
        if high_corr.any().any():
            errors_found.append("Подозрительно высокие корреляции между колонками")
    
    # 12. Проверка временных меток
    time_cols = ['exchHostTime', 'adapterTime']
    for time_col in time_cols:
        if time_col in df.columns:
            try:
                # Попытка преобразования в datetime
                time_series = pd.to_datetime(df[time_col], errors='coerce')
                valid_times = time_series.dropna()
                
                if len(valid_times) > 0:
                    # Проверка монотонности
                    if not valid_times.is_monotonic_increasing:
                        errors_found.append(f"Немонотонные временные метки в {time_col}")
                    
                    # Проверка больших временных разрывов
                    time_diffs = valid_times.diff().dropna()
                    if (time_diffs > pd.Timedelta(minutes=5)).any():
                        errors_found.append(f"Большие временные разрывы в {time_col}")
                        
            except Exception:
                errors_found.append(f"Неверный формат временных меток в {time_col}")
    
    # 13. Проверка логики книги ордеров
    # Проверка упорядоченности цен покупки (должны убывать)
    buy_price_cols = [f'px_buy_{i}' for i in range(1, 11) if f'px_buy_{i}' in df.columns]
    if len(buy_price_cols) > 1:
        for i in range(len(buy_price_cols) - 1):
            col1, col2 = buy_price_cols[i], buy_price_cols[i + 1]
            
            # Маска для строк с валидными данными
            mask = pd.notna(df[col1]) & pd.notna(df[col2])
            if mask.any():
                price1 = pd.to_numeric(df.loc[mask, col1], errors='coerce')
                price2 = pd.to_numeric(df.loc[mask, col2], errors='coerce')
                valid_mask = pd.notna(price1) & pd.notna(price2)
                
                if valid_mask.any():
                    # px_buy_1 должен быть >= px_buy_2
                    if (price1[valid_mask] < price2[valid_mask]).any():
                        errors_found.append(f"Нарушение порядка цен покупки: {col1} < {col2}")
    
    # Проверка упорядоченности цен продажи (должны возрастать)
    sell_price_cols = [f'px_sell_{i}' for i in range(1, 11) if f'px_sell_{i}' in df.columns]
    if len(sell_price_cols) > 1:
        for i in range(len(sell_price_cols) - 1):
            col1, col2 = sell_price_cols[i], sell_price_cols[i + 1]
            
            # Маска для строк с валидными данными
            mask = pd.notna(df[col1]) & pd.notna(df[col2])
            if mask.any():
                price1 = pd.to_numeric(df.loc[mask, col1], errors='coerce')
                price2 = pd.to_numeric(df.loc[mask, col2], errors='coerce')
                valid_mask = pd.notna(price1) & pd.notna(price2)
                
                if valid_mask.any():
                    # px_sell_1 должен быть <= px_sell_2
                    if (price1[valid_mask] > price2[valid_mask]).any():
                        errors_found.append(f"Нарушение порядка цен продажи: {col1} > {col2}")
    
    # 14. Проверка соответствия объемов и цен
    for i in range(1, 11):
        px_col = f'px_buy_{i}'
        amt_col = f'amt_buy_{i}'
        
        if px_col in df.columns and amt_col in df.columns:
            # Если есть цена, должен быть объем и наоборот
            px_present = pd.notna(df[px_col]) & (df[px_col] != 0)
            amt_present = pd.notna(df[amt_col]) & (df[amt_col] != 0)
            
            # Несоответствие: цена есть, объема нет или наоборот
            mismatch = px_present != amt_present
            if mismatch.any():
                errors_found.append(f"Несоответствие цены и объема в уровне {i} (покупка)")
    
    for i in range(1, 11):
        px_col = f'px_sell_{i}'
        amt_col = f'amt_sell_{i}'
        
        if px_col in df.columns and amt_col in df.columns:
            # Если есть цена, должен быть объем и наоборот
            px_present = pd.notna(df[px_col]) & (df[px_col] != 0)
            amt_present = pd.notna(df[amt_col]) & (df[amt_col] != 0)
            
            # Несоответствие: цена есть, объема нет или наоборот
            mismatch = px_present != amt_present
            if mismatch.any():
                errors_found.append(f"Несоответствие цены и объема в уровне {i} (продажа)")
    
    # 15. Проверка типов сообщений
    if 'type' in df.columns:
        # Проверка на недопустимые типы сообщений
        valid_types = ['L2', 'Trade', 'BookSnapshot', 'TradeSnapshot']  # примерные валидные типы
        invalid_types = ~df['type'].isin(valid_types)
        if invalid_types.any():
            unique_invalid = df.loc[invalid_types, 'type'].unique()
            errors_found.append(f"Неизвестные типы сообщений: {list(unique_invalid)}")
    
    # 16. Проверка на аномально большие спреды
    if 'px_buy_1' in df.columns and 'px_sell_1' in df.columns:
        mask = pd.notna(df['px_buy_1']) & pd.notna(df['px_sell_1'])
        if mask.any():
            bid = pd.to_numeric(df.loc[mask, 'px_buy_1'], errors='coerce')
            ask = pd.to_numeric(df.loc[mask, 'px_sell_1'], errors='coerce')
            valid_mask = pd.notna(bid) & pd.notna(ask)
            
            if valid_mask.any():
                spread = ask[valid_mask] - bid[valid_mask]
                relative_spread = spread / bid[valid_mask]
                
                # Спред больше 10% от цены
                if (relative_spread > 0.1).any():
                    errors_found.append("Аномально большие спреды (>10% от цены)")
    
    # 17. Проверка корректности флага moreTradesInBatch
    if 'moreTradesInBatch' in df.columns:
        # Проверим логику: если True, то следующее сообщение должно быть тоже Trade
        trades_mask = df['type'] == 'Trade'
        if trades_mask.any():
            more_trades = df.loc[trades_mask, 'moreTradesInBatch']
            # Здесь можно добавить более сложную логику проверки
    
    # Вывод результатов
    print(f"НАЙДЕНО {len(errors_found)} ТИПОВ ОШИБОК:\n")
    
    for i, error in enumerate(errors_found, 1):
        print(f"{i}. {error}")
    
    print(f"\nВсего найдено ошибок: {len(errors_found)}")
    print("\nАнализ завершен!")
    
    return errors_found

# Запуск анализа
if __name__ == "__main__":
    errors = download_and_analyze_hft_data()