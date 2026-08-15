import sqlite3
import pandas as pd
from pathlib import Path


class DatabaseManager:
    def __init__(self, db_path: str = "data/tsetmc.db"):
        self.db_path = db_path
        # اطمینان از وجود پوشه data
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """ساخت جدول‌های اولیه دیتابیس"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    close_price REAL,
                    last_price REAL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    volume INTEGER,
                    value REAL,
                    count INTEGER,
                    UNIQUE(symbol, date)
                )
            """)
            conn.commit()

    def save_history(self, symbol: str, df: pd.DataFrame):
        """ذخیره یا بروزرسانی دیتافریم قیمت تاریخی در دیتابیس"""
        if df.empty:
            return

        df_to_save = df.copy()
        df_to_save["symbol"] = symbol

        with self._get_connection() as conn:
            # ذخیره دیتا در دیتابیس
            df_to_save.to_sql(
                "price_history", conn, if_exists="append", index=False, method="ignore"
            )

    def get_history(self, symbol: str) -> pd.DataFrame:
        """خواندن تاریخچه قیمت یک نماد از دیتابیس"""
        with self._get_connection() as conn:
            query = "SELECT * FROM price_history WHERE symbol = ? ORDER BY date ASC"
            return pd.read_sql_query(query, conn, params=(symbol,))
