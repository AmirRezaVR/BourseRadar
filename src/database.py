import sqlite3
import pandas as pd
from pathlib import Path


class DatabaseManager:
    def __init__(self, db_path: str = "data/tsetmc.db"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create the initial database tables"""
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
        """Insert historical price rows into the database, skipping duplicates"""
        if df.empty:
            return

        df_to_save = df.copy()
        df_to_save["symbol"] = symbol

        cols = list(df_to_save.columns)
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                f"INSERT OR IGNORE INTO price_history ({col_names}) VALUES ({placeholders})",
                df_to_save[cols].values.tolist(),
            )
            conn.commit()

    def get_history(self, symbol: str) -> pd.DataFrame:
        """Read the price history for a symbol from the database"""
        with self._get_connection() as conn:
            query = "SELECT * FROM price_history WHERE symbol = ? ORDER BY date ASC"
            return pd.read_sql_query(query, conn, params=(symbol,))
