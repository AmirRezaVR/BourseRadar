import pandas as pd
import numpy as np


class TechnicalIndicators:
    @staticmethod
    def add_sma(
        df: pd.DataFrame, column: str = "close_price", period: int = 20
    ) -> pd.DataFrame:
        df[f"SMA_{period}"] = df[column].rolling(window=period).mean()
        return df

    @staticmethod
    def add_ema(
        df: pd.DataFrame, column: str = "close_price", period: int = 20
    ) -> pd.DataFrame:
        df[f"EMA_{period}"] = df[column].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(
        df: pd.DataFrame, column: str = "close_price", period: int = 14
    ) -> pd.DataFrame:
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame, column: str = "close_price", fast=12, slow=26, signal=9
    ) -> pd.DataFrame:
        exp1 = df[column].ewm(span=fast, adjust=False).mean()
        exp2 = df[column].ewm(span=slow, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range - used for volatility-based stop-loss sizing"""
        prev_close = df["close_price"].shift(1)
        tr1 = df["high_price"] - df["low_price"]
        tr2 = (df["high_price"] - prev_close).abs()
        tr3 = (df["low_price"] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f"ATR_{period}"] = true_range.rolling(window=period).mean()
        return df

    @staticmethod
    def add_recent_levels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Rolling recent support/resistance based on N-day high/low"""
        df[f"RECENT_LOW_{window}"] = df["low_price"].rolling(window=window).min()
        df[f"RECENT_HIGH_{window}"] = df["high_price"].rolling(window=window).max()
        return df

    @staticmethod
    def add_divergence(
        df: pd.DataFrame,
        oscillator: str,
        prefix: str,
        pivot_window: int = 3,
        lookback: int = 60,
        min_separation: int = 5,
    ) -> pd.DataFrame:
        """
        Detect regular bullish/bearish divergence using confirmed price pivots.

        A pivot is only usable after ``pivot_window`` subsequent candles have
        closed. This keeps the signal usable in a historical backtest without
        marking a divergence before it could have been observed.
        """
        bullish = [False] * len(df)
        bearish = [False] * len(df)
        if len(df) < (pivot_window * 2 + 1) or oscillator not in df:
            df[f"{prefix}_BULLISH_DIVERGENCE"] = bullish
            df[f"{prefix}_BEARISH_DIVERGENCE"] = bearish
            return df

        prices = pd.to_numeric(df["close_price"], errors="coerce").tolist()
        values = pd.to_numeric(df[oscillator], errors="coerce").tolist()
        confirmed_lows = []
        confirmed_highs = []

        for current in range(pivot_window * 2, len(df)):
            pivot = current - pivot_window
            start = max(0, pivot - pivot_window)
            end = min(len(df), pivot + pivot_window + 1)
            price_window = prices[start:end]
            price = prices[pivot]

            if pd.isna(price) or pd.isna(values[pivot]):
                continue

            if price == min(price_window):
                confirmed_lows.append(pivot)
            if price == max(price_window):
                confirmed_highs.append(pivot)

            cutoff = current - lookback
            confirmed_lows = [index for index in confirmed_lows if index >= cutoff]
            confirmed_highs = [index for index in confirmed_highs if index >= cutoff]

            if len(confirmed_lows) >= 2:
                first, second = confirmed_lows[-2:]
                if (
                    second - first >= min_separation
                    and prices[second] < prices[first]
                    and values[second] > values[first]
                ):
                    bullish[current] = True

            if len(confirmed_highs) >= 2:
                first, second = confirmed_highs[-2:]
                if (
                    second - first >= min_separation
                    and prices[second] > prices[first]
                    and values[second] < values[first]
                ):
                    bearish[current] = True

        df[f"{prefix}_BULLISH_DIVERGENCE"] = bullish
        df[f"{prefix}_BEARISH_DIVERGENCE"] = bearish
        return df

    @classmethod
    def apply_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the indicators actually used by the signal engine and reports."""
        df = cls.add_ema(df, period=20)
        df = cls.add_ema(df, period=50)
        df = cls.add_rsi(df, period=14)
        df = cls.add_macd(df)
        df = cls.add_atr(df, period=14)
        df = cls.add_recent_levels(df, window=20)
        df = cls.add_divergence(df, "RSI_14", "RSI")
        df = cls.add_divergence(df, "MACD", "MACD")
        return df
