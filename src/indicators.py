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

    @classmethod
    def apply_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all base indicators to the data"""
        df = cls.add_sma(df, period=20)
        df = cls.add_sma(df, period=50)
        df = cls.add_ema(df, period=20)
        df = cls.add_rsi(df, period=14)
        df = cls.add_macd(df)
        return df
