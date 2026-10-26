import pandas as pd


class TechnicalIndicators:
    @staticmethod
    def _wilder_average(values: pd.Series, period: int) -> pd.Series:
        """Seed Wilder smoothing with the first period's simple average."""
        result = pd.Series(float("nan"), index=values.index, dtype=float)
        seed_values = []
        average = None

        for position, value in enumerate(values):
            if pd.isna(value):
                seed_values = []
                average = None
                continue

            if average is None:
                seed_values.append(value)
                if len(seed_values) == period:
                    average = sum(seed_values) / period
                    result.iloc[position] = average
            else:
                average = ((average * (period - 1)) + value) / period
                result.iloc[position] = average

        return result

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
        gain = TechnicalIndicators._wilder_average(delta.clip(lower=0), period)
        loss = TechnicalIndicators._wilder_average(-delta.clip(upper=0), period)

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
        """Calculate ATR for volatility-based stop sizing."""
        prev_close = df["close_price"].shift(1)
        tr1 = df["high_price"] - df["low_price"]
        tr2 = (df["high_price"] - prev_close).abs()
        tr3 = (df["low_price"] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f"ATR_{period}"] = TechnicalIndicators._wilder_average(true_range, period)
        return df

    @staticmethod
    def add_recent_levels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Use prior sessions to keep current prices out of support/resistance."""
        df[f"RECENT_LOW_{window}"] = (
            df["low_price"].rolling(window=window).min().shift(1)
        )
        df[f"RECENT_HIGH_{window}"] = (
            df["high_price"].rolling(window=window).max().shift(1)
        )
        return df

    @staticmethod
    def add_breakout_signals(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        resistance = df["high_price"].rolling(window=window).max().shift(1)
        support = df["low_price"].rolling(window=window).min().shift(1)

        df[f"BREAKOUT_UP_{window}"] = (df["close_price"] > resistance).fillna(False)
        df[f"BREAKOUT_DOWN_{window}"] = (df["close_price"] < support).fillna(False)
        df[f"FAKE_BREAKOUT_UP_{window}"] = (
            (df["high_price"] > resistance) & (df["close_price"] <= resistance)
        ).fillna(False)
        df[f"FAKE_BREAKOUT_DOWN_{window}"] = (
            (df["low_price"] < support) & (df["close_price"] >= support)
        ).fillna(False)
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
        """Wait for pivot confirmation to avoid look-ahead in historical backtests."""
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
        df = cls.add_ema(df, period=20)
        df = cls.add_ema(df, period=50)
        df = cls.add_rsi(df, period=14)
        df = cls.add_macd(df)
        df = cls.add_atr(df, period=14)
        df = cls.add_recent_levels(df, window=20)
        df = cls.add_breakout_signals(df, window=20)
        df = cls.add_divergence(df, "RSI_14", "RSI")
        df = cls.add_divergence(df, "MACD", "MACD")
        return df
