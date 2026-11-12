import math

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
        active_bars: int = 10,
    ) -> pd.DataFrame:
        """Flag divergence for a short window after its second pivot is confirmed.

        A divergence is evaluated only when a new pivot is confirmed, and the flag
        stays on for active_bars bars. Without this, an old divergence would keep
        adding to the score until the next pivot appeared.
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
        last_bullish = None
        last_bearish = None

        for current in range(pivot_window * 2, len(df)):
            pivot = current - pivot_window
            start = max(0, pivot - pivot_window)
            end = min(len(df), pivot + pivot_window + 1)
            price_window = prices[start:end]
            price = prices[pivot]

            if pd.isna(price) or pd.isna(values[pivot]):
                continue

            window_complete = not any(pd.isna(p) for p in price_window)
            new_low = window_complete and price == min(price_window)
            new_high = window_complete and price == max(price_window)
            if new_low:
                confirmed_lows.append(pivot)
            if new_high:
                confirmed_highs.append(pivot)

            cutoff = current - lookback
            confirmed_lows = [index for index in confirmed_lows if index >= cutoff]
            confirmed_highs = [index for index in confirmed_highs if index >= cutoff]

            if new_low and len(confirmed_lows) >= 2:
                first, second = confirmed_lows[-2:]
                if (
                    second - first >= min_separation
                    and prices[second] < prices[first]
                    and values[second] > values[first]
                ):
                    last_bullish = current

            if new_high and len(confirmed_highs) >= 2:
                first, second = confirmed_highs[-2:]
                if (
                    second - first >= min_separation
                    and prices[second] > prices[first]
                    and values[second] < values[first]
                ):
                    last_bearish = current

            if last_bullish is not None and current - last_bullish <= active_bars:
                bullish[current] = True
            if last_bearish is not None and current - last_bearish <= active_bars:
                bearish[current] = True

        df[f"{prefix}_BULLISH_DIVERGENCE"] = bullish
        df[f"{prefix}_BEARISH_DIVERGENCE"] = bearish
        return df


    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close_price"].shift(1)
        tr1 = df["high_price"] - df["low_price"]
        tr2 = (df["high_price"] - prev_close).abs()
        tr3 = (df["low_price"] - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Wilder ADX with +DI and -DI: trend strength and direction (0-100)."""
        up = df["high_price"].diff()
        down = -df["low_price"].diff()
        valid = up.notna() & down.notna()
        plus_dm = pd.Series(
            ((up > down) & (up > 0)).astype(float) * up.clip(lower=0), index=df.index
        ).where(valid)
        minus_dm = pd.Series(
            ((down > up) & (down > 0)).astype(float) * down.clip(lower=0), index=df.index
        ).where(valid)

        true_range = TechnicalIndicators._true_range(df)
        atr = TechnicalIndicators._wilder_average(true_range, period)
        plus_avg = TechnicalIndicators._wilder_average(plus_dm, period)
        minus_avg = TechnicalIndicators._wilder_average(minus_dm, period)

        plus_di = 100 * plus_avg / atr
        minus_di = 100 * minus_avg / atr
        di_sum = plus_di + minus_di
        dx = (100 * (plus_di - minus_di).abs() / di_sum).where(di_sum > 0, 0.0)

        df[f"PLUS_DI_{period}"] = plus_di
        df[f"MINUS_DI_{period}"] = minus_di
        df[f"ADX_{period}"] = TechnicalIndicators._wilder_average(dx.where(atr.notna()), period)
        return df

    @staticmethod
    def add_bollinger(
        df: pd.DataFrame, period: int = 20, num_std: float = 2.0, squeeze_lookback: int = 120
    ) -> pd.DataFrame:
        """Bollinger bands, %B, band width and a squeeze flag (narrowest 20% of recent widths)."""
        close = df["close_price"]
        mid = close.rolling(window=period).mean()
        std = close.rolling(window=period).std(ddof=0)
        upper = mid + num_std * std
        lower = mid - num_std * std
        band = upper - lower

        width = (band / mid).where(mid > 0)
        width_floor = width.rolling(
            window=squeeze_lookback, min_periods=squeeze_lookback // 2
        ).quantile(0.2)

        df[f"BB_MID_{period}"] = mid
        df[f"BB_UPPER_{period}"] = upper
        df[f"BB_LOWER_{period}"] = lower
        df[f"BB_PCT_B_{period}"] = ((close - lower) / band).where(band > 0)
        df[f"BB_WIDTH_{period}"] = width
        df[f"BB_SQUEEZE_{period}"] = (width <= width_floor).fillna(False)
        return df

    @staticmethod
    def add_money_flow(df: pd.DataFrame, period: int = 20, mfi_period: int = 14) -> pd.DataFrame:
        """OBV flow, Chaikin Money Flow and Money Flow Index (all use only past and current bars)."""
        high, low, close = df["high_price"], df["low_price"], df["close_price"]
        volume = df["volume"]

        direction = close.diff().apply(lambda x: 0.0 if pd.isna(x) or x == 0 else (1.0 if x > 0 else -1.0))
        obv = (direction * volume).fillna(0).cumsum()
        volume_sum = volume.rolling(window=period).sum()
        df[f"OBV_FLOW_{period}"] = ((obv - obv.shift(period)) / volume_sum).where(volume_sum > 0)

        span = (high - low).where(high > low)
        multiplier = ((close - low) - (high - close)) / span
        multiplier = multiplier.fillna(0.0)
        df[f"CMF_{period}"] = (
            (multiplier * volume).rolling(window=period).sum() / volume_sum
        ).where(volume_sum > 0)

        typical = (high + low + close) / 3
        raw_flow = typical * volume
        change = typical.diff()
        positive = raw_flow.where(change > 0, 0.0).where(change.notna())
        negative = raw_flow.where(change < 0, 0.0).where(change.notna())
        pos_sum = positive.rolling(window=mfi_period).sum()
        neg_sum = negative.rolling(window=mfi_period).sum()
        ratio = pos_sum / neg_sum
        mfi = 100 - (100 / (1 + ratio))
        mfi = mfi.where(~((pos_sum == 0) & (neg_sum == 0)))
        mfi = mfi.where(~((neg_sum == 0) & (pos_sum > 0)), 100.0)
        df[f"MFI_{mfi_period}"] = mfi
        return df

    @staticmethod
    def add_vwap(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Rolling VWAP from TSETMC traded value and volume (value / volume)."""
        if "value" not in df.columns:
            df[f"VWAP_{period}"] = float("nan")
            return df
        value_sum = df["value"].rolling(window=period).sum()
        volume_sum = df["volume"].rolling(window=period).sum()
        df[f"VWAP_{period}"] = (value_sum / volume_sum).where(volume_sum > 0)
        return df

    @staticmethod
    def add_volatility_and_liquidity(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Parkinson range volatility, average traded value, Amihud illiquidity, trade-size ratio."""
        log_range_sq = (df["high_price"] / df["low_price"]).where(
            (df["high_price"] > 0) & (df["low_price"] > 0)
        ).apply(lambda x: math.log(x) ** 2 if pd.notna(x) and x > 0 else float("nan"))
        parkinson_var = log_range_sq.rolling(window=period).mean() / (4 * math.log(2))
        df[f"PARKINSON_VOL_{period}"] = parkinson_var.apply(
            lambda x: math.sqrt(x) if pd.notna(x) and x >= 0 else float("nan")
        )

        if "value" in df.columns:
            value = df["value"]
            df[f"AVG_VALUE_{period}"] = value.rolling(window=period).mean()
            abs_return = df["close_price"].pct_change().abs()
            illiquidity = (abs_return / value).where(value > 0)
            df[f"AMIHUD_{period}"] = illiquidity.rolling(window=period).mean()
        else:
            df[f"AVG_VALUE_{period}"] = float("nan")
            df[f"AMIHUD_{period}"] = float("nan")

        if "value" in df.columns and "count" in df.columns:
            trade_size = (df["value"] / df["count"]).where(df["count"] > 0)
            baseline = trade_size.rolling(window=period).mean().shift(1)
            df[f"TRADE_SIZE_RATIO_{period}"] = (trade_size / baseline).where(baseline > 0)
        else:
            df[f"TRADE_SIZE_RATIO_{period}"] = float("nan")
        return df

    @staticmethod
    def trading_week_key(dates: pd.Series) -> pd.PeriodIndex:
        """Map each date to its Iranian trading week (Saturday to Wednesday).

        Periods end on Wednesday, so Saturday through Wednesday share one key and
        the next Saturday starts a new one.
        """
        parsed = pd.to_datetime(pd.Series(dates).to_numpy(), format="%Y%m%d", errors="coerce")
        return pd.DatetimeIndex(parsed).to_period("W-WED")

    @staticmethod
    def add_weekly_trend(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
        """Trend of the last completed week: close vs EMA(span) of weekly closes.

        A trading week runs Saturday to Wednesday (Thursday and Friday are closed), so
        weeks end on Wednesday. A day only sees weeks that finished before it, so the
        current, still-forming week never leaks into its own trend value.
        Result: +1 above the weekly EMA, -1 below it, 0 when not yet available.
        """
        week = TechnicalIndicators.trading_week_key(df["date"])
        week_key = pd.Series(week, index=df.index)
        weekly_close = df["close_price"].groupby(week_key).last()
        weekly_ema = weekly_close.ewm(span=span, adjust=False).mean()
        completed_ema = weekly_ema.shift(1)

        previous = pd.Series(
            week_key.map(completed_ema).to_numpy(dtype=float), index=df.index
        )
        close = df["close_price"]
        trend = pd.Series(0.0, index=df.index)
        trend = trend.where(previous.isna(), (close > previous).astype(float) * 2 - 1)
        df[f"WEEKLY_TREND_{span}"] = trend.where(previous.notna() & close.notna(), 0.0)
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
        df = cls.add_adx(df, period=14)
        df = cls.add_bollinger(df, period=20)
        df = cls.add_money_flow(df, period=20, mfi_period=14)
        df = cls.add_vwap(df, period=20)
        df = cls.add_volatility_and_liquidity(df, period=20)
        df = cls.add_weekly_trend(df, span=10)
        return df
